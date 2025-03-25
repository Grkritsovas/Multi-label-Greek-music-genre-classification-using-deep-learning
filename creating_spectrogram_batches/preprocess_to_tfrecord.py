#!/usr/bin/env python
import os
import gc
import argparse
import json
import numpy as np
import pandas as pd
import librosa
import random
import tensorflow as tf
from tqdm import tqdm

# Audio Processing Helpers
def trim_silence(y, top_db=20):
    return librosa.effects.trim(y, top_db=top_db)[0]

def segment_audio(y, sr, window_duration=30, hop_duration=15):
    """Segment audio into overlapping segments of given duration."""
    window_length = int(window_duration * sr)
    hop_length_samples = int(hop_duration * sr)
    segments = []
    for start in range(0, len(y) - window_length + 1, hop_length_samples):
        segments.append(y[start:start + window_length])
    return segments

def augment_audio(y, sr):
    """Apply random augmentations to the audio signal for data augmentation."""
    if random.random() > 0.5:
        n_steps = random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    if random.random() > 0.5:
        rate = random.uniform(0.8, 1.2)
        y = librosa.effects.time_stretch(y=y, rate=rate)
    if random.random() > 0.5:
        noise = np.random.randn(len(y))
        noise_amplitude = 0.005 * np.max(np.abs(y))  # Scale noise to signal amplitude
        y = y + noise_amplitude * noise
    return y

def compute_mel_spectrogram(y, sr, global_mean, global_std, n_fft=2048, hop_length=512, n_mels=80):
    """Compute the mel spectrogram in dB and then standardize (z-score)."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                              hop_length=hop_length,
                                              n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - global_mean) / (global_std + 1e-10)  # Use global stats
    return mel_spec_db

# TFRecord Functions
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    """Returns an int64_list from a list of bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_example(mel_spec, target_vector, song_name, segment_idx, n_segments):
    mel_spec_tensor = tf.convert_to_tensor(mel_spec, dtype=tf.float32)
    
    mel_spec_bytes = tf.io.serialize_tensor(mel_spec_tensor).numpy()
    
    feature = {
        'mel_spectrogram': _bytes_feature(mel_spec_bytes),
        'labels': _int64_list_feature(target_vector),
        'song_name': _bytes_feature(song_name.encode('utf-8')),
        'segment_idx': _int64_feature(segment_idx),
        'total_segments': _int64_feature(n_segments),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tfrecord(examples, output_path):
    """Write TFRecord file from a list of examples."""
    with tf.io.TFRecordWriter(output_path) as writer:
        for example in examples:
            writer.write(example.SerializeToString())

def save_metadata(metadata, output_dir):
    """Save metadata about the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")
    
    # Save a TF-friendly feature description
    feature_description = {
        'mel_spectrogram': 'bytes',
        'labels': 'int64_list',
        'song_name': 'bytes',
        'segment_idx': 'int64',
        'total_segments': 'int64',
    }
    
    feature_path = os.path.join(output_dir, "feature_description.json")
    with open(feature_path, 'w') as f:
        json.dump(feature_description, f, indent=2)
    print(f"Feature description saved to {feature_path}")

def get_next_file_index(directory):
    """Get the next file index for writing TFRecord files."""
    files = [f for f in os.listdir(directory) if f.endswith('.tfrecord')]
    if not files:
        return 0
    
    # Extract the indices from filenames (train_XXXX.tfrecord or test_XXXX.tfrecord)
    existing_indices = [int(f.split('_')[1].split('.')[0]) for f in files]
    return max(existing_indices) + 1 if existing_indices else 0

def load_progress(progress_file):
    """Load the current progress from a JSON file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"current_index": 0}
    return {"current_index": 0}

def save_progress(progress_file, progress_data):
    """Save the current progress to a JSON file."""
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def process_songs_to_tfrecord(
    songs_json_path,
    audio_folders,
    downloaded_mapping_path,
    output_dir,
    split_name="train",
    global_stats_path="../json/global_stats.json",
    progress_file=None,
    apply_augmentation=False,
    sr=22050,
    window_duration=14,
    hop_duration=7,
    batch_size=50,
    examples_per_file=1000
):
    """Process a JSON file containing song names and labels into TFRecord files"""

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    

    with open(songs_json_path, 'r') as f:
        songs_data = json.load(f)
    

    with open(downloaded_mapping_path, 'r') as f:
        downloaded_mapping = json.load(f)
    
    # Create a reverse mapping from local filename to YouTube link
    filename_to_youtube = {v: k for k, v in downloaded_mapping.items()}

    with open(global_stats_path, 'r') as f:
        stats = json.load(f)
        global_mean, global_std = stats["mean"], stats["std"]
    
    if progress_file is None:
        progress_file = os.path.join(output_dir, f"{split_name}_progress.json")
    
    progress = load_progress(progress_file)
    start_index = progress["current_index"]
    
    song_names = list(songs_data.keys())
    
    if start_index >= len(song_names):
        print(f"All {len(song_names)} songs have already been processed.")
        return
    
    # Find the next file index for TFRecord files
    file_idx = get_next_file_index(split_dir)
    
    song_to_segments_mapping = {}
    examples_buffer = []
    successful_songs = []
    failed_songs = []
    
    # Process songs in batches
    for batch_start in range(start_index, len(song_names), batch_size):
        batch_end = min(batch_start + batch_size, len(song_names))
        batch_songs = song_names[batch_start:batch_end]
        
        print(f"Processing {len(batch_songs)} songs (index {batch_start} to {batch_end-1})...")
        
        for song_name in tqdm(batch_songs):

            target_vector = songs_data[song_name]
            
            file_path = None
            
            # Try to find YouTube link for the song
            youtube_link = None
            for link, local_name in downloaded_mapping.items():
                if os.path.basename(local_name) in song_name or song_name in os.path.basename(local_name):
                    youtube_link = link
                    break
            
            if youtube_link and youtube_link in downloaded_mapping:
                candidate_file = os.path.basename(downloaded_mapping[youtube_link])
                for folder in audio_folders:
                    candidate_path = os.path.join(folder, candidate_file)
                    if os.path.exists(candidate_path):
                        file_path = candidate_path
                        break
            
            # If file not found, try to match by name
            if file_path is None:
                for folder in audio_folders:
                    for filename in os.listdir(folder):
                        if song_name.lower() in filename.lower():
                            file_path = os.path.join(folder, filename)
                            break
                    if file_path:
                        break
            
            if file_path is None:
                print(f"[ERROR] File not found for '{song_name}'")
                failed_songs.append(song_name)
                continue
            
            try:
                # Load and process the audio file
                y, sr_loaded = librosa.load(file_path, sr=sr)
                y = trim_silence(y, top_db=20)
                
                # Process original segments
                segments_orig = segment_audio(y, sr_loaded, window_duration, hop_duration)
                song_to_segments_mapping[song_name] = len(segments_orig)
                
                # Process and add original segments to TFRecord
                for idx, segment in enumerate(segments_orig):
                    mel_spec = compute_mel_spectrogram(segment, sr_loaded, global_mean, global_std)
                    example = create_example(mel_spec, target_vector, song_name, idx, len(segments_orig))
                    examples_buffer.append(example)
                    
                    # Write to file if buffer is full
                    if len(examples_buffer) >= examples_per_file:
                        output_path = os.path.join(split_dir, f"{split_name}_{file_idx:04d}.tfrecord")
                        write_tfrecord(examples_buffer, output_path)
                        print(f"Wrote {len(examples_buffer)} examples to {output_path}")
                        examples_buffer = []
                        file_idx += 1
                
                # Apply augmentation (for training data)
                if apply_augmentation:
                    aug_song_name = f"{song_name}_aug"
                    
                    # Process and add augmented segments to TFRecord
                    for idx, segment in enumerate(segments_orig):
                        # Apply augmentation to each segment
                        aug_segment = augment_audio(segment, sr_loaded)
                        mel_spec = compute_mel_spectrogram(aug_segment, sr_loaded, global_mean, global_std)
                        example = create_example(mel_spec, target_vector, aug_song_name, idx, len(segments_orig))
                        examples_buffer.append(example)
                        
                        if len(examples_buffer) >= examples_per_file:
                            output_path = os.path.join(split_dir, f"{split_name}_{file_idx:04d}.tfrecord")
                            write_tfrecord(examples_buffer, output_path)
                            print(f"Wrote {len(examples_buffer)} examples to {output_path}")
                            examples_buffer = []
                            file_idx += 1
                    
                    song_to_segments_mapping[aug_song_name] = len(segments_orig)
                    successful_songs.append(aug_song_name)
                
                successful_songs.append(song_name)
                print(f"Processed '{song_name}' with {len(segments_orig)} segments.")
                
                # Clean up for memory management
                del y, sr_loaded, segments_orig
                gc.collect()
                
            except Exception as e:
                print(f"[ERROR] Processing '{song_name}': {e}")
                failed_songs.append(song_name)
                continue
        
        # Update progress after each batch
        progress["current_index"] = batch_end
        save_progress(progress_file, progress)
        print(f"Progress updated: {batch_end}/{len(song_names)} songs processed.")
    
    # Write any remaining examples
    if examples_buffer:
        output_path = os.path.join(split_dir, f"{split_name}_{file_idx:04d}.tfrecord")
        write_tfrecord(examples_buffer, output_path)
        print(f"Wrote {len(examples_buffer)} examples to {output_path}")
    
    # Save segment mapping
    mapping_path = os.path.join(output_dir, f"{split_name}_song_segments.json")
    with open(mapping_path, 'w') as f:
        json.dump(song_to_segments_mapping, f, indent=2)
    
    # Save success/failure lists
    successful_path = os.path.join(output_dir, f"{split_name}_successful_songs.json")
    with open(successful_path, 'w') as f:
        json.dump(successful_songs, f, indent=2)
    
    failed_path = os.path.join(output_dir, f"{split_name}_failed_songs.json")
    with open(failed_path, 'w') as f:
        json.dump(failed_songs, f, indent=2)
    
    print(f"Processing complete. Processed {len(successful_songs)} songs successfully.")
    print(f"Failed to process {len(failed_songs)} songs.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process songs into TFRecord files containing mel spectrograms."
    )
    parser.add_argument("--songs_json", type=str, required=True, 
                        help="Path to the JSON file with song names and labels")
    parser.add_argument("--output_dir", type=str, default="tfrecord_dataset", 
                        help="Output directory for TFRecord files")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="Data split to process (train or test)")
    parser.add_argument("--augment", action="store_true", 
                        help="Apply audio augmentation (for training data)")
    parser.add_argument("--window_duration", type=float, default=14.0,
                        help="Duration of each segment in seconds")
    parser.add_argument("--hop_duration", type=float, default=7.0,
                        help="Hop length for segmentation in seconds")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Number of songs to process at once")
    parser.add_argument("--examples_per_file", type=int, default=1000,
                        help="Number of examples to store in each TFRecord file")
    parser.add_argument("--progress_file", type=str, default=None,
                        help="Path to the progress file")
    downloaded_mapping = "../json/downloaded_mapping_wav_updated.json" #mapping of YouTube links to local files
    audio_folders= ["../../downloads_wav_1", "../../downloads_wav_2"]
    global_stats = "../json/global_stats.json" # global mean and std
    
    args = parser.parse_args()
    
    # Generate a TFRecord feature specs file if it doesn't exist
    if not os.path.exists(os.path.join(args.output_dir, "tfrecord_feature_specs.py")):
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "tfrecord_feature_specs.py"), 'w') as f:
            f.write("""import tensorflow as tf

def get_feature_description():
    feature_description = {
        'mel_spectrogram': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.VarLenFeature(tf.int64),
        'song_name': tf.io.FixedLenFeature([], tf.string),
        'segment_idx': tf.io.FixedLenFeature([], tf.int64),
        'total_segments': tf.io.FixedLenFeature([], tf.int64),
    }
    return feature_description

def parse_tfrecord_fn(example):
    feature_description = get_feature_description()
    example = tf.io.parse_single_example(example, feature_description)
    
    mel_spec = tf.io.parse_tensor(example['mel_spectrogram'], out_type=tf.float32)
    
    labels = tf.sparse.to_dense(example['labels'])
    
    return mel_spec, labels

def get_dataset(tfrecord_files, batch_size=32):
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
""")
    
    process_songs_to_tfrecord(
        songs_json_path=args.songs_json,
        audio_folders=audio_folders,
        downloaded_mapping_path=downloaded_mapping,
        output_dir=args.output_dir,
        split_name=args.split,
        global_stats_path=global_stats,
        progress_file=args.progress_file,
        apply_augmentation=args.augment,
        sr=22050,
        window_duration=args.window_duration,
        hop_duration=args.hop_duration,
        batch_size=args.batch_size,
        examples_per_file=args.examples_per_file
    )
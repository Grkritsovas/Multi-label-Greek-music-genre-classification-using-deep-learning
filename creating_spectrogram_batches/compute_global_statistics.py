#!/usr/bin/env python
import os
import json
import numpy as np
import pandas as pd
import librosa
import argparse

# ---------------- Data Loading Functions ----------------
def load_df(file_path):
    """Load a CSV or Excel dataset."""
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)

def load_downloaded_mapping(mapping_path):
    """Load the downloaded mapping from a JSON file."""
    with open(mapping_path, 'r') as fp:
        return json.load(fp)

# ---------------- Audio Processing Helpers ----------------
def trim_silence(y, sr, top_db=20):
    """Trim silence from an audio signal."""
    return librosa.effects.trim(y, top_db=top_db)[0]

def compute_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=80):
    """Compute the mel spectrogram in dB."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                              hop_length=hop_length,
                                              n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# ---------------- Main Function to Compute Global Stats ----------------
def compute_global_stats(
    df,
    audio_folders,
    sample_size=100,
    sr=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=80,
    output_file="../json/global_stats.json"
):
    """Compute global mean and std of mel spectrograms across a sample of songs."""
    means = []
    stds = []
    sample_df = df.sample(min(sample_size, len(df)))  # Use smaller of sample_size or dataset size
    for idx, row in sample_df.iterrows():
        file_path = None
        candidate_file = os.path.basename(row['local_filename'])
        for folder in audio_folders:
            candidate_path = os.path.join(folder, candidate_file)
            if os.path.exists(candidate_path):
                file_path = candidate_path
                break
        if not file_path:
            print(f"File not found for '{row['local_filename']}'")
            continue
        try:
            y, sr_loaded = librosa.load(file_path, sr=sr)
            y = trim_silence(y, sr_loaded, top_db=20)
            mel_spec_db = compute_mel_spectrogram(y, sr_loaded, n_fft, hop_length, n_mels)
            means.append(np.mean(mel_spec_db))
            stds.append(np.std(mel_spec_db))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    if means and stds:
        global_mean = np.mean(means)
        global_std = np.mean(stds)
        with open(output_file, "w") as f:
            json.dump({"mean": float(global_mean), "std": float(global_std)}, f, indent=2)
        print(f"Global mean: {global_mean}, Global std: {global_std} saved to '{output_file}'.")
    else:
        print("No valid spectrograms processed.")

# ---------------- Main Script Execution ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute global mean and std for mel spectrograms."
    )
    parser.add_argument("--sample_size", type=int, default=100, help="Number of songs to sample for stats")
    args = parser.parse_args()

    excel_file_path = "../../datasets/Greek_Music_Dataset.xlsx"
    mapping_path = '../json/downloaded_mapping_wav_updated.json'
    audio_folders = ["../../downloads_wav_1", "../../downloads_wav_2"]

    df = load_df(excel_file_path)
    downloaded_mapping = load_downloaded_mapping(mapping_path)
    df['downloaded'] = df["YouTube Link"].apply(lambda x: x in downloaded_mapping)
    df = df[df['downloaded']].copy()
    df['local_filename'] = df["YouTube Link"].map(downloaded_mapping)

    compute_global_stats(
        df=df,
        audio_folders=audio_folders,
        sample_size=args.sample_size
    )
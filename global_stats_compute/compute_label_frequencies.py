# weights_calculator.py

import tensorflow as tf
import numpy as np
import json
import os
import glob

def get_feature_description():
    feature_description = {
        'mel_spectrogram': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.VarLenFeature(tf.int64),
        'song_name': tf.io.FixedLenFeature([], tf.string),
        'segment_idx': tf.io.FixedLenFeature([], tf.int64),
        'total_segments': tf.io.FixedLenFeature([], tf.int64),
    }
    return feature_description

def parse_tfrecord_fn(example, normalize=False, mean_global=None, std_global=None):
    feature_description = get_feature_description()
    example = tf.io.parse_single_example(example, feature_description)
    
    mel_spec = tf.io.parse_tensor(example['mel_spectrogram'], out_type=tf.float32)
    
    # Extract labels only
    labels = tf.sparse.to_dense(example['labels'])
    labels = tf.ensure_shape(labels, [8])
    
    return labels

def compute_label_frequencies(tfrecord_files):
    """Compute label frequencies from tfrecord files"""
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord_fn)
    
    total_counts = np.zeros(8)
    total_samples = 0
    
    for labels in dataset:
        total_counts += labels.numpy()
        total_samples += 1
        
    return total_counts, total_samples

def compute_class_weights(total_counts, total_samples, num_labels=8):
    """Compute class weights based on frequencies"""
    class_weights = total_samples / (num_labels * (total_counts + 1e-6))
    return class_weights

def calculate_and_save_weights(data_dir, segment_length, output_dir="../json"):
    """Calculate and save weights for a specific segment length"""

    os.makedirs(output_dir, exist_ok=True)
    
    # Find all tfrecord files
    train_files = glob.glob(f"{data_dir}/train/*.tfrecord")
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    counts, total = compute_label_frequencies(train_files) #label frequencies
    
    
    class_weights = compute_class_weights(counts, total) #class weights
    
    # weight for label = 0
    neg_counts = total - counts
    neg_weights = compute_class_weights(neg_counts, total)
    
    # Combine into (num_labels, 2) format
    class_weights_combined = np.stack([neg_weights, class_weights], axis=1)
    
    results = {
        "total_examples": int(total),
        "label_counts": counts.tolist(),
        "pos_weights": class_weights.tolist(),
        "neg_weights": neg_weights.tolist(),
        "combined_weights": class_weights_combined.tolist()
    }
    
    output_file = f"{output_dir}/class_weights/class_weights_{segment_length}s.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved weights for {segment_length}s segments to {output_file}")
    return results

def main():
    # Calculate for different segment lengths
    segment_lengths = [3, 6, 10, 14, 20, 30]
    for length in segment_lengths:
        data_dir = f"/home/georgios/Music Analysis/dataset_creation/tfrecord_dataset_{length}s"
        calculate_and_save_weights(data_dir, length)

if __name__ == "__main__":
    main()
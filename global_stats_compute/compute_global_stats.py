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

def parse_tfrecord_fn(example):
    feature_description = get_feature_description()
    example = tf.io.parse_single_example(example, feature_description)
    
    mel_spec = tf.io.parse_tensor(example['mel_spectrogram'], out_type=tf.float32)
    
    
    return mel_spec

def compute_global_stats(tfrecord_files):
    # Create dataset from TFRecord files
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(lambda x: parse_tfrecord_fn(x))
    
    count = 0
    sum_values = 0
    sum_squares = 0
    
    # Process one example at a time
    for mel_spec in dataset:
        # Flatten the mel spectrogram
        mel_flat = tf.reshape(mel_spec, [-1])
        count += tf.cast(tf.size(mel_flat), tf.float64)
        sum_values += tf.reduce_sum(tf.cast(mel_flat, tf.float64))
        sum_squares += tf.reduce_sum(tf.square(tf.cast(mel_flat, tf.float64)))
    
    # Calculate mean and std
    mean_global = sum_values / count
    variance = (sum_squares / count) - tf.square(mean_global)
    std_global = tf.sqrt(variance)
    
    return mean_global.numpy(), std_global.numpy()

if __name__ == '__main__':
    # Find all TFRecord files
    segment_lengths = [3, 6, 14, 20, 30] #already did 10
    for length in segment_lengths:
        data_dir = f"/home/georgios/Music Analysis/dataset_creation/tfrecord_dataset_{length}s"

        train_files = glob.glob(f"{data_dir}/train/*.tfrecord")
    
        mean_global, std_global = compute_global_stats(train_files)
    
        print(f"Global mean: {mean_global}")
        print(f"Global standard deviation: {std_global}")
    
        stats = {
            'mean': float(mean_global),
            'std': float(std_global)
        }
        
        with open(f'../json/global_norm_stats/global_norm_stats_{length}s.json', 'w') as f:
            json.dump(stats, f)
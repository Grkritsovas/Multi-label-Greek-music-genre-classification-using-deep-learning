# utils_glob_stats.py

import json
import tensorflow as tf

def load_class_weights(segment_length, json_dir="json/class_weights"):
    """Load class weights for a specific segment length"""
    try:
        with open(f"{json_dir}/class_weights_{segment_length}s.json", "r") as f:
            data = json.load(f)
            
        # Convert to TensorFlow tensors for the model
        combined_weights = tf.constant(data["combined_weights"], dtype=tf.float32)
        
        return {
            "total_examples": data["total_examples"],
            "combined_weights": combined_weights,
            "combined_weights_numpy": data["combined_weights"],
            "pos_weights": data["pos_weights"],
            "neg_weights": data["neg_weights"],
            "label_counts": data["label_counts"]
        }
    except FileNotFoundError:
        raise ValueError(f"Weights for {segment_length}s segments not found.")
    
def load_global_stats(segment_length, json_dir="json/global_norm_stats"):
    try:
        with open(f"{json_dir}/global_norm_stats_{segment_length}s.json", "r") as f:
            data = json.load(f)
        
        return {
            "mean": data["mean"],
            "std": data["std"]
        }
    except FileNotFoundError:
        raise ValueError(f"Stats for {segment_length}s segments not found.")
    

def load_time_axis_stats(segment_length, json_dir="json/average_time_axis"):
    """Load time axis statistics for a specific segment length"""
    try:
        with open(f"{json_dir}/average_time_axis_{segment_length}s.json", "r") as f:
            data = json.load(f)
        
        return data
    except FileNotFoundError:
        raise ValueError(f"Time axis stats for {segment_length}s segments not found.")
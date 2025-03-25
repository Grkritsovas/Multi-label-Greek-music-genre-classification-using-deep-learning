#!/usr/bin/env python

import os
import json
import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

# 1) Build the list of actual songs from directories, detect duplicates

def get_song_names_from_dirs(audio_folders, extension=".wav"):
    """
    Scans the given folders for audio files with the specified extension.
    Returns:
      - song_name_list: a list of all found base filenames (no extension).
      - prints any duplicates as it finds them.
    """
    song_counts = {}
    
    for folder in audio_folders:
        if not os.path.isdir(folder):
            print(f"[WARNING] Folder does not exist: {folder}")
            continue
        
        for f in os.listdir(folder):
            if f.lower().endswith(extension):
                base_name = os.path.splitext(f)[0]
                song_counts[base_name] = song_counts.get(base_name, 0) + 1
    
    # Print duplicates in the directories
    for song, count in song_counts.items():
        if count > 1:
            print(f"[DIRECTORY DUPLICATE] The song '{song}' appears {count} times in the directories.")
    
    return list(song_counts.keys())


# 2) Clean up the DataFrame:
# Only keep rows that correspond to songs physically existing in directories
# If the same song appears multiple times in the DataFrame, keep the first

def filter_df_by_directory_songs(df, song_col, valid_songs):
    """
    1) Keep only rows whose 'song_col' is in valid_songs.
    2) Drop duplicates for the 'song_col' if they exist multiple times.
    3) Return the cleaned DataFrame.
    """
    # Filter to only songs that physically exist
    filtered_df = df[df[song_col].isin(valid_songs)].copy()
    
    # Check for duplicates in the DataFrame
    duplicated_songs = filtered_df[filtered_df.duplicated(song_col, keep=False)][song_col].unique()
    if len(duplicated_songs) > 0:
        for dup_song in duplicated_songs:
            print(f"[DATAFRAME DUPLICATE] The song '{dup_song}' appears multiple times in the DataFrame.")
    
    # Keep only the first occurrence for each song
    filtered_df.drop_duplicates(subset=[song_col], keep='first', inplace=True)
    
    return filtered_df


# 3) Convert label columns to 0/1, then do iterative train/test split

def clean_label(val):
    """Helper to convert any 'Yes'/1/True-like value into 1, else 0."""
    if pd.isna(val):
        return 0
    if isinstance(val, str):
        return 1 if val.strip().lower() in ("yes", "true", "1") else 0
    return 1 if bool(val) else 0


def stratified_split_multilabel(df, song_col, label_cols, test_size=0.15, random_seed=42):
    """
    Does a multi-label (iterative) stratified split:
      - label_cols: list of columns holding 0/1 indicators
      - test_size: fraction for test
    Returns: (train_df, test_df)
    """
    np.random.seed(random_seed)
    
    # Make label matrix
    labels = df[label_cols].applymap(clean_label).values  # shape = (num_samples, num_labels)
    
    # X is just the song names in a 2D array for iterative_train_test_split
    X = df[song_col].values.reshape(-1, 1)
    Y = labels
    
    X_train, Y_train, X_test, Y_test = iterative_train_test_split(X, Y, test_size=test_size)
    
    # Build back DataFrames
    train_songs = X_train.flatten()
    test_songs = X_test.flatten()
    
    train_df = df[df[song_col].isin(train_songs)].copy()
    test_df  = df[df[song_col].isin(test_songs)].copy()
    
    return train_df, test_df

if __name__ == "__main__":

    dataset_path = "../../datasets/Greek_Music_Dataset.xlsx"
    
    # Audio folders
    audio_folders = ["../../downloads_wav_1", "../../downloads_wav_2"]
    
    # Columns that represent the different genre labels
    genre_order = ["LAIKO", "REMPETIKO", "ENTEXNO", "ROCK", "Mod LAIKO", "POP", "ENALLAKTIKO", "HIPHOP/RNB"]
    
    if dataset_path.endswith(".xlsx"):
        df = pd.read_excel(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    
    song_col = "Song"
    
    # Step 1: Gather actual songs from directories, detect duplicates
    valid_songs = get_song_names_from_dirs(audio_folders, extension=".wav")
    print(f"[INFO] Found {len(valid_songs)} unique song files across all directories.")
    
    # Step 2: Filter the DataFrame so it only keeps those songs that exist on disk
    # and drop duplicates in the DataFrame
    filtered_df = filter_df_by_directory_songs(df, song_col, valid_songs)
    print(f"After filtering by directory presence, there are {len(filtered_df)} songs in the DataFrame.")
    
    # Step 3: Stratified multi-label split
    train_df, test_df = stratified_split_multilabel(
        filtered_df,
        song_col=song_col,
        label_cols=genre_order,
        test_size=0.15,   # 15% test
        random_seed=42
    )
    
    print(f"[INFO] Train set size: {len(train_df)} | Test set size: {len(test_df)}")
    
    # Step 4: Output to JSON files dict of {song_name: {label_col: 0/1, ...}}
    
    def df_to_song_label_dict(in_df, genre_order, song_col):
        """ Converts DataFrame to a dictionary with song names as keys and binary-encoded genre lists as values"""
        out_dict = {}
        for _, row in in_df.iterrows():
            sname = row[song_col]
            labels = [clean_label(row[g]) for g in genre_order]  # Convert to binary list
            out_dict[sname] = labels
        return out_dict

    
    train_data_dict = df_to_song_label_dict(train_df, genre_order, song_col)
    test_data_dict = df_to_song_label_dict(test_df, genre_order, song_col)

    with open("train_songs.json", "w", encoding="utf-8") as f:
        json.dump(train_data_dict, f, indent=2, ensure_ascii=False)
    print("[INFO] Wrote train_songs.json")

    with open("test_songs.json", "w", encoding="utf-8") as f:
        json.dump(test_data_dict, f, indent=2, ensure_ascii=False)
    print("[INFO] Wrote test_songs.json")

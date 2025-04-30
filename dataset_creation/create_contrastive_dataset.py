import os, json, hashlib, librosa, tensorflow as tf
from glob import glob
from tqdm import tqdm
import numpy as np

SEG_DUR   = 10        # seconds
CLIPS_PER = 4         # how many 10-s clips per song
N_MELS    = 96
SR        = 22_050

# ---------- TF helpers ----------
def _bytes(x):  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
def _int64(x):  return tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))

def make_example(mel, song_id, clip_idx):
    feat = {
        "mel":      _bytes(tf.io.serialize_tensor(tf.convert_to_tensor(mel, tf.float32)).numpy()),
        "song_id":  _bytes(song_id.encode()),
        "clip_idx": _int64(clip_idx),
    }
    return tf.train.Example(features=tf.train.Features(feature=feat))

# ---------- audio ----------
def centre_clips(y, sr, seg_sec=SEG_DUR, n_clips=CLIPS_PER):
    seg_len = seg_sec*sr
    if len(y) < seg_len*2:          # need at least 2 clips to form a pair
        return []
    # total window to grab:
    total = seg_len * n_clips
    if total > len(y):              # shrink n_clips if song too short
        n_clips = len(y)//seg_len
        total  = seg_len*n_clips
    start = (len(y) - total)//2
    clips = [y[start+i*seg_len : start+(i+1)*seg_len] for i in range(n_clips)]
    return clips

def mel_db(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512,
                                       n_mels=N_MELS)
    return librosa.power_to_db(S, ref=np.max)

# ---------- main ----------
def main(audio_root, stats_json, out_dir, per_file=1024):
    os.makedirs(out_dir, exist_ok=True)
    stats = json.load(open(stats_json))
    mean, std = stats["mean"], stats["std"]

    writer  = None
    idx_in_file = file_idx = 0

    audio_paths = sorted(sum([glob(os.path.join(audio_root, "**/*"+ext), recursive=True)
                              for ext in [".mp3",".wav",".flac",".ogg",".m4a"]], []))
    print(f"{len(audio_paths)} audio files found")

    for path in tqdm(audio_paths):
        try:
            y,_ = librosa.load(path, sr=SR)
            y   = librosa.effects.trim(y, top_db=20)[0]
            clips = centre_clips(y, SR)
            if len(clips)<2: continue

            song_id = hashlib.md5(path.encode()).hexdigest()
            for cidx,clip in enumerate(clips):
                mel = (mel_db(clip, SR) - mean)/ (std+1e-10)
                ex  = make_example(mel, song_id, cidx)

                if writer is None:
                    tfrec = os.path.join(out_dir,f"contrast_{file_idx:04}.tfrecord")
                    writer = tf.io.TFRecordWriter(tfrec)
                    idx_in_file = 0

                writer.write(ex.SerializeToString()); idx_in_file += 1
                if idx_in_file >= per_file:
                    writer.close(); writer=None; file_idx+=1
        except Exception as e:
            print("ERR",path,e)

    if writer: writer.close()
    print("Done")

def write_feature_spec(out_dir):
    spec_code = """\
import tensorflow as tf

def get_feature_description():
    return {
        'mel_spectrogram': tf.io.FixedLenFeature([], tf.string),
        'song_id':         tf.io.FixedLenFeature([], tf.string),
        'clip_idx':        tf.io.FixedLenFeature([], tf.int64),
    }

def parse_tfrecord(example):
    feat = tf.io.parse_single_example(example, get_feature_description())
    mel  = tf.io.parse_tensor(feat['mel_spectrogram'], out_type=tf.float32)
    mel  = tf.expand_dims(mel, -1)      # (96,T,1)
    return mel, feat['song_id']
"""
    path = os.path.join(out_dir, "tfrecord_feature_specs_contrastive.py")
    with open(path, "w") as f:
        f.write(spec_code)
    print(f"✓ wrote feature-specs helper →  {path}")

if __name__ == "__main__":
    import argparse, pathlib, sys
    p=argparse.ArgumentParser()
    p.add_argument("audio_root", help="folder with mp3/wav etc.")
    p.add_argument("stats_json", help='json file: {"mean":float,"std":float}')
    p.add_argument("--out_dir", default="contrast_tfrecords")
    args=p.parse_args()
    main(**vars(args))
    write_feature_spec(args.out_dir)

#!/usr/bin/env python3
"""
Create TFRecords with (mel, multi-hot label) pairs.

finds audio by YouTube-link map or fuzzy filename
segments each song (window, hop)
optional augmentation
progress resume
rolling TFRecord files  (train_0000.tfrecord …)
"""

import json, random, gc, argparse, hashlib
from pathlib import Path
from typing   import List, Dict, Tuple
from functools import partial

import librosa, numpy as np, tensorflow as tf
from tqdm import tqdm


# audio helpers
def trim(y, top_db=20):               # remove leading / trailing silence
    return librosa.effects.trim(y, top_db=top_db)[0]

def segment(y, sr, win_s, hop_s) -> List[np.ndarray]:
    """Return non-overlapping segments of length `win_s` every `hop_s`."""
    win, hop = int(win_s*sr), int(hop_s*sr)
    idx = range(0, len(y)-win+1, hop)
    return [y[i:i+win] for i in idx]

def augment_audio(y, sr):
    if random.random() > .5:
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=random.uniform(-2,2))
    if random.random() > .5:
        y = librosa.effects.time_stretch(y=y, rate=random.uniform(.8,1.2))
    if random.random() > .5:
        y = y + 0.005*y.std()*np.random.randn(len(y))
    return y

def mel_db(y, sr, n_mels=96):
    S  = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512,
                                        n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max, top_db=None)


# TF helpers
def _bytes(x):  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
def _int_list(x): return tf.train.Feature(int64_list=tf.train.Int64List(value=x))
def _int(x):   return tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))

def make_example(mel:np.ndarray,
                 labels:List[int],
                 song:str,
                 idx:int,
                 total:int)->tf.train.Example:
    mel_raw = tf.io.serialize_tensor(tf.convert_to_tensor(mel, tf.float32)).numpy()
    feat = {
        "mel":           _bytes(mel_raw),
        "labels":        _int_list(labels),
        "song":          _bytes(song.encode()),
        "segment_idx":   _int(idx),
        "total_segments":_int(total)
    }
    return tf.train.Example(features=tf.train.Features(feature=feat))


#  main pipeline
def build_path_index(audio_dirs:List[Path])->Dict[str,Path]:
    """map basename → full Path for quick lookup"""
    idx = {}
    for d in audio_dirs:
        for p in d.rglob("*"):
            if p.suffix.lower() in {".wav",".mp3",".flac",".ogg",".m4a"}:
                idx[p.name] = p
    return idx

def resolve_song(name:str,
                 yt_map:Dict[str,str],
                 path_index:Dict[str,Path])->Path|None:
    """1) exact YouTube map  2) basename match"""
    # by YouTube link
    for link, local in yt_map.items():
        if name in link or name in local:
            p = Path(local).name
            if p in path_index: return path_index[p]
    # fuzzy filename search
    for base,p in path_index.items():
        if name.lower() in base.lower(): return p
    return None


def write_buffer(buf:List[tf.train.Example], out_dir:Path,
                 split:str, file_idx:int)->int:
    """Dump buffer to TFRecord and return next index"""
    if not buf: return file_idx
    fn = out_dir/f"{split}_{file_idx:04d}.tfrecord"
    with tf.io.TFRecordWriter(str(fn)) as w:
        for ex in buf: w.write(ex.SerializeToString())
    print(f" ⇢  wrote {len(buf)} examples → {fn}")
    buf.clear()
    return file_idx+1


def process_dataset(
        songs_json:Path, audio_dirs:List[Path], yt_map_path:Path,
        out_root:Path, split:str, *,
        win=14, hop=7, sr=22_050, ex_per_file=1000,
        augment=False, resume_file=None):

    out_dir = out_root/split; out_dir.mkdir(parents=True, exist_ok=True)
    songs   = json.load(open(songs_json))
    yt_map  = json.load(open(yt_map_path))
    index   = build_path_index(audio_dirs)

    # progress
    resume_file = resume_file or out_root/f"{split}_progress.json"
    done_i = json.load(open(resume_file))["idx"] if Path(resume_file).exists() else 0

    buf, file_idx = [], len(list(out_dir.glob("*.tfrecord")))
    failed=[]
    for i,(song,labels) in enumerate(list(songs.items())[done_i:], start=done_i):
        p = resolve_song(song, yt_map, index)
        if p is None: failed.append(song); continue

        try:
            y,_ = librosa.load(p, sr=sr); y=trim(y)
            segs = segment(y,sr,win,hop)
            for sidx,s in enumerate(segs):
                for data,is_aug in ((s,False),(augment_audio(s,sr),True)) if augment else ((s,False),):
                    mel = mel_db(data,sr)
                    buf.append(make_example(mel, labels, song + ("_aug" if is_aug else ""),
                                            sidx,len(segs)))
                    if len(buf)>=ex_per_file:
                        file_idx=write_buffer(buf,out_dir,split,file_idx)
            if i%10==0:
                json.dump({"idx":i},open(resume_file,"w"))
        except Exception as e:
            print("err",song,e); failed.append(song)

    write_buffer(buf,out_dir,split,file_idx)
    json.dump({"idx":len(songs)},open(resume_file,"w"))
    json.dump(failed,open(out_root/f"{split}_failed.json","w"),indent=2)
    print(f"✓ done  {len(songs)-len(failed)} ok  /  {len(failed)} failed")


# CLI
if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--songs_json",required=True)
    pa.add_argument("--audio_dirs",nargs="+",required=True)
    pa.add_argument("--yt_map",required=True, help="downloaded_mapping_wav_updated.json")
    pa.add_argument("--output_dir",default="tfrecord_dataset")
    pa.add_argument("--split",default="train",choices=["train","test"])
    pa.add_argument("--window",type=float,default=14)
    pa.add_argument("--hop",type=float,default=7)
    pa.add_argument("--augment",action="store_true")
    pa.add_argument("--examples_per_file",type=int,default=1000)
    args=pa.parse_args()

    process_dataset(
        songs_json = Path(args.songs_json),
        audio_dirs = [Path(d) for d in args.audio_dirs],
        yt_map_path= Path(args.yt_map),
        out_root   = Path(args.output_dir),
        split      = args.split,
        win        = args.window,
        hop        = args.hop,
        augment    = args.augment,
        ex_per_file= args.examples_per_file)
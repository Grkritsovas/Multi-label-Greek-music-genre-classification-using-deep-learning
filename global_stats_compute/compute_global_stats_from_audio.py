#Pass over all audio, compute global mean / std of dB-scaled 96-mel spectrograms.
#Writes {"mean": float, "std": float} to stats.json
import librosa, numpy as np, json, os, sys
from glob import glob
from tqdm import tqdm

SR        = 22_050
N_MELS    = 96
SEG_SEC   = 10
CLIPS_PER = 6            # must match contrastive writer!

def centre_clips(y, sr):
    seg_len = SEG_SEC * sr
    if len(y) < 2*seg_len:      # need ≥2 clips → skip file
        return []
    n = min(CLIPS_PER, len(y)//seg_len)
    start = (len(y) - n*seg_len)//2
    return [y[start+i*seg_len : start+(i+1)*seg_len] for i in range(n)]

def mel_db(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048,
                                       hop_length=512, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=np.max)

def main(audio_roots, out_json):
    # Collect all audio files from all provided roots
    paths = []
    for root in audio_roots:
        for ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a"]:
            paths += glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    paths = sorted(paths)

    n_pix = 0
    mean  = 0.0
    M2    = 0.0  # for Welford's algorithm

    for p in tqdm(paths):
        try:
            y, _ = librosa.load(p, sr=SR)
            y = librosa.effects.trim(y, top_db=20)[0]
            for clip in centre_clips(y, SR):
                spec = mel_db(clip, SR).astype(np.float32)
                flat = spec.flatten()
                for x in flat:
                    n_pix += 1
                    delta = x - mean
                    mean += delta / n_pix
                    M2   += delta * (x - mean)
        except Exception as e:
            print("skip", p, e, file=sys.stderr)

    std = np.sqrt(M2 / (n_pix - 1))
    json.dump({"mean": float(mean), "std": float(std)}, open(out_json,"w"), indent=2)
    print(f"mean={mean:.4f}  std={std:.4f}  (from {n_pix:,} pixels)")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_roots", nargs="+", help="One or more directories with audio files")
    ap.add_argument("--out_json", default="../json/global_norm_stats/contrastive_global_norm_stats.json")
    args = ap.parse_args()
    main(args.audio_roots, args.out_json)
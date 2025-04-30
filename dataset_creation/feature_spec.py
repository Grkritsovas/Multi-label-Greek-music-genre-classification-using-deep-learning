import os
import tensorflow as tf

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

write_feature_spec('contrastive_tfrecords')

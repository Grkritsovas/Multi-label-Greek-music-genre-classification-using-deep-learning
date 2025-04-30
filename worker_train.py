import argparse
import os
import gc
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.optimizer.set_jit(True)  # Enable XLA
from tensorflow.keras import layers, models, regularizers
import gc
import datetime
import glob

from tensorflow.keras.mixed_precision import set_global_policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

import contrastive_utils as cu
glob_stats  = cu.load_global_stats(10)
time_stats  = cu.load_time_axis_stats(10)
mean_global = glob_stats['mean']
std_global  = glob_stats['std']
T_FIXED = int(time_stats['mean'])

print(mean_global, std_global, T_FIXED)

# parser
@tf.function
def parse_example(example):
    feat = {
        "mel": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example, feat)

    mel = tf.io.parse_tensor(ex["mel"], tf.float32)  # shape unknown

    # Fix frequency and time
    freq_dim = tf.shape(mel)[0]
    time_dim = tf.shape(mel)[1]

    def fix_freq(mel):
        return tf.cond(
            freq_dim < 96,
            lambda: tf.pad(mel, [[0, 96 - freq_dim], [0,0]]),
            lambda: mel[:96, :]
        )

    def fix_time(mel):
        return tf.cond(
            time_dim < T_FIXED,
            lambda: tf.pad(mel, [[0,0], [0, T_FIXED - time_dim]]),
            lambda: mel[:, :T_FIXED]
        )

    mel = fix_freq(mel)
    mel = fix_time(mel)

    # Set static shape so TensorFlow knows dimensions
    mel.set_shape([96, T_FIXED])

    # Normalize
    mel = (mel - mean_global) / (std_global + 1e-6)

    # Add channel dimension
    mel = mel[..., tf.newaxis]  # (96, 432, 1)

    return mel

# augmentations
@tf.function
def batch_augment(batch):
    # batch: (B, 96, 432, 1)

    def augment_single(mel):
        view1 = augment(mel)
        view2 = augment(mel)

        # Force shapes for both views
        view1 = tf.ensure_shape(view1, [96, T_FIXED, 1])
        view2 = tf.ensure_shape(view2, [96, T_FIXED, 1])

        stacked = tf.stack([view1, view2], axis=0)  # (2, 96, 432, 1)
        stacked.set_shape([2, 96, T_FIXED, 1])

        return stacked

    return tf.vectorized_map(augment_single, batch)  # (B, 2, 96, 432, 1)



@tf.function
def add_noise(x, max_noise=0.05):
    noise_level = tf.random.uniform([], 0, max_noise)
    noise = tf.random.normal(tf.shape(x), mean=0., stddev=noise_level, dtype=x.dtype)
    return x + noise

@tf.function
def time_mask(x, max_frames=80, min_masks=1, max_masks=3):
    # x: (F, T, C)
    F = tf.shape(x)[0]
    T = tf.shape(x)[1]
    C = tf.shape(x)[2]
    max_frames = tf.minimum(max_frames, T - 1)

    def _cond(i, mask):
        return i < num_masks

    def _body(i, mask):
        # draw a random mask length and start
        t = tf.random.uniform([], 0, max_frames + 1, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, T - t + 1,    dtype=tf.int32)
        # build the new mask slice
        before = mask[:, :t0, :]
        middle = tf.zeros([F, t, C], dtype=x.dtype)
        after  = mask[:, t0 + t:, :]
        mask   = tf.concat([before, middle, after], axis=1)
        return i + 1, mask

    # how many time masks?
    num_masks = tf.random.uniform([], min_masks, max_masks + 1, dtype=tf.int32)
    # initial all-ones mask
    i0, mask0 = 0, tf.ones_like(x)
    _, final_mask = tf.while_loop(_cond, _body, [i0, mask0])
    return x * final_mask


@tf.function
def freq_mask(x, max_bins=40, min_masks=1, max_masks=3):
    # x: (F, T, C)
    F = tf.shape(x)[0]
    T = tf.shape(x)[1]
    C = tf.shape(x)[2]
    max_bins = tf.minimum(max_bins, F - 1)

    def _cond(i, mask):
        return i < num_masks

    def _body(i, mask):
        f  = tf.random.uniform([], 0, max_bins + 1, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, F - f + 1,    dtype=tf.int32)
        # slice out the band
        before = mask[:f0, :, :]
        middle = tf.zeros([f, T, C], dtype=x.dtype)
        after  = mask[f0 + f:, :, :]
        mask   = tf.concat([before, middle, after], axis=0)
        return i + 1, mask

    num_masks = tf.random.uniform([], min_masks, max_masks + 1, dtype=tf.int32)
    i0, mask0 = 0, tf.ones_like(x)
    _, final_mask = tf.while_loop(_cond, _body, [i0, mask0])
    return x * final_mask


@tf.function
def pitch_shift(x, max_bins=4):
    shift = tf.random.uniform([], -max_bins, max_bins+1, dtype=tf.int32)
    F = tf.shape(x)[0]
    
    def shift_op():
        shifted = tf.roll(x, shift=shift, axis=0)
        if shift > 0:
            shifted = tf.tensor_scatter_nd_update(
                shifted,
                tf.reshape(tf.range(0, shift), [-1, 1]),
                tf.zeros([shift, tf.shape(x)[1], tf.shape(x)[2]], dtype=x.dtype)
            )
        else:
            shifted = tf.tensor_scatter_nd_update(
                shifted,
                tf.reshape(tf.range(F + shift, F), [-1, 1]),
                tf.zeros([tf.abs(shift), tf.shape(x)[1], tf.shape(x)[2]], dtype=x.dtype)
            )
        return shifted
    
    return tf.cond(shift != 0, shift_op, lambda: x)

@tf.function
def channel_drop(x, max_width=5):
    F = tf.shape(x)[0]
    W = tf.random.uniform([], 1, max_width+1, dtype=tf.int32)
    W = tf.minimum(W, F//3)
    start = tf.random.uniform([], 0, F-W+1, dtype=tf.int32)
    mask = tf.ones_like(x)
    mask = tf.tensor_scatter_nd_update(
        mask,
        tf.reshape(tf.range(start, start+W), [-1, 1]),
        tf.zeros([W, tf.shape(x)[1], tf.shape(x)[2]], dtype=x.dtype)
    )
    return x * mask

@tf.function
def time_reversal(x, prob=0.3):
    reverse = tf.random.uniform([], 0, 1.0) < prob
    return tf.cond(reverse, lambda: tf.reverse(x, axis=[1]), lambda: x)

# Combined augmentation
@tf.function
def augment(mel):
    noise_std = tf.random.uniform([], 0, 0.02)
    if tf.random.uniform([]) < 0.85:
        mel = time_mask(mel, max_frames=80, min_masks=1, max_masks=3)
    if tf.random.uniform([]) < 0.80:
        mel = freq_mask(mel, max_bins=40, min_masks=1, max_masks=3)
    extra_noise = tf.random.normal(tf.shape(mel), mean=0., stddev=0.015, dtype=mel.dtype)

    mel = mel + tf.random.normal(tf.shape(mel), mean=0., stddev=noise_std, dtype=mel.dtype)


    if tf.random.uniform([]) < 0.6:
        mel = pitch_shift(mel, max_bins=4)
    if tf.random.uniform([]) < 0.4:
        mel = channel_drop(mel, max_width=5)
    if tf.random.uniform([]) < 0.3:
        mel = tf.reverse(mel, axis=[1])

    mel = mel + tf.random.normal(tf.shape(mel), 0., 0.01)

    mel = tf.pad(mel, 
                 [[0, 96 - tf.shape(mel)[0]],
                  [0, T_FIXED - tf.shape(mel)[1]],
                  [0,0]])
    mel = mel[:96, :T_FIXED, :]
    mel.set_shape([96,T_FIXED,1])
    return mel


# Two-view generator
@tf.function
def make_views(mel):
    view1 = augment(mel)
    view2 = augment(mel)
    views = tf.stack([view1, view2], axis=0)
    return views

def residual_block(x, filters, kernel_size=3, stride=1, activation='relu', use_dropout=False, regularization=1e-7):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', 
                      kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(regularization))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    if use_dropout:
        x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', 
                      kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(regularization))(x)
    x = layers.BatchNormalization()(x)
    
    # Adaptive shortcut connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', 
                                 kernel_initializer='he_normal')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([shortcut, x])
    x = layers.Activation(activation)(x)
    return x

def attention_layer(x):
    # Channel-wise attention
    channel_mean = layers.GlobalAveragePooling2D()(x)
    channel_max = layers.GlobalMaxPooling2D()(x)
    
    # Compact attention mechanism
    attention_features = layers.Dense(x.shape[-1] // 4, activation='relu')(channel_mean)
    attention_features = layers.Dense(x.shape[-1], activation='sigmoid')(attention_features)
    
    # Spatial attention component
    spatial_attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(
        layers.Concatenate()([
            layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x),
            layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)
        ])
    )
    
    # Apply attention
    x = layers.Multiply()([x, attention_features[:, tf.newaxis, tf.newaxis, :]])
    x = layers.Multiply()([x, spatial_attention])
    
    return x


def build_encoder(input_shape=(96, T_FIXED,1)):

    inputs = layers.Input(shape=input_shape) #(96, 432, 1)

    x = layers.Conv2D(
        32, 3,
        padding='same',
        kernel_initializer='he_normal',
        use_bias=False
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.MaxPooling2D((2,4))(x)

    # 1 res
    x = residual_block(
        x, 64,
        use_dropout=True,
        regularization=1e-6
    )
    x = layers.MaxPooling2D((2,3))(x)

    # 2 res
    x = residual_block(
        x, 96,
        stride=(2,2),
        regularization=1e-5
    )
    x = layers.Dropout(0.25)(x)

    # 3 res
    x = residual_block(x, 128, stride=(2,2))
    x = attention_layer(x)
    x = layers.Dropout(0.3)(x)

    # Global pooling -> projection head
    x = layers.GlobalAveragePooling2D()(x) # 128->embedding dim

    return models.Model(inputs, x, name="encoder_CNN_Att")


LATENT = 128 # size that encoder returns (gap)
PROJ   = 16 # projection-dim for NT-Xent, smaller, to try to let the encoder do the hard work.

def build_projector(latent_dim=LATENT, proj_dim=PROJ):
    inp  = layers.Input((latent_dim,))
    x = layers.Dense(128, use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    z    = layers.Dense(proj_dim, use_bias=False)(x)
    z    = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(z)
    return tf.keras.Model(inp, z, name="projector")


@tf.function
def nt_xent(z1, z2, T=0.1):
    B  = tf.shape(z1)[0]

    z  = tf.concat([z1, z2], 0) # 2B x d
    sim = tf.matmul(z, z, transpose_b=True) / T # 2B x 2B - raw cosine scores

    mask = tf.eye(2*B, dtype=tf.bool)
    mask_val = tf.constant(-1e9, dtype=sim.dtype)
    sim = tf.where(mask, tf.fill(tf.shape(sim), mask_val), sim)

    labels = tf.range(B)
    pos    = tf.concat([labels + B, labels], 0)

    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pos,
                                                       logits=sim))


class SimCLR(tf.keras.Model):
    def __init__(self, encoder, proj_dim=64, temperature=0.07):
        super().__init__()
        self.enc   = encoder # (Batch,128)
        self.proj  = build_projector(LATENT, proj_dim)# (Batch,64)
        self.T     = temperature

    def compile(self, optimizer):
        super().compile(run_eagerly=False)
        self.opt = optimizer

    def train_step(self, data):
        # data shape: (batch, 2, 96, 432, 1)
        v1, v2 = data

        with tf.GradientTape() as tape:
            h1 = self.enc(v1, training=True)
            h2 = self.enc(v2, training=True)
            z1 = tf.math.l2_normalize(self.proj(h1, training=True), axis=1)
            z2 = tf.math.l2_normalize(self.proj(h2, training=True), axis=1)
            loss = nt_xent(z1, z2, self.T)

        vars = self.enc.trainable_weights + self.proj.trainable_weights
        self.opt.apply_gradients(zip(tape.gradient(loss, vars), vars))
        return {"contrastive_loss": loss}

# dataset loader for specific files
def get_worker_ds(file_list, batch_size, examples_per_file):
    if not file_list:
        return None, 0

    num_files = len(file_list)
    total_examples_in_worker = num_files * examples_per_file
    steps_for_worker = total_examples_in_worker // batch_size

    print(f"Worker processing {num_files} files, {total_examples_in_worker} examples, {steps_for_worker} steps.")

    ds = tf.data.TFRecordDataset(file_list, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(15000)
    ds = ds.map(lambda mel: make_views(mel), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, steps_for_worker


def get_latest_weights(weights_dir='./weights'):
    # Create weights directory if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)
    
    # Update file extensions to match the Keras .weights.h5 format
    weight_files = glob.glob(os.path.join(weights_dir, '*.weights.h5'))
    
    if not weight_files:
        print(f"No weight files found in {weights_dir}")
        return None
    
    # Sort files by modification time (most recent last)
    weight_files.sort(key=os.path.getmtime)
    latest_weights = weight_files[-1]
    
    print(f"Found latest weights: {latest_weights}")
    return latest_weights


def train_one_epoch(ds, model, steps):
    step = 0
    for batch in ds:
        v1 = batch[:, 0, ...]
        v2 = batch[:, 1, ...]
        loss = model.train_step((v1, v2))["contrastive_loss"]

        step += 1
        print(f"Step {step}/{steps} - contrastive_loss: {loss.numpy():.4f}")

        if step >= steps:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", required=True, help="list of TFRecord files")
    parser.add_argument("--save_weights", required=True, help="Path to save weights to (.h5)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--examples_per_file", type=int, default=1024)

    args = parser.parse_args()

    weights_dir = "contrastive_weights"
    os.makedirs(weights_dir, exist_ok=True)

    # Look for latest weights in the contrastive_weights directory
    
    weights_path = get_latest_weights(weights_dir=weights_dir)

    file_paths = args.files.split(',')

    # Build Model
    encoder = build_encoder()
    model = SimCLR(encoder, proj_dim=PROJ) # Use PROJ=16

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer)

    # Try to load weights if they exist
    if weights_path:
        try:
            print(f"Loading weights from {weights_path}")
            model.enc.load_weights(weights_path)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Could not load weights: {e}")
            print("Initializing weights from scratch.")
    else:
        print("No weight file found. Initializing weights from scratch.")


    # Get dataset for this worker
    worker_ds, steps = get_worker_ds(file_paths, args.batch_size, args.examples_per_file)
    
    if worker_ds and steps > 0:
        print(f"Training for {steps} steps...")

        train_one_epoch(worker_ds, model, steps)
        # Save weights just for encoder (Overwrite the file for the next worker)
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        weights_path = os.path.join("contrastive_weights", f"encoder_weights_{timestamp}.weights.h5")
        print(f"Saving encoder weights to {weights_path}")
        model.enc.save_weights(weights_path)
        print("Weights saved.")
    else:
        print("No data or steps for this worker.")

    # ensure cleanup
    del model
    del worker_ds
    del encoder
    gc.collect()
    tf.keras.backend.clear_session()
    print("Worker finished and cleaned up.")
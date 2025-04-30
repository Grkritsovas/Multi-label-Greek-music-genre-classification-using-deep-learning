import subprocess
import os
import glob
import time
import tensorflow as tf

tfrecord_dir = 'dataset_creation/contrastive_tfrecords'
all_files = sorted(tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord")))
files_per_worker = 4 # Number of files per sub-process run
total_epochs = 30
start_epoch = 0 # loads from a state file

# Load previous weights
base_checkpoint_path = "./simclr_checkpoints/worker_ckpt.weights.h5"

if not os.path.exists(os.path.dirname(base_checkpoint_path)):
    os.makedirs(os.path.dirname(base_checkpoint_path))

# Check if a starting checkpoint exists from a previous run
print(f"Found {len(all_files)} TFRecord files.")

for epoch in range(start_epoch, total_epochs):
    print(f"\n--- Starting Epoch {epoch+1}/{total_epochs} ---")
    epoch_start_time = time.time()
    
    # Shuffle file order for the epoch
    current_file_index = 0
    worker_run = 0
    while current_file_index < len(all_files):
        worker_run += 1
        start = current_file_index
        end = min(current_file_index + files_per_worker, len(all_files))
        files_for_worker = all_files[start:end]
        current_file_index = end

        print(f"Epoch {epoch+1}, Worker Run {worker_run}: Processing files {start+1}-{end}")

        # Prepare arguments for the worker script
        worker_args = [
            "python", "worker_train.py",
            "--files", ",".join(files_for_worker),
            "--load_weights", base_checkpoint_path,
            "--save_weights", base_checkpoint_path,
            "--batch_size", "64",
            "--examples_per_file", "1024"
        ]

        # Execute the worker script as a sub-process
        process = subprocess.run(worker_args, capture_output=True, text=True)

        print(f"Worker {worker_run} STDOUT:\n{process.stdout}")
        if process.returncode != 0:
            print(f"Worker {worker_run} STDERR:\n{process.stderr}")
            raise RuntimeError(f"Worker process failed with code {process.returncode}")
        else:
            print(f"Worker {worker_run} completed successfully.")
            
        time.sleep(2)

    print(f"--- Epoch {epoch+1} completed in {time.time() - epoch_start_time:.2f} seconds ---")

print("All epochs finished.")
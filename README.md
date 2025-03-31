# Multilabel Classification on Greek Music Genres
## Main methodology: CNN with attention on mel spectrograms and a dense embedding layer to help the network learn label correlations before final classification.
Raw audio files were acquired using youtube-dl, then converted to .wav format using ffmpeg.
Various preprocessing techniques are tested, including:
  1. Segment size
  2. Mel bin size
  3. Augmentation on raw audio and/or mel spectrograms
Different architectures are compared — from simple to deep — with attention mechanisms to preserve both temporal and spatial information.
Multiple loss functions and optimization strategies are evaluated.

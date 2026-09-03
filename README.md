# Multilabel Classification on Greek Music Genres

This project investigates multilabel music genre classification for Greek music using deep learning.

The main methodology combines convolutional layers for local feature extraction with CRNN architectures and attention mechanisms to preserve both temporal and spatial information. The models operate on 96 × 432 Mel-spectrograms.

The study also explores contrastive learning pretraining on unlabeled data, followed by gradual fine-tuning on the labeled dataset.

Several architectures are evaluated, ranging from simpler baseline models to deeper hybrid architectures with recurrent and attention-based components.

## Paper

[Read the full paper](https://link.springer.com/chapter/10.1007/978-3-032-11442-6_37)

## Poster

<img src="./Mitigating Western Bias in Music Genre Classification A Contrastive Learning Approach for Greek Music.png" alt="Project Poster" width="900">

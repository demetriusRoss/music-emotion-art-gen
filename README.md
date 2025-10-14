# Generating Abstract Art from Music Emotion Embeddings (Deep Learning Final)
Clustering Songs by Mood Using PCA and Autoencoders

![alt text](image.png)


## Project Overview


Music expresses a wide range of emotions, and these emotions can be represented visually. In this project, I aim to generate abstract art from the emotional characteristics of music. Building on my earlier work that clustered songs by mood, this project extends that analysis into a cross-modal representation where audio-derived emotion embeddings are translated into visual art.


The workflow includes several key steps:


1. Gathering and processing emotional audio data (using datasets such as DEAM or GTZAN)
2. Extracting Mel spectrograms using Librosa
3. Training an autoencoder to learn latent emotion embeddings from audio
4. Using a conditional variational autoencoder (VAE) to generate abstract images from the learned emotion embeddings
5. Performing model comparisons and evaluation of both the audio embeddings and the generated art


The project concludes with a discussion of results and ideas for future extensions, such as building an interactive Streamlit demo.



## Structure
- `src/` — core ML scripts
- `data/` — raw and processed data
- `app/` — Streamlit demo
- `notebooks/` — Jupyter notebook
- `tests/` — unit tests
- `models/` — trained PyTorch model weights

## Setup
```
pip install -r requirements.txt
```

```

## Data Source
**DEAM (Dataset for Emotion Analysis using Music)** 


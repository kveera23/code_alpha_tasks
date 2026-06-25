import os
import librosa
import numpy as np
from tqdm import tqdm

from data_loader import load_dataset
from augmentation import (
    add_noise,
    pitch_shift,
    time_stretch
)

# -----------------------------
# Configuration
# -----------------------------
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

N_MFCC = 40


# -----------------------------
# Extract Features
# -----------------------------
def extract_features(signal, sr):

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC
    )

    delta = librosa.feature.delta(mfcc)

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_mels=40
        )
    )

    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sr
    )

    min_frames = min(
        mfcc.shape[1],
        delta.shape[1],
        delta2.shape[1],
        mel.shape[1],
        chroma.shape[1]
    )

    mfcc = mfcc[:, :min_frames]
    delta = delta[:, :min_frames]
    delta2 = delta2[:, :min_frames]
    mel = mel[:, :min_frames]
    chroma = chroma[:, :min_frames]

    features = np.vstack([
        mfcc,
        delta,
        delta2,
        mel,
        chroma
    ])

    return features


# -----------------------------
# Load One Audio
# -----------------------------
def process_audio(audio_path):

    signal, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )

    signal, _ = librosa.effects.trim(
        signal,
        top_db=20
    )

    if len(signal) > SAMPLES:

        start = (len(signal) - SAMPLES) // 2

        signal = signal[start:start + SAMPLES]

    else:

        signal = np.pad(
            signal,
            (0, SAMPLES - len(signal))
        )

    return signal, sr


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    df = load_dataset()

    X = []
    y = []

    print("Extracting Features...\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        signal, sr = process_audio(row["audio_path"])

        versions = [

            signal,

            add_noise(signal),

            pitch_shift(signal, sr),

            time_stretch(signal)

        ]

        for audio in versions:

            if len(audio) > SAMPLES:

                audio = audio[:SAMPLES]

            else:

                audio = np.pad(
                    audio,
                    (0, SAMPLES - len(audio))
                )

            feature = extract_features(
                audio,
                sr
            )

            X.append(feature)

            y.append(row["emotion"])

    X = np.array(X)

    y = np.array(y)

    print("\nExtraction Completed!\n")

    print("Feature Shape :", X.shape)

    print("Labels Shape :", y.shape)

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    np.save(
        "data/processed/X_features.npy",
        X
    )

    np.save(
        "data/processed/y_labels.npy",
        y
    )

    print("\nFeatures Saved Successfully!")
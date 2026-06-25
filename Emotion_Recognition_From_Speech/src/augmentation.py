import numpy as np
import librosa


# -----------------------------
# Add Random Noise
# -----------------------------
def add_noise(signal, noise_factor=0.003):

    noise = np.random.randn(len(signal))

    augmented = signal + noise_factor * noise

    return augmented


# -----------------------------
# Pitch Shift
# -----------------------------
def pitch_shift(signal, sr):

    steps = np.random.uniform(-2, 2)

    return librosa.effects.pitch_shift(
        y=signal,
        sr=sr,
        n_steps=steps
    )


# -----------------------------
# Time Stretch
# -----------------------------
def time_stretch(signal):

    rate = np.random.uniform(0.9, 1.1)

    stretched = librosa.effects.time_stretch(
        y=signal,
        rate=rate
    )

    return stretched
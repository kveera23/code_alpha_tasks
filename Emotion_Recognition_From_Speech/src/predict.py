import numpy as np
import librosa
from joblib import load
from tensorflow.keras.models import load_model # type: ignore

# -----------------------------
# Configuration
# -----------------------------
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

N_MFCC = 40


# -----------------------------
# Load Model
# -----------------------------
model = load_model("models/best_model.keras")

label_encoder = load("models/label_encoder.pkl")


# -----------------------------
# Audio Processing
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
# Feature Extraction
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
# Prediction
# -----------------------------
def predict_emotion(audio_path):

    signal, sr = process_audio(audio_path)

    features = extract_features(
        signal,
        sr
    )

    # Normalize
    features = (
        features - np.mean(features)
    ) / (np.std(features) + 1e-8)

    features = np.expand_dims(
        features,
        axis=-1
    )

    features = np.expand_dims(
        features,
        axis=0
    )

    prediction = model.predict(
        features,
        verbose=0
    )

    class_index = np.argmax(prediction)

    emotion = label_encoder.inverse_transform(
        [class_index]
    )[0]

    confidence = float(prediction[0][class_index])

    # Probability for every emotion
    probabilities = {
        label: float(prob)
        for label, prob in zip(
            label_encoder.classes_,
            prediction[0]
        )
    }

    return emotion, confidence, probabilities

# -----------------------------
# Testing
# -----------------------------
if __name__ == "__main__":

    path = input("Enter audio path : ")

    emotion, confidence, probabilities = predict_emotion(path)

    print("\nPredicted Emotion :", emotion)
    print(f"Confidence : {confidence*100:.2f}%")

    print("\nAll Probabilities\n")

    for emotion_name, score in probabilities.items():
        print(f"{emotion_name:<12}: {score*100:.2f}%")
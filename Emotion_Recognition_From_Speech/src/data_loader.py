import os
import pandas as pd

# -----------------------------
# Dataset Path
# -----------------------------
DATASET_PATH = os.path.join("data", "raw")

# -----------------------------
# Emotion Mapping
# -----------------------------
EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise"
}


def load_dataset():

    audio_paths = []
    emotions = []

    for actor_folder in sorted(os.listdir(DATASET_PATH)):

        actor_path = os.path.join(DATASET_PATH, actor_folder)

        if not os.path.isdir(actor_path):
            continue

        for audio_file in sorted(os.listdir(actor_path)):

            if not audio_file.endswith(".wav"):
                continue

            emotion_code = audio_file.split("-")[2]

            emotion = EMOTION_MAP.get(emotion_code)

            audio_paths.append(os.path.join(actor_path, audio_file))
            emotions.append(emotion)

    dataset = pd.DataFrame({
        "audio_path": audio_paths,
        "emotion": emotions
    })

    return dataset


if __name__ == "__main__":

    df = load_dataset()

    print("=" * 60)
    print("RAVDESS DATASET INFORMATION")
    print("=" * 60)

    print(f"\nTotal Audio Files : {len(df)}")

    print("\nEmotion Distribution\n")
    print(df["emotion"].value_counts())

    print("\nFirst Five Samples\n")
    print(df.head())
# 🎤 Speech Emotion Recognition using CNN + BiLSTM

---

## Description

Speech Emotion Recognition using CNN + BiLSTM with Streamlit UI for real-time emotion prediction from speech.

---

## Topics

speech-emotion-recognition
deep-learning
cnn
bilstm
tensorflow
keras
streamlit
python
machine-learning
audio-processing
librosa
ravdess

---

## 📌 Overview

Speech Emotion Recognition (SER) is a Deep Learning application that identifies human emotions from speech recordings. This project utilizes a hybrid **CNN + Bidirectional LSTM (BiLSTM)** architecture to classify speech into eight emotional categories using acoustic features extracted from audio signals.

The application also includes a modern **Streamlit Web Interface** that enables users to upload a speech recording and instantly obtain the predicted emotion along with confidence scores.

---

## 🚀 Features

* Upload WAV audio files
* CNN + BiLSTM Deep Learning Model
* MFCC Feature Extraction
* Delta & Delta-Delta Features
* Mel Spectrogram Features
* Chroma Features
* Interactive Streamlit Dashboard
* Emotion Confidence Scores
* Confusion Matrix Visualization
* Accuracy & Loss Curves

---

## 😊 Supported Emotions

* Angry 😠
* Calm 😌
* Disgust 🤢
* Fear 😨
* Happy 😊
* Neutral 😐
* Sad 😢
* Surprise 😲

---

## 📂 Dataset

**Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

* Total Samples: **5760**
* Number of Classes: **8**
* Sample Rate: **22050 Hz**

---

## 📥 Dataset Setup

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

### Step 1: Download the Dataset

Download the dataset from:

https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

### Step 2: Extract the Dataset

Extract the downloaded dataset into the following directory:

```
data/
└── raw/
    ├── Actor_01/
    ├── Actor_02/
    ├── ...
    └── Actor_24/
```

### Step 3: Generate Processed Features

Run the feature extraction script to generate the processed dataset:

```bash
python src/feature_extraction.py
```

This will automatically create:

```
data/
└── processed/
    ├── X_features.npy
    └── y_labels.npy
```

### Step 4: Train the Model

```bash
python src/train.py
```

### Step 5: Evaluate the Model

```bash
python src/evaluate.py
```

### Step 6: Launch the Application

```bash
streamlit run app.py
```

---

## 🧠 Model Architecture

Speech Audio (.wav)

↓

Preprocessing

↓

Feature Extraction

* MFCC
* Delta MFCC
* Delta-Delta MFCC
* Mel Spectrogram
* Chroma Features

↓

CNN Layers

↓

Bidirectional LSTM

↓

Dense Layer

↓

Softmax Classifier

↓

Emotion Prediction

---

## 📊 Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **85.76%** |
| Precision | **85.99%** |
| Recall    | **85.76%** |
| F1 Score  | **85.83%** |

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* Librosa
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Joblib

---

## 📁 Project Structure

```text
Emotion_Recognition_From_Speech/

│── app.py
│── README.md
│── requirements.txt
│── style.css

├── assets/
│   ├── home.png
│   ├── prediction.png
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_model.keras
│   ├── cnn_emotion_model.keras
│   ├── history.pkl
│   └── label_encoder.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── classification_report.txt
│   └── metrics.txt
│
├── src/
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── predict.py
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Emotion_Recognition_From_Speech.git
```

Navigate to the project

```bash
cd Emotion_Recognition_From_Speech
```

Install the dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ How to Run

Train the model

```bash
python src/train.py
```

Evaluate the model

```bash
python src/evaluate.py
```

Generate visualizations

```bash
python src/visualize.py
```

Launch the Streamlit application

```bash
streamlit run app.py
```

---

## 📷 Screenshots

Add the following screenshots inside the **assets** folder.

* Home Page
* Prediction Dashboard
* Confusion Matrix
* Accuracy Curve
* Loss Curve

---

## 🎯 Future Enhancements

* Real-time microphone emotion recognition
* Transformer-based speech emotion recognition
* Multi-language support
* Cloud deployment
* Docker support

---

## 👨‍💻 Author

**K Veera**

Artificial Intelligence & Machine Learning Enthusiast

Developed as part of the **CodeAlpha Artificial Intelligence Internship**.

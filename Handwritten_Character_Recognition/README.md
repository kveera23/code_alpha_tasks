# ✍️ Handwritten Character Recognition using CNN

A Deep Learning-based web application that recognizes handwritten English alphabets using a Convolutional Neural Network (CNN). The model is trained on the EMNIST Letters dataset and deployed through a modern Streamlit web application.

---

## 📌 Project Overview

Handwritten Character Recognition is an important computer vision task used in applications such as document digitization, postal mail sorting, bank cheque processing, and educational tools.

This project uses a Convolutional Neural Network (CNN) to classify handwritten English alphabets (A–Z). Users can upload an image of a handwritten character through an interactive Streamlit interface and instantly receive the predicted character with its confidence score.

---

## 🚀 Features

- Handwritten English alphabet recognition (A–Z)
- CNN-based deep learning model
- Trained on the EMNIST Letters dataset
- Image preprocessing and normalization
- Real-time prediction from uploaded images
- Prediction confidence score
- Modern and responsive Streamlit UI
- Model evaluation with multiple performance metrics
- Training history visualization
- Confusion matrix
- Sample prediction visualization
- Professional project structure

---

## 🧠 Technologies Used

- Python 3.12
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit

---

## 📂 Project Structure

```text
Handwritten_Character_Recognition/
│
├── app.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── app/
│   ├── __init__.py
│   ├── predictor.py
│   └── utils.py
│
├── assets/
│   └── style.css
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── data/
│
├── models/
│   └── cnn_model.keras
│
└── outputs/
    ├── plots/
    └── reports/
```

---

## 📊 Dataset

**Dataset:** EMNIST Letters

- 26 English alphabet classes (A–Z)
- 124,800 training images
- 20,800 testing images
- Image Size: 28 × 28 pixels
- Grayscale handwritten character images

This project uses the **EMNIST Letters** dataset.

Due to GitHub file size limitations, the dataset is **not included** in this repository.

Download the EMNIST Letters dataset and place the files in:

```text
data/raw/emnist/
```

Required files:

- emnist-letters-train-images-idx3-ubyte.gz
- emnist-letters-train-labels-idx1-ubyte.gz
- emnist-letters-test-images-idx3-ubyte.gz
- emnist-letters-test-labels-idx1-ubyte.gz

---

## 🏗️ Model Architecture

The project uses a Convolutional Neural Network consisting of:

- Conv2D
- Batch Normalization
- MaxPooling2D
- Dropout
- Conv2D
- Batch Normalization
- MaxPooling2D
- Dropout
- Conv2D
- Batch Normalization
- Flatten
- Dense (256)
- Dropout
- Dense (26 Output Classes)

---

## 📈 Model Performance

| Metric | Value |
|---------|------:|
| Test Accuracy | 94.81% |
| Model | CNN |
| Classes | 26 |
| Dataset | EMNIST Letters |

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/kveera23/Handwritten_Character_Recognition.git
```

Navigate to the project

```bash
cd Handwritten_Character_Recognition
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🎯 Future Enhancements

- Support handwritten words
- Support sentence recognition
- CRNN-based sequence modeling
- Mobile-friendly deployment
- Real-time webcam recognition
- ONNX / TensorFlow Lite deployment

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- EMNIST Dataset
- TensorFlow
- Streamlit
- Scikit-learn
- OpenCV

---

## 👨‍💻 Developer

**Veeranjaneyulu Kopparapu**

Machine Learning Internship Project

CodeAlpha AI Internship
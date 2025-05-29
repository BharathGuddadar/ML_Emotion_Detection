# 😄 Emotion Detection from Facial Expressions Using CNN

This project detects **seven human emotions** from facial expressions using a **Convolutional Neural Network (CNN)**. The system classifies real-time webcam input into the following emotions:

- Happy
- Sad
- Neutral
- Surprise
- Fear
- Disgust
- Angry

---

## 📌 Model Overview

We built a deep learning model using **TensorFlow** and **Keras**. The model takes grayscale images of size **48x48** pixels and performs **multi-class classification** using the **softmax activation** function.

### 📊 Features
- **7 emotion categories** classified from facial expressions.
- **Trained on FER-2013-like dataset** (images from `EmotionData/train` and `EmotionData/test` directories).
- Uses **ImageDataGenerator** for real-time image augmentation and normalization.
- Includes **webcam integration** for real-time emotion prediction.

---

## 🧠 Algorithm and Architecture

We used a **Convolutional Neural Network (CNN)** with the following architecture:

- **Conv2D(32)** + ReLU + MaxPooling
- **Conv2D(64)** + ReLU + MaxPooling
- **Conv2D(128)** + ReLU + MaxPooling
- Flatten + Dense(128) + Dropout(0.5)
- Dense(7, softmax)

**Optimizer**: Adam  
**Loss Function**: Categorical Crossentropy  
**Evaluation Metric**: Accuracy

---

## 🏋️ Training

- **Epochs**: 25  
- **Batch Size**: 32  
- **Input Size**: 48x48x3  
- **Training & Validation Split**: Predefined folders

### 📈 Accuracy Graph

![image](https://github.com/user-attachments/assets/6222aeca-c7f9-437a-9726-330a546d9a6c)


### 📉 Loss Graph

![image](https://github.com/user-attachments/assets/a0b83984-3ff5-4850-a3bf-4367ba90cd37)

---

## 📦 Model Saving

The trained model is saved in:
```

emotion\_detection\_model.keras

````

To load the model:
```python
from tensorflow.keras.models import load_model
model = load_model('emotion_detection_model.keras')
````

---

## 🎥 Real-Time Emotion Detection

You can use OpenCV to capture input from your webcam and predict emotions using the trained model.

```bash
python webcam_emotion.py
```

---

## 📚 Dependencies

* TensorFlow
* OpenCV
* Matplotlib
* NumPy

Install using:

```bash
pip install tensorflow opencv-python matplotlib numpy
```

---

## 💡 Future Work

* Improve accuracy by using deeper CNNs or transfer learning (e.g., MobileNet, EfficientNet).
* Support for multiple faces in a frame.
* Deploy as a web or mobile app.

---

## 📬 Contact

For any queries or suggestions, feel free to reach out at bharathps821@gmail.com

```

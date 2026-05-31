# 🚗🚲 Car vs Bike Image Classification | CNN & TensorFlow

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![CNN](https://img.shields.io/badge/Model-CNN-5C3EE8?style=flat-square)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaelsayed7/cars-bikes)

> Binary CNN image classifier achieving **92.7% validation accuracy** —
> end-to-end pipeline from raw images to real-time single predictions.

---

## 📌 Project Overview

Built a binary CNN classifier using TensorFlow & Keras to distinguish between
cars and bikes, trained on a real-world image dataset with an **80/10/10 split**
and full data augmentation. Includes a deployable single-image prediction function.
Directly applicable in:

- 🚦 Smart traffic monitoring systems
- 🏙️ Urban mobility analytics
- 🅿️ Automated parking classification
- 📹 Vehicle detection in surveillance

---

## 🔧 Full Pipeline

```
Raw Image Dataset
    ↓
Split: 80% Train / 10% Validation / 10% Test
    ↓
ImageDataGenerator — rescale + augmentation (train only)
    ↓
CNN Model (2 Conv layers + MaxPooling + Dense)
    ↓
Binary Crossentropy Loss | Adam Optimizer
    ↓
Training (5 epochs, batch_size=32)
    ↓
Single Image Prediction (car / bike)
    ↓
92.7% Validation Accuracy
```

---

## 🏗️ Architecture

```
Input (64×64×3)
    ↓
Conv2D(32, 3×3, relu) → MaxPooling2D(2×2, stride=2)
    ↓
Conv2D(32, 3×3, relu) → MaxPooling2D(2×2, stride=2)
    ↓
Flatten
    ↓
Dense(128, relu)
    ↓
Dense(1, sigmoid) → Car or Bike
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Dataset | Car vs Bike Classification Dataset |
| Validation Accuracy | **92.7%** |
| Split | 80% train / 10% val / 10% test |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Epochs | 5 |
| Batch Size | 32 |
| Input Size | 64×64×3 |

---

## 🔧 Data Augmentation (Train Set Only)

| Technique | Value |
|---|---|
| Rescale | 1/255 |
| Shear Range | 0.2 |
| Zoom Range | 0.2 |
| Horizontal Flip | Enabled |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `tensorflow` / `keras` | Model building & training |
| `ImageDataGenerator` | Augmentation & data loading |
| `splitfolders` | 80/10/10 dataset splitting |
| `numpy` | Array operations |
| `matplotlib` | Training curves visualization |
| `visualkeras` | Architecture visualization |
| `kagglehub` | Dataset download |

---

## ▶️ How to Run

1. View on Kaggle (dataset pre-loaded):

[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaelsayed7/cars-bikes)

2. Install dependencies:
```bash
pip install tensorflow numpy matplotlib split-folders visualkeras kagglehub
```

3. Download the dataset:
```python
import kagglehub
path = kagglehub.dataset_download('utkarshsaxenadn/car-vs-bike-classification-dataset')
```

4. To predict on a single image:
```python
# Load and preprocess image
test_image = image.load_img('your_image.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Predict
result = cnn.predict(test_image)
prediction = 'car' if result[0][0] == 1 else 'bike'
print(prediction)
```

---

## 📁 Project Structure

```
car-vs-bike-classification/
│
├── notebook.ipynb     # Full CNN pipeline
└── README.md          # Project documentation
```

---

## 👤 Author

**Mostafa El-Sayed** — Data Scientist | Computer Vision Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/mostafa-nfc)
[![Gmail](https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:m.e.2172000@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/MostafaELsayed-217)

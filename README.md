# 🌳 Forest Change Detection & Reforestation Recommendation System

## 📌 Project Overview
This project detects forest cover changes using satellite images and deep learning.  
It identifies deforestation or afforestation by comparing images and calculates forest coverage percentage.  
The system also suggests suitable tree species for reforestation based on environmental conditions.

The system is deployed as a web application using Flask and hosted on Render Cloud.

---

## 🎯 Objectives
- Detect forest regions from satellite images  
- Calculate forest cover percentage  
- Identify deforestation or afforestation  
- Train a deep learning segmentation model  
- Provide tree recommendations  
- Deploy the application on cloud  

---

## 🧠 Technologies Used

| Component | Technology |
|----------|-----------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV |
| Backend | Flask |
| Frontend | HTML, CSS |
| Deployment | Render |
| Version Control | GitHub |

---

## 🏗️ SystemSatellite Images
↓
Preprocessing (Resize, Normalize)
↓
Deep Learning Model (CNN - U-Net Inspired)
↓
Forest Mask Prediction
↓
Forest Percentage Calculation
↓
Change Detection (Deforestation / Afforestation)
↓
Tree Recommendation System 🌱
↓
Web Interface (Flask) Architecture


---

## 🧪 Methodology

### 1️⃣ Dataset
- Satellite images with corresponding mask images  
- Used for supervised learning  

### 2️⃣ Preprocessing
- Image resizing  
- Normalization  
- Mask conversion  

### 3️⃣ Model
- CNN-based segmentation model  
- U-Net inspired architecture  
- Binary classification (forest / non-forest)  

### 4️⃣ Training
- Loss Function: Binary Cross-Entropy  
- Optimizer: Adam  
- Epochs: 3  
- Batch Size: 2–8  

### 5️⃣ Prediction
- Model predicts forest areas  
- Calculates forest percentage  
- Detects change between images  

---

## 🚀 Features

- 🌳 Forest detection using machine learning  
- 📊 Forest percentage calculation  
- 🔄 Change detection (Before vs After images)  
- 🌱 Tree recommendation system  
- 🌐 Web-based interface  
- ☁️ Cloud deployment  

---

## 📂 Project Structure
forest-change-detection/
│
├── app.py
├── detector.py
├── train_model.py
├── forest_segmentation_model.h5
├── requirements.txt
│
├── static/
│ ├── uploads/
│ ├── outputs/
│ └── style.css
│
├── templates/
│ ├── index.html
│ ├── result.html
│ └── single_result.html
│
└── README.md


---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/forest-change-detection.git
cd forest-change-detection
### 2️⃣ Install Dependencies
pip install -r requirements.txt
### 3️⃣ Run Application
python app.py
### 4️⃣ Open Browser
http://127.0.0.1:5000


---

## 🌍 Deployment

The application is deployed on Render Cloud.

🔗 Live URL:  
https://forest-change-detection.onrender.com

---

## 📊 Results

- Successfully detects forest regions  
- Calculates forest coverage percentage  
- Identifies deforestation and afforestation  
- Provides tree recommendations  

---

## 🔮 Future Scope

- Integration with real-time satellite APIs  
- Improved model accuracy using large datasets  
- Mobile application development  
- Region-based intelligent recommendations  

---

## 📚 References

- TensorFlow Documentation  
- OpenCV Documentation  
- U-Net Research Paper  
- Environmental Monitoring Research  

---

## 👨‍💻 Author

Vedant Shelar

---

## ⭐ Conclusion

This project demonstrates how deep learning and image processing can be used for environmental monitoring.  
It provides an effective solution for detecting forest changes and supporting reforestation planning.

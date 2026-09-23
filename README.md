# 🩺 Diabetic Retinopathy Explainable AI Screening System

An AI-powered web application for **diabetic retinopathy screening from retinal/eye images**, with explainable AI visualization using **Grad-CAM**.

The system is designed to support preliminary screening in **Primary Health Centers and Community Healthcare settings** by combining automated eye detection, deep-learning-based classification, and visual explanations.

> **Note:** This project is intended for educational, research, and screening purposes. It is **not a replacement for professional medical diagnosis**.

---

## 📌 Project Overview

Diabetic retinopathy is a diabetes-related eye condition that can cause vision loss if it is not detected and managed early.

This project provides a simple workflow where a healthcare worker can:

1. Verify or register a patient.
2. Capture **one eye photograph** using the device camera.
3. Automatically detect and crop the eye region using a local AI-based eye detection module.
4. Analyze the cropped image using a deep-learning diabetic retinopathy model.
5. Display the predicted diabetic retinopathy stage and confidence.
6. Generate a **Grad-CAM heatmap** to explain which regions influenced the AI prediction.
7. View previous patient screening history.
8. Generate and download a PDF screening report.
9. Send the report to the patient's registered email address.

---

## ✨ Key Features

### 👤 Patient Management

* Patient verification using Patient ID and name
* New patient registration
* Mobile number and email storage
* Patient screening history
* Patient-specific image and report storage

### 📷 One-Photo Eye Screening

* Single camera capture workflow
* Automatic eye detection
* Automatic eye-region cropping
* No manual image cropping required
* Designed for simple use in community healthcare environments

### 🤖 AI-Based Classification

The system classifies diabetic retinopathy into five stages:

| Stage   | Classification                     |
| ------- | ---------------------------------- |
| Stage 0 | No Diabetic Retinopathy            |
| Stage 1 | Mild Diabetic Retinopathy          |
| Stage 2 | Moderate Diabetic Retinopathy      |
| Stage 3 | Severe Diabetic Retinopathy        |
| Stage 4 | Proliferative Diabetic Retinopathy |

### 🔍 Explainable AI

The system uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide a visual explanation of the model's prediction.

The heatmap helps indicate the regions of the eye image that contributed to the model's classification.

### 📊 Confidence Analysis

The application displays:

* Predicted stage
* Prediction confidence
* Confidence warning for uncertain predictions
* Recommendation for professional ophthalmologist review when appropriate

### 📄 PDF Reports

The system can generate a structured screening report containing:

* Patient information
* Eye image
* AI prediction
* Confidence
* Grad-CAM visualization
* AI-generated clinical guidance
* Clinical notes
* Medical disclaimer

### 📧 Email Report

The generated report can be sent to the **email address registered with the patient's profile**, avoiding the need to enter the email address again for every report.

---

## 🧠 AI Architecture

The system uses a deep-learning classification model based on **EfficientNet-B0** with transfer-learning architecture.

### Processing Pipeline

```text
Patient Verification
        ↓
Camera Capture
        ↓
Eye Detection
        ↓
Automatic Eye Cropping
        ↓
Image Preprocessing
        ↓
EfficientNet-B0
        ↓
DR Stage Classification
        ↓
Confidence Analysis
        ↓
Grad-CAM Explanation
        ↓
Result & Screening Report
```

---

## 🔬 Explainable AI Workflow

```text
Input Eye Image
      ↓
Preprocessing
      ↓
Deep Learning Model
      ↓
Predicted DR Stage
      ↓
Grad-CAM
      ↓
Heatmap Generation
      ↓
Visual Explanation
```

Grad-CAM is used to visualize the areas of the image that have a stronger influence on the model's prediction.

---

## 🛠️ Technologies Used

### Frontend / Web Application

* Python
* Streamlit
* HTML/CSS through Streamlit components

### Artificial Intelligence

* PyTorch
* Torchvision
* EfficientNet-B0
* Grad-CAM
* NumPy
* OpenCV

### Eye Detection

* MediaPipe Tasks
* Face/Eye landmark detection

### Image Processing

* OpenCV
* Pillow

### Report Generation

* ReportLab

### AI Explanation

* Google Gemini API

### Text-to-Speech

* gTTS

### Data Storage

* JSON-based patient records
* Local image storage

---

## 📂 Project Structure

```text
explainable-ai-diabetics-retinopathy/
│
├── app/
│   ├── ui.py
│   ├── eye_ai.py
│   └── api.py
│
├── src/
│   ├── model.py
│   └── xai.py
│
├── data/
│   └── dataset.py
│
├── models/
│   └── dr_model.pth
│
├── patient_database/
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/JeevanGowda18/explainable-ai-diabetics-retinopathy.git
```

### 2. Open the project

```bash
cd explainable-ai-diabetics-retinopathy
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file locally.

Example:

```text
GEMINI_API_KEY=your_gemini_api_key
```

### Important

Do **not** upload `.env` to GitHub.

Add it to `.gitignore`:

```text
.env
.streamlit/secrets.toml
venv/
.venv/
__pycache__/
*.pyc
```

For cloud deployment, use the hosting platform's secure Secrets system instead of committing API keys to the repository.

---

## ▶️ Run the Application Locally

From the project root:

```bash
python -m streamlit run app/ui.py
```

The application will open in your browser.

Typically:

```text
http://localhost:8501
```

---

## ☁️ Deployment

The application can be deployed using **Streamlit Community Cloud**.

Basic deployment configuration:

```text
Repository:
JeevanGowda18/explainable-ai-diabetics-retinopathy

Branch:
main

Main file:
app/ui.py
```

After deployment, Streamlit provides a public web URL that can be opened from:

* Desktop computers
* Android phones
* iPhones
* Tablets

The application is designed to provide a responsive browser-based screening interface.

---

## 📱 Mobile Workflow

The application can be accessed directly from a smartphone browser.

```text
Open Web Link
      ↓
Patient Verification
      ↓
Add / Verify Patient
      ↓
Take Eye Photograph
      ↓
Automatic Eye Detection
      ↓
AI Analysis
      ↓
DR Classification
      ↓
Grad-CAM Explanation
      ↓
Generate Report
      ↓
Download / Email Report
```

No separate mobile application installation is required for the web version.

---

## 📊 Model Output

The model produces a probability distribution across five diabetic retinopathy classes.

The application uses the highest-probability class as the predicted stage and displays the associated confidence.

For low-confidence predictions, the system presents an uncertainty warning and recommends professional clinical review.

---

## 🧪 Screening Disclaimer

This application is an **AI-assisted screening and educational research project**.

It should not be used as the sole basis for:

* Medical diagnosis
* Treatment decisions
* Medication decisions
* Emergency medical decisions

AI predictions should be reviewed by a qualified healthcare professional, particularly when the model reports low confidence or uncertain results.

---

## 🔒 Privacy Considerations

The application may process sensitive patient information such as:

* Patient name
* Patient ID
* Mobile number
* Email address
* Eye images
* Screening results
* Screening history

When deploying the application publicly:

* Do not use real patient data for demonstrations.
* Use appropriate access controls.
* Protect API credentials.
* Do not commit secrets to GitHub.
* Follow applicable healthcare data-protection and privacy requirements.

---

## 🎯 Project Objectives

The main objectives of the project are:

* Develop an accessible diabetic retinopathy screening interface.
* Reduce complexity in the image-capture process.
* Automatically identify the eye region from a captured image.
* Apply deep learning for diabetic retinopathy classification.
* Improve model interpretability using Explainable AI.
* Provide understandable screening results.
* Generate structured screening reports.
* Explore the potential of AI-assisted screening in community healthcare environments.

---

## 🚀 Future Enhancements

Possible future improvements include:

* Integration with electronic health records
* Secure cloud database
* Multi-user healthcare-center accounts
* Doctor/ophthalmologist dashboard
* Improved retinal image quality assessment
* Larger clinically validated datasets
* Multi-language voice assistance
* Advanced patient analytics
* Secure cloud storage
* Clinical validation with ophthalmologists
* Dedicated Android/iOS application

---

## 👨‍💻 Project

**Diabetic Retinopathy Explainable AI Screening System**

Developed as an AI and healthcare technology project focusing on:

**Deep Learning + Explainable AI + Computer Vision + Healthcare Screening**

---

## ⚠️ Important Notice

This project is provided for **educational, research, and prototype screening purposes**.

The predictions generated by the system are not guaranteed to be medically accurate and should not replace examination or diagnosis by a qualified ophthalmologist or other appropriate healthcare professional.

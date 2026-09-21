import sys
import os

# Append project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import io
import json
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from gtts import gTTS
from google import genai

# Import local backend modules
from src.model import DRClassifier
from src.xai import generate_gradcam_heatmap
from data.dataset import get_data_transforms
from app.api import send_report_via_email

# ---------------------------------------------------------------------
# GEMINI AI & MULTILINGUAL CONFIGURATION
# ---------------------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "title": "Diabetic Retinopathy Explainable AI Screening System",
        "subtitle": "Primary Health Center & Community Healthcare",
        "step1": "Step 1: Patient Information & Facial Eye Capture",
        "patient_name": "Patient Name",
        "patient_id": "Patient ID",
        "cam_label": "Capture Facial/Eye Photo via Camera (Gemini AI Vision Active)",
        "no_eye_warning": "❌ Gemini Vision AI: No clear eye structure detected in photo. Please align patient face properly.",
        "eye_detected": "✅ Gemini Vision AI: Patient eyes detected & cropped successfully. Proceeding with AI analysis...",
        "history_title": "📜 Patient Medical History & Prior Visits",
        "visit_count": "Total Visits Recorded:",
        "step2": "Step 2: Diagnostic Findings & Explainable Heatmap Scan",
        "orig_scan": "Cropped Eye Scan",
        "xai_scan": "Explainable AI (Grad-CAM Heatmap)",
        "diag_results": "Diagnostic Results:",
        "stage": "DR Severity Stage",
        "confidence": "Model Confidence",
        "audio_label": "🔊 Audio Explanation for Patient (Mother Tongue)",
        "gemini_title": "🤖 Gemini Medical Explanation & Advice",
        "step3": "Step 3: Patient Report Options & Prescription",
        "rx_label": "Doctor's Prescription & Clinical Notes (AI Auto-Generated & Editable)",
        "download_btn": "📄 Download Hard Copy (PDF)",
        "round_send_btn": "📧 Send Soft Copy to Patient",
        "modal_title": "Send Soft Copy Report via Email",
        "email_label": "Enter Patient Email Address",
        "modal_send": "Confirm & Send Email",
        "lang_code": "en",
        "stages": [
            "Stage 0: No Diabetic Retinopathy",
            "Stage 1: Mild Diabetic Retinopathy",
            "Stage 2: Moderate Diabetic Retinopathy",
            "Stage 3: Severe Diabetic Retinopathy",
            "Stage 4: Proliferative Diabetic Retinopathy"
        ]
    },
    "Kannada": {
        "title": "ಡಯಾಬಿಟಿಕ್ ರೆಟಿನೋಪತಿ ವಿಶ್ಲೇಷಣಾತ್ಮಕ AI ತಪಾಸಣಾ ವ್ಯವಸ್ಥೆ",
        "subtitle": "ಪ್ರಾಥಮಿಕ ಆರೋಗ್ಯ ಕೇಂದ್ರ ಮತ್ತು ಸಮುದಾಯ ಆರೋಗ್ಯ",
        "step1": "ಹಂತ 1: ರೋಗಿಯ ಮಾಹಿತಿ ಮತ್ತು ಮುಖದ ಫೋಟೋ",
        "patient_name": "ರೋಗಿಯ ಹೆಸರು",
        "patient_id": "ರೋಗಿಯ ಐಡಿ",
        "cam_label": "ಕ್ಯಾಮೆರಾ ಮೂಲಕ ಮುಖ/ಕಣ್ಣಿನ ಚಿತ್ರವನ್ನು ಸೆರೆಹಿಡಿಯಿರಿ (ಜೆಮಿನಿ AI ವಿಷನ್)",
        "no_eye_warning": "❌ ಜೆಮಿನಿ AI: ಫೋಟೋದಲ್ಲಿ ಕಣ್ಣುಗಳು ಸ್ಪಷ್ಟವಾಗಿ ಪತ್ತೆಯಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ಮುಖವನ್ನು ಸರಿಯಾಗಿ ಜೋಡಿಸಿ.",
        "eye_detected": "✅ ಜೆಮಿನಿ AI: ಕಣ್ಣಿನ ಭಾಗವನ್ನು ಯಶಸ್ವಿಯಾಗಿ ಗುರುತಿಸಲಾಗಿದೆ...",
        "history_title": "📜 ರೋಗಿಯ ಹಳೆಯ ವೈದ್ಯಕೀಯ ದಾಖಲೆಗಳು",
        "visit_count": "ಒಟ್ಟು ಭೇಟಿಗಳ ಸಂಖ್ಯೆ:",
        "step2": "ಹಂತ 2: ರೋಗನಿರ್ಣಯದ ಫಲಿತಾಂಶಗಳು ಮತ್ತು AI ಹೀಟ್‌ಮ್ಯಾಪ್ ಸ್ಕ್ಯಾನ್",
        "orig_scan": "ಕಣ್ಣಿನ ಸ್ಕ್ಯಾನ್",
        "xai_scan": "ವಿವರಣಾತ್ಮಕ AI (Grad-CAM ಹೀಟ್‌ಮ್ಯಾಪ್)",
        "diag_results": "ರೋಗನಿರ್ಣಯದ ಫಲಿತಾಂಶಗಳು:",
        "stage": "DR ತೀವ್ರತೆಯ ಹಂತ",
        "confidence": "ಮಾಡೆಲ್ ಆತ್ಮವಿಶ್ವಾಸದ ಶೇಕಡಾವಾರು",
        "audio_label": "🔊 ರೋಗಿಗಾಗಿ ಧ್ವನಿ ವಿವರಣೆ (ಮಾತೃಭಾಷೆಯಲ್ಲಿ)",
        "gemini_title": "🤖 ಜೆಮಿನಿ ವೈದ್ಯಕೀಯ ವಿವರಣೆ ಮತ್ತು ಸಲಹೆ",
        "step3": "ಹಂತ 3: ವೈದ್ಯಕೀಯ ವರದಿ ಆಯ್ಕೆಗಳು ಮತ್ತು ಪ್ರಿಸ್ಕ್ರಿಪ್ಷನ್",
        "rx_label": "ವೈದ್ಯರ ಪ್ರಿಸ್ಕ್ರಿಪ್ಷನ್ ಮತ್ತು ಟಿಪ್ಪಣಿಗಳು (AI ಸ್ವಯಂಚಾಲಿತವಾಗಿ ರಚಿಸಿದೆ)",
        "download_btn": "📄 ವೈದ್ಯಕೀಯ ವರದಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ (PDF)",
        "round_send_btn": "📧 ಇಮೇಲ್ ಮೂಲಕ ಸಾಫ್ಟ್ ಕಾಪಿ ಕಳುಹಿಸಿ",
        "modal_title": "ಇಮೇಲ್ ಮೂಲಕ ವೈದ್ಯಕೀಯ ವರದಿ ಕಳುಹಿಸಿ",
        "email_label": "ರೋಗಿಯ ಇಮೇಲ್ ವಿಳಾಸವನ್ನು ನಮೂದಿಸಿ",
        "modal_send": "ಖಚಿತಪಡಿಸಿ ಮತ್ತು ಕಳುಹಿಸಿ",
        "lang_code": "kn",
        "stages": [
            "ಹಂತ 0: ಡಯಾಬಿಟಿಕ್ ರೆಟಿನೋಪತಿ ಇಲ್ಲ",
            "ಹಂತ 1: ಹಗುರವಾದ ಡಯಾಬಿಟಿಕ್ ರೆಟಿನೋಪತಿ",
            "ಹಂತ 2: ಮಧ್ಯಮ ಡಯಾಬಿಟಿಕ್ ರೆಟಿನೋಪತಿ",
            "ಹಂತ 3: ತೀವ್ರ ಡಯಾಬಿಟಿಕ್ ರೆಟಿನೋಪತಿ",
            "ಹಂತ 4: ಪ್ರೊಲಿಫರೇಟಿವ್ ಡಯಾಬಿಟಿಕ್ ರೆಟಿನೋಪತಿ"
        ]
    },
    "Hindi": {
        "title": "डायबिटिक रेटिनोपैथी व्याख्यात्मक AI जांच प्रणाली",
        "subtitle": "प्राथमिक स्वास्थ्य केंद्र और सामुदायिक स्वास्थ्य देखरेख",
        "step1": "चरण 1: रोगी की जानकारी और चेहरा कैप्चर",
        "patient_name": "रोगी का नाम",
        "patient_id": "रोगी की आईडी",
        "cam_label": "कैमरे के माध्यम से चेहरे/आंख का चित्र लें (जेमिनी AI विजन)",
        "no_eye_warning": "❌ जेमिनी AI: फोटो में आंखें स्पष्ट नहीं मिलीं। कृपया चेहरा सही करें।",
        "eye_detected": "✅ जेमिनी AI: आंख का क्षेत्र सफलतापूर्वक पहचाना गया...",
        "history_title": "📜 रोगी का पिछला चिकित्सा इतिहास",
        "visit_count": "कुल दौरों की संख्या:",
        "step2": "चरण 2: नैदानिक निष्कर्ष और व्याख्यात्मक हीटमैप स्कैन",
        "orig_scan": "क्रॉप किया गया आंख का स्कैन",
        "xai_scan": "व्याख्यात्मक AI (Grad-CAM हीटमैप)",
        "diag_results": "नैदानिक परिणाम:",
        "stage": "DR गंभीरता का चरण",
        "confidence": "मॉडल आत्मविश्वास प्रतिशत",
        "audio_label": "🔊 रोगी के लिए ऑडियो स्पष्टीकरण (मातृभाषा)",
        "gemini_title": "🤖 जेमिनी चिकित्सा स्पष्टीकरण और सलाह",
        "step3": "चरण 3: रोगी रिपोर्ट विकल्प और पर्ची",
        "rx_label": "डॉक्टर की पर्ची और नोट्स (AI ऑटो-जनरेटेड)",
        "download_btn": "📄 मेडिकल रिपोर्ट डाउनलोड करें (PDF)",
        "round_send_btn": "📧 ईमेल के माध्यम से सॉफ्ट कॉपी भेजें",
        "modal_title": "ईमेल द्वारा रिपोर्ट भेजें",
        "email_label": "रोगी का ईमेल पता दर्ज करें",
        "modal_send": "पुष्टि करें और भेजें",
        "lang_code": "hi",
        "stages": [
            "चरण 0: कोई डायबिटिक रेटिनोपैथी नहीं",
            "चरण 1: हल्का डायबिटिक रेटिनोपैथी",
            "चरण 2: मध्यम डायबिटिक रेटिनोपैथी",
            "चरण 3: गंभीर डायबिटिक रेटिनोपैथी",
            "चरण 4: प्रोलिफेरेटिव डायबिटिक रेटिनोपैथी"
        ]
    }
}

# ---------------------------------------------------------------------
# CONFIGURATION & MODEL LOADING
# ---------------------------------------------------------------------
st.set_page_config(page_title="Diabetic Retinopathy Screening System", layout="wide")

selected_lang = st.sidebar.selectbox("🌐 Select Language / ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ / भाषा चुनें", ["English", "Kannada", "Hindi"])
txt = TRANSLATIONS[selected_lang]

MODEL_PATH = os.path.join("models", "dr_model.pth")
DB_DIR = "patient_database"

@st.cache_resource
def load_dr_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRClassifier(num_classes=5, pretrained=False).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

# ---------------------------------------------------------------------
# GEMINI VISION AI EYE VERIFICATION & CROPPER
# ---------------------------------------------------------------------
def analyze_and_crop_eye_with_gemini(pil_image):
    try:
        client = genai.Client()
        prompt = (
            "Analyze this camera photo carefully. Check if a human face or human eye is clearly visible. "
            "Reply strictly with JSON formatted as: {\"eye_detected\": true, \"box_2d\": [ymin, xmin, ymax, xmax]} "
            "where box_2d normalizes coordinates from 0 to 1000 around the primary eye. "
            "If no human eyes are visible (e.g., room background, wall, object), reply: {\"eye_detected\": false}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_image, prompt]
        )
        
        clean_json = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_json)

        if data.get("eye_detected", False):
            box = data.get("box_2d", [300, 300, 700, 700])
            width, height = pil_image.size
            ymin = int((box[0] / 1000.0) * height)
            xmin = int((box[1] / 1000.0) * width)
            ymax = int((box[2] / 1000.0) * height)
            xmax = int((box[3] / 1000.0) * width)

            if ymax > ymin and xmax > xmin:
                cropped_eye = pil_image.crop((xmin, ymin, xmax, ymax))
                return True, np.array(cropped_eye)
            else:
                return True, np.array(pil_image)
        else:
            return False, None
    except Exception:
        img_np = np.array(pil_image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        if len(eyes) > 0:
            x, y, w, h = eyes[0]
            cropped = img_np[y:y+h, x:x+w]
            return True, cropped
        return False, None

# ---------------------------------------------------------------------
# PATIENT DATABASE & VISUAL VISIT HISTORY FUNCTIONS
# ---------------------------------------------------------------------
def get_patient_history(patient_id):
    os.makedirs(DB_DIR, exist_ok=True)
    record_file = os.path.join(DB_DIR, f"{patient_id}.json")
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            return json.load(f)
    return []

def save_patient_visit(patient_id, patient_name, diagnosis_stage, confidence, rx_text, orig_img_rgb, heatmap_img_rgb):
    os.makedirs(DB_DIR, exist_ok=True)
    history = get_patient_history(patient_id)
    visit_num = len(history) + 1

    img_dir = os.path.join(DB_DIR, "images", patient_id)
    os.makedirs(img_dir, exist_ok=True)
    
    orig_path = os.path.join(img_dir, f"visit_{visit_num}_orig.jpg")
    heat_path = os.path.join(img_dir, f"visit_{visit_num}_heat.jpg")
    
    # Save image arrays explicitly
    cv2.imwrite(orig_path, cv2.cvtColor(orig_img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(heat_path, cv2.cvtColor(heatmap_img_rgb, cv2.COLOR_RGB2BGR))

    record_file = os.path.join(DB_DIR, f"{patient_id}.json")
    
    new_visit = {
        "visit_number": visit_num,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_name": patient_name,
        "diagnosis_stage": diagnosis_stage,
        "confidence": confidence,
        "prescription": rx_text,
        "orig_img_path": orig_path,
        "heat_img_path": heat_path
    }
    history.append(new_visit)
    with open(record_file, "w") as f:
        json.dump(history, f, indent=4)
    return visit_num

# ---------------------------------------------------------------------
# GEMINI AI EXPLANATION & AUTO-PRESCRIPTION GENERATOR
# ---------------------------------------------------------------------
@st.cache_data
def generate_gemini_explanation(stage_name, language):
    try:
        client = genai.Client()
        prompt = (
            f"You are an empathetic eye specialist. Explain what '{stage_name}' means to a patient in simple terms "
            f"in {language} language. Keep it under 4 sentences and include simple lifestyle or medical advice."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception:
        return f"Eye Examination Finding: {stage_name}. Please consult an ophthalmologist for a thorough evaluation."

@st.cache_data
def generate_ai_medications(stage_name):
    try:
        client = genai.Client()
        prompt = (
            f"Provide concise medical recommendations and clinical medication guidance for a patient diagnosed with '{stage_name}'. "
            f"List 3 brief bullet points focusing on glycemic control, ocular drops/treatment, and follow-up timing."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception:
        return "• Strict blood sugar and HbA1c control.\n• Routine dilated eye examination.\n• Refer to retinal specialist if vision declines."

@st.cache_data
def generate_ai_prescription(stage_name):
    try:
        client = genai.Client()
        prompt = (
            f"Act as an ophthalmologist drafting a formal medical prescription for a patient with '{stage_name}'. "
            f"List 3 specific numbered prescription items including drug name, dosage, frequency, and follow-up schedule."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception:
        return "1. Tab. Metformin 500mg - 1-0-1 after meals\n2. Lubricating Eye Drops - 1 drop thrice daily\n3. Follow up in 3 months with HbA1c report."

@st.cache_data
def generate_tts_audio(text, lang_code, patient_id):
    os.makedirs("temp_uploads", exist_ok=True)
    output_path = os.path.join("temp_uploads", f"audio_{patient_id}_{lang_code}.mp3")
    if os.path.exists(output_path):
        return output_path
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(output_path)
        return output_path
    except Exception:
        return None

# ---------------------------------------------------------------------
# ZERO-GAP PDF REPORT GENERATOR
# ---------------------------------------------------------------------
def generate_pdf_report(patient_name, patient_id, visit_num, diagnosis_stage, confidence, orig_img_rgb, heatmap_img_rgb, ai_meds, typed_rx, output_path, is_hardcopy=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    c = canvas.Canvas(output_path, pagesize=letter)
    
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(306, 765, "DIABETIC EYE EXAMINATION REPORT")
    c.setFont("Helvetica", 9)
    c.drawCentredString(306, 752, "(Primary Health Center & Community Healthcare Screening System)")
    c.line(30, 745, 582, 745)
    
    c.rect(30, 625, 270, 115)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(35, 725, "Patient Information:")
    c.setFont("Helvetica", 9)
    c.drawString(35, 705, f"Name: {patient_name}")
    c.drawString(35, 685, f"Patient ID: {patient_id} (Visit #{visit_num})")
    c.drawString(35, 665, f"Exam Date: {datetime.now().strftime('%Y-%m-%d')}")
    c.drawString(35, 645, f"Target Eye: Both Eyes (OD/OS)")

    c.rect(310, 625, 272, 115)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(315, 727, "Original Scan")
    c.drawString(450, 727, "Grad-CAM Heatmap")

    pil_orig = Image.fromarray(orig_img_rgb)
    pil_heat = Image.fromarray(heatmap_img_rgb)
    c.drawImage(ImageReader(pil_orig), 315, 632, width=90, height=90)
    c.drawImage(ImageReader(pil_heat), 450, 632, width=90, height=90)

    c.rect(30, 485, 552, 135)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, 605, "Exam Findings & AI Severity Grading:")
    
    stages = [
        "Stage 0: No Diabetic Retinopathy",
        "Stage 1: Mild Diabetic Retinopathy",
        "Stage 2: Moderate Diabetic Retinopathy",
        "Stage 3: Severe Diabetic Retinopathy",
        "Stage 4: Proliferative Diabetic Retinopathy"
    ]
    
    y = 585
    for stg in stages:
        c.setFont("Helvetica", 9)
        checkbox = "[X]" if stg.startswith(diagnosis_stage.split(":")[0]) else "[  ]"
        c.drawString(45, y, f"{checkbox}  {stg}")
        y -= 19
        
    c.setFont("Helvetica-Bold", 9)
    c.drawString(330, 585, f"Model Confidence: {confidence:.2f}%")
    
    c.rect(30, 355, 552, 125)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, 465, "AI Clinical Guidance & Suggested Medications:")
    c.setFont("Helvetica", 8)
    
    med_lines = ai_meds.split('\n')
    med_y = 448
    for line in med_lines[:5]:
        c.drawString(45, med_y, line[:95])
        med_y -= 16

    c.rect(30, 115, 552, 235)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, 335, "Doctor's Prescription & Clinical Notes (Rx):")
    c.setFont("Helvetica", 9)
    
    if is_hardcopy:
        line_y = 310
        for _ in range(8):
            c.line(40, line_y, 570, line_y)
            line_y -= 25
    else:
        rx_lines = typed_rx.split('\n')
        rx_y = 315
        for rx_line in rx_lines[:9]:
            c.drawString(45, rx_y, rx_line[:95])
            rx_y -= 22

    if is_hardcopy:
        c.line(370, 55, 560, 55)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(370, 42, "Doctor Signature / Seal")
    else:
        c.setFont("Helvetica-BoldOblique", 9)
        c.drawString(35, 45, "* This is machine generated and signature is not required .")
        
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(35, 25, "This report is generated by an Explainable AI Screening System to assist healthcare professionals.")
    
    c.save()

# ---------------------------------------------------------------------
# ROUND BUTTON POPUP DIALOG FOR EMAIL DISPATCH
# ---------------------------------------------------------------------
@st.dialog(txt["modal_title"])
def send_email_modal(patient_name, report_path):
    email_val = st.text_input(txt["email_label"])
    if st.button(txt["modal_send"], use_container_width=True):
        if not email_val or not email_val.strip():
            st.warning("Please enter a valid email address.")
        else:
            with st.spinner("Sending soft copy..."):
                success, msg = send_report_via_email(email_val, patient_name, report_path)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

# ---------------------------------------------------------------------
# STREAMLIT DASHBOARD UI
# ---------------------------------------------------------------------
st.title(txt["title"])
st.subheader(txt["subtitle"])

st.markdown(f"### {txt['step1']}")

col_input1, col_input2 = st.columns(2)
with col_input1:
    patient_name_val = st.text_input(txt["patient_name"], value="Ramesh Gowda DR")
with col_input2:
    patient_id_val = st.text_input(txt["patient_id"], value="P-1002")

# Visual Patient History Log
history = get_patient_history(patient_id_val)
if history:
    with st.expander(f"{txt['history_title']} ({txt['visit_count']} {len(history)})"):
        for visit in history:
            st.markdown(f"#### Visit #{visit['visit_number']} ({visit['date']})")
            st.write(f"**Diagnosis:** {visit['diagnosis_stage']} | **Confidence:** {visit['confidence']:.2f}%")
            
            # Load images directly into PIL to guarantee rendering in history
            orig_path = visit.get("orig_img_path", "")
            heat_path = visit.get("heat_img_path", "")
            
            if os.path.exists(orig_path) and os.path.exists(heat_path):
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.image(Image.open(orig_path), caption="Prior Retinal Scan", use_container_width=True)
                with col_h2:
                    st.image(Image.open(heat_path), caption="Prior Grad-CAM Heatmap", use_container_width=True)
                    
            st.text(f"Prescription Notes:\n{visit['prescription']}")
            st.markdown("---")

# Camera Capture Input
camera_file = st.camera_input(txt["cam_label"])

if camera_file is not None:
    os.makedirs("temp_uploads", exist_ok=True)
    pil_captured = Image.open(camera_file).convert("RGB")

    with st.spinner("Gemini AI analyzing photo for eye structure..."):
        has_eye, cropped_eye_np = analyze_and_crop_eye_with_gemini(pil_captured)

    if not has_eye:
        st.error(txt["no_eye_warning"])
    else:
        st.success(txt["eye_detected"])
        
        resized_rgb = cv2.resize(cropped_eye_np, (224, 224))
        _, val_transform = get_data_transforms()
        input_tensor = val_transform(resized_rgb).unsqueeze(0)

        model, device = load_dr_model()
        input_tensor = input_tensor.to(device)

        with st.spinner("Analyzing Eye Scan & Generating Grad-CAM Heatmap..."):
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class].item() * 100

            heatmap_overlay = generate_gradcam_heatmap(model, input_tensor, resized_rgb)
            diagnosis_stage = txt["stages"][pred_class]

        st.markdown("---")
        st.markdown(f"### {txt['step2']}")

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(resized_rgb, caption=txt["orig_scan"], use_container_width=True)
        with col_img2:
            st.image(heatmap_overlay, caption=txt["xai_scan"], use_container_width=True)

        st.markdown(f"#### {txt['diag_results']}")
        st.info(f"**{txt['stage']}:** {diagnosis_stage}\n\n**{txt['confidence']}:** {confidence:.2f}%")

        st.markdown(f"#### {txt['gemini_title']}")
        gemini_text = generate_gemini_explanation(diagnosis_stage, selected_lang)
        st.success(gemini_text)

        st.markdown(f"#### {txt['audio_label']}")
        audio_path = generate_tts_audio(gemini_text, txt["lang_code"], patient_id_val)
        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path, format="audio/mp3")

        # Generate AI Medication Guidance & Gemini Auto-Prescription
        english_stage_str = TRANSLATIONS["English"]["stages"][pred_class]
        ai_meds = generate_ai_medications(english_stage_str)
        ai_rx_default = generate_ai_prescription(english_stage_str)

        # ---------------------------------------------------------------------
        # SECTION 3: Prescription Input & Hard/Soft Copy Options
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown(f"### {txt['step3']}")

        typed_rx = st.text_area(
            txt["rx_label"], 
            value=ai_rx_default,
            height=130
        )

        # Save visit details & images to persistent database
        visit_count = save_patient_visit(
            patient_id_val, patient_name_val, english_stage_str, confidence, typed_rx, resized_rgb, heatmap_overlay
        )

        # Generate Reports
        hardcopy_path = os.path.join("temp_uploads", f"HardCopy_{patient_id_val}.pdf")
        softcopy_path = os.path.join("temp_uploads", f"SoftCopy_{patient_id_val}.pdf")
        
        generate_pdf_report(patient_name_val, patient_id_val, visit_count, english_stage_str, confidence, resized_rgb, heatmap_overlay, ai_meds, typed_rx, hardcopy_path, is_hardcopy=True)
        generate_pdf_report(patient_name_val, patient_id_val, visit_count, english_stage_str, confidence, resized_rgb, heatmap_overlay, ai_meds, typed_rx, softcopy_path, is_hardcopy=False)

        col_act1, col_act2 = st.columns(2)

        with col_act1:
            if os.path.exists(hardcopy_path):
                with open(hardcopy_path, "rb") as pdf_file:
                    st.download_button(
                        label=txt["download_btn"],
                        data=pdf_file.read(),
                        file_name=f"Report_{patient_id_val}_Print.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

        with col_act2:
            if st.button(txt["round_send_btn"], use_container_width=True, type="primary"):
                send_email_modal(patient_name_val, softcopy_path)
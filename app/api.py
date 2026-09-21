import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Diabetic Retinopathy Screening API")

# =====================================================================
# CONFIGURATION
# =====================================================================
SENDER_EMAIL = "helloeyediognatics@gmail.com"  
SENDER_PASSWORD = "pyjn rxzr yngx qcxh"

# =====================================================================
# REQUEST MODELS
# =====================================================================
class EmailRequest(BaseModel):
    recipient_email: str
    patient_name: str
    report_file_path: str

# =====================================================================
# DELIVERY FUNCTIONS
# =====================================================================
def send_report_via_email(recipient_email: str, patient_name: str, report_file_path: str):
    if not recipient_email or not recipient_email.strip():
        return False, "No email address provided."

    if not os.path.exists(report_file_path):
        return False, f"Report file not found at: {report_file_path}"

    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = f"Diabetic Eye Examination Report - {patient_name}"

        body = (
            f"Dear {patient_name},\n\n"
            f"Please find attached your Diabetic Eye Examination Report.\n\n"
            f"Best regards,\n"
            f"Primary Health Center Screening Team"
        )
        msg.attach(MIMEText(body, 'plain'))

        with open(report_file_path, "rb") as attachment:
            part = MIMEApplication(attachment.read(), _subtype="pdf")
            filename = os.path.basename(report_file_path)
            part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
            msg.attach(part)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())

        return True, f"Soft copy successfully emailed to {recipient_email}"

    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

# =====================================================================
# API ENDPOINTS
# =====================================================================
@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/send-email")
def send_email_endpoint(payload: EmailRequest):
    success, message = send_report_via_email(
        payload.recipient_email, payload.patient_name, payload.report_file_path
    )
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "success", "message": message}
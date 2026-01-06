import os
os.environ["GRADIO_DISABLE_RELOAD"] = "true"

import gradio as gr
import numpy as np
from PIL import Image
import cv2
import smtplib
from email.mime.text import MIMEText

# ================= EMAIL CONFIG =================
SENDER_EMAIL = "marineedunet8203@gmail.com"
APP_PASSWORD = "aaiecegdqemqnayd"
RECEIVER_EMAIL = "authority1alert@gmail.com"  # Authority mail

# ================= EMAIL FUNCTION =================
def send_email_alert(spill_percent):
    subject = "🚨 MARINE OIL SPILL ALERT"
    body = f"""
ALERT 🚨

Marine Oil Spill Detected!

Spill Area: {spill_percent:.2f} %

Immediate action required.
Please initiate containment and cleanup procedures.

— AI Marine Oil Spill Detection System
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)

# ================= SAR IMAGE VALIDATION =================
def is_valid_sar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # SAR texture variance (SAR images are noisy)
    variance = np.var(gray)

    # SAR images are almost grayscale
    color_diff = np.mean(
        np.abs(img[:, :, 0] - img[:, :, 1]) +
        np.abs(img[:, :, 1] - img[:, :, 2])
    )

    if variance < 200 or color_diff > 15:
        return False
    return True

# ================= MAIN DETECTION =================
def detect_oil_spill(image):
    image = image.convert("RGB")
    image = image.resize((256, 256))
    img = np.array(image)

    # ❌ Invalid Image
    if not is_valid_sar(img):
        return (
            Image.fromarray(img),
            "❌ INVALID INPUT IMAGE\n\n"
            "This does not appear to be a SAR / Sea surface image.\n"
            "Please upload a valid SAR satellite image."
        )

    # ✅ SAR Processing
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

    output = img.copy()
    output[mask > 0] = [0, 100, 0]       # Oil spill (dark green)
    output[mask == 0] = [135, 206, 235]  # Sea (sky blue)

    spill_percent = np.sum(mask > 0) / mask.size * 100

    if spill_percent > 5:
        send_email_alert(spill_percent)
        msg = (
            "🚨 OIL SPILL DETECTED\n\n"
            f"Spill Area: {spill_percent:.2f}%\n"
            "⚠️ Immediate action required.\n"
            "📧 Email alert sent to authority."
        )
    else:
        msg = (
            "✅ NO SIGNIFICANT OIL SPILL\n\n"
            f"Spill Area: {spill_percent:.2f}%\n"
            "Sea surface appears normal."
        )

    return Image.fromarray(output), msg

# ================= UI =================
with gr.Blocks() as demo:
    gr.Markdown("## 🌊 Marine Oil Spill Detection & Alert System")

    with gr.Row():
        inp = gr.Image(type="pil", label="Upload SAR / Sea Image")
        out_img = gr.Image(label="Detection Output")
        out_txt = gr.Textbox(
            label="🖥️ System Status",
            lines=7,
            interactive=False
        )

    btn = gr.Button("🔍 Analyze Image")
    btn.click(detect_oil_spill, inp, [out_img, out_txt])

demo.launch(server_name="0.0.0.0", server_port=7860)

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from watermark_utils import apply_watermark_with_model,set_models
from model_loader import load_models

# Set page configuration FIRST
st.set_page_config(page_title="🔐 Watermark ML/DL App", layout="centered")

# Load and set all models only once
with st.spinner("🔄 Loading models..."):
    svm, gbm, pca_x, pca_y, cnn, xception, gan = load_models()
    set_models(svm, gbm, pca_x, pca_y, cnn, xception, gan)

# App Title and Intro
st.title("Multi-Technique Watermarking System")
st.markdown("""
Upload a cover image and optionally provide a text watermark.  
Choose a model, click **Apply Watermark**, and view results.
""")

# Sidebar Controls
st.sidebar.title("⚙️ Watermark Options")
mode = st.sidebar.radio("Choose Mode", ["Pretrained Watermark", "Custom Watermark"])
model_choice = st.sidebar.selectbox("Choose Model", ["SVM", "GBM", "CNN", "Xception", "GAN"])

# Upload Section
uploaded_file = st.file_uploader("Upload a Cover Image", type=["jpg", "jpeg", "png"])
cover_image = None
custom_watermark = None

if uploaded_file:
    cover_image = Image.open(uploaded_file).convert("RGB")
    st.image(cover_image, caption="Uploaded Cover Image", use_column_width=True)

    if mode == "Custom Watermark":
        watermark_text = st.text_input("Enter Watermark Text", max_chars=20)
        if watermark_text:
            custom_watermark = Image.new("L", (64, 64), color=0)
            draw = ImageDraw.Draw(custom_watermark)
            font = ImageFont.load_default()
            draw.text((10, 25), watermark_text, fill=255, font=font)

            # Show overlay preview
            preview = cover_image.copy().convert("RGB")
            draw_overlay = ImageDraw.Draw(preview)
            draw_overlay.text((10, 25), watermark_text, fill=(255, 0, 0), font=font)
            st.image(preview, caption="Preview: Text Watermark on Cover", use_column_width=True)

# Watermarking Button
if uploaded_file and st.button("🚀 Apply Watermark"):
    try:
        with st.spinner("🖼️ Applying watermark..."):
            watermarked_img, metrics = apply_watermark_with_model(
                model=model_choice,
                cover_image=cover_image,
                custom_watermark_img=custom_watermark
            )

        # Output Image
        st.subheader("🔍 Watermarked Image")
        st.image(watermarked_img, caption=f"Watermarked using {model_choice}", use_column_width=True)

        # Download button
        buf = BytesIO()
        watermarked_img.save(buf, format="PNG")
        st.download_button(
            label="📥 Download Watermarked Image",
            data=buf.getvalue(),
            file_name="watermarked_image.png",
            mime="image/png"
        )

        # Metrics Display
        st.subheader("📊 Evaluation Metrics")
        st.write(f"**MSE:** {metrics['mse']:.4f}")
        st.write(f"**PSNR:** {metrics['psnr']:.2f}")
        st.write(f"**SSIM:** {metrics['ssim']:.4f}")

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

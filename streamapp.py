import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from utils.watermark_utils import apply_watermark_with_model

# Set page configuration first
st.set_page_config(page_title="🔐 Watermark ML/DL App", layout="centered")

# App Title and Intro
st.title("Multi-Technique Watermarking System")
st.markdown("""
Welcome! Upload a cover image and optionally add a custom text watermark.  
Select a model and click **Apply Watermark** to view and evaluate results.
""")

# Sidebar Options
st.sidebar.title("⚙️ Watermark Options")
mode = st.sidebar.radio("Choose Mode", ["Pretrained Watermark", "Custom Watermark"])
model_choice = st.sidebar.selectbox("Choose Model", ["SVM", "GBM", "CNN", "Xception", "GAN"])

# Upload Image
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

            overlay_preview = cover_image.copy().convert("RGB")
            draw_overlay = ImageDraw.Draw(overlay_preview)
            draw_overlay.text((10, 25), watermark_text, fill=(255, 0, 0), font=font)
            st.image(overlay_preview, caption="Preview: Text Watermark on Cover", use_column_width=True)

# Apply Watermark Button
if uploaded_file and st.button("🚀 Apply Watermark"):
    try:
        if cover_image is None:
            st.error("Please upload a valid cover image.")
        else:
            with st.spinner("Applying watermark and generating results..."):
                watermarked_img, metrics = apply_watermark_with_model(
                    model=model_choice,
                    cover_image=cover_image,
                    custom_watermark_img=custom_watermark
                )

            # Display the watermarked result
            st.subheader("Watermarked Image")
            st.image(watermarked_img, caption=f"Watermarked using {model_choice}", use_column_width=True)

            # Download Button
            buf = BytesIO()
            watermarked_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Watermarked Image",
                data=byte_im,
                file_name="watermarked_image.png",
                mime="image/png"
            )

            # Show Evaluation Metrics
            st.subheader("Evaluation Metrics")
            st.write(f"**MSE:** {metrics['mse']:.4f}")
            st.write(f"**PSNR:** {metrics['psnr']:.2f}")
            st.write(f"**SSIM:** {metrics['ssim']:.4f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

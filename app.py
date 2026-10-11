import io
import os

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Optional Gemini SDK
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


st.set_page_config(
    page_title="Pixel Revive AI",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Pixel Revive AI")
st.write(
    "Restore faded, noisy, and slightly blurry photos. "
    "Upload your image and adjust the restoration settings."
)


# ---------------- SIDEBAR SETTINGS ----------------

with st.sidebar:
    st.header("Restoration Settings")

    preset = st.selectbox(
        "Choose restoration type",
        [
            "Balanced",
            "Faded Photo",
            "Noisy Photo",
            "Blurry Photo"
        ]
    )

    auto_contrast = st.checkbox(
        "Automatic contrast correction",
        value=True
    )

    denoise = st.checkbox(
        "Remove image noise",
        value=True
    )

    sharpen = st.checkbox(
        "Sharpen image",
        value=True
    )

    brightness = st.slider(
        "Brightness",
        0.7, 1.5, 1.0, 0.05
    )

    contrast = st.slider(
        "Contrast",
        0.7, 2.0, 1.15, 0.05
    )

    saturation = st.slider(
        "Color correction",
        0.0, 1.8, 1.05, 0.05
    )

    sharpness = st.slider(
        "Sharpening strength",
        0.0, 2.0, 0.8, 0.1
    )

    upscale = st.checkbox(
        "Upscale image 2×",
        value=False
    )


# ---------------- IMAGE RESTORATION ----------------

def restore_image(uploaded_image):
    # Correct image rotation from camera metadata
    image = ImageOps.exif_transpose(uploaded_image).convert("RGB")

    # Work with a copy of the image
    image_array = np.array(image)

    # Gentle automatic contrast enhancement
    if auto_contrast:
        image = ImageOps.autocontrast(image, cutoff=0.5)

    # Preset-specific settings
    denoise_strength = 4

    if preset == "Faded Photo":
        contrast_value = max(contrast, 1.25)
        saturation_value = max(saturation, 1.05)
        denoise_strength = 3

    elif preset == "Noisy Photo":
        contrast_value = contrast
        saturation_value = saturation
        denoise_strength = 6

    elif preset == "Blurry Photo":
        contrast_value = contrast
        saturation_value = saturation
        denoise_strength = 2

    else:
        contrast_value = contrast
        saturation_value = saturation

    # Correct brightness, contrast, and color
    image = ImageEnhance.Brightness(image).enhance(brightness)

    image = ImageEnhance.Contrast(image).enhance(
        contrast_value
    )

    image = ImageEnhance.Color(image).enhance(
        saturation_value
    )

    # Reduce color noise without excessive smoothing
    if denoise:
        image_array = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

        image_array = cv2.fastNlMeansDenoisingColored(
            image_array,
            None,
            denoise_strength,
            denoise_strength,
            7,
            21
        )

        image_array = cv2.cvtColor(
            image_array,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(image_array)

    # Sharpen edges carefully
    if sharpen and sharpness > 0:
        image = image.filter(
            ImageFilter.UnsharpMask(
                radius=1.2,
                percent=int(60 + sharpness * 80),
                threshold=4
            )
        )

    # Optional 2× resize
    if upscale:
        new_size = (
            image.width * 2,
            image.height * 2
        )

        image = image.resize(
            new_size,
            Image.Resampling.LANCZOS
        )

    return image


# ---------------- GEMINI RECOMMENDATION ----------------

def get_ai_recommendation(image):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        return (
            "AI recommendations are disabled. Add GEMINI_API_KEY "
            "in your hosting environment to enable them."
        )

    if not GEMINI_AVAILABLE:
        return (
            "The Gemini SDK is not installed. "
            "Install google-genai from requirements.txt."
        )

    try:
        client = genai.Client(api_key=api_key)

        prompt = (
            "Give three short, conservative recommendations for "
            "restoring a damaged photograph using image-processing "
            "techniques. Do not claim that missing image details "
            "can always be recovered. Image size: "
            f"{image.width}x{image.height}."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text or "No recommendation was returned."

    except Exception as error:
        return f"AI recommendation unavailable: {error}"


# ---------------- MAIN APPLICATION ----------------

uploaded_file = st.file_uploader(
    "Upload damaged photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        original = Image.open(uploaded_file)
        original.load()

        enhanced = restore_image(original)

        st.caption(
            f"Original resolution: "
            f"{original.width} × {original.height}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Photo")
            st.image(
                ImageOps.exif_transpose(original),
                use_container_width=True
            )

        with col2:
            st.subheader("Restored Preview")
            st.image(
                enhanced,
                use_container_width=True
            )

        # Download as PNG to avoid additional JPEG compression
        output = io.BytesIO()
        enhanced.save(output, format="PNG", optimize=True)

        st.download_button(
            label="Download Restored Photo",
            data=output.getvalue(),
            file_name="pixel_revive_restored.png",
            mime="image/png",
            use_container_width=True
        )

        st.divider()

        st.subheader("AI Restoration Recommendations")

        if st.button("Get AI Recommendations"):
            with st.spinner("Generating recommendations..."):
                st.session_state["ai_advice"] = (
                    get_ai_recommendation(original)
                )

        if "ai_advice" in st.session_state:
            st.info(st.session_state["ai_advice"])

        st.warning(
            "This version improves contrast, noise, and edge clarity. "
            "Severe blur, missing areas, and deep scratches may require "
            "a dedicated AI photo-restoration model."
        )

    except Exception as error:
        st.error(f"Could not process this photo: {error}")

else:
    st.info("Upload a photo to start restoration.")


st.divider()
st.caption("Pixel Revive AI | Photo Enhancement Project")

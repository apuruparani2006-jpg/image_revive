
import io
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Pixel Revive AI",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Pixel Revive AI")
st.write(
    "Enhance old photos and use an AI agent to explain "
    "images, Verilog HDL, and 8086 assembly code."
)


# --------------------------------------------------
# GEMINI CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "gemini-3.8-flash"


def get_api_key():
    """Read the API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    return os.environ.get("GEMINI_API_KEY", "")


def ask_agent(prompt, image=None):
    """Ask Gemini to generate an answer for the current task."""
    if not GEMINI_AVAILABLE:
        return (
            "The Google Gen AI package is missing. "
            "Run: pip install --upgrade google-genai"
        )

    api_key = get_api_key()
    if not api_key:
        return (
            "API key not found. Add GEMINI_API_KEY to "
            ".streamlit/secrets.toml and restart the app."
        )

    try:
        client = genai.Client(api_key=api_key)

        contents = [prompt]
        if image is not None:
            contents.append(image.convert("RGB"))

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
        )

        answer = response.text
        if not answer:
            return "The AI returned no text. Please try again."

        return answer

    except Exception as exc:
        return (
            "The Gemini request failed.\n\n"
            f"Error: {exc}\n\n"
            "Check your API key, internet connection, model "
            "availability, and API access."
        )


# --------------------------------------------------
# IMAGE ENHANCEMENT
# --------------------------------------------------

def enhance_photo(pil_image, denoise_strength=5,
                  contrast=1.08, brightness=1.02):
    """
    Perform conservative image enhancement.

    This function does not automatically erase scratches.
    That is intentional: an automatic scratch mask can
    mistake eyes, hair, skin texture, and other details
    for damage.
    """
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Gentle denoising; keep the original image dimensions.
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=denoise_strength,
        hColor=denoise_strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # Adjust contrast and brightness.
    adjusted = cv2.convertScaleAbs(
        denoised,
        alpha=contrast,
        beta=(brightness - 1.0) * 255,
    )

    # Mild sharpening, avoiding aggressive edge enhancement.
    blurred = cv2.GaussianBlur(adjusted, (0, 0), 1.0)
    sharpened = cv2.addWeighted(
        adjusted, 1.15, blurred, -0.15, 0
    )

    output_rgb = cv2.cvtColor(
        sharpened, cv2.COLOR_BGR2RGB
    )
    return Image.fromarray(output_rgb)


# --------------------------------------------------
# PHOTO RESTORATION INTERFACE
# --------------------------------------------------

st.header("1. Photo Enhancement")

uploaded_photo = st.file_uploader(
    "Upload an old or damaged photo",
    type=["png", "jpg", "jpeg", "webp"],
    key="photo_uploader",
)

if uploaded_photo is not None:
    try:
        original = Image.open(uploaded_photo).convert("RGB")

        st.subheader("Enhancement settings")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            denoise = st.slider(
                "Denoising", 0, 15, 5
            )

        with col_b:
            contrast = st.slider(
                "Contrast", 0.80, 1.40, 1.08, 0.01
            )

        with col_c:
            brightness = st.slider(
                "Brightness", 0.80, 1.30, 1.02, 0.01
            )

        if st.button("Enhance Photo", type="primary"):
            with st.spinner("Enhancing your photo..."):
                result = enhance_photo(
                    original,
                    denoise_strength=denoise,
                    contrast=contrast,
                    brightness=brightness,
                )
                st.session_state["enhanced_photo"] = result

        result = st.session_state.get("enhanced_photo")

        if result is not None:
            left, right = st.columns(2)

            with left:
                st.subheader("Original")
                st.image(original, use_container_width=True)

            with right:
                st.subheader("Enhanced")
                st.image(result, use_container_width=True)

            buffer = io.BytesIO()
            result.save(buffer, format="PNG")

            st.download_button(
                "Download Enhanced Photo",
                data=buffer.getvalue(),
                file_name="pixel_revive_enhanced.png",
                mime="image/png",
            )
        else:
            st.image(
                original,
                caption="Uploaded photo",
                use_container_width=True,
            )

    except Exception as exc:
        st.error(f"Could not open this image: {exc}")


# --------------------------------------------------
# AI AGENT: IMAGE ANALYSIS
# --------------------------------------------------

st.divider()
st.header("2. AI Agent — Image Analysis")

if uploaded_photo is not None:
    if st.button("Analyze Uploaded Photo"):
        try:
            photo_for_ai = Image.open(
                uploaded_photo
            ).convert("RGB")

            prompt = """
            Act as a photo restoration assistant.
            Describe the visible condition of this photo.
            Identify likely issues such as fading, noise,
            scratches, blur, or low contrast.
            Recommend conservative restoration steps.
            Do not claim to have restored pixels just by
            describing the image.
            """
            with st.spinner("AI is analyzing the photo..."):
                st.markdown(ask_agent(prompt, photo_for_ai))
        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")
else:
    st.info("Upload a photo above to enable image analysis.")


# --------------------------------------------------
# AI AGENT: VERILOG EXPLANATION
# --------------------------------------------------

st.divider()
st.header("3. AI Agent — Verilog HDL")

default_verilog = """module brightness_adder (
    input  wire [7:0] pixel,
    input  wire [7:0] brightness,
    output wire [7:0] result
);
    wire [8:0] sum;
    assign sum = {1'b0, pixel} + {1'b0, brightness};
    assign result = sum[8] ? 8'hFF : sum[7:0];
endmodule
"""

verilog_code = st.text_area(
    "Paste your Verilog code",
    value=default_verilog,
    height=230,
    key="verilog_input",
)

if st.button("Explain Verilog Code", type="primary"):
    if not verilog_code.strip():
        st.warning("Please enter some Verilog code first.")
    else:
        prompt = f"""
You are an experienced digital-design tutor.
Explain the user's actual Verilog code below.

Organize the answer into:
1. Purpose of the module
2. Inputs and outputs, including bit widths
3. Explanation of important statements
4. How the logic behaves for example inputs
5. Potential bugs, width issues, or synthesis concerns
6. A short summary in beginner-friendly language

Be accurate. If something cannot be determined from
the code, say so. Do not invent signals or behavior.

VERILOG CODE:
{verilog_code}
"""
        with st.spinner("AI agent is reading the Verilog code..."):
            st.markdown(ask_agent(prompt))


# --------------------------------------------------
# AI AGENT: 8086 ASSEMBLY EXPLANATION
# --------------------------------------------------

st.divider()
st.header("4. AI Agent — 8086 Assembly")

default_asm = """; Example only: add a value to AL with saturation
MOV AL, 100
MOV BL, 30
ADD AL, BL
JNC NO_OVERFLOW
MOV AL, 255
NO_OVERFLOW:
HLT
"""

asm_code = st.text_area(
    "Paste your 8086 assembly code",
    value=default_asm,
    height=230,
    key="asm_input",
)

if st.button("Explain 8086 Code", type="primary"):
    if not asm_code.strip():
        st.warning("Please enter some assembly code first.")
    else:
        prompt = f"""
You are an experienced 8086 assembly-language tutor.
Explain the user's actual code below.

Organize the answer into:
1. What the program is intended to do
2. Explanation of each instruction in order
3. Registers and flags that are affected
4. A step-by-step trace using the actual instructions
5. Potential errors or assumptions about assembler syntax
6. A short beginner-friendly summary

Do not pretend to execute the program. If the intended
behavior is ambiguous, explain the uncertainty.

8086 ASSEMBLY CODE:
{asm_code}
"""
        with st.spinner("AI agent is reading the assembly code..."):
            st.markdown(ask_agent(prompt))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Pixel Revive AI | AI-generated explanations may contain "
    "errors. Verify code behavior before using it in a project."
)

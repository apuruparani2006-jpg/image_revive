import os
import io
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import streamlit as st

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

st.set_page_config(page_title="Pixel Revive AI", page_icon="🖼️", layout="wide")
st.title("🖼️ Pixel Revive AI")
st.subheader("AI-assisted photo enhancement with Verilog and 8086 ALP demonstrations")
st.write("Upload a photo, adjust enhancement settings, compare the images, and inspect example ECE implementations.")

with st.sidebar:
    st.header("Restoration settings")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.10, 0.05)
    contrast = st.slider("Contrast", 0.5, 2.5, 1.35, 0.05)
    color = st.slider("Color saturation", 0.0, 2.0, 1.10, 0.05)
    sharpen = st.checkbox("Apply light sharpening", value=True)
    denoise = st.checkbox("Apply light noise reduction", value=False)
    st.caption("These are conventional image-processing operations, not a trained photo-reconstruction model.")

def restore_image(image):
    result = ImageOps.exif_transpose(image).convert("RGB")
    result = ImageEnhance.Brightness(result).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)
    result = ImageEnhance.Color(result).enhance(color)
    if denoise:
        result = result.filter(ImageFilter.MedianFilter(size=3))
    if sharpen:
        result = result.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    return result

def gemini_recommendation(image_description):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or not GEMINI_AVAILABLE:
        return ("Basic enhancement workflow selected: adjust brightness and contrast, "
                "then optionally apply light denoising and sharpening. "
                "No Gemini agent is connected; set GEMINI_API_KEY to enable AI recommendations.")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = (
            "You are the image-restoration decision agent in an ECE student demo. "
            "Recommend a conservative sequence of conventional operations for this image. "
            "Do not claim to reconstruct missing details or identify people. "
            "Keep the answer to 3 short bullet points. "
            f"Image metadata: {image_description}"
        )
        response = model.generate_content(prompt)
        return response.text or "No recommendation was returned."
    except Exception as exc:
        return f"AI recommendation unavailable: {exc}"

uploaded = st.file_uploader("Upload a JPG, JPEG, or PNG photo", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        original = Image.open(uploaded)
        processed = restore_image(original)

        left, right = st.columns(2)
        with left:
            st.markdown("#### Original image")
            st.image(original, use_container_width=True)
        with right:
            st.markdown("#### Enhanced preview")
            st.image(processed, use_container_width=True)

        buffer = io.BytesIO()
        processed.save(buffer, format="PNG")
        st.download_button("Download enhanced image", buffer.getvalue(),
                           file_name="pixel_revive_enhanced.png", mime="image/png")

        st.markdown("### Restoration agent")
        image_info = f"format={original.format}, size={original.width}x{original.height}, mode={original.mode}"
        if st.button("Analyze image and recommend operations"):
            with st.spinner("Preparing recommendation..."):
                st.session_state["agent_recommendation"] = gemini_recommendation(image_info)
        if "agent_recommendation" in st.session_state:
            st.info(st.session_state["agent_recommendation"])

        st.markdown("### Verilog example — pixel brightness adjustment")
        verilog = """// 8-bit grayscale pixel brightness adjustment with saturation
module pixel_brightness (
    input  wire [7:0] pixel_in,
    input  wire [7:0] brightness_add,
    output wire [7:0] pixel_out
);
    wire [8:0] sum;
    assign sum = {1'b0, pixel_in} + {1'b0, brightness_add};
    assign pixel_out = sum[8] ? 8'd255 : sum[7:0];
endmodule"""
        st.code(verilog, language="verilog")

        st.markdown("### 8086 ALP example — one pixel value")
        asm8086 = "; 8086 ALP: add a brightness offset to one 8-bit pixel\n; Input: AL = grayscale pixel, BL = brightness offset\n; Output: AL = saturated result (maximum 255)\n        ADD AL, BL\n        JNC NO_OVERFLOW\n        MOV AL, 0FFH\nNO_OVERFLOW:\n        ; AL now contains the adjusted pixel value"
        st.code(asm8086, language="asm")

        st.markdown("### Project notes")
        st.write(
            "- Python/Pillow performs the actual preview enhancement.\n"
            "- Gemini is optional and provides text recommendations when GEMINI_API_KEY is configured.\n"
            "- Verilog and 8086 ALP are demonstrations of a simple pixel arithmetic operation.\n"
            "- This prototype does not claim AI face reconstruction or recovery of missing photo details."
        )
    except Exception as exc:
        st.error(f"Could not open or process this image: {exc}")
else:
    st.info("Upload an image to begin. Image enhancement works without a Gemini API key.")

st.divider()
st.caption("Pixel Revive AI • Student ECE demonstration • Software-only prototype")

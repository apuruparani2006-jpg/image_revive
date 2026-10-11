
import io
import os
import textwrap

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance

# Gemini agent
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


st.set_page_config(
    page_title="Pixel Revive AI",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Pixel Revive AI")
st.subheader("AI-assisted photo restoration and hardware-code explanation")

st.write(
    "Restore damaged photos, get AI-generated restoration recommendations, "
    "and ask the agent to explain Verilog and 8086 assembly code."
)


# ---------------------------------------------------------
# API configuration
# ---------------------------------------------------------

def get_api_key():
    key = os.getenv("GEMINI_API_KEY", "").strip()

    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "").strip()
        except Exception:
            key = ""

    return key


def ask_agent(prompt, image=None):
    """Ask Gemini to generate a response dynamically."""
    api_key = get_api_key()

    if not GEMINI_AVAILABLE:
        return (
            "Gemini SDK is not installed. Install google-genai "
            "using the requirements.txt file."
        )

    if not api_key:
        return (
            "Gemini API key is missing. Add GEMINI_API_KEY to your "
            "environment variables or Streamlit secrets."
        )

    try:
        client = genai.Client(api_key=api_key)

        contents = [prompt]
        if image is not None:
            contents.append(image.convert("RGB"))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )

        answer = getattr(response, "text", None)
        return answer or "The AI agent returned an empty response."

    except Exception as exc:
        return (
            "The AI agent could not complete the request. "
            f"Details: {exc}"
        )


# ---------------------------------------------------------
# Photo restoration
# ---------------------------------------------------------

def restore_photo(
    image,
    scratch_strength=5,
    denoise=True,
    brightness=1.0,
    contrast=1.05,
    sharpening=0,
):
    """
    Attempt to remove thin scratch-like marks using OpenCV inpainting.
    The automatic mask can mistakenly select image details.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")

    # Limit working dimensions to control memory usage.
    image.thumbnail((1800, 1800))

    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Detect bright and dark thin marks.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (17, 17)
    )

    bright_marks = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, kernel
    )
    dark_marks = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, kernel
    )

    marks = cv2.bitwise_or(bright_marks, dark_marks)

    # Higher strength uses a lower threshold and detects more marks.
    threshold = max(12, 65 - scratch_strength * 5)

    _, mask = cv2.threshold(
        marks, threshold, 255, cv2.THRESH_BINARY
    )

    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        ),
        iterations=1,
    )

    # Discard very small isolated detections.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    clean_mask = np.zeros_like(mask)

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 3:
            clean_mask[labels == i] = 255

    # Repair selected areas from their surrounding pixels.
    restored_bgr = cv2.inpaint(
        bgr, clean_mask, 3, cv2.INPAINT_TELEA
    )

    restored_rgb = cv2.cvtColor(
        restored_bgr, cv2.COLOR_BGR2RGB
    )
    result = Image.fromarray(restored_rgb)

    if denoise:
        array = np.array(result)
        array_bgr = cv2.cvtColor(
            array, cv2.COLOR_RGB2BGR
        )
        array_bgr = cv2.fastNlMeansDenoisingColored(
            array_bgr, None, 5, 5, 7, 21
        )
        result = Image.fromarray(
            cv2.cvtColor(array_bgr, cv2.COLOR_BGR2RGB)
        )

    result = ImageEnhance.Brightness(result).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)

    if sharpening > 0:
        result_array = np.array(result)
        result_bgr = cv2.cvtColor(
            result_array, cv2.COLOR_RGB2BGR
        )
        result_bgr = cv2.detailEnhance(
            result_bgr, sigma_s=10, sigma_r=0.15
        )
        enhanced = cv2.cvtColor(
            result_bgr, cv2.COLOR_BGR2RGB
        )
        # Blend lightly to avoid excessive artificial sharpening.
        result = Image.blend(
            result,
            Image.fromarray(enhanced),
            min(sharpening / 100.0, 0.5),
        )

    return result, Image.fromarray(clean_mask)


# ---------------------------------------------------------
# Verilog and 8086 examples
# The agent explains these strings dynamically.
# ---------------------------------------------------------

VERILOG_CODE = textwrap.dedent("""
module pixel_brightness (
    input  wire [7:0] pixel_in,
    input  wire [7:0] brightness_add,
    output wire [7:0] pixel_out
);

    wire [8:0] sum;

    assign sum = {1'b0, pixel_in}
               + {1'b0, brightness_add};

    assign pixel_out = sum[8] ? 8'd255 : sum[7:0];

endmodule
""").strip()


ASM_CODE = textwrap.dedent("""
; Input: AL = original pixel value
;        BL = brightness offset
; Output: AL = adjusted pixel value

MOV AH, 00H
ADD AL, BL
JNC NO_OVERFLOW
MOV AL, 0FFH

NO_OVERFLOW:
; AL contains the saturated result
""").strip()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Restoration settings")

    scratch_strength = st.slider(
        "Scratch detection strength",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values detect more marks but can alter details.",
    )

    denoise = st.checkbox(
        "Reduce image noise",
        value=True,
    )

    brightness = st.slider(
        "Brightness",
        min_value=0.7,
        max_value=1.5,
        value=1.0,
        step=0.05,
    )

    contrast = st.slider(
        "Contrast",
        min_value=0.7,
        max_value=1.8,
        value=1.05,
        step=0.05,
    )

    sharpening = st.slider(
        "Detail enhancement",
        min_value=0,
        max_value=30,
        value=0,
        step=5,
    )

    show_mask = st.checkbox(
        "Show detected scratch areas",
        value=False,
    )


# ---------------------------------------------------------
# Main photo restoration interface
# ---------------------------------------------------------

uploaded = st.file_uploader(
    "Upload a damaged photo",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is not None:
    try:
        original = ImageOps.exif_transpose(
            Image.open(uploaded)
        ).convert("RGB")

        with st.spinner("Processing the photo..."):
            restored, scratch_mask = restore_photo(
                original,
                scratch_strength=scratch_strength,
                denoise=denoise,
                brightness=brightness,
                contrast=contrast,
                sharpening=sharpening,
            )

        left, right = st.columns(2)

        with left:
            st.subheader("Original photo")
            st.image(original, use_container_width=True)

        with right:
            st.subheader("Restored preview")
            st.image(restored, use_container_width=True)

        output = io.BytesIO()
        restored.save(output, format="PNG")

        st.download_button(
            "Download restored photo",
            data=output.getvalue(),
            file_name="pixel_revive_restored.png",
            mime="image/png",
        )

        if show_mask:
            st.subheader("Detected scratch areas")
            st.caption(
                "White regions are selected for repair. "
                "If facial details are selected, reduce the strength."
            )
            st.image(scratch_mask, use_container_width=True)

        st.divider()
        st.header("🤖 Photo Restoration Agent")

        st.write(
            "Ask the AI agent to inspect the photo and recommend "
            "a restoration workflow."
        )

        if st.button("Analyze photo and recommend restoration"):
            with st.spinner("The AI agent is analyzing the photo..."):
                prompt = """
You are the restoration decision agent inside Pixel Revive AI.

Inspect the supplied photo and provide:
1. Visible image-quality problems, such as scratches, fading,
   noise, blur, or contrast issues.
2. A recommended order of restoration operations.
3. Which settings in this application should be adjusted.
4. Limitations and areas where automatic repair could damage detail.

Be honest about uncertainty. Do not claim that missing information
has been recovered. Distinguish visible observations from guesses.
Keep the response practical and concise.
"""
                st.session_state["photo_agent_result"] = ask_agent(
                    prompt,
                    image=original,
                )

        if "photo_agent_result" in st.session_state:
            st.markdown(st.session_state["photo_agent_result"])

        st.caption(
            "The agent recommends actions; the restoration function "
            "performs the actual image processing."
        )

    except Exception as exc:
        st.error(f"Could not process the photo: {exc}")

else:
    st.info("Upload a photo to start restoring it.")


# ---------------------------------------------------------
# Agentic Verilog explainer
# ---------------------------------------------------------

st.divider()
st.header("⚙️ Verilog Code and AI Explanation")

st.write(
    "The agent will explain the code below when requested. "
    "You can also edit the code before asking for an explanation."
)

verilog_code = st.text_area(
    "Verilog source code",
    value=VERILOG_CODE,
    height=260,
    key="verilog_source",
)

if st.button("🤖 Explain Verilog Code"):
    if not verilog_code.strip():
        st.warning("Enter Verilog code first.")
    else:
        prompt = f"""
You are the Verilog tutor agent in Pixel Revive AI.

Explain the user's actual Verilog source code, not a generic example.

Source code:
```verilog
{verilog_code}
```

Organize the answer into:
1. Purpose of the circuit.
2. Explanation of every input and output.
3. Line-by-line explanation of the important statements.
4. How the logic works.
5. At least three input/output examples when possible.
6. Any bugs, limitations, or synthesis concerns you can identify.
7. How the circuit relates to pixel processing in the photo app.

If you cannot determine something from the source, say so.
Do not claim the Verilog is connected to the Python app unless
the provided code demonstrates that connection.
Use clear language suitable for a student project presentation.
"""
        with st.spinner("Verilog agent is preparing its explanation..."):
            st.session_state["verilog_explanation"] = ask_agent(prompt)

if "verilog_explanation" in st.session_state:
    st.markdown(st.session_state["verilog_explanation"])


# ---------------------------------------------------------
# Agentic 8086 assembly explainer
# ---------------------------------------------------------

st.divider()
st.header("💻 8086 Assembly Code and AI Explanation")

asm_code = st.text_area(
    "8086 assembly source code",
    value=ASM_CODE,
    height=220,
    key="asm_source",
)

if st.button("🤖 Explain 8086 Assembly Code"):
    if not asm_code.strip():
        st.warning("Enter assembly code first.")
    else:
        prompt = f"""
You are the 8086 assembly tutor agent in Pixel Revive AI.

Explain this actual assembly source code:
```asm
{asm_code}
```

Include:
1. Purpose of the program.
2. Registers used and what each stores.
3. Explanation of each instruction.
4. Control flow and flags used.
5. A simple worked example.
6. Any limitations or errors in the code.
7. How the example relates conceptually to pixel processing.

Do not invent instructions that are not present. Use simple,
presentation-friendly language.
"""
        with st.spinner("Assembly agent is preparing its explanation..."):
            st.session_state["asm_explanation"] = ask_agent(prompt)

if "asm_explanation" in st.session_state:
    st.markdown(st.session_state["asm_explanation"])


# ---------------------------------------------------------
# Project notes
# ---------------------------------------------------------

st.divider()
st.subheader("Project architecture")

st.markdown("""
- **Python and OpenCV:** perform scratch-mask detection, inpainting,
  denoising, and image adjustments.
- **Gemini agent:** analyzes the supplied image and dynamically
  generates recommendations and code explanations.
- **Verilog:** demonstrates a digital pixel-brightness circuit.
- **8086 Assembly:** demonstrates processor instructions for
  brightness adjustment.
- **Integration status:** the Verilog and 8086 examples are
  educational demonstrations; they are not connected to the
  Python image-processing pipeline.
""")

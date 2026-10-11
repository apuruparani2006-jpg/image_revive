
import io
import json
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from google import genai
except ImportError:
    genai = None

st.set_page_config(
    page_title="Pixel Revive AI",
    page_icon="🖼️",
    layout="wide",
)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_URLS = {
    "prototxt": "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
    "points": "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy",
    "weights": "https://github.com/richzhang/colorization/raw/caffe/colorization_release_v2.caffemodel",
}


def setting(name, default=""):
    try:
        value = st.secrets.get(name, "")
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


def stats(image):
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "saturation": float(np.mean(hsv[:, :, 1])),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def plan(image):
    s = stats(image)
    fallback = {
        "colorize": s["saturation"] < 12,
        "denoise": True,
        "sharpen": s["sharpness"] < 180,
        "reason": "Local image-statistics fallback",
        "planner": "Local fallback",
    }

    key = setting("GEMINI_API_KEY")
    if not key or genai is None:
        fallback["reason"] += (
            "; Gemini key/SDK unavailable, so automatic local planning was used."
        )
        return fallback

    model = setting("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = f"""
You are a photo restoration planning agent.
Image statistics: {json.dumps(s)}
Return only JSON with boolean fields colorize, denoise, sharpen,
and a short string reason.
Colorize only if grayscale. Denoise old/noisy images.
Sharpen only if soft. Be conservative. Do not recommend aggressive
scratch removal because it may destroy facial features.
"""

    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=[prompt, image.convert("RGB")],
            config={"response_mime_type": "application/json"},
        )
        decision = json.loads(response.text or "{}")
        for field in ("colorize", "denoise", "sharpen"):
            if not isinstance(decision.get(field), bool):
                raise ValueError(f"Invalid agent decision: {field}")
        decision["reason"] = str(decision.get("reason", "No reason provided"))
        decision["planner"] = f"Gemini ({model})"
        return decision
    except Exception as exc:
        fallback["reason"] += f"; Gemini failed, using fallback: {exc}"
        return fallback


def download(url, path):
    if path.exists() and path.stat().st_size > 1000:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "PixelReviveAI"}
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            with open(temp, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        if temp.stat().st_size < 1000:
            raise RuntimeError(f"Invalid model download: {path.name}")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def colorize(image):
    proto = MODEL_DIR / "colorization_deploy_v2.prototxt"
    points_file = MODEL_DIR / "pts_in_hull.npy"
    weights = MODEL_DIR / "colorization_release_v2.caffemodel"

    download(MODEL_URLS["prototxt"], proto)
    download(MODEL_URLS["points"], points_file)
    download(MODEL_URLS["weights"], weights)

    net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
    points = np.load(str(points_file)).transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [
        points.astype(np.float32)
    ]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full((1, 313), 2.606, dtype=np.float32)
    ]

    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    normalized = bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

    blob = cv2.dnn.blobFromImage(
        normalized, 1.0, (224, 224), (50, 50, 50),
        swapRB=False, crop=False
    )
    net.setInput(blob)
    ab = net.forward("class8_ab")[0].transpose(1, 2, 0)
    ab = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC)

    output_lab = np.concatenate([lab[:, :, 0, None], ab], axis=2)
    output = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    output = np.clip(output, 0, 1)
    output = cv2.cvtColor(
        (output * 255).astype(np.uint8), cv2.COLOR_BGR2RGB
    )
    return Image.fromarray(output)


def denoise(image):
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    result = cv2.fastNlMeansDenoisingColored(
        bgr, None, 4, 4, 7, 21
    )
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def sharpen(image):
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    result = cv2.addWeighted(bgr, 1.12, blur, -0.12, 0)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def quality_check(original, result, should_colorize):
    s = stats(result)
    issues = []
    if result.size != original.size:
        issues.append("Image dimensions changed.")
    if should_colorize and s["saturation"] < 8:
        issues.append("Output remains nearly grayscale.")
    if not issues:
        issues.append("Basic checks passed; visually inspect the result.")
    return issues, s


def restore(original):
    decision = plan(original)
    result = original.convert("RGB")
    log = []

    if decision["colorize"]:
        result = colorize(result)
        log.append("Colorization executed")
    else:
        log.append("Colorization skipped: image already has color")

    if decision["denoise"]:
        result = denoise(result)
        log.append("Denoising executed")

    if decision["sharpen"]:
        result = sharpen(result)
        log.append("Sharpening executed")

    checks, output_stats = quality_check(
        original, result, decision["colorize"]
    )
    retried = False

    # One retry from the original if colorization remains nearly grayscale.
    if decision["colorize"] and output_stats["saturation"] < 8:
        retried = True
        log.append("Colorization quality check failed; retrying once")
        result = colorize(original)
        if decision["denoise"]:
            result = denoise(result)
        checks, output_stats = quality_check(
            original, result, decision["colorize"]
        )

    return result, decision, log, checks, output_stats, retried


st.title("🖼️ Pixel Revive AI")
st.write(
    "Automatically plan and run photo restoration tools, check the output, "
    "and download the result."
)

uploaded = st.file_uploader(
    "Upload a black-and-white or damaged color photo",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    original = Image.open(uploaded).convert("RGB")
    upload_id = f"{uploaded.name}-{uploaded.size}"

    if st.session_state.get("upload_id") != upload_id:
        st.session_state["upload_id"] = upload_id
        for key in ("result", "decision", "log", "checks", "output_stats", "retried"):
            st.session_state.pop(key, None)

    st.subheader("Original photograph")
    st.image(original, use_container_width=True)

    if st.button("Restore photo automatically", type="primary"):
        with st.spinner("Agent is analyzing and restoring your photo..."):
            try:
                result, decision, log, checks, output_stats, retried = restore(original)
                st.session_state["result"] = result
                st.session_state["decision"] = decision
                st.session_state["log"] = log
                st.session_state["checks"] = checks
                st.session_state["output_stats"] = output_stats
                st.session_state["retried"] = retried
            except Exception as exc:
                st.error(
                    "Restoration failed. Check internet access for the first "
                    f"model download, then try again.\n\nDetails: {exc}"
                )

    result = st.session_state.get("result")
    if result is not None:
        left, right = st.columns(2)
        with left:
            st.subheader("Original")
            st.image(original, use_container_width=True)
        with right:
            st.subheader("Restored")
            st.image(result, use_container_width=True)

        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        st.download_button(
            "Download restored photo",
            data=buffer.getvalue(),
            file_name="pixel_revive_restored.png",
            mime="image/png",
        )

        with st.expander("Agent plan and quality report", expanded=True):
            st.write("**Planner:**", st.session_state["decision"]["planner"])
            st.write("**Reason:**", st.session_state["decision"]["reason"])
            st.write("**Tool log:**")
            for line in st.session_state["log"]:
                st.write("- " + line)
            st.write("**Quality checks:**")
            for line in st.session_state["checks"]:
                st.write("- " + line)
            if st.session_state["retried"]:
                st.info("The agent retried colorization once.")

st.caption(
    "Colorization estimates plausible colors, not historically exact colors. "
    "This version uses conservative enhancement and does not automatically "
    "erase scratches, because that can damage facial details."
)

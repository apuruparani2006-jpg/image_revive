
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

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Corrected configuration and cluster-point locations.
# The large model weights use a direct-download link.
MODEL_URLS = {
    "prototxt": (
        "https://raw.githubusercontent.com/richzhang/"
        "colorization/master/colorization/models/"
        "colorization_deploy_v2.prototxt"
    ),
    "points": (
        "https://github.com/richzhang/colorization/blob/"
        "master/colorization/resources/pts_in_hull.npy?raw=true"
    ),
    "weights": (
        "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/"
        "colorization_release_v2.caffemodel?dl=1"
    ),
}


def setting(name, default=""):
    try:
        value = st.secrets.get(name, "")
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


def image_stats(image):
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "saturation": float(np.mean(hsv[:, :, 1])),
        "sharpness": float(cv2.Laplacian(
            gray, cv2.CV_64F
        ).var()),
        "contrast": float(np.std(gray)),
    }


def plan_restoration(image):
    s = image_stats(image)
    plan = {
        "colorize": s["saturation"] < 12,
        "denoise": True,
        "sharpen": s["sharpness"] < 180,
        "reason": "Local image-statistics fallback.",
        "planner": "Local fallback",
    }

    key = setting("GEMINI_API_KEY")
    if not key or genai is None:
        plan["reason"] += " Gemini is not configured."
        return plan

    model = setting("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = f"""
You are the planning agent for a photo restoration app.

Image statistics: {json.dumps(s)}

Return ONLY JSON with boolean fields:
colorize, denoise, sharpen
and a string field reason.

Colorize grayscale photos only.
Denoise noisy or aged images.
Sharpen only if the image is soft.
Do not claim you have inspected details you cannot verify.
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

        decision["reason"] = str(decision.get("reason", "No reason given"))
        decision["planner"] = f"Gemini ({model})"
        return decision

    except Exception as exc:
        plan["reason"] += f" Gemini planning failed: {exc}"
        return plan


def download_model(name):
    destination = MODEL_DIR / name
    if destination.exists() and destination.stat().st_size > 1000:
        return destination

    url = MODEL_URLS[name]
    temporary = destination.with_suffix(destination.suffix + ".tmp")

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "PixelReviveAI/1.0"},
        )

        with urllib.request.urlopen(request, timeout=180) as response:
            with open(temporary, "wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)

        if temporary.stat().st_size < 1000:
            raise RuntimeError(
                f"Downloaded file {name} is too small or invalid."
            )

        temporary.replace(destination)
        return destination

    except Exception as exc:
        raise RuntimeError(
            f"Download failed for '{name}'. URL: {url}. "
            f"Details: {exc}"
        ) from exc

    finally:
        if temporary.exists():
            temporary.unlink()


@st.cache_resource
def load_colorization_model():
    proto = download_model("prototxt")
    points_path = download_model("points")
    weights = download_model("weights")

    net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
    points = np.load(str(points_path), allow_pickle=False)

    if points.shape != (313, 2):
        raise RuntimeError(
            f"Unexpected cluster-points shape: {points.shape}"
        )

    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [
        points.astype(np.float32)
    ]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full((1, 313), 2.606, dtype=np.float32)
    ]
    return net


def colorize_photo(image):
    net = load_colorization_model()

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]

    # Convert to RGB Lab so lightness is preserved.
    rgb_float = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]

    resized_l = cv2.resize(
        lightness, (224, 224), interpolation=cv2.INTER_AREA
    )
    resized_l -= 50

    blob = cv2.dnn.blobFromImage(resized_l)
    net.setInput(blob)
    ab = net.forward("class8_ab")[0].transpose(1, 2, 0)
    ab = cv2.resize(ab, (width, height), interpolation=cv2.INTER_CUBIC)

    output_lab = np.concatenate(
        [lightness[:, :, None], ab], axis=2
    )
    output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)
    output_rgb = np.clip(output_rgb, 0, 1)

    return Image.fromarray(
        (output_rgb * 255).astype(np.uint8)
    )


def denoise_photo(image):
    bgr = cv2.cvtColor(
        np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR
    )
    output = cv2.fastNlMeansDenoisingColored(
        bgr, None, 4, 4, 7, 21
    )
    return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


def sharpen_photo(image):
    rgb = np.asarray(image.convert("RGB"))
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    output = cv2.addWeighted(rgb, 1.12, blur, -0.12, 0)
    return Image.fromarray(output)


def check_quality(original, result, colorization_requested):
    before = image_stats(original)
    after = image_stats(result)
    issues = []

    if original.size != result.size:
        issues.append("Output dimensions changed.")

    if colorization_requested and after["saturation"] < 8:
        issues.append("Output is still almost grayscale.")

    if after["sharpness"] < before["sharpness"] * 0.2:
        issues.append("Output became substantially softer.")

    if not issues:
        issues.append("Basic statistical checks passed; inspect visually.")

    return issues, after


def restore_photo(original):
    decision = plan_restoration(original)
    result = original.convert("RGB")
    log = []

    if decision["colorize"]:
        result = colorize_photo(result)
        log.append("Colorization executed.")
    else:
        log.append("Colorization skipped: photo appears to contain color.")

    if decision["denoise"]:
        result = denoise_photo(result)
        log.append("Denoising executed.")

    if decision["sharpen"]:
        result = sharpen_photo(result)
        log.append("Sharpening executed.")

    checks, stats = check_quality(
        original, result, decision["colorize"]
    )
    retried = False

    if decision["colorize"] and stats["saturation"] < 8:
        retried = True
        log.append("Quality check failed; retrying colorization once.")
        result = colorize_photo(original)

        if decision["denoise"]:
            result = denoise_photo(result)

        checks, stats = check_quality(original, result, True)

    return result, decision, log, checks, stats, retried


st.title("🖼️ Pixel Revive AI")
st.write(
    "Automatic photo restoration with agent-selected tools, "
    "quality checks, and retry logic."
)

uploaded = st.file_uploader(
    "Upload a black-and-white or damaged color photo",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    try:
        original = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read this image: {exc}")
        st.stop()

    upload_id = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("upload_id") != upload_id:
        st.session_state["upload_id"] = upload_id
        for key in (
            "result", "decision", "log",
            "checks", "stats", "retried"
        ):
            st.session_state.pop(key, None)

    st.subheader("Original photo")
    st.image(original, use_container_width=True)

    if st.button("Restore photo automatically", type="primary"):
        with st.spinner(
            "Agent is planning tools and restoring your photo..."
        ):
            try:
                result, decision, log, checks, stats, retried = (
                    restore_photo(original)
                )
                st.session_state["result"] = result
                st.session_state["decision"] = decision
                st.session_state["log"] = log
                st.session_state["checks"] = checks
                st.session_state["stats"] = stats
                st.session_state["retried"] = retried
            except Exception as exc:
                st.error(str(exc))

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

        with st.expander("Agent decisions and quality report", expanded=True):
            decision = st.session_state["decision"]
            st.write("**Planner:**", decision["planner"])
            st.write("**Reason:**", decision["reason"])

            st.write("**Tool log**")
            for line in st.session_state["log"]:
                st.write("- " + line)

            st.write("**Quality checks**")
            for line in st.session_state["checks"]:
                st.write("- " + line)

            st.write("**Output statistics**", st.session_state["stats"])

            if st.session_state["retried"]:
                st.info("Colorization was retried once.")

st.caption(
    "Colorization estimates plausible colors, not exact historical colors. "
    "This version does not perform advanced scratch reconstruction."
)

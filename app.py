import os
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Logic Diagram AI Tutor",
    page_icon="🔌",
    layout="wide",
)

st.title("🔌 Logic Diagram AI Tutor")
st.write(
    "Ask questions, upload images, and learn about digital "
    "logic or any other subject shown in an image."
)


# --------------------------------------------------
# GEMINI CONFIGURATION
# --------------------------------------------------

def get_setting(name, default=None):
    """Read a setting from Streamlit secrets or environment."""
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    return os.getenv(name, default)


def get_gemini_client():
    api_key = get_setting("GEMINI_API_KEY")

    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        raise ValueError(
            "Gemini API key is missing. Add GEMINI_API_KEY "
            "to .streamlit/secrets.toml."
        )

    return genai.Client(api_key=api_key)


def get_model_name():
    return get_setting(
        "GEMINI_MODEL",
        "gemini-3.8-flash",
    )


# --------------------------------------------------
# AI TUTOR INSTRUCTIONS
# --------------------------------------------------

SYSTEM_PROMPT = """
You are Logic Diagram AI Tutor, an intelligent and careful AI tutor.

Your job is to answer text questions and explain uploaded images.

IMPORTANT: FIRST IDENTIFY THE INPUT.

Determine whether the user has provided:
1. A text-only question.
2. A digital logic problem or diagram.
3. Another type of image.
4. An unclear or mixed image.

RULES FOR TEXT-ONLY QUESTIONS:
- Answer the user's question directly.
- Do not require an image.
- Explain concepts clearly at an appropriate learning level.
- Include examples when useful.
- Only introduce digital logic, Verilog, or 8086 assembly when relevant.

RULES FOR DIGITAL LOGIC IMAGES:
If the image contains a logic gate diagram, digital circuit,
Boolean expression, truth table, or related digital electronics
question, explain the actual problem.

When appropriate:
- Identify inputs, outputs, gates, and connections.
- Explain how the circuit works.
- Derive the Boolean expression.
- Provide a truth table.
- Generate suitable Verilog HDL.
- Generate 8086 assembly code that calculates the same
  Boolean function in software.
- Explain each important line of the code.
- Show example inputs and expected outputs.
- Mention ambiguity or uncertainty in the diagram.

Do not generate every type of output if it is unnecessary
for the particular question.

RULES FOR NON-LOGIC IMAGES:
If the image is not related to digital logic:
- Explain the image based on what is actually visible.
- Describe relevant objects, text, diagrams, or activities.
- Answer the user's specific question about the image.
- If useful, explain the image's likely purpose.
- Do NOT provide a truth table, Boolean expression, Verilog,
  8086 assembly, or logic-gate analysis unless the user
  specifically asks for a relevant connection.
- Never invent objects, text, or details that cannot be seen.

OTHER IMAGE TYPES:
For a mathematics worksheet, explain the visible mathematics.
For a science diagram, explain the relevant science.
For a photograph, describe the visible scene.
For handwritten text, read what is legible and mention
when any part is uncertain.
For an unrelated chart or diagram, explain its actual content.

WHEN BOTH A QUESTION AND AN IMAGE ARE PROVIDED:
- Treat the question as the user's main request.
- Use the image as context.
- If the image does not match the question, explain that
  mismatch rather than making up a connection.

VERILOG AND 8086:
- Verilog HDL describes hardware.
- 8086 assembly performs calculations in software.
- Do not claim that assembly code creates physical hardware.
- State any assumptions about register assignments or inputs.
- Use valid-looking syntax and explain any platform assumptions.

ACCURACY:
- Do not claim that generated code has been compiled or
  simulated unless that has actually happened.
- Do not pretend to read unclear text with certainty.
- Clearly separate visible facts from reasonable inferences.

RESPONSE FORMAT:
Choose headings that fit the actual task.
Use simple language, clear steps, and examples when useful.
Do not force the digital-logic format onto unrelated questions.
"""


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------

def analyze_input(image=None, user_question=""):
    client = get_gemini_client()
    model = get_model_name()

    question = user_question.strip()

    if image is None and not question:
        raise ValueError(
            "Please enter a question or upload an image."
        )

    prompt = SYSTEM_PROMPT

    if image is not None:
        prompt += (
            "\n\nThe user has provided an image. Inspect it first. "
            "Classify its actual content before deciding how to "
            "respond. Do not assume that every image is a logic diagram."
        )

    if question:
        prompt += (
            "\n\nUSER'S QUESTION:\n"
            + question
        )
    else:
        prompt += (
            "\n\nThe user has not entered a separate question. "
            "Explain the uploaded image appropriately."
        )

    contents = [prompt]

    if image is not None:
        contents.append(image.convert("RGB"))

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=8192,
        ),
    )

    answer = response.text

    if not answer:
        raise RuntimeError(
            "The AI returned an empty response. Please try again."
        )

    return answer


# --------------------------------------------------
# INPUT MODE
# --------------------------------------------------

st.subheader("Choose how to interact")

mode = st.radio(
    "Input method",
    [
        "Ask a question",
        "Upload an image",
        "Ask + upload image",
    ],
    horizontal=True,
)

user_question = ""
uploaded_file = None
image = None


# --------------------------------------------------
# TEXT QUESTION
# --------------------------------------------------

if mode in ("Ask a question", "Ask + upload image"):
    user_question = st.text_area(
        "Your question",
        placeholder=(
            "Examples:\n"
            "- Explain an AND gate.\n"
            "- What does this diagram mean?\n"
            "- Explain the difference between Verilog and assembly.\n"
            "- Describe the uploaded image."
        ),
        height=120,
    )


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

if mode in ("Upload an image", "Ask + upload image"):
    uploaded_file = st.file_uploader(
        "Upload your image",
        type=["png", "jpg", "jpeg", "webp"],
        help=(
            "You can upload a logic diagram, circuit, worksheet, "
            "handwritten question, photograph, or another image."
        ),
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image.verify()

            # Reopen after verification.
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True,
            )

        except Exception:
            image = None
            st.error(
                "This image could not be opened. "
                "Please upload a valid PNG, JPG, JPEG, or WEBP image."
            )


# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button(
    "Analyze with AI Tutor",
    type="primary",
    use_container_width=True,
):
    has_question = bool(user_question.strip())
    has_image = image is not None

    if not has_question and not has_image:
        st.warning(
            "Enter a question, upload an image, or do both."
        )

    elif (
        mode == "Upload an image"
        and uploaded_file is not None
        and not has_image
    ):
        st.warning("Please upload a valid image.")

    else:
        try:
            with st.spinner(
                "Analyzing your input and preparing an explanation..."
            ):
                answer = analyze_input(
                    image=image,
                    user_question=user_question,
                )

            st.session_state["last_answer"] = answer
            st.session_state["last_question"] = user_question

        except Exception as error:
            st.error(f"Unable to analyze your input: {error}")


# --------------------------------------------------
# DISPLAY ANSWER
# --------------------------------------------------

if st.session_state.get("last_answer"):
    st.divider()
    st.subheader("AI Tutor's Explanation")

    st.markdown(st.session_state["last_answer"])

    st.download_button(
        label="Download explanation",
        data=st.session_state["last_answer"],
        file_name="ai_tutor_explanation.md",
        mime="text/markdown",
        use_container_width=True,
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI-generated explanations may contain mistakes. "
    "Check important technical results before relying on them."
)

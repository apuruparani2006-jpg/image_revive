import os
import json
import io
import streamlit as st
from PIL import Image
from google import genai

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Logic Diagram AI Tutor",
    page_icon="🔌",
    layout="wide"
)

st.title("🔌 Logic Diagram AI Tutor")
st.write(
    "Upload a logic diagram, circuit image, truth table, "
    "or handwritten digital-logic question. The AI agent "
    "analyzes it and explains the circuit, Verilog HDL, "
    "and 8086 assembly where appropriate."
)

# --------------------------------------------------
# API CONFIGURATION
# --------------------------------------------------

def get_setting(name, default=""):
    try:
        value = st.secrets.get(name, "")
        if value:
            return value
    except Exception:
        pass

    return os.environ.get(name, default)


def get_gemini_client():
    api_key = get_setting("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key is missing. Add GEMINI_API_KEY "
            "to .streamlit/secrets.toml or your environment."
        )

    return genai.Client(api_key=api_key)


def get_model_name():
    return get_setting(
        "GEMINI_MODEL",
        "gemini-3.8-flash"
    )


# --------------------------------------------------
# AGENT INSTRUCTIONS
# --------------------------------------------------

SYSTEM_PROMPT = """
You are Logic Diagram AI Tutor, a careful AI agent specializing
in digital logic, Verilog HDL, and Intel 8086 assembly.

The user uploads an image. Analyze the actual image before
answering. Do not assume every image is a logic circuit.

Follow these sections in your response.

## 1. IMAGE ANALYSIS
- Identify the type of image.
- Describe the visible symbols, gates, components, labels,
  inputs, outputs, and wire connections.
- Distinguish visible facts from assumptions.
- If a wire or symbol is unclear, explicitly say so.
- Never invent a connection just to complete the answer.

## 2. CIRCUIT IDENTIFICATION
- Explain what the circuit appears to be.
- Identify whether it is combinational logic, sequential
  logic, a processor-related diagram, or something else.
- Explain the practical purpose of the circuit.
- State the Boolean expression when it can be determined.

## 3. TRUTH TABLE
- Create a truth table when the available information allows it.
- Explain how the output is calculated for each input combination.
- For large circuits, use representative examples if a full
  truth table would be impractical.
- Do not invent unknown inputs or outputs.

## 4. VERILOG HDL
- Explain whether Verilog is appropriate for this image.
- When sufficient information is available, generate a complete
  synthesizable Verilog module for the inferred digital circuit.
- Include module name, input and output declarations, and logic.
- Use appropriate bit widths.
- Explain the code line by line.
- For sequential circuits, identify clock, reset, and state
  assumptions. Do not silently invent timing behavior.

## 5. INTEL 8086 ASSEMBLY
- Explain whether the logic can be represented by an 8086 program.
- When appropriate, generate an 8086-compatible assembly example
  that calculates the same Boolean function in software.
- State the assembler and input/output assumptions.
- Prefer widely supported 8086 instructions.
- Explain every instruction and show sample register values.
- Do not claim that assembly creates physical hardware.
- For sequential circuits, explain which timing or hardware
  behavior the software example does not model.

For simple combinational circuits, you may use this convention:
AL contains the packed input bits, BL receives the result,
unless another convention is more suitable. Clearly define
which bit of AL corresponds to each named input.

## 6. WORKING EXAMPLE
- Walk through at least one input example.
- Show how the circuit produces its output.
- Show how the Verilog and assembly represent that behavior.

## 7. FINAL COMPARISON
Compare:
- What the real circuit does
- What the Verilog describes
- What the 8086 program calculates
- Any assumptions or limitations

## 8. CONFIDENCE AND VALIDATION
- List uncertain symbols or connections.
- Suggest what needs to be confirmed.
- Never claim code was compiled, simulated, or tested unless
  that actually happened.

If the image is unrelated to digital logic, explain what it
appears to show and why Verilog or 8086 might not apply.

Use beginner-friendly explanations, headings, tables, and
separate code fences for Verilog and assembly.
Be accurate rather than forcing an answer.
"""


# --------------------------------------------------
# AGENT: ANALYZE IMAGE
# --------------------------------------------------

def analyze_image(image, user_question, output_mode):
    client = get_gemini_client()
    model = get_model_name()

    prompt = f"""
{SYSTEM_PROMPT}

USER'S ADDITIONAL QUESTION:
{user_question or "No additional question."}

OUTPUT PREFERENCE:
{output_mode}

Analyze the attached image and follow the instructions.
If the image does not contain enough information to produce
reliable code, explain what is missing and provide code only
for the parts that can be inferred safely.
"""

    response = client.models.generate_content(
        model=model,
        contents=[
            prompt,
            image.convert("RGB")
        ]
    )

    answer = getattr(response, "text", None)

    if not answer:
        raise RuntimeError(
            "The AI returned an empty answer. Try again with "
            "a clearer image or a different supported model."
        )

    return answer


# --------------------------------------------------
# OPTIONAL: STRUCTURED CIRCUIT SUMMARY
# --------------------------------------------------

def extract_summary(answer):
    """
    A lightweight local helper. It does not pretend to verify
    the AI's interpretation; it simply keeps the full response
    available for display and export.
    """
    return {
        "analysis": answer,
        "generated_by": "Gemini vision model",
        "verified_by_simulator": False
    }


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("Agent settings")

    st.write(
        "The agent inspects the image and generates an explanation "
        "and code based on what it can identify."
    )

    output_mode = st.selectbox(
        "Explanation mode",
        [
            "Full analysis: circuit, truth table, Verilog and 8086",
            "Focus on Verilog HDL",
            "Focus on 8086 assembly",
            "Focus on circuit identification and truth table"
        ]
    )

    st.info(
        "Generated code is a starting point. Verify the circuit "
        "and test code in the appropriate simulator or assembler."
    )

    st.caption(
        "Model: " + get_model_name()
    )


# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a circuit or logic image",
    type=["png", "jpg", "jpeg", "webp"],
    help=(
        "Use a clear image with readable gate symbols, "
        "labels, truth tables, and connections."
    )
)

user_question = st.text_area(
    "What would you like to know? (optional)",
    placeholder=(
        "Example: Identify this circuit, derive its Boolean "
        "expression, and generate Verilog and 8086 code."
    )
)


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded image")
        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True
        )

        analyze_button = st.button(
            "🧠 Analyze with AI agent",
            type="primary",
            use_container_width=True
        )

        if analyze_button:
            with st.spinner(
                "The agent is inspecting the image and preparing "
                "the circuit explanation and code..."
            ):
                try:
                    answer = analyze_image(
                        image,
                        user_question,
                        output_mode
                    )

                    result = extract_summary(answer)

                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_filename"] = (
                        uploaded_file.name
                    )

                except Exception as error:
                    st.error(
                        "The analysis could not be completed."
                    )
                    st.code(str(error))

        result = st.session_state.get("analysis_result")

        if (
            result
            and st.session_state.get("analysis_filename")
            == uploaded_file.name
        ):
            st.divider()
            st.header("📘 Agent analysis")

            st.markdown(result["analysis"])

            st.download_button(
                label="⬇️ Download complete explanation",
                data=result["analysis"],
                file_name="logic_diagram_analysis.md",
                mime="text/markdown",
                use_container_width=True
            )

            with st.expander("About this result"):
                st.write(
                    "AI-generated interpretation. It has not been "
                    "independently verified by a circuit simulator."
                )

    except Exception as error:
        st.error("Could not open the uploaded image.")
        st.code(str(error))

else:
    st.info(
        "Upload an image to start. You can use a logic-gate "
        "diagram, digital circuit, truth table, or handwritten "
        "digital-logic question."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Logic Diagram AI Tutor | Verilog describes hardware; "
    "8086 assembly performs software calculations. "
    "AI-generated interpretations and code should be verified."
)

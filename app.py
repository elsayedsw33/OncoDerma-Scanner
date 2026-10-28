import os
import hashlib
import json
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv


ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
MODEL_PATH = ROOT / "best_model_final.keras"
CLASS_NAMES_PATH = ROOT / "class_names.json"
IMAGE_SIZE = (300, 300)

LESION_INFO = {
    "akiec": ("Actinic keratoses and intraepithelial carcinoma", "A group of sun-related lesions, including actinic keratoses and Bowen disease."),
    "bcc": ("Basal cell carcinoma", "A common skin cancer that often grows slowly and should be assessed by a clinician."),
    "bkl": ("Benign keratosis-like lesions", "A group that includes several non-cancerous keratosis-like skin lesions."),
    "df": ("Dermatofibroma", "A usually benign, firm skin growth."),
    "mel": ("Melanoma", "A serious skin cancer that needs prompt assessment by a qualified clinician."),
    "nv": ("Melanocytic nevus", "A common mole formed by pigment-producing cells."),
    "scc": ("Squamous cell carcinoma", "A skin cancer that should be assessed by a qualified clinician."),
    "vasc": ("Vascular lesion", "A group of lesions related to blood vessels, such as angiomas."),
}

st.set_page_config(page_title="OncoDerma Scanner", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #07111f; color: #edf5ff; }
    [data-testid="stSidebar"] { background: #0b1829; border-right: 1px solid #1b3552; }
    .hero { padding: 1.35rem 1.5rem; border: 1px solid #173e60; border-radius: 18px;
        background: linear-gradient(120deg, #10243a, #0b1930 65%, #082b40); margin-bottom: 1.2rem; }
    .hero h1 { margin: 0; color: #f5f9ff; font-size: 2rem; }
    .hero p { color: #9dc9e8; margin: .4rem 0 0; }
    .result-card { background: #10243a; padding: 1rem 1.2rem; border: 1px solid #1a4969;
        border-radius: 14px; margin: .5rem 0; }
    .muted { color: #a8bbcf; }
    div[data-testid="stMetric"] { background: #10243a; padding: 1rem; border: 1px solid #1a4969; border-radius: 12px; }
    .stButton button, .stFormSubmitButton button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def get_class_names():
    with CLASS_NAMES_PATH.open(encoding="utf-8") as file:
        names = json.load(file)["class_names"]
    return names


def prepare_image(image):
    image = image.convert("RGB")
    tensor = tf.convert_to_tensor(np.asarray(image))
    tensor = tf.image.resize(tensor, IMAGE_SIZE)
    return tf.expand_dims(tf.cast(tensor, tf.float32), axis=0)


def get_setting(name, default=""):
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name) or default).strip()


def gemini_error(error, api_key):
    code = getattr(error, "code", None)
    messages = {
        400: "Gemini rejected the request. Check your API key and whether the API is available for your project and region.",
        401: "Gemini authentication failed. Check GEMINI_API_KEY in Streamlit secrets.",
        403: "Gemini access was denied. Check the key's restrictions, project permissions, and billing status in Google AI Studio.",
        404: "The configured Gemini model is unavailable. Set GEMINI_MODEL in Streamlit secrets to a model available to your API project.",
        429: "Gemini's rate limit or quota was reached. Check your quota and billing in Google AI Studio, then retry when quota is available.",
    }
    message = messages.get(code, "Gemini is temporarily unavailable. Please try again shortly.")
    if code:
        detail = str(getattr(error, "message", "")).replace(api_key, "[hidden]")[:500]
        return f"{message}\n\nAPI error {code}: {detail}"
    return message


def ask_gemini(api_key, instruction):
    from google import genai
    from google.genai import types

    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model=get_setting("GEMINI_MODEL", "gemini-3.8-flash"),
            contents=[
                {"role": "model" if item["role"] == "assistant" else "user",
                 "parts": [{"text": item["content"]}]}
                for item in st.session_state.chat_history
            ],
            config=types.GenerateContentConfig(system_instruction=instruction),
        )
    return response.text


def prediction_context():
    result = st.session_state.get("latest_prediction")
    if not result:
        return "No scan has been made in this session."
    details = {key: result[key] for key in ["label", "full_name", "confidence", "explanation"]}
    details["top_three"] = [
        {"class": label, "score_percent": score, "name": LESION_INFO[label][0],
         "description": LESION_INFO[label][1]}
        for label, score in result["top_three"]
    ]
    return json.dumps(details)


def discuss_results():
    st.session_state.page = "AI Chatbot"


def render_header():
    st.markdown(
        '<div class="hero"><h1>OncoDerma Scanner</h1>'
        '<p>AI-Powered Skin Lesion Classification</p></div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    if (ROOT / "Samsung-Innovation-Campus.jpg").exists():
        st.image(str(ROOT / "Samsung-Innovation-Campus.jpg"), use_container_width=True)
    st.markdown("### OncoDerma Scanner")
    page = st.radio("Navigate", ["Scanner", "AI Chatbot", "About the Model"],
                    key="page", label_visibility="collapsed")
    st.caption("Samsung Innovation Campus · 8th Edition")


if page == "Scanner":
    render_header()
    st.write("Upload a clear skin lesion image to see the model's classification scores.")
    uploaded = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])

    if uploaded:
        try:
            upload_id = hashlib.sha256(uploaded.getvalue()).hexdigest()
            image = Image.open(uploaded).convert("RGB")
            left, right = st.columns([1, 1.15], gap="large")
            with left:
                st.image(image, caption="Uploaded image", use_container_width=True)
                if st.button("Analyze image", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image…"):
                        model = load_model()
                        class_names = get_class_names()
                        output_count = int(model.output_shape[-1])
                        if output_count != len(class_names):
                            st.error(
                                f"Model returns {output_count} classes, but class_names.json contains "
                                f"{len(class_names)}. Please verify the model and class mapping."
                            )
                        else:
                            scores = np.asarray(model.predict(prepare_image(image), verbose=0))[0]
                            order = np.argsort(scores)[::-1][:3]
                            label = class_names[int(order[0])]
                            full_name, explanation = LESION_INFO.get(label, (label, ""))
                            st.session_state.latest_prediction = {
                                "label": label,
                                "image_id": upload_id,
                                "filename": uploaded.name,
                                "full_name": full_name,
                                "confidence": float(scores[order[0]] * 100),
                                "explanation": explanation,
                                "top_three": [(class_names[int(i)], float(scores[i] * 100)) for i in order],
                            }
        except (UnidentifiedImageError, OSError, ValueError) as error:
            st.error(f"Could not read that image. Please upload a valid JPG or PNG. ({error})")
        except Exception as error:
            st.error(f"The scan could not be completed: {error}")

    result = st.session_state.get("latest_prediction")
    if result:
        with st.container(border=True):
            st.subheader("Latest result")
            st.caption(f"Analyzed image: {result.get('filename', 'Previous scan')}")
            st.markdown(f"### {result['full_name']} · `{result['label']}`")
            st.metric("Model confidence score", f"{result['confidence']:.1f}%")
            st.write(result["explanation"])
            st.markdown("**Top 3 model predictions**")
            for label, score in result["top_three"]:
                full_name = LESION_INFO.get(label, (label, ""))[0]
                st.write(f"{full_name} (`{label}`) — {score:.1f}%")
                st.progress(min(max(score / 100, 0.0), 1.0))
            st.button("Discuss Results with AI", type="primary", on_click=discuss_results)
            st.caption(
                "These scores describe the model's output for this image; they are not calibrated medical probabilities. "
                "This tool is for education only and cannot diagnose skin conditions. For a changing, painful, bleeding, "
                "or concerning lesion, consult a qualified dermatologist."
            )
elif page == "AI Chatbot":
    render_header()
    st.subheader("Ask about skin health")
    st.caption("The assistant can use your latest scan result as context. It cannot diagnose or replace a clinician.")
    result = st.session_state.get("latest_prediction")
    if result:
        st.info(f"Discussing: {result['full_name']} — {result['confidence']:.1f}% model score. "
                f"Image: {result.get('filename', 'Previous scan')}")
    else:
        st.info("You can ask general skin-health questions, or analyze an image on the Scanner page first.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    clear_col, _ = st.columns([1, 4])
    with clear_col:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    suggestion = None
    for column, question in zip(st.columns(2), ["What does my result mean?", "Should I see a dermatologist?"]):
        if column.button(question, use_container_width=True):
            suggestion = question
    prompt = st.chat_input("Ask a question about skin health or your result…") or suggestion
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        provider = get_setting("AI_PROVIDER", "gemini").lower()
        key_name = "OPENROUTER_API_KEY" if provider == "openrouter" else "GEMINI_API_KEY"
        api_key = get_setting(key_name)
        successful = False
        if not api_key:
            answer = f"Add {key_name} to your Streamlit secrets or .env file to enable the chatbot."
        else:
            try:
                instruction = (
                    "You are a supportive skin-health information assistant. Give general, clear, cautious information. "
                    "Never diagnose, claim certainty from an image-classifier score, recommend delaying care, or prescribe treatment. "
                    "Encourage a qualified dermatologist for personal concerns and urgent care for severe symptoms. "
                    "Explain that predictions are not medical diagnoses and scores are not disease probabilities. "
                    "Use the latest scan for questions about the current result; older messages may refer to a different image. "
                    "Ask for clarification if the user refers to an earlier scan. "
                    "The latest scan context is: " + prediction_context()
                )
                if provider == "openrouter":
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "X-Title": "OncoDerma Scanner"},
                        json={
                            "model": get_setting("OPENROUTER_MODEL", "openrouter/free"),
                            "messages": [{"role": "system", "content": instruction}] +
                                        st.session_state.chat_history,
                            "max_tokens": 1024,
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()
                    if "error" in data:
                        raise ValueError("OpenRouter returned an error. Check the model and account quota in your dashboard.")
                    answer = data["choices"][0]["message"]["content"]
                    successful = isinstance(answer, str) and bool(answer.strip())
                    if not successful:
                        answer = "OpenRouter returned no text. Please retry or choose another OPENROUTER_MODEL."
                elif provider == "gemini":
                    answer = ask_gemini(api_key, instruction)
                    successful = bool(answer)
                    if not successful:
                        answer = "Gemini returned no text. Please try again."
                else:
                    answer = "Set AI_PROVIDER to gemini or openrouter in Streamlit secrets."
            except requests.HTTPError as error:
                code = error.response.status_code
                guidance = {
                    401: "Check OPENROUTER_API_KEY.",
                    402: "Check your OpenRouter credit balance or select an available free model.",
                    403: "Check your OpenRouter account permissions and model access.",
                    404: "Check OPENROUTER_MODEL against the model ID in the OpenRouter catalog.",
                    429: "Your OpenRouter rate limit or quota was reached. Wait and retry.",
                }
                answer = f"OpenRouter API error {code}. " + guidance.get(code, "The provider may be busy; retry or choose another model.")
            except requests.RequestException:
                answer = "OpenRouter could not be reached or timed out. Please retry shortly."
            except Exception as error:
                answer = gemini_error(error, api_key) if provider == "gemini" else "OpenRouter could not return a valid response. Check your model and account status."
        if successful:
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.session_state.chat_history.pop()
        with st.chat_message("assistant"):
            st.markdown(answer)
else:
    render_header()
    st.subheader("About the Model")
    st.write("**EfficientNetB6 · transfer learning with an eight-class softmax classifier**")
    st.caption("Source: skin-classifier.ipynb, final model summary and training cells; best_model_final.keras.")
    st.write(
        "The notebook initializes EfficientNetB6 with ImageNet weights and average pooling. "
        "Its classifier has dense layers of 512, 256, and 64 units, each followed by batch normalization, "
        "ReLU activation, and 20% dropout. The final layer produces eight class scores."
    )

    st.subheader("Dataset")
    st.write(
        "HAM10000 is a public research benchmark containing 10,015 dermatoscopic images in seven lesion categories. "
        "[Read the original dataset paper](https://www.nature.com/articles/sdata2018161). "
        "This notebook loads a separate Kaggle meta.csv with 24,930 records and eight labels, including SCC. "
        "The provided files do not establish whether or how HAM10000 contributed to that collection."
    )
    st.write("Recorded stratified split: **17,451 training · 3,739 validation · 3,740 test** images "
             "(70% / 15% / 15%, random seed 42). The notebook merges the label `ak` into `akiec`.")

    st.subheader("Supported classes")
    st.dataframe([
        {"Code": label, "Medical name": LESION_INFO[label][0], "Description": LESION_INFO[label][1]}
        for label in get_class_names()
    ], hide_index=True, use_container_width=True)

    st.subheader("From image to result")
    st.write(
        "1. Convert the uploaded image to three RGB channels.\n"
        "2. Resize to 300 × 300 with TensorFlow's default bilinear resize and cast to float32.\n"
        "3. Keep pixel values on the 0–255 scale; EfficientNet's saved layers perform rescaling internally.\n"
        "4. Run the saved model and rank its eight softmax scores to show the top three classes."
    )
    st.caption("The JSON metadata says 192 pixels, but both the notebook pipeline and saved model input specify 300 × 300; the app follows those verified inputs.")

    st.subheader("Training recorded in the notebook")
    st.write(
        "Training uses batches of 8 and balanced class weights, with sparse categorical cross-entropy loss. "
        "Augmentation includes horizontal and vertical flips, random 90° rotations, padding to 330 × 330 "
        "followed by a random 300 × 300 crop, brightness changes up to 20, contrast 0.8–1.2, "
        "and saturation 0.85–1.15, with pixels clipped to 0–255. Augmentation is applied only during training."
    )
    st.write(
        "The notebook trains the classifier and progressively unfreezes backbone blocks. "
        "The final stage trains backbone layers whose names start with top_, while keeping backbone batch normalization frozen. "
        "It uses Adam with an initial learning rate of 0.00001 and a maximum of 30 epochs. "
        "Early stopping monitors validation loss with patience 8 and restores the best weights; "
        "learning-rate reduction uses factor 0.5, patience 2, and minimum rate 0.000001. "
        "The checkpoint saves the model with the lowest validation loss."
    )
    st.caption("The saved training log completes 14 epochs and records a manual interruption during epoch 15. It does not show a completed 30-epoch run.")

    st.subheader("Recorded evaluation")
    st.caption("Notebook final evaluation cell, 3,740 test images. Precision, recall and F1 below use weighted averages.")
    for column, name, value in zip(st.columns(4), ["Accuracy", "Precision", "Recall", "F1-score"],
                                   [85.32, 85.61, 85.32, 85.39]):
        column.metric(name, f"{value:.2f}%")
    st.info(
        "These are the notebook's recorded results, not a new evaluation of the supplied checkpoint. "
        "The evaluation uses the in-memory model after an interrupted training run; "
        "the files do not prove that these metrics are identical for the saved best checkpoint."
    )
    try:
        notebook = json.loads((ROOT / "skin-classifier.ipynb").read_text(encoding="utf-8"))
        report = "".join(
            "".join(output.get("text", []))
            for output in notebook["cells"][-1].get("outputs", [])
            if output.get("name") == "stdout"
        )
        if "Classification Report:" in report:
            classification, matrix = report.split("Classification Report:", 1)[1].split("Confusion Matrix:")
            with st.expander("Per-class precision, recall, F1 and support"):
                st.code(classification.strip(), language=None)
            st.markdown("**Confusion matrix**")
            st.caption("Rows: actual classes. Columns: predicted classes. Order: " + ", ".join(get_class_names()))
            st.code(matrix.strip(), language=None)
        else:
            st.caption("The detailed evaluation output is unavailable in this notebook copy.")
    except (OSError, ValueError, KeyError, IndexError):
        st.caption("The detailed notebook evaluation could not be read.")

    st.subheader("Limitations")
    st.write(
        "The classifier always chooses among its eight supported classes, including for unrelated or poor-quality images. "
        "Lighting, blur, image framing and differences from training images can affect scores. "
        "The notebook does not provide clinical validation, score calibration, or performance broken down by skin tone. "
        "It splits by image records; patient-level or lesion-level separation is not documented. "
        "The model cannot replace a clinical examination or rule out cancer."
    )
    st.warning("Educational use only. Predictions are not medical diagnoses. Consult a qualified dermatologist about personal skin concerns.")

# OncoDerma Scanner

AI-Powered Skin Lesion Classification — Samsung Innovation Campus 8th Edition project.

## Run locally

1. Install Python 3.10–3.13.
2. From this folder, install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add a Gemini API key. The scanner works without a key; the chatbot needs one. On Streamlit Community Cloud, add `GEMINI_API_KEY` under **App settings → Secrets** instead.
4. Start the app:

   ```bash
   streamlit run app.py
   ```

On Windows, you can also double-click `run.bat`. It uses the project's `.venv` when present.

Gemini can be configured in Streamlit secrets using TOML:

```toml
GEMINI_API_KEY = "your_api_key"
GEMINI_MODEL = "gemini-3.8-flash"
```

Streamlit secrets take precedence over `.env`. If your project cannot use the default model, set `GEMINI_MODEL` to an available text model shown in Google AI Studio. The chatbot displays the API error code and provider message for rejected requests; failed requests are excluded from conversation history.

Keep `best_model_final.keras`, `class_names.json`, and `Samsung-Innovation-Campus.jpg` beside `app.py` when deploying.

For Streamlit Community Cloud, select **Python 3.13** in **Advanced settings** before deploying. The requirements pin the versions tested in the local Python 3.13 environment, including Keras for model loading. TensorFlow 2.20.0 does not provide Python 3.14 wheels. If an existing deployment uses the wrong Python version, preserve its secrets and app settings, then delete and redeploy it with Python 3.13; editing requirements does not change the cloud interpreter.

The model file is larger than GitHub's regular per-file limit. Before pushing the project to GitHub, track it with Git LFS (`git lfs install` and `git lfs track "best_model_final.keras"`), then commit the generated `.gitattributes` together with the model. Streamlit Community Cloud supports Git LFS.

## Model input and labels

The notebook's inference pipeline decodes images as RGB, resizes them to 300×300, and casts pixel values to `float32`. The saved model accepts `(300, 300, 3)` and includes its own input rescaling, so the app does not normalize pixels a second time. Class names and output ordering come from `class_names.json`; the app checks that the number of labels matches the model output.

The model score is an image-classification score, not a medical probability or diagnosis. Results are for educational use only. Users should ask a qualified dermatologist about skin concerns.

After a scan, choose **Discuss Results with AI** to open the chatbot with the latest class, score, top three predictions, and lesion descriptions. Results and chat history persist while navigating between all three pages. A new analysis updates the context used for subsequent questions.

The **About the Model** page documents EfficientNetB6, training settings, and the notebook's recorded evaluation (85.32% test accuracy). It includes the full classification report and confusion matrix, explains the uncertain dataset provenance, and distinguishes the notebook's in-memory evaluation from a fresh evaluation of the supplied checkpoint.

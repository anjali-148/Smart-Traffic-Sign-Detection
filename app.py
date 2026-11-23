import html
import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from predict import load_artifacts, predict


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SignScope | Traffic Sign Recognition",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                ellipse at 90% 0%,
                rgba(28, 93, 76, 0.30),
                transparent 42%
            ),
            #090f18;
        color: #e8eef7;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background-color: #101925;
        border-right: 1px solid #263345;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #edf3fa !important;
        letter-spacing: -0.04em;
    }

    p, label {
        color: #bac7d6;
    }

    .brand {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: #f0f5fb;
        margin-bottom: 5px;
    }

    .brand span {
        color: #82e8be;
    }

    .brand-caption {
        font-size: 10px;
        color: #8396ac;
        letter-spacing: 2px;
        margin-bottom: 25px;
    }

    .badge {
        display: inline-block;
        padding: 8px 14px;
        border: 1px solid #2c5c4c;
        border-radius: 30px;
        background: #122a24;
        color: #8de7c2;
        font-size: 11px;
        letter-spacing: 1.5px;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: clamp(36px, 5vw, 64px);
        font-weight: 800;
        line-height: 1.08;
        letter-spacing: -2.5px;
        color: #f0f5fb;
        margin-bottom: 18px;
    }

    .hero-title span {
        color: #82e8be;
    }

    .hero-description {
        color: #9aacc0;
        font-size: 16px;
        line-height: 1.8;
        max-width: 740px;
        margin-bottom: 30px;
    }

    .stat-card {
        background: #111d2b;
        border: 1px solid #27364a;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 22px;
        min-height: 115px;
    }

    .stat-value {
        font-size: 29px;
        font-weight: 750;
        color: #edf4fb;
        line-height: 1.3;
    }

    .stat-label {
        font-size: 12px;
        color: #8fa3b9;
        margin-top: 8px;
    }

    .section-number {
        color: #82e8be;
        font-size: 11px;
        letter-spacing: 2px;
        margin-top: 12px;
        margin-bottom: 6px;
    }

    .section-title {
        color: #edf3fa;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .section-description {
        color: #93a7bd;
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 20px;
    }

    .prediction-card {
        background: linear-gradient(135deg, #163b31, #122331);
        border: 1px solid #356350;
        border-radius: 18px;
        padding: 27px;
        margin: 15px 0 22px;
    }

    .prediction-label {
        color: #96e5c4;
        font-size: 11px;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .prediction-name {
        color: #f1f8f5;
        font-size: 31px;
        font-weight: 750;
        line-height: 1.25;
        margin-bottom: 14px;
        overflow-wrap: anywhere;
    }

    .prediction-score {
        color: #8af0c4;
        font-size: 43px;
        font-weight: 800;
    }

    .prediction-note {
        color: #a3b9b0;
        font-size: 12px;
        margin-top: 5px;
    }

    .empty-state {
        background: #111c2a;
        border: 1px dashed #34485f;
        border-radius: 18px;
        text-align: center;
        padding: 65px 25px;
        margin-top: 15px;
    }

    .empty-symbol {
        color: #82e8be;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .empty-title {
        color: #eaf1f9;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .empty-text {
        color: #91a5bb;
        font-size: 13px;
        line-height: 1.8;
    }

    .step {
        border-left: 2px solid #3b7260;
        padding: 0 0 25px 18px;
        margin-left: 5px;
    }

    .step-title {
        color: #e9f1fa;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .step-text {
        color: #95a9bf;
        font-size: 14px;
        line-height: 1.8;
    }

    .footer {
        border-top: 1px solid #263448;
        margin-top: 40px;
        padding-top: 22px;
        color: #7e92a9;
        font-size: 12px;
        line-height: 1.8;
    }

    [data-testid="stFileUploader"] {
        background: #111d2b;
        border: 1px solid #304157;
        border-radius: 15px;
        padding: 12px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    .stButton > button[kind="primary"] {
        background: #82e8be;
        border: 1px solid #82e8be;
        color: #08271c;
    }

    .stButton > button[kind="primary"]:hover {
        background: #a1f3d2;
        border-color: #a1f3d2;
        color: #08271c;
    }

    .stDownloadButton > button {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
@st.cache_resource
def get_model():
    return load_artifacts()


def show_stat(value, label):
    safe_value = html.escape(str(value))
    safe_label = html.escape(str(label))

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">{safe_value}</div>
            <div class="stat-label">{safe_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_section(number, title, description):
    st.markdown(
        f"""
        <div class="section-number">{html.escape(number)}</div>
        <div class="section-title">{html.escape(title)}</div>
        <div class="section-description">
            {html.escape(description)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_empty_result():
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-symbol">[ S ]</div>
            <div class="empty-title">Ready when you are.</div>
            <div class="empty-text">
                Upload a cropped traffic sign image,<br>
                then select Analyze traffic sign.<br>
                Your predictions will appear here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
model = None
labels = []
model_error = None

try:
    model, labels = get_model()
except Exception as exc:
    model_error = str(exc)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">Sign<span>Scope.</span></div>
        <div class="brand-caption">
            TRAFFIC SIGN INTELLIGENCE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Recognition studio",
            "Model insights",
            "Project guide",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Model status")

    if model is not None:
        st.success("Trained model loaded")
        st.caption(f"{len(labels)} traffic sign classes")
    else:
        st.warning("Model unavailable")
        st.caption(
            "Train the model to enable image predictions."
        )

    st.divider()

    if st.button("Reload model", use_container_width=True):
        get_model.clear()
        st.session_state.pop("prediction_result", None)
        st.rerun()

    st.caption(
        "Use Reload model after retraining or changing labels.json."
    )

    st.divider()

    st.caption("Indian traffic sign recognition")
    st.caption("CNN-based educational project")
    st.caption("Not intended for driving decisions.")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="badge">COMPUTER VISION / DEEP LEARNING</div>
    <div class="hero-title">
        Small signs.<br>
        <span>Smarter recognition.</span>
    </div>
    <div class="hero-description">
        Explore Indian traffic signs with your trained CNN.
        Upload an image, inspect the top predictions,
        and understand your model's performance.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Recognition studio
# ---------------------------------------------------------
if page == "Recognition studio":
    col1, col2, col3 = st.columns(3)

    with col1:
        show_stat(
            len(labels) if labels else "Not loaded",
            "Traffic sign classes",
        )

    with col2:
        show_stat("32 × 32", "Model input resolution")

    with col3:
        show_stat("CNN", "Recognition architecture")

    left, right = st.columns([1, 1], gap="large")

    image = None
    upload_signature = None

    with left:
        show_section(
            "01 / INPUT",
            "Add your traffic sign",
            "Choose a close-up image containing one clearly visible sign.",
        )

        uploaded_file = st.file_uploader(
            "Upload traffic sign image",
            type=[
                "png",
                "jpg",
                "jpeg",
                "ppm",
                "bmp",
                "webp",
            ],
        )

        if uploaded_file is not None:
            import hashlib

            upload_signature = hashlib.sha256(
                uploaded_file.getvalue()
            ).hexdigest()

            try:
                image = ImageOps.exif_transpose(
                    Image.open(uploaded_file)
                ).convert("RGB")

                st.image(
                    image,
                    caption=uploaded_file.name,
                    width=300,
                )

                st.caption(
                    f"{image.width} × {image.height} pixels | RGB"
                )

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as exc:
                st.error(f"Unable to open this image: {exc}")

        analyze = st.button(
            "Analyze traffic sign",
            type="primary",
            use_container_width=True,
            disabled=image is None or model is None,
        )

        with st.expander("Tips for better predictions"):
            st.write(
                """
                - Crop the image around the traffic sign.
                - Keep the sign upright.
                - Avoid excessive blur, glare, or obstruction.
                - Use one sign per image.
                - Full road scenes require a separate sign detector.
                """
            )

        if model_error:
            st.warning(
                "The model could not be loaded. "
                "Train it first or check the details below."
            )

            with st.expander("Model loading details"):
                st.code(model_error, language="text")

                st.code(
                    'python train.py --data-dir '
                    '"D:/Downloads/Indian-Traffic Sign-Dataset"',
                    language="bat",
                )

    # Clear stale results when the uploaded image changes.
    existing_result = st.session_state.get("prediction_result")

    if (
        existing_result is not None
        and existing_result["signature"] != upload_signature
    ):
        st.session_state.pop("prediction_result", None)

    if analyze and image is not None and model is not None:
        try:
            with st.spinner("Analyzing visual features..."):
                predictions = predict(image, model, labels)

            st.session_state["prediction_result"] = {
                "signature": upload_signature,
                "filename": uploaded_file.name,
                "predictions": predictions,
            }

        except Exception as exc:
            st.session_state.pop("prediction_result", None)
            st.error(f"Prediction failed: {exc}")

    with right:
        show_section(
            "02 / OUTPUT",
            "Recognition results",
            "Compare the top three predictions and download your report.",
        )

        result = st.session_state.get("prediction_result")

        if result is None:
            show_empty_result()

        else:
            predictions = result["predictions"]
            top = predictions[0]

            safe_name = html.escape(str(top["name"]))
            safe_class = html.escape(str(top["class_id"]))

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">
                        TOP PREDICTION / CLASS {safe_class}
                    </div>
                    <div class="prediction-name">
                        {safe_name}
                    </div>
                    <div class="prediction-score">
                        {top["score"]:.1%}
                    </div>
                    <div class="prediction-note">
                        Model score, not a guarantee of correctness
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Top predictions")

            for rank, item in enumerate(predictions, start=1):
                score = float(item["score"])

                st.write(
                    f"{rank}. {item['name']} "
                    f"— Class {item['class_id']} "
                    f"— {score:.1%}"
                )

                st.progress(
                    max(0.0, min(1.0, score))
                )

            report = {
                "application": "SignScope",
                "filename": result["filename"],
                "predictions": predictions,
                "note": (
                    "Model scores are not calibrated guarantees "
                    "of correctness. This application is an "
                    "educational prototype."
                ),
            }

            st.download_button(
                "Download prediction report",
                data=json.dumps(report, indent=2),
                file_name="signscope_prediction.json",
                mime="application/json",
                use_container_width=True,
            )

            st.caption(
                "The classifier always chooses from its trained "
                "classes. Unrelated images can still receive "
                "high scores."
            )


# ---------------------------------------------------------
# Model insights
# ---------------------------------------------------------
elif page == "Model insights":
    st.subheader("Understand your model")

    st.write(
        "This page shows saved evaluation results from your "
        "training run. No sample accuracy is used."
    )

    metrics_path = MODELS_DIR / "metrics.json"

    if metrics_path.exists():
        try:
            metrics = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )

            columns = st.columns(4)

            stats = [
                (
                    f"{metrics['test_accuracy']:.1%}",
                    "Held-out test accuracy",
                ),
                (
                    metrics["train_images"],
                    "Training images",
                ),
                (
                    metrics["validation_images"],
                    "Validation images",
                ),
                (
                    metrics["test_images"],
                    "Test images",
                ),
            ]

            for column, (value, label) in zip(columns, stats):
                with column:
                    show_stat(value, label)

        except (OSError, ValueError, KeyError) as exc:
            st.warning(f"Unable to read evaluation metrics: {exc}")

    else:
        st.info(
            "No saved metrics found yet. Complete training "
            "to generate evaluation results."
        )

    curves_path = MODELS_DIR / "training_curves.png"

    if curves_path.exists():
        st.subheader("Training and validation curves")
        st.image(str(curves_path))

    matrix_path = MODELS_DIR / "confusion_matrix.png"

    if matrix_path.exists():
        st.subheader("Confusion matrix")
        st.image(str(matrix_path))

    report_path = MODELS_DIR / "classification_report.txt"

    if report_path.exists():
        with st.expander("View per-class classification report"):
            report_text = report_path.read_text(
                encoding="utf-8"
            )

            st.code(report_text, language="text")

            st.download_button(
                "Download classification report",
                data=report_text,
                file_name="classification_report.txt",
                mime="text/plain",
            )

    st.warning(
        "Near-duplicate images in different splits can inflate "
        "measured accuracy. Stronger evaluation keeps images "
        "of the same physical sign or scene in one split."
    )


# ---------------------------------------------------------
# Project guide
# ---------------------------------------------------------
else:
    st.subheader("Your recognition workflow")

    steps = [
        (
            "01 / Prepare the dataset",
            "Keep Images/ and traffic_sign.csv together. "
            "Each numbered folder inside Images represents "
            "one traffic sign class.",
        ),
        (
            "02 / Train the CNN",
            "Run train.py with your dataset directory. "
            "The script saves the trained model, labels, "
            "evaluation reports, and plots.",
        ),
        (
            "03 / Analyze a sign",
            "Open Recognition studio, upload a cropped sign, "
            "and select Analyze traffic sign.",
        ),
    ]

    for title, description in steps:
        st.markdown(
            f"""
            <div class="step">
                <div class="step-title">
                    {html.escape(title)}
                </div>
                <div class="step-text">
                    {html.escape(description)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Training command")

    st.code(
        'python train.py --data-dir '
        '"D:/Downloads/Indian-Traffic Sign-Dataset"',
        language="bat",
    )

    st.subheader("Launch command")

    st.code(
        "python -m streamlit run app.py",
        language="bat",
    )

    st.subheader("Class directory")

    if labels:
        st.dataframe(
            [
                {
                    "Class ID": item["class_id"],
                    "Sign name": item["name"],
                }
                for item in labels
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "The class directory will appear after a trained "
            "model and labels.json are available."
        )

    st.caption(
        "If names appear as Class N, update only the name "
        "fields in models/labels.json using your dataset CSV. "
        "Then select Reload model in the sidebar."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        SignScope / Indian Traffic Sign Recognition<br>
        Built for learning, experimentation, and transparent evaluation.
        Not an autonomous-driving safety system.
    </div>
    """,
    unsafe_allow_html=True,
)
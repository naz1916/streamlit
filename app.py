import os
import gdown
import streamlit as st
import cv2
from PIL import Image, ImageOps
from ultralytics import YOLO

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Potato Leaves Early and Late-Blight detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.title("Potato Leaves Early and Late-Blight detection")

    st.divider()

    page = st.radio(
        "Navigation",
        ["About", "Detection"],
        index=0
    )
    st.divider()

    st.markdown("**Research Team**")
    for member in [
        "Acampado, Carmelo",
        "Agustin, Jayson",
        "Alfaro, Tricia",
        "Adel, Irene",
    ]:
        st.caption(member)

    st.divider()

    st.caption("Computer Vision Research Project")
    st.caption("YOLOv26 • Instance Segmentation")

# --------------------------------------------------
# LOAD YOLO ONNX MODEL
# --------------------------------------------------

MODEL_PATH = "best.onnx"

# From a link like https://drive.google.com/file/d/<FILE_ID>/view
# the file ID is the part between /d/ and /view.
GDRIVE_FILE_ID = "1iZVEfjtvy1Y48A4Us0Fp6iHTcV7zmZSD"

@st.cache_resource
def load_onnx_model():
    # Download the model from Google Drive on first run only
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first run only)..."):
            gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model download failed. Check the Google Drive file ID and "
            "make sure sharing is set to 'Anyone with the link'."
        )

    return YOLO(MODEL_PATH, task="segment")


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

if page == "About":

    st.title("Identification and Localization of Early and Late Blight-Affected Regions in Potato Leaves Using YOLOv26")

    st.divider()

    st.subheader("Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### 🟡 Early Blight")
            st.write("""
            Early blight is a potato disease commonly associated
            with lesions that may exhibit concentric ring patterns
            and surrounding yellowing.
            """)

    with col2:
        with st.container(border=True):
            st.markdown("#### 🟣 Late Blight")
            st.write("""
            Late blight can produce irregular, dark or
            water-soaked lesions on potato leaves, potentially
            leading to extensive tissue damage.
            """)

    st.divider()

    st.subheader("Project Objectives")

    objectives = [
        "Identify potential early blight and late blight symptoms in potato leaves.",
        "Localize affected regions through instance segmentation.",
        "Generate visual segmentation masks for detected regions.",
        "Provide confidence scores for model predictions.",
        "Develop an accessible web-based interface for image analysis."
    ]

    for objective in objectives:
        st.markdown(f"- {objective}")

    st.divider()

    st.subheader("Technology Stack")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### YOLOv26")
            st.caption("Object detection and segmentation")

    with col2:
        with st.container(border=True):
            st.markdown("### ONNX")
            st.caption("Model deployment format")

    with col3:
        with st.container(border=True):
            st.markdown("### Streamlit")
            st.caption("Web application framework")

    st.divider()

    st.info(
        "The system provides model-based visual predictions "
        "and is intended for research and educational purposes. "
        "It does not replace professional plant disease diagnosis."
    )

# --------------------------------------------------
# DETECTION PAGE
# --------------------------------------------------

elif page == "Detection":

    st.write(
        "Upload a potato leaf image to identify and localize "
        "potential early blight and late blight symptoms."
    )

    st.divider()

    st.subheader("Detection Settings")

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.01,
        max_value=0.90,
        value=0.20,
        step=0.01,
        format="%.2f",
        help="Adjust the minimum confidence threshold for predictions."
    )

    st.caption(
        "Lower confidence thresholds may display more predictions, "
        "while higher thresholds may filter out uncertain predictions."
    )

    st.divider()

    # Load model
    try:
        model = load_onnx_model()

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(
            "Check that the Google Drive file ID is correct and the file "
            "is shared as 'Anyone with the link'."
        )
        st.stop()

    # Image uploader
    uploaded_file = st.file_uploader(
        "Upload Potato Leaf Image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, and WEBP."
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image)   # fix phone-photo rotation
            image = image.convert("RGB")             # drop alpha / force 3 channels

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Segmentation Result")

                with st.spinner("Analyzing potato leaf..."):
                    # Pass the PIL image directly (no NumPy RGB/BGR mix-up)
                    results = model.predict(
                        source=image,
                        conf=confidence,
                        imgsz=640,   # must match the size used in the ONNX export
                        verbose=False
                    )

                    first_result = results[0]
                    annotated_image = first_result.plot()   # returns BGR

                    annotated_image_rgb = cv2.cvtColor(
                        annotated_image,
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(annotated_image_rgb, use_container_width=True)

            st.divider()

            st.subheader("Prediction Summary")

            if (
                first_result.boxes is not None
                and len(first_result.boxes) > 0
            ):

                total_detections = len(first_result.boxes)

                st.success(
                    f"Detected {total_detections} potential "
                    "disease-affected region(s)."
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Detected Regions", total_detections)

                with col2:
                    st.metric("Confidence Threshold", f"{confidence:.0%}")

                st.subheader("Detection Details")

                for i, box in enumerate(first_result.boxes):

                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    conf = float(box.conf[0])

                    with st.container(border=True):

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"**Region {i + 1}**")
                            st.write(f"Predicted Class: {class_name}")

                        with col2:
                            st.metric("Confidence", f"{conf:.2%}")

            else:

                st.warning(
                    "No disease-affected regions were detected. "
                    "Try another image or adjust the confidence threshold."
                )

            st.divider()

            st.caption(
                "Predictions indicate visual patterns learned by the model "
                "and should not be treated as a definitive plant disease diagnosis."
            )

        except Exception as e:
            st.error(f"An error occurred during image processing: {e}")

    else:

        st.info("Please upload a potato leaf image to begin segmentation.")

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import cv2 # OpenCV for drawing bounding boxes

# --- Configuration ---
# Since app.py is in the base_dir, we can use relative paths or just the current dir
APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is D:\yolo_training_tb_only
MODEL_SEARCH_DIR = APP_ROOT_DIR # Search for models in the same directory as app.py

# Try to find a suitable model automatically
model_files = [f for f in os.listdir(MODEL_SEARCH_DIR) if (f.startswith("best_") or f.startswith("last_")) and f.endswith(".pt")]
# Prioritize 'best' models
best_model_files = [f for f in model_files if f.startswith("best_")]

if best_model_files:
    best_model_files.sort(reverse=True) # Simple sort, might need refinement if names are complex
    DEFAULT_MODEL_NAME = best_model_files[0]
elif model_files: # Fallback to any .pt file if no 'best_' found
    model_files.sort(reverse=True)
    DEFAULT_MODEL_NAME = model_files[0]
else:
    # If no models are auto-detected, you might want to prompt the user or have a fixed fallback
    DEFAULT_MODEL_NAME = "your_model_name.pt" # REPLACE THIS if auto-detection fails
    st.warning(f"No models auto-detected. Please ensure a .pt file is in {MODEL_SEARCH_DIR} or select one manually.")


# Path to the initially selected model
DEFAULT_MODEL_PATH = os.path.join(MODEL_SEARCH_DIR, DEFAULT_MODEL_NAME) if DEFAULT_MODEL_NAME != "your_model_name.pt" else ""


CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# --- Load YOLO Model ---
@st.cache_resource
def load_model(model_path):
    if not model_path or not os.path.exists(model_path):
        st.error(f"Model path is invalid or model not found: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        st.success(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model from {model_path}: {e}")
        return None

# --- Main App ---
st.set_page_config(page_title="TB Lesion Detection", layout="wide")
st.title("🔬 TB Lesion Detector")
st.write("Upload an X-ray image to detect potential Tuberculosis (TB) lesions using a YOLOv8s model.")

# --- Sidebar for Model Selection and Parameters ---
st.sidebar.header("⚙️ Configuration")

# List all .pt files in the MODEL_SEARCH_DIR
available_models_in_dir = [f for f in os.listdir(MODEL_SEARCH_DIR) if f.endswith(".pt")]

if not available_models_in_dir:
    st.sidebar.error(f"No .pt model files found in the application directory: {MODEL_SEARCH_DIR}")
    st.error("Please place your trained .pt model file in the same directory as this app or select it using the file uploader.")
    # Option to upload model if not found
    uploaded_model_file = st.sidebar.file_uploader("Or upload your .pt model file here", type=["pt"])
    if uploaded_model_file is not None:
        # Save the uploaded model temporarily or to a fixed location
        temp_model_path = os.path.join(APP_ROOT_DIR, uploaded_model_file.name)
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_model_file.getbuffer())
        model_path_to_load = temp_model_path
        st.sidebar.success(f"Uploaded model: {uploaded_model_file.name}")
    else:
        st.stop()
else:
    # Determine the default index for the selectbox
    default_model_index = 0
    if DEFAULT_MODEL_NAME in available_models_in_dir:
        default_model_index = available_models_in_dir.index(DEFAULT_MODEL_NAME)

    selected_model_name_sidebar = st.sidebar.selectbox(
        "Select Trained Model",
        available_models_in_dir,
        index=default_model_index
    )
    model_path_to_load = os.path.join(MODEL_SEARCH_DIR, selected_model_name_sidebar)

# Load the selected model
model = load_model(model_path_to_load)

if model is None:
    st.warning("Model could not be loaded. Please check the selection or upload a model.")
    st.stop()

confidence_thresh = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05
)
iou_thresh = st.sidebar.slider(
    "IoU Threshold (NMS)", 0.0, 1.0, IOU_THRESHOLD, 0.05
)

# --- Image Upload ---
uploaded_file = st.file_uploader("📁 Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        pil_image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image:")
            st.image(pil_image, use_column_width=True)

        if st.button("🔍 Detect Lesions"):
            if model:
                with st.spinner("Detecting..."):
                    results = model.predict(pil_image, conf=confidence_thresh, iou=iou_thresh)
                    result = results[0]
                    annotated_image_np = np.array(pil_image).copy()

                    if len(result.boxes) > 0:
                        detection_summary = []
                        for i, box in enumerate(result.boxes):
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls_id = int(box.cls[0].cpu().numpy())
                            class_name = model.names[cls_id]

                            cv2.rectangle(
                                annotated_image_np,
                                (int(xyxy[0]), int(xyxy[1])),
                                (int(xyxy[2]), int(xyxy[3])),
                                (0, 255, 0), 2
                            )
                            label = f"{class_name}: {conf:.2f}"
                            cv2.putText(
                                annotated_image_np, label,
                                (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                            )
                            detection_summary.append(f"  - Lesion {i+1}: **{class_name}** (Conf: {conf:.2f})")
                        
                        annotated_pil_image = Image.fromarray(annotated_image_np)
                        with col2:
                            st.subheader("Detection Results:")
                            st.image(annotated_pil_image, caption=f"{len(result.boxes)} lesions detected", use_column_width=True)
                        
                        st.markdown("---")
                        st.markdown("#### Detected Lesions Summary:")
                        for item in detection_summary:
                            st.markdown(item)
                    else:
                        with col2:
                            st.success("✅ No lesions detected with the current thresholds.")
                            st.image(pil_image, caption="Original Image (No Detections)", use_column_width=True)
            else:
                st.error("Model not loaded. Cannot perform detection.")
    except Exception as e:
        st.error(f"An error occurred processing the image: {e}")
        st.error("Please ensure the uploaded file is a valid image.")
else:
    st.info("👆 Please upload an image to begin.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for TB Lesion Detection Demo.")
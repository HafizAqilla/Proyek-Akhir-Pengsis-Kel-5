import streamlit as st
import torch
from torchvision import transforms
# <<< NEW: Import torchvision.models
import torchvision.models as tv_models
from PIL import Image
# import timm # We might not need timm for loading if we use torchvision directly

import os
from collections import OrderedDict # For handling state_dict keys

# --- Configuration (Derived from your vit_training.py) ---
MODEL_WEIGHTS_FILENAME = "vit_model.pth"
MODEL_WEIGHTS_PATH = os.path.join("weights", MODEL_WEIGHTS_FILENAME)

# MODEL_NAME = 'vit_base_patch16_224' # We'll use torchvision's model directly
IMAGE_SIZE = 224
NUM_CLASSES = 3
CLASS_NAMES = ["health", "sick no tb", "tb"]

NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

# --- Model Loading ---
@st.cache_resource
def load_trained_vit_model(weights_path, num_classes_model): # <<< Renamed function for clarity
    try:
        if not os.path.exists(weights_path):
            st.error(f"Model weights file not found at: {weights_path}")
            return None

        # <<< CHANGED: Use torchvision.models.vit_b_16
        # Create the model structure as it was in your training script before saving
        # Use weights=None for random initialization as we'll load our own.
        # For older torchvision versions, you might use pretrained=False
        model = tv_models.vit_b_16(weights=None) # Or tv_models.vit_b_16(pretrained=False)

        # IMPORTANT: Replicate the head modification from your training script
        # model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        # This line from training: model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        # The `in_features` for ViT-B/16 head is typically 768
        in_features = model.heads.head.in_features # Get in_features from the default head
        model.heads.head = torch.nn.Linear(in_features, num_classes_model)
        # If the above line for in_features gives an error, and you know it's 768 for vit_b_16:
        # model.heads.head = torch.nn.Linear(768, num_classes_model)


        st.info(f"Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

        # Handle potential 'module.' prefix if saved from DataParallel or DDP
        # (Your training script didn't seem to use DataParallel, but this is good practice)
        cleaned_state_dict = OrderedDict()
        is_ddp_model = False
        for k, v in state_dict.items():
            if k.startswith('module.'):
                is_ddp_model = True
                name = k[7:]  # remove `module.`
                cleaned_state_dict[name] = v
            else:
                cleaned_state_dict[k] = v
        
        if is_ddp_model:
            st.info("Handled 'module.' prefix from DDP model weights.")
            state_dict_to_load = cleaned_state_dict
        else:
            state_dict_to_load = state_dict

        model.load_state_dict(state_dict_to_load)
        model.eval()
        st.success(f"TorchVision ViT B/16 model loaded successfully from '{weights_path}'.")
        return model

    except Exception as e:
        st.error(f"Error loading model with TorchVision: {e}")
        st.exception(e) # Print full traceback
        st.error(f"Weights path: '{weights_path}'.")
        st.error("Tips: \n"
                 "1. Ensure TorchVision is installed correctly.\n"
                 "2. Verify NUM_CLASSES ({num_classes_model}) matches the output layer of your saved model.\n"
                 "3. The model structure in this script must exactly match how it was when `torch.save` was called.")
        return None

# --- Image Preprocessing (remains the same) ---
def preprocess_image(image_pil, image_size):
    transform_list = [
        transforms.Resize((image_size, image_size)),
    ]
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    transform = transforms.Compose(transform_list)
    return transform(image_pil).unsqueeze(0)

# --- Main Streamlit App ---
st.set_page_config(page_title="Chest X-Ray Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 Chest X-Ray Analysis with Vision Transformer")
st.markdown("""
Upload a PNG, JPG, or JPEG X-ray image of lungs. The Vision Transformer (ViT) model
will predict the likelihood of different health conditions.
""")

# <<< CHANGED: Call the new loading function
model = load_trained_vit_model(MODEL_WEIGHTS_PATH, NUM_CLASSES)

if model is None:
    st.warning("Model could not be loaded. Please check the console for errors and verify configurations in `app.py`.")
    # ... (rest of the UI if model is None remains similar) ...
    st.stop()

# ... (The rest of your Streamlit UI code for col1, col2, sidebar, etc. remains the same) ...
# Just ensure the parts that use MODEL_NAME in the sidebar are updated if you remove it globally.
# For example, in the sidebar:
# st.sidebar.markdown(f"**Model Architecture:** `TorchVision ViT B/16`")


# --- UI Elements ---
col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.header("📤 Upload X-ray Image")
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width="always")
        except Exception as e:
            st.error(f"Error opening or displaying image: {e}")
            uploaded_file = None
    else:
        st.info("Awaiting X-ray image upload.")


with col2:
    st.header("💡 Prediction Result")
    if uploaded_file is not None:
        if st.button("🔬 Analyze X-ray", use_container_width=True, type="primary"):
            with st.spinner("🧠 Analyzing image... Please wait."):
                try:
                    input_tensor = preprocess_image(image, IMAGE_SIZE)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)

                    predicted_class_name = CLASS_NAMES[predicted_idx.item()]
                    confidence_score = confidence.item()

                    st.subheader(f"**Model Prediction: {predicted_class_name.upper()}**")

                    if predicted_class_name == "tb":
                        st.error(f"Condition: Tuberculosis (TB) Signs Detected", icon="⚠️")
                        st.markdown("""
                        **Interpretation:** The model indicates a likelihood of Tuberculosis.
                        **Recommendation:** URGENT consultation with a medical professional is strongly advised for accurate diagnosis and treatment.
                        """)
                    elif predicted_class_name == "sick no tb":
                        st.warning(f"Condition: Sickness (Non-TB) Signs Detected", icon="🤒")
                        st.markdown("""
                        **Interpretation:** The model suggests signs of a lung condition that is not Tuberculosis.
                        **Recommendation:** Please consult a medical professional for further evaluation and diagnosis.
                        """)
                    elif predicted_class_name == "health":
                        st.success(f"Condition: Healthy", icon="✅")
                        st.markdown("""
                        **Interpretation:** The model predicts no clear signs of TB or other analyzed sicknesses.
                        **Recommendation:** Maintain regular health check-ups.
                        """)
                    else:
                        st.info(f"Prediction: {predicted_class_name}")


                    st.metric(label=f"Confidence in '{predicted_class_name.upper()}'", value=f"{confidence_score*100:.2f}%")

                    st.subheader("Detailed Class Probabilities:")
                    prob_data = {CLASS_NAMES[i]: probabilities[0][i].item() for i in range(NUM_CLASSES)}
                    st.bar_chart(prob_data)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.exception(e)
                    st.error("Please ensure the image is valid and model configurations are correct.")
        elif not uploaded_file:
            st.info("Upload an image and click 'Analyze X-ray'.")
    else:
        st.info("Upload an image to enable the 'Analyze X-ray' button.")


st.sidebar.header("ℹ️ About this App")
st.sidebar.info(
    # <<< CHANGED description
    f"This application uses a Vision Transformer (TorchVision ViT B/16) model, "
    "trained on the TBX11K dataset (or similar), to analyze chest X-ray images. "
    "It predicts one of three states: 'health', 'sick no tb', or 'tb'."
)
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Configuration")
# <<< CHANGED description
st.sidebar.markdown(f"**Model Architecture:** `TorchVision ViT B/16`")
st.sidebar.markdown(f"**Expected Input Size:** `{IMAGE_SIZE}x{IMAGE_SIZE}` pixels")
st.sidebar.markdown(f"**Output Classes ({NUM_CLASSES}):** `{', '.join(CLASS_NAMES)}`")
st.sidebar.markdown(f"**Weights File:** `{MODEL_WEIGHTS_FILENAME}`")
st.sidebar.markdown(f"**Normalization Mean:** `{NORM_MEAN}`")
st.sidebar.markdown(f"**Normalization Std:** `{NORM_STD}`")
st.sidebar.markdown("---")
st.sidebar.warning(
    "**⚕️ Medical Disclaimer:**\n"
    "This tool is for educational and research purposes ONLY. "
    "It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. "
    "Always seek the advice of your physician or other qualified health provider."
)
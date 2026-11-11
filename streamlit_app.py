import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # Safe for running headlessly without GUI clashes
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datetime import datetime

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Brain MRI Diagnostics Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Initialize Session State for tracking analysis history
if "history" not in st.session_state:
    st.session_state.history = []

# --- 1. CONFIGURATION & CONSTANTS ---
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Stroke", "Tumor"]
best_weight = 0.40  # DenseNet Weight determined during optimization

# Fixed benchmark accuracies from the notebook evaluation
DENSENET_ACC = 83.45
EFFICIENTNET_ACC = 97.24
ENSEMBLE_ACC = 97.47

# --- 2. LOAD PRE-TRAINED MODELS ---
@st.cache_resource
def load_diagnostic_models():
    try:
        densenet_model = tf.keras.models.load_model("best_class_weighted.keras")
        efficientnet_model = tf.keras.models.load_model("best_efficientnet.keras")

        def find_base_model(model):
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    return layer
            return None

        densenet_base = find_base_model(densenet_model)
        efficientnet_base = find_base_model(efficientnet_model)

        densenet_target_layer = [l.name for l in densenet_base.layers if 'conv' in l.name or 'relu' in l.name][-1]
        efficientnet_target_layer = [l.name for l in efficientnet_base.layers if 'conv' in l.name or 'activation' in l.name or 'add' in l.name][-1]

        return densenet_model, efficientnet_model, densenet_base, efficientnet_base, densenet_target_layer, efficientnet_target_layer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

densenet_model, efficientnet_model, densenet_base, efficientnet_base, densenet_target_layer, efficientnet_target_layer = load_diagnostic_models()

# --- 3. CORE PROCESSING FUNCTIONS ---
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

def get_nested_gradcam_heatmap(img_array, outer_model, base_layer, target_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[base_layer.input],
        outputs=[base_layer.get_layer(target_layer_name).output, base_layer.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, base_predictions = grad_model(img_array)
        x = base_predictions
        found_base = False
        for layer in outer_model.layers:
            if layer == base_layer:
                found_base = True
                continue
            if found_base:
                x = layer(x)
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    return superimposed_img

def generate_accuracy_plot():
    fig, ax = plt.subplots(figsize=(6, 4.2))
    models = ['DenseNet121', 'EfficientNetB0', 'Ensemble']
    accuracies = [DENSENET_ACC, EFFICIENTNET_ACC, ENSEMBLE_ACC]
    colors = ['#4C72B0', '#DD8452', '#55A868']
    bars = ax.bar(models, accuracies, color=colors, width=0.45, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Test Accuracy Comparison', fontsize=13, weight='bold', pad=12)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + 2, f'{height:.2f}%', ha='center', va='bottom', fontsize=10, weight='bold')
    plt.tight_layout()
    return fig

# --- 4. STREAMLIT USER INTERFACE ---
st.title("🧠 Brain MRI Stroke & Tumor Diagnostics Ensemble Dashboard")
st.write("Upload a brain MRI scan to predict whether it is **Normal**, represents a **Stroke**, or contains a **Tumor**.")

col1, col2 = st.columns([1.1, 1.3])

with col1:
    st.header("📷 Upload & Settings")
    uploaded_file = st.file_uploader("Choose a Brain MRI Scan Image...", type=["png", "jpg", "jpeg"])

    cam_type = st.radio(
        "Select Interpretability Heatmap Type:",
        ("DenseNet CAM", "EfficientNet CAM", "Ensemble Blended CAM"),
        index=2
    )

    info_markdown = (
        "### 💡 How to Interpret the Grad-CAM Visuals:\n"
        "* **🔴 Red/Orange Regions:** Highest diagnostic focus; where the model paid key attention.\n"
        "* **🟡 Yellow/Green Regions:** Secondary features supporting classification.\n"
        "* **🔵 Blue/Dark Regions:** Minimal or zero influence on final prediction."
    )
    st.markdown(info_markdown)

with col2:
    st.header("📊 Diagnostic Output")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_resized = image.resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_tensor = np.expand_dims(img_array, axis=0)

        if densenet_model is not None and efficientnet_model is not None:
            with st.spinner("Running diagnostic calculations..."):
                dens_input = densenet_preprocess(img_tensor.copy())
                dens_pred_prob = densenet_model.predict(dens_input)[0]
                eff_pred_prob = efficientnet_model.predict(img_tensor.copy())[0]

                ensemble_pred_prob = (dens_pred_prob * best_weight) + (eff_pred_prob * (1.0 - best_weight))
                pred_class_idx = np.argmax(ensemble_pred_prob)
                predicted_class = CLASS_NAMES[pred_class_idx]
                confidence = ensemble_pred_prob[pred_class_idx] * 100

                # Create session state history log record if not already added
                history_entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "File Name": uploaded_file.name,
                    "Predicted Diagnosis": predicted_class,
                    "Confidence Score": f"{confidence:.2f}%"
                }
                if not any(h["File Name"] == uploaded_file.name for h in st.session_state.history):
                    st.session_state.history.insert(0, history_entry)

                if cam_type == "DenseNet CAM":
                    heatmap = get_nested_gradcam_heatmap(dens_input, densenet_model, densenet_base, densenet_target_layer)
                elif cam_type == "EfficientNet CAM":
                    heatmap = get_nested_gradcam_heatmap(img_tensor.copy(), efficientnet_model, efficientnet_base, efficientnet_target_layer)
                else:
                    dens_h = get_nested_gradcam_heatmap(dens_input, densenet_model, densenet_base, densenet_target_layer)
                    eff_h = get_nested_gradcam_heatmap(img_tensor.copy(), efficientnet_model, efficientnet_base, efficientnet_target_layer)
                    heatmap = (dens_h * best_weight) + (eff_h * (1.0 - best_weight))
                    heatmap = np.maximum(heatmap, 0)
                    heatmap = heatmap / (np.max(heatmap) + 1e-8)

                cam_image = display_gradcam(img_array, heatmap)

            # Show diagnostic results side-by-side with accuracy stats
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                st.image(cam_image, caption="Model Attention (Grad-CAM Overlay)", use_container_width=True)
                st.subheader("Classification Probabilities")
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = float(ensemble_pred_prob[i])
                    st.write(f"**{class_name}**: {prob*100:.2f}%")
                    st.progress(prob)

            with res_col2:
                fig = generate_accuracy_plot()
                st.pyplot(fig)
        else:
            st.error("Models are not loaded. Please verify weights files exist.")
    else:
        st.info("Awaiting MRI scan upload on the left panel.")

# --- 5. HISTORY SECTION ---
st.write("---")
st.header("📜 Session Upload & Diagnosis History")
if len(st.session_state.history) > 0:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    if st.button("🗑️ Clear Session History"):
        st.session_state.history = []
        st.rerun()
else:
    st.write("No images have been analyzed in this session yet.")
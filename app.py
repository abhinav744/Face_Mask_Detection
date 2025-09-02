import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

# --------------------------
# Page Config (must be first Streamlit command)
# --------------------------
st.set_page_config(page_title="😷 Face Mask Detection", page_icon="😷", layout="centered")

# --------------------------
# Load your trained model
# --------------------------
@st.cache_resource
def load_model():
    model = keras.models.load_model("mask_detection_model.h5")  # save your model after training
    return model

model = load_model()

# --------------------------
# UI Layout
# --------------------------
st.title("😷 Face Mask Detection System")
st.write("Upload an image to check whether the person is **wearing a mask or not**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)  # or any width you prefer

    # Preprocess image
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (128,128))
    img_scaled = img_resized / 255.0
    img_reshaped = np.reshape(img_scaled, [1,128,128,3])

    # Prediction
    prediction = model.predict(img_reshaped)
    pred_label = np.argmax(prediction)

    # Show result
    if pred_label == 1:
        st.success("✅ The person is **wearing a mask**.")
    else:
        st.error("❌ The person is **not wearing a mask**.")

    # Debug info (optional toggle)
    with st.expander("🔍 Prediction Details"):
        st.write("Raw Prediction:", prediction)
        st.write("Predicted Label:", pred_label)

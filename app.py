import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("Villain Classifier 🔮")
st.write("Upload an image to classify which villain it belongs to.")

# Load model
model = tf.keras.models.load_model('transfer_model.h5', compile=False)
st.success("✅ Model loaded successfully!")

# Define classes (replace with your real ones)
class_names = ['Villain_A', 'Villain_B', 'Villain_C', 'Villain_D', 'Villain_E']

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")

    image = image.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    st.write("### Prediction Result:")
    st.write(f"Predicted Class: **{class_names[np.argmax(score)]}**")
    st.write(f"Confidence: **{100 * np.max(score):.2f}%**")

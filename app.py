import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("Villain Character Classifier")
st.write("Upload an image to identify the villain character.")

# Load model
model = tf.keras.models.load_model('transfer_model.h5', compile=False)
st.success("✅ Model loaded successfully!")

class_names = ['Venom', 'Darth Vader', 'Green Goblin', 'Thanos', 'Joker']

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.write("---")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(img, caption="Image for Classification", use_container_width=True)
    
    with col2:
        st.subheader("Classification Result")
        st.write("Processing...")


        img_resized = img.resize((128, 128))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)


        st.metric(label="Predicted Class", value=predicted_class)
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

        st.markdown("---")
        st.write("Top Predictions:")
        top_indices = np.argsort(score)[::-1][:3]
        for i in top_indices:
            st.write(f"- {class_names[i]}: {100 * score[i]:.2f}%")

st.write("---") 

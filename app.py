import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("Villain Character Classifier")
# CORRECTED GRAMMAR: Clear and concise.
st.write("Upload an image to identify the villain character.")

# Load model
# compile=False is necessary for loading the model successfully in the Streamlit environment.
model = tf.keras.models.load_model('transfer_model.h5', compile=False)
st.success("✅ Model loaded successfully!")

# Define classes (5 classes total)
class_names = ['Venom', 'Darth Vader', 'Green Goblin', 'Thanos', 'Joker']

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.write("---") 

    # GUI ADJUSTMENT: Create two columns for a side-by-side layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(img, caption="Image for Classification", use_container_width=True)
    
    with col2:
        st.subheader("Classification Result")
        st.write("Processing...")

        # --- Image Preprocessing ---
        # Fixed the variable conflict (img = img.resize) and ensured 128x128 shape.
        img_resized = img.resize((128, 128))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # --- Model Prediction ---
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        # --- Display Main Results ---
        st.metric(label="Predicted Class", value=predicted_class)
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        
        # --- Display Top Predictions (Fixes the IndexError) ---
        st.markdown("---")
        st.write("Top Predictions:")
        
        # FIX: Safely determine the top indices by ensuring we don't exceed the list bounds (5 classes).
        top_k = min(3, len(class_names))
        top_indices = np.argsort(score)[::-1][:top_k]
        
        for i in top_indices:
            # This line now uses guaranteed valid index 'i', resolving the IndexError.
            st.write(f"- {class_names[i]}: {100 * score[i]:.2f}%")

st.write("---")

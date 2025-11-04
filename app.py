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

# Define classes (5 classes total)
class_names = ['Venom', 'Darth Vader', 'Green Goblin', 'Thanos', 'Joker']
NUM_CLASSES = len(class_names) # 5

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.write("---") 

    # GUI: Two columns for a side-by-side layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(img, caption="Image for Classification", use_container_width=True)
    
    with col2:
        st.subheader("Classification Result")
        st.write("Processing...")

        # --- Image Preprocessing ---
        img_resized = img.resize((128, 128))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # --- Model Prediction ---
        predictions = model.predict(img_array)
        # Flatten the score array to ensure it's 1D, which handles model output inconsistencies
        score = tf.nn.softmax(predictions[0]).numpy().flatten() 
        
        # FINAL INDEX CHECK: Ensure the score array size matches the number of classes
        if score.size != NUM_CLASSES:
            st.error(f"Prediction Error: Model output size ({score.size}) does not match the defined class count ({NUM_CLASSES}).")
            st.stop() # Stops the execution cleanly
            
        # --- Display Main Results ---
        predicted_index = np.argmax(score)
        predicted_class = class_names[predicted_index]
        confidence = 100 * score[predicted_index]

        st.metric(label="Predicted Class", value=predicted_class)
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        
        # --- Display Top Predictions (Robust Python Sorting Fix) ---
        st.markdown("---")
        st.write("Top Predictions:")
        
        # 1. Combine class names and scores into a list of (score, name) tuples
        combined_results = list(zip(score, class_names))
        
        # 2. Sort the list by score in descending order
        combined_results.sort(key=lambda x: x[0], reverse=True)
        
        # 3. Display the top 3 results directly
        for score_val, class_name in combined_results[:3]:
            st.write(f"- {class_name}: {100 * score_val:.2f}%")

st.write("---")

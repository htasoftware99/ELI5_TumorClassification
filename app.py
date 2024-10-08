import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import requests

# Ollama API üzerinden gemma modeline istek gönderme
def get_gemma_explanation(prompt):
    try:
        ollama_endpoint = "http://localhost:11434/api/generate"
        payload = json.dumps({"model": "gemma:2b", "prompt": prompt, "stream": False})
        response = requests.post(ollama_endpoint, data=payload)
        response.raise_for_status()
        return response.json().get("response", "No response from Ollama.")
    except requests.exceptions.RequestException as e:
        return f"Error contacting Ollama API: {str(e)}"

def app():
    model = tf.keras.models.load_model('Brain Tumors Classifier.h5', compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    
    def preprocess_image(image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  
        img_array = np.expand_dims(img_array, 0)
        return img_array

    
    st.title("MRI Brain Tumor Detection")

    # ELI5 seçeneğini ekleyin
    eli5_mode = st.checkbox("Explain like I'm 5 (ELI5)")

    
    uploaded_file = st.file_uploader("Upload a MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        img_array = preprocess_image(image)

        # Tahminleri yapın
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]

        if predicted_class == 'No Tumor':
            display_class = 'Tumor does not exist'
            st.success(f"Result: **{display_class}**")
            if eli5_mode:
                prompt = "No tumor detected in the MRI scan. Please explain like I'm 5 years old."
            else:
                prompt = "No tumor detected in the MRI scan. Please explain."
        else:
            display_class = 'Tumor exists'
            st.error(f"Result: **{display_class}**")
            if eli5_mode:
                prompt = f"A {predicted_class} tumor has been detected in the MRI scan. Please explain like I'm 5 years old."
            else:
                prompt = f"A {predicted_class} tumor has been detected in the MRI scan. Please explain the implications."

        explanation = get_gemma_explanation(prompt)
        st.write(explanation)

if __name__ == "__main__":
    app()

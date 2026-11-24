
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configure the Streamlit page
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Define model and class-name file locations
MODEL_PATH = "crop_disease_model_final.keras"
CLASS_PATH = "class_names.txt"

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Load the class names
@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r") as file:
        return [line.strip() for line in file.readlines()]

# Load model and categories
model = load_model()
class_names = load_classes()

# Display application title and description
st.title("🌿 Multi-Crop Disease Detection System")

st.write(
    "Upload a leaf image to identify its crop and predict "
    "its possible disease category."
)

st.info("Supported crops: Tomato, Potato and Bell Pepper")

# Allow users to upload a leaf image
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# Process the uploaded image
if uploaded_file is not None:

    # Open the uploaded image and convert it to RGB
    image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    # Resize the image to the model's expected input dimensions
    resized_image = image.resize((224, 224))

    # Convert the image into a NumPy array
    image_array = np.array(resized_image, dtype=np.float32)

    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Generate predictions
    predictions = model.predict(image_array, verbose=0)[0]

    # Find the highest probability class
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]

    # Calculate prediction confidence
    confidence = float(predictions[predicted_index]) * 100

    # Identify the crop type
    if predicted_class.startswith("Tomato"):
        crop_name = "Tomato"
    elif predicted_class.startswith("Potato"):
        crop_name = "Potato"
    elif predicted_class.startswith("Pepper"):
        crop_name = "Bell Pepper"
    else:
        crop_name = "Unknown"

    # Convert the class name into readable text
    readable_disease = predicted_class.replace("_", " ")
    readable_disease = readable_disease.replace("  ", " ")

    # Display prediction results
    st.subheader("Prediction Results")

    st.success(f"Predicted Crop: {crop_name}")
    st.write(f"**Predicted Category:** {readable_disease}")
    st.metric("Prediction Confidence", f"{confidence:.2f}%")

    # Display the three most likely categories
    st.subheader("Top 3 Predictions")

    top_indices = np.argsort(predictions)[-3:][::-1]

    for index in top_indices:
        st.write(
            f"{class_names[index].replace('_', ' ')}: "
            f"{predictions[index] * 100:.2f}%"
        )

    # Display an important usage disclaimer
    st.warning(
        "This AI prediction is a screening aid, not a definitive "
        "plant diagnosis. Please consult an agricultural expert "
        "before taking treatment decisions."
    )

# Display project information
st.divider()
st.caption("Multi-Crop Disease Detection | Deep Learning with TensorFlow")
st.caption("SDG 2 – Zero Hunger")

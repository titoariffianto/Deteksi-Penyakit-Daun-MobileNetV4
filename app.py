import streamlit as st
from PIL import Image

# Title of the app
st.title("Image Detection App")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detection button
    if st.button("Detect"):
        # Placeholder for detection logic
        st.write("Detection in progress...")
        # Example output
        st.success("Detection complete! No objects detected.")
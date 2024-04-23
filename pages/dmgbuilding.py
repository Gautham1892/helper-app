import streamlit as st
from PIL import Image, ImageDraw
from io import BytesIO
from inference_sdk import InferenceHTTPClient

# Streamlit app title
st.title("Building Damage Inference App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Create a BytesIO object from the uploaded file
    bytes_data = uploaded_file.getvalue()
    image_file = BytesIO(bytes_data)
    
    # Open the image using PIL
    image = Image.open(image_file)
    
    # Create a copy of the original image for drawing bounding boxes
    image_with_boxes = image.copy()
    
    # Create a draw object
    draw = ImageDraw.Draw(image_with_boxes)
    
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="CVwxAyZDrlzg38W16MCa"
    )
    
    result = CLIENT.infer(image, model_id="damaged-building-jytkn/1")
    
    # Draw bounding boxes on the image
    for prediction in result['predictions']:
        x_center = prediction['x']
        y_center = prediction['y']
        width = prediction['width']
        height = prediction['height']
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    
    # Display the original uploaded image
    st.image(image, caption="Original Image", use_column_width=True)
    
    # Display the image with bounding boxes
    st.image(image_with_boxes, caption="Inference Result", use_column_width=True)
    
    # Display the inference result data
    st.write("Inference Result:")
    st.write(f"Time taken: {result['time']} seconds")
    st.write("Image dimensions:")
    st.write(f"Width: {result['image']['width']}, Height: {result['image']['height']}")
    st.write("Predictions:")
    for idx, prediction in enumerate(result['predictions']):
        st.write(f"Prediction {idx + 1}:")
        st.write(f"Class: {prediction['class']}")
        st.write(f"Confidence: {prediction['confidence']}")
        st.write(f"Bounding box coordinates:")
        st.write(f"x: {prediction['x']}, y: {prediction['y']}, width: {prediction['width']}, height: {prediction['height']}")

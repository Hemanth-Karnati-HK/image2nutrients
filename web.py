import streamlit as st
from PIL import Image
import os
import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from io import BytesIO
import zipfile
import os


# Load the pre-trained model
import requests


def download_model(model_url, model_path):
    response = requests.get(model_url)
    response.raise_for_status()
    with open(model_path, "wb") as f:
        f.write(response.content)


# Call the function to download the model
model_url = "https://drive.google.com/drive/folders/1gaM01eV6Ms2lAcwwjgy0UQQstoW_Aq5S?usp=sharing"
model_path = "finetuned_model"
download_model(model_url, model_path)

# Then you can load the model from the downloaded file
model = VisionEncoderDecoderModel.from_pretrained(model_path)

model.eval()

# Define the feature extractor
feature_extractor = ViTImageProcessor.from_pretrained(
    'nlpconnect/vit-gpt2-image-captioning')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'nlpconnect/vit-gpt2-image-captioning')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transfer the model to GPU if available
model = model.to(device)

# Set prediction arguments
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to predict ingredients from images


def predict_step(image_files, model, feature_extractor, tokenizer, device, gen_kwargs):
    images = []
    for image_file in image_files:
        if image_file is not None:
            # Create a BytesIO object from the UploadedFile (image_file)
            byte_stream = BytesIO(image_file.getvalue())
            image = Image.open(byte_stream)
            if image.mode != "RGB":
                image = image.convert(mode="RGB")
            images.append(image)

    if not images:
        return None

    inputs = feature_extractor(images=images, return_tensors="pt")
    inputs.to(device)
    output_ids = model.generate(inputs["pixel_values"], **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# Streamlit app code


def main():
    st.title("Image2Nutrients: Food Ingredient Recognition")
    st.write("Upload an image of your food to recognize the ingredients!")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform ingredient recognition
        preds = predict_step([uploaded_file], model,
                             feature_extractor, tokenizer, device, gen_kwargs)

        preds = preds[0].split('-')
        # remove numbers
        preds = [x for x in preds if not any(c.isdigit() for c in x)]
        # remove empty strings
        preds = list(filter(None, preds))
        # remove duplicates
        preds = list(dict.fromkeys(preds))

        # Display the recognized ingredients
        st.subheader("Recognized Ingredients:")
        for ingredient in preds:
            st.write(ingredient)


if __name__ == "__main__":
    main()

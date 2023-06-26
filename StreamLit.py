import nltk
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor, GPT2LMHeadModel, GPT2Tokenizer
import torch
from PIL import Image
import streamlit as st
from nltk.corpus import stopwords
import os
from io import BytesIO

# os.system('pip install --upgrade transformers')
# os.system('pip install nltk')
nltk.download('stopwords')

# Load the pre-trained model
model = VisionEncoderDecoderModel.from_pretrained(
    "SumanthKarnati/Image2Ingredients")
model.eval()

# Define the feature extractor
feature_extractor = ViTImageProcessor.from_pretrained(
    'nlpconnect/vit-gpt2-image-captioning')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'nlpconnect/vit-gpt2-image-captioning')

# Load GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

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


# Get the list of English stop words
stop_words = set(stopwords.words('english'))

# Function to remove stop words from a list of words


def remove_stop_words(word_list):
    return [word for word in word_list if word not in stop_words]

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

        preds = remove_stop_words(preds)

        # Display the recognized ingredients
        st.subheader("Recognized Ingredients:")
        for ingredient in preds:
            st.write(ingredient)

        preds_str = ', '.join(preds)

        # Prepare the prompt
        prompt = f"You are a knowledgeable assistant that provides nutritional advice based on a list of ingredients. The identified ingredients are: {preds_str}. Note that some ingredients may not make sense, so use the ones that do. Can you provide a nutritional analysis and suggestions for improvement?"

        # Encode and add special tokens
        input_ids = gpt2_tokenizer.encode(prompt, return_tensors='pt')

        # Generate a sequence of text
        output = gpt2_model.generate(
            input_ids, max_length=200, temperature=0.7, pad_token_id=gpt2_tokenizer.eos_token_id)

        # Decode the output
        suggestions = gpt2_tokenizer.decode(
            output[:, input_ids.shape[-1]:][0], clean_up_tokenization_spaces=True)

        st.subheader("Nutritional Analysis and Suggestions:")
        st.write(suggestions)


if __name__ == "__main__":
    main()

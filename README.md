# Project_N
Project_N is an application developed with the aim to extract nutritional insights from images of food, with no additional information required. The application operates using artificial intelligence to analyze images of meals, identify the ingredients present, and then provide a comprehensive nutritional breakdown. Here are the steps taken in the development of this application:

Fine-tuning the VisionEncoderDecoderModel: We started by fine-tuning a VisionEncoderDecoderModel, a pre-trained model, on a specific food dataset. The model was capable of interpreting food images and identifying key ingredients.

Feature Extraction and Tokenization: We utilized the Vision Transformer (ViT) and AutoTokenizer from the same pre-trained model ("nlpconnect/vit-gpt2-image-captioning") to handle the process of feature extraction and tokenization.

Dataset Preparation: We meticulously gathered, cleaned, and preprocessed a dataset of food images and their corresponding ingredients, which were stored in a CSV file. All images were ensured to be in the correct format and irrelevant entries were removed.

Model Training: Training the model was a complex process carried out on Google Colab Pro due to its high computational capacity. We ran batches of images and their corresponding captions (ingredient lists) from our dataset through the feature extractor and tokenizer. This transformed the images and text into a format that the model could understand. The inputs were then fed into the model, and the loss between the model's predictions and the actual labels was calculated. This loss was used to adjust the model's parameters through backpropagation and gradient descent. This process was repeated over 10 epochs, with the learning rate managed by the Adam optimizer.

Monitoring Training Progress: To keep track of the training progress, we implemented a condition to print the loss every 10 batches.

Creating User Interface with Streamlit: We used Streamlit to design a user-friendly web interface for Image2Nutrients. Streamlit allowed us to develop a simple and effective UI where users can upload images of their food and receive detailed nutritional information.

Code Management with GitHub: GitHub was employed for version control and code management. It helped us maintain a systematic record of different versions of our project, track changes, and collaborate effectively.

Despite facing challenges such as time-consuming model training and storage limitations of the Streamlit hosting environment, we successfully developed Image2Nutrients, a tool that can revolutionize how people understand and interact with their food.

Looking ahead, we plan to enhance the precision of our ingredient identification and nutritional analysis further. Our future plans include refining our model with additional data, introducing more nuanced parameters for analysis, and offering personalized dietary recommendations based on factors like age, gender, lifestyle, and health conditions. Our goal is to make Image2Nutrients an essential tool for personalized nutrition, promoting healthier eating habits, and fostering a healthier society globally.


![Progress](Example.png)
![Food Item 1](Example-1.png)
![Food Item 2](Example-2.png)
![Food Item 3](Example-3.png)
![Food Item 4](Example-4.png)
![Food Item 5](Example-5.png)


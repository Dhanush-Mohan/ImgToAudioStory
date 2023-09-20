import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests
from langchain import PromptTemplate, LLMChain, OpenAI
import os
from PIL import Image

#globalvar
audio_name=""

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# sub modules
# img2text


def img2text(image):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(image)[0]['generated_text']
    return text

# llm


def generate_story(scenario):
    template = """
    You are  a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 120 words;

    CONTEXT: {scenario}
    STORY: 
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)
    return story

# text2speech


def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open(f'{audio_name}.mp3', 'wb') as file:
        file.write(response.content)



# Title and description
st.title("Image to Story Converter")
st.write("Upload an image, and we'll generate a short story based on it!")

# Sidebar for image input
st.sidebar.header("Image Input")
uploaded_image = st.sidebar.file_uploader(
    "Upload an Image:", type=["jpg", "jpeg", "png"])

# Button to trigger image processing
if st.sidebar.button("Generate Story"):
    if uploaded_image:
        # Display uploaded image
        st.image(uploaded_image, caption="Uploaded Image",
                 use_column_width=True)

        # Image to Text
        st.subheader("Image to Text")
        image_to_text = pipeline(
            "image-to-text", model="Salesforce/blip-image-captioning-base")
        image = Image.open(uploaded_image)
        text = image_to_text(image)[0]['generated_text']
        audio_name+=text
        st.write(text)

        # Generate Story
        st.subheader("Generate Story")
        story = generate_story(text)
        #story = "Once upon a time, a group of friends set sail on a magnificent boat for an adventure of a lifetime. They shared laughter, food, and dreams as they sailed away into the endless sea. As the wind carried them forward, a sudden storm struck, causing the boat to rock violently. Fear gripped their hearts as waves crashed against their vessel. But their unity kept them strong. Amidst the chaos, they discovered a message in a bottle, thrown into the sea by a lonely soul. Intrigued, they followed the map leading to a hidden island. There, they found a treasure chest filled with priceless memories. With newfound strength, they returned to the boat, braving the storm with renewed hope. Together, they made it back to shore, forever grateful for the bonds forged during their perilous journey."
        st.write(story)

        # Text to Speech
        st.subheader("Text to Speech")
        text2speech(story)
        st.audio(f'{audio_name}.mp3', format='audio/mp3')


# Check if audio file exists and provide download link
if os.path.exists(f"{audio_name}.mp3"):
    st.subheader("Download Audio")
    download_button = st.download_button(label="Download Audio File", data=f"{audio_name}.mp3", key="download_audio")

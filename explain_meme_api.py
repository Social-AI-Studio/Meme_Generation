import argparse
import json
import os
import random

from concurrent import futures
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
from PIL import Image
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from tqdm import tqdm

from utils.prompts import *

# Define model configurations
MODEL_TYPE = {
    'qwen': {
        'base_url': "https://api.fireworks.ai/inference/v1", 
        'api_key': 'FIREWORKS_API_KEY',
        'model_name': 'accounts/fireworks/models/qwen2p5-vl-32b-instruct'
    },
    'gemini': {
        'base_url': "", 
        'api_key': 'GEMINI_API_KEY',
        'model_name': 'gemini-2.0-flash-001'
    },
}

load_dotenv()

# ### --------------------------------------------------------------------------------------------------------- ###
# ### ----------------------------------Specify required command line arguments-------------------------------- ###
# ### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Infer information on how meme templates should be used")
parser.add_argument(
    "--model",
    required=True,
    help="Model name for inference",
    choices=['qwen', 'gemini']
)
parser.add_argument(
    "--inputfile",
    help="Absolute file path of .json file containing URLs of memes with placeholder text overlays"
)
parser.add_argument(
    "--outputpath",
    required=True,
    help="Relative file path of .json file containing information on how meme templates should be used"
)

args = parser.parse_args()

MODEL = args.model
API_KEY = os.getenv(MODEL_TYPE[MODEL]['api_key'])
BASE_URL = MODEL_TYPE[MODEL]['base_url']

INPUT_FILE = args.inputfile
OUTPUT_PATH = args.outputpath

if INPUT_FILE is None:
    INPUT_FILE = f'./explain_meme/scraped_memes_stage0002A_{MODEL}_completed.json'

### --------------------------------------------------------------------------------------------------------- ###
### ----Create a directory to store .json file containing information on how meme templates should be used--- ###
### --------------------------------------------------------------------------------------------------------- ###
os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------Initialize Gemini API client-------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
def prompt_consolidation(
        user_text_prompt: str, 
        image_paths: list,
        encode: bool
    ):
    '''
    Combine text prompt with the images

    Parameters:
        user_text_prompt (str): Formatted user prompt according to the meme data
        image_paths (list): List of meme paths.
                        - Requires the first path as template meme
    
    Returns:
        messages (list[dict]): prompt message to be processed
    '''
    # Due to some model constraints on context length, we limit the number of images to 5. Do adjust according to the model used.
    MAX_NUM_IMG = min(6, len(image_paths))

    image_paths = image_paths[:MAX_NUM_IMG]
    if encode:
        image_paths = [encode_image(image_path) for image_path in image_paths]

    user_prompt_with_image = [
        {"type": "text", "text": user_text_prompt},
        {"type": "text", "text": 'This is the template meme.'},
        {"type": "image_url", "image_url": {'url': f'data:image/jpeg;base64,{image_paths[0]}'}}, 
        {"type": "text", "text": 'Here are the example memes:'},
    ]

    for i in range(1, len(image_paths)):
        user_prompt_with_image.append({"type": "image_url", "image_url": {'url': f'data:image/jpeg;base64,{image_paths[i]}'}},)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": EXPLAIN_MEME_SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": user_prompt_with_image
        }
    ]

    return messages

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------Function to call API------------------------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
def inference(
        client,
        messages: list
    ):
    """    
    Parameters:
        client (OpenAI Client): OpenAI client for API inference
        messages (list): List of messages to be sent to the API, containing user prompt and images.

    Returns:
        tuple containing

            - response_json (str): Response from calling the Gemini API, in JSON string format.
            - response_text (str): Text content of the response.
    """

    model_response = client.chat.completions.create(
        model=MODEL_TYPE[MODEL]['model_name'],
        max_tokens=1024,
        temperature=1e-6,
        top_p=1,
        messages=messages
    )

    response_json = json.dumps(model_response, default=lambda obj: obj.__dict__, indent=4)
    response_text = clean_json_format(model_response.choices[0].message.content)

    return response_json, response_text

@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(10)
)
def gemini_inference(
        client,
        user_text_prompt: str,
        image_paths: list
    ):
    """
    This function calls the Gemini API.
    This leverages code from https://ai.google.dev/gemini-api/docs/text-generation?lang=python AND https://ai.google.dev/gemini-api/docs/vision?lang=python.
            
    Parameters:
        user_text_prompt (str): User text message passed to the Gemini API.
        image_paths (list): List of local file paths to images, passed to the Gemini API.

    Returns:
        tuple containing

            - gemini_response_json (str): Response from calling the Gemini API, in JSON string format.
            - gemini_response_text (str): Text content of the response.
    """
    # Due to some model constraints on context length, we limit the number of images to 5. Do adjust according to the model used.
    MAX_NUM_IMG = min(6, len(image_paths))

    image_paths = image_paths[:MAX_NUM_IMG]

    contents = [user_text_prompt]
    for image_path in image_paths:
        contents.append(Image.open(image_path))

    # Invoke the chat completion endpoint of the Gemini API.
    gemini_response = client.models.generate_content(
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            seed=42,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
            system_instruction=EXPLAIN_MEME_SYSTEM_PROMPT,
            temperature=1e-6,
            top_p=1
        ),
        contents=contents,
        model="gemini-2.0-flash-001"
    )

    gemini_response_json = json.dumps(gemini_response, default=lambda obj: obj.__dict__, indent=4)
    gemini_response_text = gemini_response.text

    gemini_response_text = clean_json_format(gemini_response_text)
    return gemini_response_json, gemini_response_text

### --------------------------------------------------------------------------------------------------------- ###
### --------------------------------Define a function to update templates_data------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
def update_templates_data(
        client,
        val: dict
    ):
    """
    This function calls the inference function to infer information on how meme templates should be used, and updates templates_data with that information.

    Parameters:
        client (OpenAI or Gemini Client): OpenAI or Gemini client for API inference.
        val (dict): A dictionary containing information about a meme template.
    """
    box_count = val["box_count"]
    numbered_texts = (
        " and ".join([", ".join([f"Text {i}" for i in range(1, box_count)]),f"Text {box_count}"])
        if box_count > 1 else "Text 1"
    )
    seed = 42
    if val.get("knowyourmeme_about"):
        user_text_prompt = EXPLAIN_MEME_USER_PROMPT_WITH_ABOUT.format(
            about=val.get("knowyourmeme_about"),
            numbered_texts=numbered_texts
        )
    else:
        user_text_prompt = EXPLAIN_MEME_USER_PROMPT_WITHOUT_ABOUT.format(
            numbered_texts=numbered_texts
        )

    image_paths=[val["placeholder_meme_path"]] + [path for path in val["eg_list_local"]]

    if len(image_paths) <= 1:
        print(f"Not enough images for inference: {image_paths}")
        return 

    # Model for inference
    if MODEL == "gemini":
        response_json, response_text = gemini_inference(
            client=client,
            user_text_prompt=user_text_prompt,
            image_paths=image_paths
        )
    else:
        messages = prompt_consolidation(
            user_text_prompt=user_text_prompt, 
            image_paths=image_paths,
            encode=True
        )

        response_json, response_text = inference(
            client=client,
            messages=messages,
        )

    val.update({
        # "seed": seed,
        "response_json": response_json,
        "response_text": response_text
    })


### --------------------------------------------------------------------------------------------------------- ###
### -----------------Define main function that applies the update_templates_data function-------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
def main():
    # Read .json file containing URLs of memes with placeholder text overlays
    with open(INPUT_FILE, "r") as file:
        templates_data = json.load(file)
    
    with open('./scraped_memes_stage0001C_completed.json') as f:
        filtered_data = json.load(f)
    filtered_keys = list(filtered_data.keys())

    if MODEL == "gemini":
        client = genai.Client(api_key=API_KEY)
    else:
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

    COUNT = 0
    for key, val in tqdm(templates_data.items()):
        if key not in filtered_keys:
            continue

        update_templates_data(client, val)
        COUNT += 1
        
        if COUNT % 10 == 0:
            with open(f'{OUTPUT_PATH}/scraped_memes_stage0002B_{MODEL}_temp.json', 'w') as file:
                json.dump(templates_data, file)

    with open(f"{OUTPUT_PATH}/scraped_memes_stage0002B_{MODEL}_completed.json", "w") as file:
        json.dump(templates_data, file)

if __name__ == "__main__":
    main()

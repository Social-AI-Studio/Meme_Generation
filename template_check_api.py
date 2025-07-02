'''
This code only helps to filter out some of the example memes, which could be wrong.
'''

import argparse
import json
import os
import base64
from tqdm import tqdm

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

parser = argparse.ArgumentParser(description="Infer information on how meme templates should be used")
parser.add_argument(
    "--model",
    required=True,
    help="Model name for inference",
    choices=['qwen', 'gemini']
)
parser.add_argument(
    "--inputfile",
    required=True,
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

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)

def encode_image(
    image_path: str
):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def prompt_consolidation(
    data: dict
):
    '''
    Consolidate prompt for OpenAI API
    Parameters:
        data (dict): Data of meme template and example data to be verified

    Return:
        messages (list[list]): List of consolidate prompt message
    '''
    messages = []

    template_image = encode_image(data['placeholder_meme_path'])

    for example_meme_path in data['eg_list_local']:
        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": MEME_TEMPLATE_CHECK_SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is the template meme:"},
                    {"type": "image_url", "image_url": {'url': f'data:image/jpeg;base64,{template_image}'}},
                    {"type": "text", "text": "This is the example meme:"},
                    {"type": "image_url", "image_url": {'url': f'data:image/jpeg;base64,{encode_image(example_meme_path)}'}},
                    {"type": "text", "text": MEME_TEMPLATE_CHECK_USER_PROMPT}
                ]
            }
        ]

        messages.append(message)
    return messages

def gemini_prompt_consolidation(
    data: dict
):
    '''
    Consolidate prompt for Gemini API
    Parameters:
        data (dict): Data of meme template and example data to be verified

    Return:
        messages (list[list]): List of consolidate prompt message
    '''
    messages = []

    for example_meme_path in data['eg_list_local']:
        message = [
            "This is the template meme:",
            Image.open(data['placeholder_meme_path']),
            "This is the example meme:",
            Image.open(example_meme_path),
            MEME_TEMPLATE_CHECK_USER_PROMPT
        ]

        messages.append(message)
    return messages

@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(10)
)
def gemini_inference(
        client,
        data: dict,
        messages: list
    ):
    """
    Perform inferences with LLM

    Parameters:
        client (Gemini Client): Gemini client for API inference
        data (dict): Data of meme template and example data to be verified
        messages (list[list]): List of consolidate prompt message
    """

    valid_memes = []
    full_responses = []

    # Invoke the chat completion endpoint of the Gemini API.
    for message in messages:
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
                system_instruction=MEME_TEMPLATE_CHECK_SYSTEM_PROMPT,
                temperature=1e-6,
                top_p=1
            ),
            contents=message,
            model="gemini-2.0-flash-001"
        )

        gemini_response_json = json.dumps(gemini_response, default=lambda obj: obj.__dict__, indent=4)
        gemini_response_text = gemini_response.text
        if gemini_response_text is None:
            print("No response from Gemini API")
            return

        gemini_response_text = clean_json_format(gemini_response_text)
        try:
            result = json.loads(gemini_response_text)
            if 'match' not in result:
                print("Invalid response format", gemini_response_text)
                raise ValueError("Invalid response format")
        except:
            print("Invalid response format", gemini_response_text)
            result = {'response': gemini_response_text, 'match': False}

        valid_memes.append(result['match'])
        full_responses.append(gemini_response_json)

    example_meme_path = data['eg_list_local']

    valid_memes = [path for path, valid in zip(example_meme_path, valid_memes) if valid]

    data.update({
        'eg_list_local': valid_memes,
        'template_check_json': full_responses
    })

def inference(
        client,
        data: dict,
        messages: list
    ):
    """
    Perform inferences with LLM

    Parameters:
        client (OpenAI Client): OpenAI client for API inference
        data (dict): Data of meme template and example data to be verified
        messages (list[list]): List of consolidate prompt message
    """

    valid_memes = []
    full_responses = []

    for message in messages:
        model_response = client.chat.completions.create(
            model=MODEL_TYPE[MODEL]['model_name'],
            max_tokens=1024,
            temperature=1e-6,
            top_p=1,
            messages=message
        )

        response_json = json.dumps(model_response, default=lambda obj: obj.__dict__, indent=4)

        response = clean_json_format(model_response.choices[0].message.content)
        try:
            result = json.loads(response)
            if 'match' not in result:
                print("Invalid response format", response)
                raise ValueError("Invalid response format")
        except:
            print("Invalid response format", response)
            result = {'response': response, 'match': False}

        valid_memes.append(result['match'])
        full_responses.append(response_json)

    example_meme_path = data['eg_list_local']

    valid_memes = [path for path, valid in zip(example_meme_path, valid_memes) if valid]

    data.update({
        'eg_list_local': valid_memes,
        'template_check_json': full_responses
    })

def main():
    with open(INPUT_FILE, 'r') as f:
        template_data = json.load(f)

    if MODEL == "gemini":
        client = genai.Client(api_key=API_KEY)

        for key, val in tqdm(template_data.items()):
            if 'template_check_json' in val.keys():
                print(f"Already processed {key}")
                continue
            messages = gemini_prompt_consolidation(val)
            gemini_inference(client, val, messages)
    else:
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

        for key, val in tqdm(template_data.items()):
            messages = prompt_consolidation(val)
            inference(client, val, messages)

    with open(f'{OUTPUT_PATH}/scraped_memes_stage0002A_{MODEL}_completed.json', 'w') as f:
        json.dump(template_data, f)

if __name__ == "__main__":
    main()
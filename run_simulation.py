import os
import json
import argparse
import asyncio

from pydantic import BaseModel
from PIL import Image
from tenacity import retry, wait_random_exponential, stop_after_attempt
from inspect import iscoroutinefunction

from google import genai
from google.genai import types
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from utils.environment import Environment

import logging

load_dotenv()

# AWS credentials
AWS_ACCESS_KEY=os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

args = argparse.ArgumentParser(description="Environment for meme generation simulation")
args.add_argument("--rounds", type=int, default=1, help="Number of rounds to simulate")
args.add_argument("--article-num", type=int, default=1, help="Article number to simulate (1-5)")
args.add_argument("--vlm-model", type=str, default="gemini", choices=['gemini', 'qwen'], help="Vision-Language Model to use")
args.add_argument("--with-stance", action='store_true', help="Whether to include stance in the simulation")
args.add_argument("--continue-simulation", action='store_true', help="Whether to continue from the last simulation state")
args.add_argument("--upload", action='store_true', help="Whether to upload files to S3 after simulation")

args = args.parse_args()

class FilterMemeResponse(BaseModel):
    is_safe: bool

def run_simulation(
    rounds: int = 1, 
    article_num: int = 1, 
    vlm_model: str = "gemini", 
    with_stance: bool = False, 
    continue_simulation: bool = False
):
    env = Environment(rounds=rounds, article_num=article_num, vlm_model=vlm_model, with_stance=with_stance, continue_simulation=continue_simulation)

    asyncio.run(env.run_simulation())

# The code is same as VLMClient, but different safety settings, here with certain level of moderation
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(5)
)
async def gemini_inference(
    client,
    system_prompt: str,
    user_text_prompt: str, 
    image_path: str,
    output_schema: object,
    temperature: float = 1e-6,
    top_p: float = 1.0
):
    contents = [user_text_prompt]

    if image_path:
        contents.append(Image.open(image_path))
    else:
        return None, None

    response = await client.aio.models.generate_content(
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            seed=42,
            # safety settings are set to block medium and above for harmful content
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
            system_instruction=system_prompt,
            temperature=temperature,
            top_p=top_p,
            response_mime_type="application/json",
            response_schema=output_schema,
        ),
        contents=contents,
        model="gemini-2.0-flash-001",
    )

    response_json = json.dumps(response, default=lambda obj: obj.__dict__, indent=4)
    response_text = response.parsed

    # validate output against schema
    validate_output(response_text, output_schema)

    return response_json, response_text

def validate_output(
    model_response: dict, 
    output_schema: object
):
    try:
        # If already an instance, use as is; else, construct from dict
        if isinstance(model_response, output_schema):
            model_output = model_response
        else:
            model_output = output_schema(**model_response)
        # Check for empty or None fields
        if any(val is None or val == "" for val in model_output.model_dump().values()):
            raise ValueError("Model response contains empty fields.")
        return model_output
    except Exception as e:
        return False

async def safe_inference(
    client,
    system_prompt: str,
    user_text_prompt: str, 
    image_path: str,
    output_schema: object
):
    if iscoroutinefunction(gemini_inference):
        return await gemini_inference(
            client=client,
            system_prompt=system_prompt,
            user_text_prompt=user_text_prompt,
            image_path=image_path,
            output_schema=output_schema
        )
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: gemini_inference(
                client=client,
                system_prompt=system_prompt,
                user_text_prompt=user_text_prompt,
                image_path=image_path,
                output_schema=output_schema
            )
        )

async def filter_meme(
    client,
    meme_path: str
):
    result_json, result = await safe_inference(
        client=client,
        system_prompt="You are an expert content moderator in memes.",
        user_text_prompt="""You will be given a meme image. Your task is to detect whether this meme is safe for public viewing. If the meme is harmful or hateful, return 'False'. If the meme is safe, return 'True'. Strictly return only 'True' or 'False' without any additional text.""",
        image_path=meme_path,
        output_schema=FilterMemeResponse
    )
    if result is None:
        # If model fails to response, set that as unsafe
        return None, False, None
    return result_json, result.is_safe, meme_path

async def filter_meme_main(
    article_num:int, 
    model_name:str, 
    with_stance: bool = False
):
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    path_prefix = f"./simulation/article_{article_num}/{model_name}{'_stance' if with_stance else ''}"

    with open(f"{path_prefix}/meme_pool.json", "r") as f:
        meme_pool = json.load(f)

    safe_memes = []

    tasks = [filter_meme(client, meme['path']) for meme in meme_pool]
    results = await asyncio.gather(*tasks)

    try:
        for result in results:
            response_json, response, meme_path = result
            if response:
                safe_memes.append(meme_path)
            logging.info(f"Processed meme: {meme_path} - Safe: {response}")
    except Exception as e:
        logging.error(f"Error processing meme: {e}")

    with open(f"{path_prefix}/safe_memes.json", 'w') as f:
        json.dump(safe_memes, f)

    logging.info('Saved safe memes')

# Uploading files to S3 bucket
def upload_file(
    file_name: str, 
    bucket: str, 
    object_name: str = None
):
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3',
                             aws_access_key_id=AWS_ACCESS_KEY,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                            )
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        print(e)
        return False
    return True

def upload_simulation_files(
    article_num: int, 
    model_name: str, 
    with_stance: bool = False
):
    path_prefix = f"./simulation/article_{article_num}/{model_name}{'_stance' if with_stance else ''}"
    s3_path = f'article_{article_num}/{model_name}{"_stance" if with_stance else ""}'

    with open(f'{path_prefix}/safe_memes.json', 'r') as f:
        data = json.load(f)

    with open(f'{path_prefix}/meme_pool.json', 'r') as f:
        meme_pool = json.load(f)

    for meme in meme_pool:
        if meme['path'] not in data:
            meme_pool.remove(meme)

    with open(f'{path_prefix}/safe_meme_pool.json', 'w') as f:
        json.dump(meme_pool, f, indent=4, ensure_ascii=False)

    # Upload simulation files to S3 -> only final simulation result and meme pool are uploaded
    upload_file(f"{path_prefix}/full_simulation.json", 'meme-generation', f'{s3_path}/full_simulation.json')
    upload_file(f"{path_prefix}/full_simulation_json.json", 'meme-generation', f'{s3_path}/full_simulation_json.json')
    upload_file(f"{path_prefix}/meme_pool.json", 'meme-generation', f'{s3_path}/meme_pool.json')
    upload_file(f"{path_prefix}/safe_meme_pool.json", 'meme-generation', f'{s3_path}/safe_meme_pool.json')

    URL = []
    for meme_path in data:
        filename = meme_path.replace('simulation/', '')
        file_path = os.path.join('./simulation', filename)
        # print(file_path)
        upload_file(file_path, 'meme-generation', filename)

        # S3 bucket URL
        url='https://meme-generation.s3.ap-southeast-1.amazonaws.com/' + filename
        URL.append(url)

    with open(f'img_link/article_{article_num}_{model_name}{"_stance" if with_stance else ""}_link.json', 'w') as f:
        json.dump(URL, f)

if __name__ == "__main__":
    from time import time

    # Start simulation
    start_time = time()
    logging.info(f"Starting simulation with {args.rounds} rounds using {args.vlm_model} model...")
    run_simulation(rounds=args.rounds, article_num=args.article_num, vlm_model=args.vlm_model, with_stance=args.with_stance, continue_simulation=args.continue_simulation)
    end_time = time()
    logging.info(f"Simulation completed in {end_time - start_time} seconds.")

    # Filter generated memes
    logging.info("Filtering memes...")
    asyncio.run(filter_meme_main(article_num=args.article_num, model_name=args.vlm_model, with_stance=args.with_stance))

    if args.upload:
        # Upload safe memes to S3
        logging.info("Uploading simulation files to S3...")
        upload_simulation_files(article_num=args.article_num, model_name=args.vlm_model, with_stance=args.with_stance)
        logging.info("Simulation files uploaded successfully.")


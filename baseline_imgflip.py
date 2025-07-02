import requests
import json
from itertools import product
import os
import asyncio

from dotenv import load_dotenv

import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

os.makedirs("baseline_imgflip", exist_ok=True)

def create_imgflip_meme(topic, template_id):
    response = requests.post("https://api.imgflip.com/ai_meme", data={
        "username": os.getenv("IMGFLIP_USERNAME"),
        "password": os.getenv("IMGFLIP_PASSWORD"),
        "model": "openai",
        "template_id": template_id,
        "prefix_text": topic # basically the viewpoint
    })

    logging.info(f"Creating meme with template {template_id} for topic: {topic}")

    result = response.json()
    if result.get("success"):
        logging.info(f"Meme created successfully: {result['data']['url']}")
        return result, result["data"]["url"]
    else: # cuz sometimes may trigger {'success': False, 'error_message': 'Imgflip does not have enough example images to feed Open AI for this meme'}
        return result, ""

def save_meme(
    meme_url: str,
    template_id: str,
    key: str,
):
    extension = ".jpg"
    filename = f"{key}_{template_id}{extension}"
    folder = "baseline_imgflip"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    logging.info(f"Saving meme image to {path}")

    response = requests.get(meme_url)
    content_type = response.headers['content-type']
    img_bytes = response.content

    with open(path, "wb") as f:
        f.write(img_bytes)

    return path

def process_meme(key:str, viewpoint:str, template_id:str):
    try:
        result, result_url = create_imgflip_meme(viewpoint, template_id)
        if result_url:
            return result, save_meme(
                meme_url=result_url,
                template_id=template_id,
                key=key
            )
        return result, None
    except Exception as e:
        logging.error(f"Error processing meme for key {key} with template {template_id}: {e}")
        return None, None

def main():
    # Load meme templates
    with open(f'./explain_meme/scraped_memes_stage0002C_gemini_completed.json') as f:
        meme_templates_data = json.load(f)

    template_ids = [val['template_id'] for val in meme_templates_data.values()][:300]

    viewpoints = []
    with open(f'./simulation/selected_article.json') as f:
        viewpoints_data = json.load(f)

    for article_num, viewpoint in enumerate(viewpoints_data):
        viewpoints.append((f'{article_num}_pos', viewpoint['viewpoint_response']['positive']))
        viewpoints.append((f'{article_num}_neg', viewpoint['viewpoint_response']['negative']))
        viewpoints.append((f'{article_num}_neu', viewpoint['viewpoint_response']['neutral']))

    full_responses = []
    
    for (key, viewpoint), template_id in product(viewpoints, template_ids):
        res, path = process_meme(key, viewpoint, template_id)

        full_responses.append({
            "response": res,
            "path": path
        })

    with open('./baseline_imgflip/meme_paths.json', 'w') as f:
        json.dump(full_responses, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
    
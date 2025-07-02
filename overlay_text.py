import json
import os

import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path
from slugify import slugify
from tqdm import tqdm

load_dotenv('.env')

INPUT_FILE = './scraped_memes_stage0000_completed.json'
OUTPUT_PATH = './meme_template'

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)

def create_meme_with_placeholder_texts(
    username: str,
    password: str,
    template_id: int,
    box_count: int
):
    """
    This function invokes the /caption_image endpoint of the ImgFlip API to generate a meme that has placeholder texts and returns the URL pointing to the generated meme.

    Parameters:
        username (str): Username of the ImgFlip account.
        password (str): Password of the ImgFlip account.
        template_id (int): Template ID of the meme template.
        box_count (int): Number of text boxes associated with the meme template.

    Returns:
        placeholder_meme_url (str): URL pointing to the generated meme with placeholder text overlays such as "Text 1 is here."
    """
    # Prepare payload to invoke the /caption_image endpoint of the ImgFlip API.
    payload = {
        "username": username,
        "password": password,
        "template_id": template_id,
    }

    assert box_count is not None, print("Box count is not provided.")
    
    for i in range(1, box_count + 1):
        payload[f"boxes[{i}][text]"] = f"Text {i} is here."
    
    # Invoke the /caption_image endpoint of the ImgFlip API.
    response_caption = requests.post("https://api.imgflip.com/caption_image", data=payload)
    try:
        placeholder_meme_url = response_caption.json()["data"]["url"]
        return placeholder_meme_url
    except:
        print(f"Error: {response_caption.json()}")
        return None

def main():
    with open(INPUT_FILE, "r") as file:
        templates_data = json.load(file)

    error_names = []

    for name, meme_data in tqdm(templates_data.items()):
        template_id = meme_data["template_id"]
        box_count = meme_data["box_count"]
        placeholder_meme_url = create_meme_with_placeholder_texts(
            template_id=template_id,
            box_count=box_count,
            username=os.getenv("USERNAME"),
            password=os.getenv("PASSWORD")
        )
        if placeholder_meme_url is None:
            error_names.append(name)
            continue
        r = requests.get(placeholder_meme_url)
        local_path = OUTPUT_PATH + "/" + os.path.basename(placeholder_meme_url)
        if r.ok:
            with open(local_path, "wb") as f:
                f.write(r.content)
        meme_data["placeholder_meme_path"] = local_path

    for name in error_names:
        del templates_data[name]

    with open(f"./scraped_memes_stage0001_completed.json", "w") as file:
        json.dump(templates_data, file, indent=4)

if __name__ == "__main__":
    main()

import json
import requests
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

os.makedirs("baseline_supermeme", exist_ok=True)

def create_supermemes(text):
    response = requests.post(
        "https://app.supermeme.ai/api/v2/meme/image",
        headers={"Authorization": f"Bearer {os.getenv('SUPERMEME_KEY')}"},
        json={"text": text, "count": 12}, # max count is 12
    )
    result = response.json()
    if result.get("memes"):
        return result["memes"] # url to memes
    else:
        return []

def save_meme(
    meme_url: str,
    key: str,
):
    extension = ".jpg"
    filename = f"{key}{extension}"
    folder = "baseline_supermeme"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    logging.info(f"Saving meme image to {path}")

    response = requests.get(meme_url)
    content_type = response.headers['content-type']
    img_bytes = response.content

    with open(path, "wb") as f:
        f.write(img_bytes)

    return path

def main():
    with open('./simulation/selected_article_paraphrased_viewpoints.json') as f:
        viewpoints = json.load(f)

    sentiment = {
        0: "positive",
        1: "negative",
        2: "neutral"
    }

    with open('./baseline_supermeme/meme_paths.json') as f:
        full_responses = json.load(f)

    for i, viewpoint in enumerate(viewpoints):
        logging.info(f"Processing viewpoint {i+1}/{len(viewpoints)}: {viewpoint}")
        if i <= 1:
            continue

        article_num = i // 27 + 1
        sentiment_index = i % 27 // 9
        viewpoint_num = i % 9 + 1

        # Create memes using Supermeme
        memes = create_supermemes(viewpoint)
        
        if not memes:
            logging.warning(f"No memes created for viewpoint {i+1}.")
            continue
        
        # Save each meme
        for j, meme_url in enumerate(memes):
            key = f"article_{article_num}_{sentiment[sentiment_index]}_{viewpoint_num}_id{j}"

            meme_path = save_meme(meme_url, key=key)
            logging.info(f"Saved meme {j+1} for viewpoint {i+1} at {meme_path}")
            full_responses.append({
                "meme_id": key,
                "viewpoint": viewpoint,
                "path": meme_path
            })

    with open('./baseline_supermeme/meme_paths.json', 'w') as f:
        json.dump(full_responses, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
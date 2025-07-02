import argparse
import json
import os

import requests

from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path
from slugify import slugify

load_dotenv('.env')

EXAMPLE_OUTPUT_PATH = './example_meme'
BLANK_MEME_OUTPUT_PATH = './blank_memes'

os.makedirs(BLANK_MEME_OUTPUT_PATH, exist_ok=True)
os.makedirs(EXAMPLE_OUTPUT_PATH, exist_ok=True)

def extract_memes(
        page_number: int
    ):
    """
    This function scrapes a particular page of the Top Meme Templates of All Time section on ImgFlip, and stores the names of the meme templates as well as their associated URL suffixes.

    Parameters:
        page_number (int): The particular page number of the Top Meme Templates of All Time section to be scraped.

    Returns:
        templates_data (dict): 
            A nested dictionary with the keys of the outer dictionary as the names of the scraped meme templates, and the values as inner dictionaries. Each inner dictionary has one key, "template_url_suffix", with the value being the associated URL suffix of the meme template.
            - "<name of meme template>": {"template_url_suffix": "<URL suffix of meme template>"}
    """
    # Initialize a dictionary to store meme template names and their associated URL suffixes
    name_and_suffix = {}

    # URL for a page of Top Meme Templates in Last 30 days.
    page_url = f"https://imgflip.com/memetemplates?page={page_number}"

    # Send GET request to retrieve the Top Meme Templates of All Time webpage and parse associated html content
    response_templates = requests.get(page_url)
    soup_templates = BeautifulSoup(response_templates.text, "html.parser")

    # Find all meme titles (wrapped in <h3> tags with class "mt-title")
    for h3_tag in soup_templates.find_all("h3", class_="mt-title"):
        # Retrieve the name and the URL suffix of the meme template
        name = h3_tag.find("a").get_text(strip=True)
        template_url_suffix = h3_tag.find("a").get("href")
        name_and_suffix[name] = {"template_url_suffix": template_url_suffix}

    return name_and_suffix

def search_memes(
        username: str,
        password: str,
        name: str
    ):
    """
    This function invokes the /search_memes endpoint of the ImgFlip API to retrieve details about a meme template, given the name of the meme template.

    Parameters:
        username (str): Username of the ImgFlip account.
        password (str): Password of the ImgFlip account.
        name (str): Name of the meme template of interest.

    Returns:
        tuple containing

            - template_id (int | None): Template ID of a given meme template if details about the meme template was returned, else None.
            - box_count (int | None): Number of text boxes associated with a given meme template if details about the meme template was returned, else None.
            - template_url_full (str | None): URL pointing to the image of a given blank meme template if details about the meme template was returned, else None.
            - width (int | None): Width of a given blank meme template, in px, if details about the meme template was returned, else None.
            - height (int | None): Height of a given blank meme template, in px, if details about the meme template was returned, else None.
    """
    # Prepare payload to invoke the /search_memes endpoint of the ImgFlip API
    payload = {
        "username": username,
        "password": password,
        "query": name,
    }

    # Invoke the /search_memes endpoint of the ImgFlip API
    response_search = requests.post(
        "https://api.imgflip.com/search_memes",
        data=payload
    )
    memes_details = response_search.json()["data"].get("memes")

    # Retrieve details about the meme template if they exist
    if memes_details:
        template_id = memes_details[0]["id"]
        box_count = memes_details[0]["box_count"]
        template_url_full = memes_details[0]["url"]
        width = memes_details[0]["width"]
        height = memes_details[0]["height"]
        return template_id, box_count, template_url_full, width, height
    else:
        return None, None, None, None, None

def download_blank_meme(
    meme_url: str, 
    file_dir: str
):
    """
    This function takes in the meme's url and save the blank meme image.

    Parameters:
        meme_url (str): URL for the meme template image.
        file_dir (str): File path where the blank meme image will be saved.

    Returns:
        - bool: True if the image was downloaded successfully, False otherwise.
    """
    response = requests.get(meme_url)

    if response.status_code == 200:
        directory = os.path.dirname(file_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_dir, "wb") as fp:
            fp.write(response.content)
        return True
    else:
        print(f"Failed to download the image. Status code: {response.status_code}")
        return False

def get_meme_details(
        template_url_suffix: str
    ):
    """
    This function retrieves the list of example memes and alternative names (if any) of a meme template, given the URL suffix of a meme template.

    Parameters:
        template_url_suffix (str): URL suffix of a given meme template, e.g. "meme/Drake-Hotline-Bling."

    Returns:
        tuple containing

            - egmeme_list (list): List of URLs pointing to example memes, scraped from the meme template's ALL Memes section on ImgFlip.
            - alt_names (str | None): Comma-separated string of alternative names of the meme template, scraped from the meme template's ALL Memes section on ImgFlip.
    """
    # Retrieve example memes from the template page
    response_allmemes = requests.get(
        "https://imgflip.com/" + template_url_suffix)
    soup_allmemes = BeautifulSoup(response_allmemes.text, "html.parser")

    # Retrieve example meme URLs
    egmeme_list = []
    for meme_img in soup_allmemes.find_all("img", class_="base-img"):
        src = meme_img.get("src")
        egmeme_list.append(src)

    # Retrieve alternative names for the meme template, if any
    alt_names_div = soup_allmemes.find("div", class_="alt-names")
    if alt_names_div:
        alt_names_text = alt_names_div.get_text(strip=True)
        alt_names = alt_names_text[len("aka: "):]
    else:
        alt_names = None

    return egmeme_list, alt_names

def get_kym_about(
        candidates: str
    ):
    """
    This function retrieves the text of the ABOUT section (if any) of the Know Your Meme page for a meme, given a list of candidate names for the meme.

    Parameters:
        candidates (str): List of candidate names for the meme, e.g. Drake Hotline Bling, drakeposting, drakepost, drake hotline approves, drake no yes, drake like dislike, drake faces.

    Returns:
        tuple containing

            - kym_url (str | None): URL of the Know Your Meme page dedicated to the meme if Know Your Meme page exists and ABOUT section exists else None.
            - kym_about (str | None): Text of the ABOUT section of the Know Your Meme page for the meme, if Know Your Meme page exists and ABOUT section exists else None.

    """
    for candidate in candidates:
        slug_candidate = slugify(candidate).replace("'", "")
        kym_url = f"https://knowyourmeme.com/memes/{slug_candidate}"

        # Retrieve information about meme from "ABOUT" section
        response_kym = requests.get(kym_url)

        if response_kym.ok:
            soup_kym = BeautifulSoup(response_kym.text, "html.parser")
            about_header = soup_kym.find("h2", id="about")

            if about_header:
                about_paragraph = about_header.find_next_sibling("p")
                kym_about = about_paragraph.get_text()
                return kym_url, kym_about
    return None, None

def main():
    SCRAPED_MEMES = []
    full_template_data = {}

    # number of pages to scrape
    NUM_PAGES = 20
    for page_number in tqdm(range(1, NUM_PAGES + 1), desc="Scraping pages"):
        templates_data = extract_memes(page_number)
        for name in templates_data.copy():
            template_id, box_count, template_url_full, width, height = search_memes(
                username=os.getenv("USERNAME"),
                password=os.getenv("PASSWORD"),
                name=name)
            # To handle memes that are not found
            if template_id is None:
                del templates_data[name]
                continue
            
            # Check if the template is scraped before
            if template_url_full in SCRAPED_MEMES:
                print(f"{name} is scraped, skipping...")
                del templates_data[name]
                continue

            # Update the template data
            SCRAPED_MEMES.append(template_url_full)

            templates_data[name].update({
                "template_id": template_id,
                "box_count": box_count,
                "template_url_full": template_url_full,
                "width": width,
                "height": height
            })

            # Download blank meme
            filename = template_url_full.split('/')[-1]
            if not (filename.endswith('.jpg') or filename.endswith('.png')):
                filename = key + '.jpg'
            file_dir = os.path.join(BLANK_MEME_OUTPUT_PATH, filename)
            # If blank meme cannot be downloaded, remove the template
            if not download_blank_meme(template_url_full, file_dir):
                print(f"Failed to download image from {template_url_full}")
                file_dir = None
                del templates_data[name]
                continue

            eg_list_local = []
            eg_list, alt_names = get_meme_details(templates_data[name].get("template_url_suffix"))
            for url in eg_list:
                local_url = EXAMPLE_OUTPUT_PATH + "/" + os.path.basename(url)
                r = requests.get("https:" + url)
                if r.ok:
                    with open(local_url, "wb") as f:
                        f.write(r.content)
                    eg_list_local.append(local_url)
            templates_data[name].update({
                "eg_list_local": eg_list_local,
                "alt_names": alt_names,
                "blank_meme_path": file_dir,
            })
            candidates = [name]
            if alt_names:
                candidates += [alt.strip() for alt in alt_names.split(",")]
            kym_url, kym_about = get_kym_about(candidates)
            templates_data[name].update({
                "knowyourmeme_url": kym_url,
                "knowyourmeme_about": kym_about
            })

            full_template_data.update({
                name: templates_data[name]
            })

    with open(f"./scraped_memes_stage0000_completed.json", "w") as file:
        json.dump(full_template_data, file, indent=4)

if __name__ == "__main__":
    main()

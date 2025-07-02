import os
import re
import time
import json

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options

from bs4 import BeautifulSoup

import argparse

parser = argparse.ArgumentParser(description="Scrape articles from news website.")
parser.add_argument(
    "--source",
    default='cna',
    choices=['cna', 'st'],
    help="News website to scrape articles from."
)
parser.add_argument(
    "--outputpath",
    help="File path of .json file containing articles and its metadata."
)
args = parser.parse_args()

if args.source == 'cna':
    source = 'channelnewsasia'
elif args.source == 'st':
    source = 'straitstimes'

# Create simulation directory
if args.outputpath.startswith('simulation'):
    os.makedirs(
        './simulation',
        exist_ok=True
    )

def get_page_source(
    driver, 
    link: str
):
    """
    Access page using Selenium and get page data (full html) using Beautiful soup.

    Parameters:
        driver (webdriver): Selenium WebDriver instance.
        link (str): URL of the page to scrape.
    
    Returns:
        soup (BeautifulSoup): BeautifulSoup object containing the page's HTML.
    """
    driver.get(link)

    driver.implicitly_wait(5)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    return soup

def find_cna_articles(
    soup
):
    """
    Find all articles in the CNA Singapore section.

    Parameters:
        soup (BeautifulSoup): BeautifulSoup object containing the page's HTML.
    Returns:
        all_articles (list): List of article links found on the page.
    """
    all_articles = soup.find_all('a', class_='list-object__heading-link')
    return all_articles

def extract_article(
    driver, 
    article: str, 
    source: str
):
    """
    Search for all articles in the CNA Singapore section and extract metadata.

    Parameters:
        driver (webdriver): Selenium WebDriver instance.
        article (soup): Article component from soup.
        source (str): News source ('channelnewsasia' or 'straitstimes').
    Returns:
        article_data (dict): Dictionary containing article metadata such as title, author, date, link, and content.
    """
    link = f'https://www.{source}.com' + article.get('href')

    if source == 'channelnewsasia' and not link.startswith('https://www.channelnewsasia.com/singapore'):
        print("Skipping non-Singapore articles:", link)
        return None

    title = article.get_text(strip=True)

    soup = get_page_source(driver, link)

    if source == 'channelnewsasia':
        # Singapore articles
        print(f"Processing article: {title} at {link}")

        author_div = soup.find('div', class_='author-card__body')
        author = author_div.get_text(strip=True) if author_div else ""
        date_div = soup.find('div', class_='article-publish')
        date = date_div.get_text(strip=True) if date_div else ""

        content = soup.find('section', {'data-title': 'Content'}).find('div', class_='text')
        content = content.get_text(separator=' ', strip=True)
    else:
        author = soup.find('a', class_='byline-name').get_text(strip=True)
        date = soup.find('div', {"data-testid": 'timestamp-container'}).get_text(strip=True)

        content = soup.find('section', class_='article-content').find_all('p', class_='paragraph-base')
        content = ' '.join([p.get_text(separator=' ', strip=True) for p in content])

    return {
        'title': title,
        'author': author,
        'date': date,
        'link': link,
        'content': content
    }

def find_straits_times_articles(
    soup
):
    most_popular_block = soup.find('div', class_='block_most_popular')
    
    articles = most_popular_block.find_all('a', class_='card-link')
    return articles

def scrape(source: str):
    """
    Scrape articles from news website.

    Parameters:
        source (str): News source ('channelnewsasia' or 'straitstimes').
    """
    # Setting up Selenium WebDriver
    options = Options()
    options.add_argument("--headless")  # Disable extensions

    driver = webdriver.Firefox(options=options)

    link = f'https://www.{source}.com/singapore'  # Default URL for CNA
    soup = get_page_source(driver, link)

    all_articles = []

    if source == 'straitstimes':
        articles = find_straits_times_articles(soup)
    elif source == 'channelnewsasia':
        articles = find_cna_articles(soup)

    for article in articles:
        data = extract_article(driver, article, source)

        if data is None:
            continue
        
        all_articles.append(data)

    driver.quit()

    output_path = args.outputpath if args.outputpath else f'{source}_articles.json'
    if not output_path.endswith('.json'):
        output_path += '.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    scrape(source)
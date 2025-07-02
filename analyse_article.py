import os
import json
import asyncio

from pydantic import BaseModel
from dotenv import load_dotenv

from utils.vlm_client import VLMClient
from utils.prompts import ANALYZE_ARTICLE_VIEWPOINT

load_dotenv()

def find_best_match(
    selected_title: str, 
    available_titles: list
):
    # Case-insensitive substring match
    for article_title in available_titles:
        if selected_title.lower() in article_title.lower():
            return article_title
    return None

async def analyze_article(vlm_client: VLMClient):
    analyzed_file_path = f"./simulation/selected_article.json"

    # Initialize response models
    class ArticleSelectionResponse(BaseModel):
        title: str
        summary: str

    class AllArticleResponse(BaseModel):
        articles: list[ArticleSelectionResponse]

    class ViewpointResponse(BaseModel):
        positive: str
        negative: str
        neutral: str

    # Analyze articles
    with open('./simulation/cna_articles.json') as f:
        articles = json.load(f)

    article_consolidation = {}

    for article in articles:
        article_title = article['title']
        article_content = article['content']
        article_consolidation[article_title] = article_content
    
    article_json = json.dumps(article_consolidation, indent=4, ensure_ascii=False)

    # Inference all articles and select one
    article_response_json, selected_articles = await vlm_client.safe_inference(
        system_prompt="""You are a creative social media user tasked to analyze several articles, select 5 articles that are must be in Singapore context while you think best fits to generate memes and provide a summary for each article.

The user will provide the articles in the following JSON format:
{
    "Article 1's title": "content of article",
    "Article 2's title": "content of article",
    ...
}

Task:
1. Analyze the articles and select 5 articles that is relevant to Singapore context and best fits to generate memes.
2. Provide a summary for each selected article.

No additional output required.""",
        user_text_prompt=article_json, 
        image_paths=[], 
        output_schema=AllArticleResponse
    )

    print(f"Selected articles: {selected_articles.articles}")
    articles = selected_articles.articles
    if len(articles) != 5:
        raise ValueError(f"Expected 5 articles, but got {len(articles)}. Please check the input articles.")

    full_response = []
    for selected_article in articles:
        match = find_best_match(selected_article.title, article_consolidation.keys())
        if match:
            article = ArticleSelectionResponse(
                title=match,
                summary=selected_article.summary,
            )
            article_content=article_consolidation[match]
        else:
            raise ValueError(f"Selected article '{selected_article.title}' does not match any available articles.")

        # Generate viewpoints based on the selected article
        viewpoint_response_json, viewpoints = await vlm_client.safe_inference(
            system_prompt=f'''You are a creative social media user tasked to analyze an article and generates multiple viewpoints according to the following instructions.
{ANALYZE_ARTICLE_VIEWPOINT}''',
            user_text_prompt=article_content,
            image_paths=[],
            output_schema=ViewpointResponse
        )

        full_response.append(
            {
                "article_response_json": article_response_json,
                "article_response": article.model_dump(),
                "viewpoint_response_json": viewpoint_response_json,
                "viewpoint_response": viewpoints.model_dump()
            }
        )

    print(full_response)

    with open(analyzed_file_path, 'w') as f:
        json.dump(full_response, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(analyze_article(VLMClient(model_name='gemini', api_key=os.getenv('GEMINI_API_KEY'))))

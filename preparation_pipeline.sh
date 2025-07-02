#!/bin/bash

# PARTS THAT ONLY NEED TO RUN ONCE
echo "=====SCRAPE NEWS====="
python scrape_article.py --source cna --outputpath simulation/cna_articles.json

echo "=====ANALYZE NEWS====="
python analyze_article.py

echo "=====SCRAPING FROM IMGFLIP====="
python scrape_imgflip.py

echo "=====OVERLAY TEXT ON TEMPLATES====="
python overlay_text.py

# RUN FOR EACH MODEL (closed-source APIs)
for model in "qwen" "gemini"
do
    echo "==================================================="
    echo "PREPARATION STAGE STARTED FOR $model"
    echo $model

    echo "=====CHECK TEMPLATE====="
    python template_check_api.py --model $model --inputfile './scraped_memes_stage0001_completed.json' --outputpath './explain_meme'

    echo "=====EXPLAIN MEME TEMPLATE====="
    python explain_meme_api.py --model $model --outputpath './explain_meme'

    echo "PREPARATION STAGE FINISHED FOR $model"
    echo "==================================================="
done

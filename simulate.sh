#!/bin/bash

for article_num in {1..5}
do
    echo "==================================================="
    echo "STARTING SIMULATION FOR ARTICLE $article_num"

    for settings in "" "--with-stance"
    do
        if [ -z "$settings" ]; then
            log_suffix=""
            echo "=====START SIMULATION FOR ARTICLE $article_num (without stance)====="
            python run_simulation.py --vlm-model qwen --rounds 30 --article-num $article_num > "simulate_article_${article_num}_qwen.log" 2>&1 &
            python run_simulation.py --vlm-model gemini --rounds 30 --article-num $article_num > "simulate_article_${article_num}_gemini.log" 2>&1 &
        else
            log_suffix="_stance"
            echo "=====START SIMULATION FOR ARTICLE $article_num (with stance)====="
            python run_simulation.py --vlm-model qwen --rounds 30 $settings --article-num $article_num > "simulate_article_${article_num}_qwen_stance.log" 2>&1 &
            python run_simulation.py --vlm-model gemini --rounds 30 $settings --article-num $article_num > "simulate_article_${article_num}_gemini_stance.log" 2>&1 &
        fi
        wait
    done
    echo "SIMULATION STAGE FINISHED FOR ARTICLE $article_num"
    echo "==================================================="
done

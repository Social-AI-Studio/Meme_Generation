# Meme Generation

Repository for meme generation through agent-based simulation. Current models are Gemini 2.0 Flash and Qwen 2.5 VL 32B.

## Getting Started

Create an `.env` file consists of the following API keys
```bash
IMGFLIP_USERNAME="<IMGFLIP_USERNAME>"
IMGFLIP_PASSWORD="<IMGFLIP_PASSWORD>"
SUPERMEME_KEY="<SUPERMEME_KEY>"                 # Baseline comparison
GEMINI_API_KEY="<GEMINI_API_KEY>"
FIREWORKS_API_KEY="<FIREWORKS_API_KEY>"         # For Qwen2.5-VL model
OPENAI_API_KEY="<OPENAI_API_KEY>"
AWS_ACCESS_KEY="<AWS_ACCESS_KEY>"               # Storage
AWS_SECRET_ACCESS_KEY="<AWS_SECRET_ACCESS_KEY>" # Storage
```

Install python requirements
```bash
pip install -r requirements.txt
```

Preparation stage (Scrape meme templates, articles etc.)
```bash
bash preparation_pipeline.sh
```

Run simulation
```bash
bash simulate.sh
```
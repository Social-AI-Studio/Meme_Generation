import os
import base64
import json
import asyncio
from inspect import iscoroutinefunction

from PIL import Image
from openai import OpenAI, AsyncOpenAI, LengthFinishReasonError
from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from json_repair import repair_json

import logging

logging.basicConfig(
    level=logging.INFO,  # Capture all messages from INFO and up
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

class VLMClient:
    def __init__(
        self,
        model_name: str,
        api_key: str
    ):
        self.name = model_name
        if model_name == 'gemini':
            self.client = genai.Client(api_key=api_key)
            self.model_name = 'gemini-2.0-flash-001'
            self.inference_function = self.gemini_inference
        elif model_name == 'qwen':
            # self.client = OpenAI(
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url='https://api.fireworks.ai/inference/v1'
            )
            self.model_name = 'accounts/fireworks/models/qwen2p5-vl-32b-instruct'
            self.inference_function = self.inference
        else:
            raise ValueError("Unsupported VLM model. Choose 'gemini' or 'qwen'.")
    
    def encode_image(
        self, 
        image_path: str
    ):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def clean_json_format(
        self, 
        text: str
    ):
        text = text.strip()

        # Simple Cleaning (Remove ```json and ```)
        if text.startswith('```json'):  
            text = text.replace('```json', '')
            text = text.replace('```', '')
            text = text.strip()

        if not text.startswith('{') and '{' in text:
            # Remove everything before the first '{'
            text = text[text.index('{'):]

        # Remove output after "}" token
        if not text.startswith('}') and '}' in text:
            text = text[:text.rfind('}') + 1]

        # Repair bad JSON
        text = repair_json(text)
        return text

    async def safe_inference(
        self, 
        *args, 
        **kwargs
    ):
        if iscoroutinefunction(self.inference_function):
            return await self.inference_function(
                system_prompt=kwargs["system_prompt"],
                user_text_prompt=kwargs["user_text_prompt"],
                image_paths=kwargs.get("image_paths", []),
                output_schema=kwargs["output_schema"]
            )
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.inference_function(
                    system_prompt=kwargs["system_prompt"],
                    user_text_prompt=kwargs["user_text_prompt"],
                    image_paths=kwargs.get("image_paths", []),
                    output_schema=kwargs["output_schema"]
                )
            )

    @retry(
        wait=wait_random_exponential(min=1, max=100),
        stop=stop_after_attempt(5)
    )
    async def inference(
        self,
        system_prompt: str,
        user_text_prompt: str, 
        image_paths: list,
        output_schema: object,
    ):
        user_prompt = [
            {"type": "text", "text": user_text_prompt},
        ]

        if image_paths:
            for image_path in image_paths:
                encoded_image = self.encode_image(image_path)
                user_prompt.append(
                    {"type": "image_url", "image_url": {'url': f'data:image/jpeg;base64,{encoded_image}'}}
                )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        try:
            model_response = await self.client.beta.chat.completions.parse(
                model=self.model_name,
                max_tokens=1024,
                temperature=1e-6,
                top_p=1,
                messages=messages,
                response_format=output_schema
            )
        # DEBUG: An issue happened, and tries to solve, not guaranteed to work
        except LengthFinishReasonError:
            logging.info(f'System prompt: {messages[0]}')
            logging.info(f'User prompt: {[message for message in messages[1]["content"] if message["type"] == "text"]}')

            logging.error("Original output error, calling using original method")
            output_schema_json = output_schema.model_json_schema()

            messages[1]["content"][0]["text"] += f'\n\nResponse in the following JSON format: {str(output_schema_json)}'

            model_response = await self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=1e-6,
                top_p=1,
                messages=messages
            )

            response_json = json.dumps(model_response, default=lambda obj: obj.__dict__, indent=4)
            response_text = model_response.choices[0].message.content

            response_text = self.clean_json_format(response_text)
            response_text = json.loads(response_text)

            json_schema = output_schema.model_json_schema()['properties'].keys()
            if output_schema_json['properties'].keys() == response_text.keys():
                response_class = output_schema(**response_text)
                logging.info("Successfully construct output schema")
                return response_json, response_class
            else:
                logging.error("Tried to construct output schema, but response does not fit output schema...")
                return None, None

        response_json = json.dumps(model_response, default=lambda obj: obj.__dict__, indent=4)

        response_text = self.clean_json_format(model_response.choices[0].message.content)
        response_text = json.loads(response_text)
        
        # validate output against schema
        response_class = self._validate_output(response_text, output_schema)

        # return response_json, response_text
        return response_json, response_class

    @retry(
        wait=wait_random_exponential(min=1, max=100),
        stop=stop_after_attempt(5)
    )
    async def gemini_inference(
        self,
        system_prompt: str,
        user_text_prompt: str, 
        image_paths: list,
        output_schema: object,
        temperature: float = 1e-6,
        top_p: float = 1.0
    ):
        contents = [user_text_prompt]
        if image_paths:
            for image_path in image_paths:
                contents.append(Image.open(image_path))

        # response = self.client.models.generate_content(
        response = await self.client.aio.models.generate_content(
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                seed=42,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
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
            model=self.model_name,
        )

        response_json = json.dumps(response, default=lambda obj: obj.__dict__, indent=4)
        response_text = response.parsed

        # validate output against schema
        self._validate_output(response_text, output_schema)

        return response_json, response_text

    def _validate_output(
        self, 
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
            if any(val is None or val == "" for val in model_output.dict().values()):
                logging.error(f"Model response contains empty fields.")
                raise ValueError("Model response contains empty fields.")
            return model_output
        except Exception as e:
            logging.error(f"Output validation failed: {e}")
            return False

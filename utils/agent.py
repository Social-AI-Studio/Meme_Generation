import os
import base64
import requests
import json
import random

from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from utils.meme import Meme
from utils.prompts import *

import logging

logging.basicConfig(
    level=logging.INFO,  # Capture all messages from INFO and up
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

class Agent:
    def __init__(
        self, 
        id: int, 
        vlm_client: object, 
        generation_client: object, 
        article_num: int,
        persona: str, 
        strategy: str, 
        neighbors: list,
        viewpoint: str, 
        **kwargs
    ):
        # initialize clients
        self.vlm_client = vlm_client
        self.vlm_model_name = vlm_client.name

        self.generation_client = generation_client

        self.article_num = article_num

        # Initialize memory
        self.memory = []
        self.json_memory = {}

        # agent basic attributes
        self.id = id
        self.persona = persona
        self.strategy = strategy
        self.neighbors = neighbors
        
        # ImgFlip credentials
        self.imgflip_username = kwargs.get('imgflip_username', None)
        self.imgflip_password = kwargs.get('imgflip_password', None)

        # Initialize prompts
        self._get_prompts()

        # viewpoint will be empty string if we are simulating without stance
        self.viewpoint = viewpoint

        if self.viewpoint == '':
            self.with_stance = False
        else:
            self.with_stance = True

        class EvaluationSchema(BaseModel):
            score: int
            explanation: str

        self.eval_schema = EvaluationSchema

    def __repr__(self):
        return f"Agent(id={self.id}, persona={self.persona}, strategy={self.strategy}, viewpoint={self.viewpoint})"

    def to_dict(self):
        return {
            "id": self.id,
            "persona": self.persona,
            "strategy": self.strategy,
            "viewpoint": self.viewpoint,
            "memory": self.memory,
        }

    def _get_prompts(self):
        """
        Get prompts based on persona and strategy.
        """
        # system prompt
        self.persona_prompt = PERSONA_PROMPT[self.persona]
        
        # generation strategy prompt
        self.strategy_prompt = STRATEGY_PROMPT[self.strategy]

    async def _inference(
        self, 
        user_text_prompt: str,
        output_schema: object, 
        image_paths: list = []
    ):
        model_response = await self.vlm_client.safe_inference(
            system_prompt=self.persona_prompt,
            user_text_prompt=user_text_prompt,
            image_paths=image_paths,
            output_schema=output_schema
        )
        if model_response:
            return model_response
        else:
            #FIXME: how to solve this?
            logging.critical("Model response is None or empty.")
            # raise ValueError("Model response is None or empty.")
            return None, None

    @retry(
        wait=wait_random_exponential(min=1, max=100),
        stop=stop_after_attempt(5)
    )
    def caption_meme(
        self, 
        template_id: int, 
        texts: list
    ):
        # Prepare payload to invoke the /caption_image endpoint of the ImgFlip API
        boxes = [{"text": text} for text in texts]
        payload = {
            "template_id": template_id,
            "username": self.imgflip_username,
            "password": self.imgflip_password,
            "boxes": json.dumps(boxes)
        }
        for i, text in enumerate(texts):
            payload[f"boxes[{i}][text]"] = text

        # Invoke the /caption_image endpoint of the ImgFlip API
        response_caption = requests.post("https://api.imgflip.com/caption_image", data=payload)
        try:
            meme_url = response_caption.json()["data"]["url"]
            return "", meme_url
        except Exception as e:
            logging.error(f"Error generating meme for agent {self.id}: {e}")
            return None, None

    async def create_meme(
        self, 
        prompt: str
    ):
        # Other meme creation strategies
        try:
            result = await self.generation_client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                moderation="low",
                quality="high", # FIXME: change to high when ready
                n=1
            )
            # return None, None
        except Exception as e:
            logging.error(f"Error generating meme for agent {self.id}: {e}")
            return None, None
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        result_json = json.dumps(result.usage, default=lambda obj: obj.__dict__, indent=4)
        return result_json, image_bytes

    async def edit_meme(
        self,
        input_path:str,
        prompt:str,
    ):
        # Edit meme strategy
        with open(input_path, "rb") as img:
            try:
                result = await self.generation_client.images.edit(
                    model="gpt-image-1",
                    image=img,
                    prompt=prompt,
                    quality="high", #FIXME: change to high when ready
                    n=1,
                )
                # return None, None
            except Exception as e:
                logging.error(f"Error editing meme for agent {self.id}: {e}")
                return None, None
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        result_json = json.dumps(result.usage, default=lambda obj: obj.__dict__, indent=4)
        return result_json, image_bytes

    def _save_meme(
        self, 
        raw_meme: bytes | str, 
        current_round: int
    ):
        # save meme locally
        if isinstance(raw_meme, bytes):
            img_bytes = raw_meme
            extension = ".png"
        else:
            response = requests.get(raw_meme)
            content_type = response.headers['content-type']
            img_bytes = response.content
            extension = ".jpg"

        # since the node number (agent's id) will points to a specific agent, no need to store that much info in the filename
        filename = f"{current_round:02d}_node{self.id}{extension}"
        path = os.path.join(f"simulation/article_{self.article_num}/{self.vlm_model_name}{'_stance' if self.with_stance else ''}/meme_images", filename)

        with open(path, "wb") as f:
            f.write(img_bytes)
        return path

    async def _prompt_generate_meme(
        self,
        prompt: str,
        output_schema: object,
        inspiration_memes: list[object] = None,
        **kwargs
    ):
        meme_description_json, meme_description = await self._inference(
            user_text_prompt=prompt,
            image_paths=inspiration_memes,
            output_schema=output_schema
        )

        # DEBUG: Check if meme_description is None
        if meme_description is None:
            logging.warning(f"Agent {self.id} failed to generate a meme description. Retrying...")
            meme_description_json, meme_description = await self._inference(
                user_text_prompt=prompt + "\n\nThe meme description should be less sensitive. Answer without any explanation or preamble.",
                image_paths=inspiration_memes,
                output_schema=output_schema
            )

        if meme_description is None:
            logging.warning(f"Agent {self.id} failed to generate a meme description.")
            return None, None, None, None

        # Generate meme based on strategy
        if self.strategy == "caption":
            # NOTE: raw_meme here is a URL, not bytes (for naming consistency)
            model_response, raw_meme = self.caption_meme(
                template_id=kwargs['template_id'],
                texts=list(meme_description.dict().values())
            )
        elif self.strategy == "edit":
            model_response, raw_meme = await self.edit_meme(
                input_path=inspiration_memes[0],  # Should only pass 1 meme here
                prompt=meme_description.description
            )
        else:
            model_response, raw_meme = await self.create_meme(
                prompt=meme_description.description
            )
        return meme_description_json, meme_description, model_response, raw_meme

    async def generate_meme(
        self, 
        current_round: int,
        summarized_article: str,
        output_schema: object,
        inspiration_memes: list[object] = None,
        **kwargs
    ):
        if inspiration_memes is not None and len(inspiration_memes) > 0:
            inspiration = "Read all the memes (images) for inspiration first. "

            inspiration_memes = [meme.path for meme in inspiration_memes if meme is not None]
        else:
            inspiration = ''

        viewpoint = f'Viewpoint after reading the article: {self.viewpoint}\n' if self.viewpoint else ''
        generate_meme_prompt = f'{self.strategy_prompt.format(viewpoint=viewpoint, inspiration=inspiration, summarized_article=summarized_article, **kwargs)}'

        meme_description_json, meme_description, model_response, raw_meme = await self._prompt_generate_meme(prompt=generate_meme_prompt, output_schema=output_schema, inspiration_memes=inspiration_memes, **kwargs)

        if raw_meme is None:
            logging.info(f"Agent {self.id} failed to generate a meme in round {current_round}. Retrying with a new description...")
            if self.strategy == "caption":
                reprompt_meme_description = f"{generate_meme_prompt}\n\nThe previous attempt with {meme_description.model_dump()} failed to generate a meme. \nPlease rewrite the meme description and make sure it is less sensitive. Answer without any explanation or preamble."
            else:
                reprompt_meme_description = f"{generate_meme_prompt}\n\nThe previous attempt with {meme_description.description} failed to generate a meme. \nPlease rewrite the meme description and make sure it is less sensitive. Answer without any explanation or preamble."

            meme_description_json, meme_description, model_response, raw_meme = await self._prompt_generate_meme(prompt=reprompt_meme_description, output_schema=output_schema, inspiration_memes=inspiration_memes, **kwargs)

        # DEBUG: After retry, meme is still None, we will share the memes with second highest score
        if raw_meme is None:
            logging.warning(f"Agent {self.id} failed to generate a meme after retrying. No meme will be created for this round.")
            return None, None

        # Save the meme to a file
        path = self._save_meme(raw_meme, current_round)

        generated_meme = Meme(
            meme_id=None, # Meme ID will be assigned later
            author_id=self.id,
            path=path,
            generated_round=current_round,
            description=meme_description.dict(),
            viewpoint=self.viewpoint,
            scores={},
            times_shared=0,
            generation_json=meme_description_json
        )

        # Store current round in memory
        self.memory.append(generated_meme)
        response_json = {
            "meme_description": meme_description_json,
            "meme_generation_usage": model_response,
        }

        return response_json, generated_meme

    async def evaluate_meme(
        self,
        memes: list,
        current_round: int
    ):
        if len(memes) == 0:
            logging.info(f"Agent {self.id} has no memes to evaluate in round {current_round}.")
            return None, None, None

        # To keep track highest score
        highest_score = 0
        eval_scores = {}

        evaluate_json_list = []
        
        for meme in memes:
            # Handling edge cases
            if meme is None:
                logging.info(f'Agent {self.id} has empty meme to evaluate, could be "edit" strategy from previous round...')
                continue
            if meme.author_id == self.id:
                logging.info(f"Meme author, skipping this...")
                continue
            if self.id in meme.scores.keys():
                logging.info(f"Meme already evaluated by agent {self.id}, skipping this...")
                continue

            viewpoint = f'Your viewpoint held is: {self.viewpoint}' if self.viewpoint else ''
            evaluate_prompt = MEME_EVALUATION_PROMPT.format(strategy_desc=STRATEGY_DESCRIPTION_PROMPT[self.strategy], number_of_shares=meme.times_shared, viewpoint=viewpoint)

            evaluate_json, evaluate_response = await self._inference(
                user_text_prompt=evaluate_prompt, 
                image_paths=[meme.path],
                output_schema=self.eval_schema
            )

            evaluate_json_list.append(evaluate_json)
            score = evaluate_response.score

            # Check score is within range [1, 7]
            if score < 1:
                logging.warning(f"Evaluation score {score} is less than 1. Setting to 1.")
                score = 1
            elif score > 7:
                logging.warning(f"Evaluation score {score} is greater than 7. Setting to 7.")
                score = 7

            meme._assign_score(self.id, evaluate_response.dict())
            eval_scores[meme.author_id] = meme

            if score > highest_score:
                highest_score = score

        # If no memes were evaluated, return None
        if len(eval_scores) == 0:
            logging.info(f"No memes were evaluated by agent {self.id} in this round.")
            return None, None, None

        # Select the best meme based on scores
        best_memes = [meme for meme in eval_scores.values() if meme.scores[self.id]['score'] == highest_score]
        
        # DEBUG: In case some error happened, all memes are filtered out, which should not be happening
        if len(best_memes) == 0:
            logging.error(f"No memes scored {highest_score} by agent {self.id}. Getting the highest score again...")

            highest_score = max([meme.scores[self.id]['score'] for meme in eval_scores.values() if self.id in meme.scores])
            best_memes = [meme for meme in eval_scores.values() if meme.scores[self.id]['score'] == highest_score]

        # when there are multiple top scores, filter by viewpoint (same will get priority)
        # when there is no stance, will not do filtering
        if len(best_memes) > 1 and self.viewpoint != '':
            filtered = [meme for meme in best_memes if meme.viewpoint == self.viewpoint]
            best_memes = filtered or best_memes
        best_meme = random.choice(best_memes) if len(best_memes) > 1 else best_memes[0]

        # Store the highest scoring meme in memory
        if highest_score >= 6:
            best_meme.times_shared += 1
            self.memory.append(best_meme)

        return evaluate_json_list, highest_score, best_meme
        
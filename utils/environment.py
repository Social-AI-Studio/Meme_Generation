import sys
import os
import json
import random
import itertools
from operator import itemgetter
import asyncio
from time import time, sleep

from dotenv import load_dotenv
import networkx as nx
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, create_model

from utils.agent import Agent
from utils.meme import Meme
from utils.vlm_client import VLMClient
from utils.prompts import ARTICLE_SELECTION_SYSTEM_PROMPT, ANALYZE_ARTICLE_VIEWPOINT

import logging

logging.basicConfig(
    level=logging.INFO,  # Capture all messages from INFO and up
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

load_dotenv('.env')

class Environment:
    def __init__(
        self, 
        rounds: int = 1,
        article_num: int = 1, 
        vlm_model: str = "gemini", 
        with_stance: bool = False, 
        continue_simulation: bool = False
    ):
        # Directory for simulation outputs (simulation history and stores generated memes)
        os.makedirs(f"simulation/article_{article_num}/{vlm_model}{'_stance' if with_stance else ''}/meme_images", exist_ok=True)

        # initialize environment params
        self.total_rounds = rounds
        self.current_round = 1

        self.with_stance = with_stance
        self.vlm_model_name = vlm_model

        # We are looking at 5 articles only
        if article_num not in range(1, 6):
            raise ValueError(f"Article number must be between 1 and 5, got {article_num}.")

        self.article_num = article_num

        # initialize storage
        self.meme_pool = []
        self.model_response_json = {}
        self.simulation_history = {}

        if continue_simulation:
            self._load_previous_simulation()

        # Only working on Gemini and Qwen models for now
        vlm_api_key = os.getenv("GEMINI_API_KEY") if vlm_model == "gemini" else os.getenv("FIREWORKS_API_KEY")
        if not vlm_api_key:
            raise ValueError("VLM API key is not set. Please set GEMINI_API_KEY or FIREWORKS_API_KEY in your environment variables.")

        self.vlm_client = VLMClient(model_name=vlm_model, api_key=vlm_api_key)
        self.generation_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _load_previous_simulation(self):
        try:
            with open(f"./simulation/article_{self.article_num}/{self.vlm_model_name}{'_stance' if self.with_stance else ''}/full_simulation_json.json", 'r') as f:
                self.model_response_json = json.load(f)

            self.current_round = len(self.model_response_json) + 1
            
            if self.current_round > self.total_rounds:
                logging.warning(f"Current round {self.current_round} exceeds total rounds {self.total_rounds}. Ending simulation.")
                sys.exit()
            
            with open(f"./simulation/article_{self.article_num}/{self.vlm_model_name}{'_stance' if self.with_stance else ''}/full_simulation.json", 'r') as f:
                self.simulation_history = json.load(f)
            self.simulation_history = {int(round_num): {author_id: (Meme(**meme_detail) if meme_detail else None) for author_id, meme_detail in round_history.items()} for round_num, round_history in self.simulation_history.items()}
            
            with open(f"./simulation/article_{self.article_num}/{self.vlm_model_name}{'_stance' if self.with_stance else ''}/meme_pool.json", 'r') as f:
                meme_pool = json.load(f)

            for meme in meme_pool:
                if meme is None:
                    logging.warning("Found a None meme in the pool. This should not happen.")
                    # continue
                meme_id = meme['meme_id']
                generated_round = meme['generated_round']

                latest_appearance = None
                for round_num in range(generated_round, self.current_round):
                    for meme in self.simulation_history[round_num].values():
                        if meme is None:
                            continue
                        if meme.meme_id == meme_id:
                            latest_appearance = meme
                            break
                self.meme_pool.append(latest_appearance)

            logging.info(f"Continuing from round {self.current_round} with {len(self.meme_pool)} memes in the pool.")
        except SystemExit:
            raise  # Re-raise so the program actually exits
        except Exception as e:
            logging.error(f"Error loading previous simulation state: {e}")
            sys.exit()

    async def _init(
        self, 
        seed: int = 42
    ):
        rng = random.Random(42)

        # Load article
        self._load_article()

        # Agents' attributes
        strategies = ["caption", "objlab", "multipanel", "imgmacro", "edit"]
        personas = [
            {"persona": "fun", "sentiment": "positive"}, 
            {"persona": "irony", "sentiment": "neutral"}, 
            {"persona": "wit", "sentiment": "positive"}, 
            {"persona": "sarcasm", "sentiment": "negative"}, 
            {"persona": "humor", "sentiment": "positive"}, 
            {"persona": "satire", "sentiment": "negative"}, 
            {"persona": "nonsense", "sentiment": "neutral"}, 
            {"persona": "cynicism", "sentiment": "negative"}
        ]

        # Create all combinations of strategies and personas
        self.combinations = list(itertools.product(strategies, personas))
        rng.shuffle(self.combinations)  # Shuffle combinations

        self.graph = self._initialize_graph(len(self.combinations), k=4, p=0.05, seed=42)

        # Initialize agents
        self._initialize_agents()

        # Load meme templates
        with open(f'./explain_meme/scraped_memes_stage0002B_{self.vlm_model_name}_completed.json') as f:
            self.meme_templates = json.load(f)

    def _load_article(self):
        class ArticleSelectionResponse(BaseModel):
            title: str
            summary: str

        class ViewpointResponse(BaseModel):
            positive: str
            negative: str
            neutral: str

        with open('./simulation/selected_article.json') as f:
            data = json.load(f)

        article = data[self.article_num - 1]

        self.article = ArticleSelectionResponse(**article['article_response'])
        if self.with_stance:
            self.viewpoints = ViewpointResponse(**article['viewpoint_response'])
        else:
            self.viewpoints = ViewpointResponse(positive="", negative="", neutral="")

    def _initialize_graph(
        self, 
        n: int, 
        k: int = 4, 
        p: float = 0.05, 
        seed: int = 42
    ):
        G = nx.connected_watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
        G.remove_edges_from(nx.selfloop_edges(G))

        assert all(G.degree(n) > 1 for n in G.nodes)
        assert all(not G.has_edge(n, n) for n in G.nodes)  # no self-loops
        return G

    def _output_schema(
        self,
        box_count: int = None
    ):
        # Create a dynamic Pydantic model based on the number of text boxes: Caption strategy only
        if box_count:
            return create_model(
                "CaptionResponse",
                **{f'Text{i}': str for i in range(1, box_count + 1)}
            )
        # Other strategies will return a description
        else:
            return create_model(
                "MemeDescription",
                description=str
            )

    def _initialize_agents(self):
        for node, (strategy, persona) in zip(self.graph.nodes, self.combinations):
            kwargs = {}
            if strategy == "caption":
                kwargs = {
                    "imgflip_username": os.getenv("IMGFLIP_USERNAME"),
                    "imgflip_password": os.getenv("IMGFLIP_PASSWORD")
                }

            # Because the logic is reversed in the graph, i.e. the agent 'receives' memes from its neighbors, we need to use predecessors
            neighbors = list(self.graph.neighbors(node))

            self.graph.nodes[node]['agent'] = Agent(
                id=node, 
                vlm_client=self.vlm_client,
                generation_client=self.generation_client,
                article_num=self.article_num,
                persona=persona["persona"], 
                strategy=strategy, 
                neighbors=neighbors,
                viewpoint=getattr(self.viewpoints, persona["sentiment"]) if self.with_stance else '',
                **kwargs
            )

    def _save_history(
        self, 
        save_round: bool = True
    ):
        meme_pool = [meme.to_dict() if isinstance(meme, Meme) else meme for meme in self.meme_pool]

        if save_round:
            filename = f'round_{self.current_round}'
            json_content = self.model_response_json[self.current_round]
            simulation_history = self.simulation_history[self.current_round].copy()

            for agent_id in simulation_history:
                if simulation_history[agent_id] is None:
                    simulation_history[agent_id] = None
                else:
                    simulation_history[agent_id] = simulation_history[agent_id].to_dict()
        else:
            filename = 'full_simulation'
            json_content = self.model_response_json
            simulation_history = self.simulation_history.copy()

            for round_num, round_history in simulation_history.items():
                for agent_id in round_history:
                    if round_history[agent_id] is None:
                        round_history[agent_id] = None
                    else:
                        round_history[agent_id] = round_history[agent_id].to_dict()

        with open(f"./simulation/article_{self.article_num}/{self.vlm_model_name}{'_stance' if self.with_stance else ''}/{filename}.json", 'w') as f:
            json.dump(simulation_history, f, indent=4, ensure_ascii=False)

        with open(f"./simulation/article_{self.article_num}/{self.vlm_model_name}{'_stance' if self.with_stance else ''}/{filename}_json.json", 'w') as f:
            json.dump(json_content, f, indent=4, ensure_ascii=False)

        with open(f"./simulation/article_{self.article_num}/{self.vlm_model_name}{'_stance' if self.with_stance else ''}/meme_pool.json", 'w') as f:
            json.dump(meme_pool, f, indent=4, ensure_ascii=False)

    async def _process_agent(
        self, 
        node, 
        attributes
    ):
        try:
            agent = attributes['agent']
        except:
            logging.warning("Agent not found in node, skipping...")
            return "", best_neighbor_meme, response_json
        agent_id = str(agent.id)

        highest_score = None
        best_neighbor_meme = None

        response_json = {}

        # Evaluate memes
        if self.current_round != 1:
            # Get memes from neighbors
            agent_neighbors = agent.neighbors
            memes_from_neighbors = [
                self.simulation_history[self.current_round - 1][str(neighbor)]
                for neighbor in agent_neighbors
            ]

            # Filter out None values (in case some neighbors did not generate memes)
            memes_from_neighbors = [meme for meme in memes_from_neighbors if meme is not None]

            # Evaluate memes
            evaluate_responses, highest_score, best_neighbor_meme = await agent.evaluate_meme(
                memes=memes_from_neighbors,
                current_round=self.current_round
            )

            response_json["evaluate_responses"] = evaluate_responses

        if highest_score is None or highest_score < 6:
            kwargs = {}

            if agent.strategy == "edit":
                if best_neighbor_meme is None or self.current_round == 1:
                    logging.info(f"Agent {agent_id} found no suitable memes to edit.")
                    return agent_id, None, None

                inspiration_memes = [best_neighbor_meme]
                output_schema = self._output_schema()

            if agent.strategy == "caption":
                template, information = random.choice(list(self.meme_templates.items()))
                getter = itemgetter('template_id', 'response_text', 'blank_meme_path', 'box_count')
                template_id, response_text, blank_meme_path, box_count = getter(information)
                labels = (
                    '"Text1"' if box_count == 1
                    else ", ".join(f'"Text{i}"' for i in range(1, box_count)) + f' and "Text{box_count}"'
                )

                output_schema = self._output_schema(box_count=box_count)
                inspiration_memes = None

                kwargs = {
                    "template_id": template_id,
                    "information": response_text,
                    "numbered_texts": labels,
                }
            else:
                if self.current_round == 1:
                    memes_from_neighbors = None
                inspiration_memes = memes_from_neighbors
                output_schema = self._output_schema()

            generation_json, generated_meme = await agent.generate_meme(
                current_round=self.current_round, 
                summarized_article=self.article.summary, 
                inspiration_memes=inspiration_memes,
                output_schema=output_schema,
                **kwargs
            )

            # DEBUG: Fallback policy
            if self.current_round != 1:
                if generated_meme is None and best_neighbor_meme is not None:
                    # Here, best_neighbor_meme is the second best meme from neighbors
                    logging.info(f"Agent {agent_id} did not generate a meme, falling back to next best neighbor meme {best_neighbor_meme.meme_id}.")
                    return agent_id, best_neighbor_meme, response_json
                elif generated_meme is None and best_neighbor_meme is None:
                    logging.info(f"Agent {agent_id} did not generate a meme, and no suitable neighbor meme is available.")
                    return agent_id, None, None

            response_json["generation_json"] = generation_json
            return agent_id, generated_meme, response_json

        # Highest score is 6 or 7
        elif highest_score >= 6:
            if best_neighbor_meme is None:
                logging.critical(f"Agent {agent_id} found no suitable memes from neighbors, but highest score is {highest_score}.")
                raise ValueError(f"Agent {agent_id} found no suitable memes from neighbors, but highest score is {highest_score}.")
            # Share best meme from neighbor
            return agent_id, best_neighbor_meme, response_json

    async def run_round(self):
        current_round_history = {}
        current_round_json = {}

        # Launch all agent tasks in parallel
        tasks = [self._process_agent(node, attributes) for node, attributes in self.graph.nodes(data=True)]
        results = await asyncio.gather(*tasks)

        try:
            # Collect results
            for agent_id, generated_meme, response_json in results:
                # print(generated_meme)
                if generated_meme is None:
                    logging.info(f"Agent {agent_id} did not generate a meme this round.")
                    continue
                current_round_history[agent_id] = generated_meme
                current_round_json[agent_id] = response_json

                # New generated meme
                assert generated_meme is not None, "Why here has a None meme?"
                if generated_meme.meme_id is None:
                    generated_meme.meme_id = len(self.meme_pool) + 1
                    self.meme_pool.append(generated_meme)
        except Exception as e:
            logging.error(f"Error processing agent results, will try to save current round: {e}")
            # If any error raised, save the current round's history
            # Handle missing agents (if any)
            if len(current_round_history) != len(self.combinations):
                missing_agents = {str(i) for i in range(len(self.combinations))} - set(current_round_history)
                for agent_id in missing_agents:
                    logging.info(f"Agent {agent_id} did not generate a meme this round.")
                    current_round_history[agent_id] = None

            self.simulation_history[self.current_round] = current_round_history
            self.model_response_json[self.current_round] = current_round_json

        # Handle missing agents (if any)
        if len(current_round_history) != len(self.combinations):
            missing_agents = {str(i) for i in range(len(self.combinations))} - set(current_round_history)
            for agent_id in missing_agents:
                logging.info(f"Agent {agent_id} did not generate a meme this round.")
                current_round_history[agent_id] = None

        # self.previous_round_history = current_round_history
        self.simulation_history[self.current_round] = current_round_history
        self.model_response_json[self.current_round] = current_round_json

    async def run_simulation(self):
        logging.info("Load articles data and initialize agents")
        await self._init()

        for meme in self.meme_pool:
            if meme is None:
                logging.warning("Found a None meme in the pool. This should not happen.")
                self.meme_pool = [meme for meme in self.meme_pool if meme is not None]

        while self.current_round <= self.total_rounds:
            try:
                logging.info(f"Running round {self.current_round}...")
                start_time = time()
                await self.run_round()

                time_taken = time() - start_time

                logging.info(f"Round {self.current_round} ended, and took {time_taken} seconds. Saving history...")
                self._save_history(save_round=True)

                if len(os.listdir(f"./simulation/article_{self.article_num}/{self.vlm_model_name}{'_stance' if self.with_stance else ''}/meme_images")) != len(self.meme_pool):
                    logging.critical("Meme images directory has different number of memes saved...")
                    sys.exit()

                self.current_round += 1
                if time_taken < 60:
                    sleep(60 - time_taken)  # Ensure at least 1 minute between rounds, prevent hit RPM limits
            except Exception as e:
                logging.error(f"Error during round {self.current_round}: {e}", exc_info=True)

                # Still try to save history even if an error occurs
                try:
                    self._save_history(save_round=True)
                except:
                    logging.error("Failed to save round history...", exc_info=True)
                    break
        self._save_history(save_round=False)

if __name__ == "__main__":
    start_time = time()
    logging.info(f"Starting simulation with {args.rounds} rounds using {args.vlm_model} model...")
    env = Environment(rounds=args.rounds, vlm_model=args.vlm_model, with_stance=args.with_stance, continue_simulation=args.continue_simulation)
    logging.info(f"Initialized environment took {time() - start_time} seconds.")

    asyncio.run(env.run_simulation())

    end_time = time()
    logging.info(f"Simulation completed in {end_time - start_time} seconds.")

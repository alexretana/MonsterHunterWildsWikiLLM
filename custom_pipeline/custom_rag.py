"""
title: Llama Index Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
import requests
import asyncio
import os

KNOWLEDGE_LIST = ['Weapons', 'Armor', 'Items', 'Decorations', 'Misc']
_API_KEY = None

def read_dot_api_key():
    global _API_KEY
    if _API_KEY is None:
        try:
            with open('../../wikiproject/.open_webui_api_key', 'r') as f:
                _API_KEY = f.read().strip()
            if not _API_KEY:
                raise ValueError("API key file is empty")
        except FileNotFoundError:
            raise FileNotFoundError("API key file not found at '../../wikiproject/.open_webui_api_key'")
    return _API_KEY

def make_headers(extra_headers: dict = None):
    api_key = read_dot_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers

def print_response_error(response):
    error_message = f"Recieved Non-Successful Status Code({response.status_code}), and message :{response.text}"
    print(error_message)
    return error_message

async def get_knowledge_list():
    OPEN_WEBUI_DOMAIN_NAME = os.getenv("OPEN_WEBUI_DOMAIN_NAME", "http://localhost")
    full_url = OPEN_WEBUI_DOMAIN_NAME + ":8080/api/v1/knowledge/list"
    headers = make_headers()
    
    print(f"Pipeline attempting to call: {full_url}")
    
    try:
        response = requests.get(url=full_url, headers=headers, timeout=10)
        
        if response.status_code not in range(200, 299):
            print_response_error(response)
            response.raise_for_status()

        response_json = response.json()
        print(f"Successfully retrieved {len(response_json)} knowledge bases")
        return {knowledge['name']: knowledge['id'] for knowledge in response_json}
        
    except requests.exceptions.Timeout as e:
        print(f"Request timed out: {e}")
        raise e
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        raise e
    except Exception as e:
        print(f"Error getting knowledge list: {e}")
        raise e

class Pipeline:
    class Valves(BaseModel):
        pass

    def __init__(self):
        # Initialize the valves - CRITICAL for Open WebUI integration!
        self.valves = self.Valves()
        
        self.collection_texts = {
            "Weapons": "Weapons in Monster Hunter: Wilds define your combat style, with 14 unique types ranging from nimble Dual Blades to massive Greatswords and technical options like the Charge Blade or Hunting Horn, each with their own combos, mechanics, and upgrade trees. Great Sword Sword & Shield Dual Blades Long Sword Hunting Horn Lance Gunlance Hammer Switch Axe Charge Blade Insect Glaive  Bow  Light Bowgun  Heavy Bowgun",
            "Armor": "Armor provides both defense and passive skills, crafted from monster parts and upgraded over time; full sets can grant set bonuses, and mixing pieces lets hunters optimize builds around specific resistances and abilities. Each set includes hemls chests arms waists and legs.",
            "Items": "Items include consumables like Potions, Traps, Bombs, and Ammo used for survival, utility, and strategy during hunts, with many crafted from gathered materials in the field or managed through the item box and crafting lists.",
            "Decorations": "Decorations are slottable jewels that enhance skills on your gear, offering flexible build customization once you've progressed far enough to unlock high-rank gear and start farming tempered monsters for rare drops.",
            "Misc": "Misc covers everything else, including NPCs like the Smithy or Handler, quest types (story, optional, investigations, events), endemic life, locations like Astera and the Guiding Lands, and systems like bounties, melding, and the ecosystem interactions that make the world feel alive."
        }
        self.model = None
        self.collection_embeddings = None
        self.collection_ids = None
        print("Successful __init__()")
        print("Custom pipeline loaded with working directory:", dir())

    async def on_startup(self):
        # Precompute embeddings at startup
        print("Starting 'on_startup()'")
        self.model = SentenceTransformer("all-MiniLm-L6-v2")
        print("Finished loading SentenceTransformer")
        self.collection_embeddings = {
            name: self.model.encode(desc)
            for name, desc in self.collection_texts.items()
        }
        print("Finished encoding descriptions")
        try:
            self.collection_ids = await get_knowledge_list()
            print("Successfully got knowledge list")
            print(f"Available knowledge bases: {list(self.collection_ids.keys())}")
        except Exception as e:
            print("Failed to get_knowledge_list()")
            print(f"Error: {e}")
            print("Using fallback: pipeline will work without knowledge base integration")
            # Fallback: use empty dict so pipeline doesn't crash
            self.collection_ids = {}

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass
    
    def get_top_collections(self, query: str, top_k: int = 2):
        query_vec = self.model.encode(query)
        scores = {
            name: cosine_similarity(
                [query_vec], [self.collection_embeddings[name]]
            )[0][0]
            for name in self.collection_embeddings
        }
        sorted_names = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return sorted_names

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:

        top_collection = self.get_top_collections(user_message)

        return  {
            "prompt": user_message,
            "collection": [
                {"type": "collection", "id": self.collection_ids[name]}
                for name in top_collection
            ],
        }

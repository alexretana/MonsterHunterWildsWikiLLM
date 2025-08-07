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
import asyncio

KNOWLEDGE_LIST = ['Weapons', 'Armor', 'Items', 'Decorations', 'Misc']
_API_KEY = None

def read_dot_api_key():
    global _API_KEY
    if _API_KEY is None:
        # Assuming your API key is in a file named api_key.txt
        try:
            with open('.open_webui_api_key', 'r') as f:
                _API_KEY = f.read().strip()
            if _API_KEY is None:
                raise ValueError("API key not found in file or environment.")
        except FileNotFoundError:
            raise FileNotFoundError("'.open_webui_api_key' file not found.")
    return _API_KEY

def make_headers(extra_headers: dict = None):
    api_key = read_dot_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers

def print_response_error(response):
    error_message = f"Recieved Non-Successful Status Code({response.status_code}), and message :{response.text}"
    print(error_message)
    return error_message

def get_knowledge_list():
    # Check if collections already exists
    full_url = OPEN_WEBUI_DOMAIN_NAME + ":8080/api/v1/knowledge/list"
    headers = make_headers()
    response = requests.get(url=full_url, headers=headers)
    # Exit early if api call fails
    if response.status_code not in range(200,299):
        return print_response_error(response)

    # Parse response for list of exiting knowledges. See which are missing
    response_json = response.json()
    return { knowledge['name']: knowledge['id'] for knowledge in response_json }

class Pipeline:
	def __init__(self):
		self.model = SentenceTransformer("all-MiniLm-L6-v2")
		self.collection_texts = {
			"Weapon": "Weapons in Monster Hunter: Wilds define your combat style, with 14 unique types ranging from nimble Dual Blades to massive Greatswords and technical options like the Charge Blade or Hunting Horn, each with their own combos, mechanics, and upgrade trees. Great Sword Sword & Shield Dual Blades Long Sword Hunting Horn Lance Gunlance Hammer Switch Axe Charge Blade Insect Glaive  Bow  Light Bowgun  Heavy Bowgun",
			"Armor": "Armor provides both defense and passive skills, crafted from monster parts and upgraded over time; full sets can grant set bonuses, and mixing pieces lets hunters optimize builds around specific resistances and abilities. Each set includes hemls chests arms waists and legs.",
			"Items": "Items include consumables like Potions, Traps, Bombs, and Ammo used for survival, utility, and strategy during hunts, with many crafted from gathered materials in the field or managed through the item box and crafting lists.",
			"Decorations": "Decorations are slottable jewels that enhance skills on your gear, offering flexible build customization once you’ve progressed far enough to unlock high-rank gear and start farming tempered monsters for rare drops.",
			"Misc": "Misc covers everything else, including NPCs like the Smithy or Handler, quest types (story, optional, investigations, events), endemic life, locations like Astera and the Guiding Lands, and systems like bounties, melding, and the ecosystem interactions that make the world feel alive."
		}
		self.collection_embeddings = None
		self.collection_ids = get_knowledge_list()

    async def on_startup(self):
        # Precompute embeddings at startup
		self.collection_embeddings= {
			name: self.model.encode(desc)
			for name, desc in self.collection_texts.items()
		}

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass
	
	def get_top_collections(self, query: str, top_k: int = 2):
		query_vec = self.model.encode(query)
		scores = {
			name: cosine_similarity(
				[query_vec], [self.collecition_embeddings[name]]
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
				for name in top_collections
			],
		}

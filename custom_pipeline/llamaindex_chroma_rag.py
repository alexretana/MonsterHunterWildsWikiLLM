"""
title: LlamaIndex Chroma RAG Pipeline
author: alexr
date: 2025-01-08
version: 1.0
license: MIT
description: A pipeline that uses LlamaIndex with persistent Chroma vector store for advanced RAG querying of Monster Hunter: Wilds wiki data.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-vector-stores-chroma
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import os

class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str
        CHROMA_PERSIST_DIR: str
        SIMILARITY_TOP_K: int

    def __init__(self):
        # Initialize the valves - CRITICAL for Open WebUI integration!
        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
                "SIMILARITY_TOP_K": int(os.getenv("SIMILARITY_TOP_K", "5")),
            }
        )
        
        # LlamaIndex components
        self.embed_model = None
        self.llm = None
        self.query_engine = None
        
        print("LlamaIndex Chroma RAG Pipeline initialized")

    async def on_startup(self):
        print("Starting LlamaIndex Chroma RAG Pipeline startup...")
        
        try:
            # Initialize LlamaIndex components with Ollama
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.llms.ollama import Ollama
            from llama_index.vector_stores.chroma import ChromaVectorStore
            from llama_index.core import VectorStoreIndex, Settings
            
            print("Initializing LlamaIndex components...")
            
            # Initialize embedding model
            self.embed_model = OllamaEmbedding(
                model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
                base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            )
            
            # Initialize LLM
            self.llm = Ollama(
                model=self.valves.LLAMAINDEX_MODEL_NAME,
                base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            )
            
            # Set global settings
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            
            # Initialize Chroma vector store from persistent directory
            print(f"Loading Chroma vector store from {self.valves.CHROMA_PERSIST_DIR}")
            chroma_store = ChromaVectorStore(persist_dir=self.valves.CHROMA_PERSIST_DIR)
            
            # Create index from existing vector store
            index = VectorStoreIndex.from_vector_store(
                vector_store=chroma_store,
                embed_model=self.embed_model
            )
            
            # Create query engine
            self.query_engine = index.as_query_engine(
                similarity_top_k=self.valves.SIMILARITY_TOP_K,
                llm=self.llm
            )
            
            print("LlamaIndex Chroma RAG Pipeline startup complete!")
            
        except Exception as e:
            print(f"Error during startup: {e}")
            print("Pipeline will not be functional without proper initialization")

    async def on_shutdown(self):
        print("LlamaIndex Chroma RAG Pipeline shutting down...")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"\n=== LlamaIndex Chroma RAG Pipeline ===")
        print(f"User message: {user_message}")
        
        if not self.query_engine:
            print("Query engine not initialized - pipeline is not functional")
            return "Sorry, the RAG system is not properly initialized. Please check the logs and ensure Chroma database exists."
        
        try:
            # Query the Chroma vector store using LlamaIndex
            print("Querying Chroma vector store...")
            response = self.query_engine.query(user_message)
            
            print("Successfully retrieved response from Chroma RAG")
            return str(response)
            
        except Exception as e:
            print(f"Error querying Chroma vector store: {e}")
            return f"Sorry, I encountered an error while searching the knowledge base: {str(e)}"

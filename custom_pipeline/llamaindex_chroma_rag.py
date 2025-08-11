"""
title: LlamaIndex Chroma RAG Pipeline
author: alexr
date: 2025-01-08
version: 1.0
license: MIT
description: A pipeline that uses LlamaIndex with persistent Chroma vector store for advanced RAG querying of Monster Hunter: Wilds wiki data.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-vector-stores-chroma
"""

from typing import List, Union, Generator, Iterator, Dict
from pydantic import BaseModel
import os

def patch_ollama_embedding():
    """Patch OllamaEmbedding class to handle version compatibility issues."""
    from llama_index.embeddings.ollama import OllamaEmbedding
    
    # Store the original method
    original_get_general_text_embedding = OllamaEmbedding.get_general_text_embedding
    
    def fixed_get_general_text_embedding(self, texts: str):
        """Get Ollama embedding with proper response handling."""
        result = self._client.embed(
            model=self.model_name, 
            input=texts, 
            options=self.ollama_additional_kwargs
        )
        
        # Handle different response formats
        if hasattr(result, 'embeddings') and result.embeddings:  # EmbedResponse object
            # embeddings is a list of lists, we want the first (and typically only) embedding
            return result.embeddings[0]
        elif hasattr(result, 'embedding'):  # EmbeddingsResponse object (older format)
            return result.embedding
        elif isinstance(result, dict) and "embedding" in result:  # Dictionary format
            return result["embedding"]
        else:
            # Fallback to original method if we can't handle the response
            try:
                return original_get_general_text_embedding(self, texts)
            except Exception as e:
                raise ValueError(f"Unexpected response format from Ollama: {type(result)}, {result}. Original error: {e}")
    
    # Monkey patch the method
    OllamaEmbedding.get_general_text_embedding = fixed_get_general_text_embedding
    print("[SUCCESS] OllamaEmbedding patched for version compatibility")

class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str
        CHROMA_PERSIST_DIR: str
        SIMILARITY_TOP_K: int
        DEBUG_MODE: bool

    def __init__(self):
        # Initialize the valves - CRITICAL for Open WebUI integration!
        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3:8b"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR", "../../chroma_db"),
                "SIMILARITY_TOP_K": int(os.getenv("SIMILARITY_TOP_K", "10")),
                "DEBUG_MODE": os.getenv("DEBUG_MODE", "false").lower() in ["true", "1", "yes"],
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
            import chromadb
            
            print("Initializing LlamaIndex components...")
            if self.valves.DEBUG_MODE:
                print(f"Current valves configuration:")
                print(f"  - LLM Model: {self.valves.LLAMAINDEX_MODEL_NAME}")
                print(f"  - Embedding Model: {self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME}")
                print(f"  - Ollama URL: {self.valves.LLAMAINDEX_OLLAMA_BASE_URL}")
                print(f"  - Chroma Dir: {self.valves.CHROMA_PERSIST_DIR}")
                print(f"  - Top K: {self.valves.SIMILARITY_TOP_K}")
                print(f"  - Debug Mode: {self.valves.DEBUG_MODE}")
            
            # Apply the embedding fix
            patch_ollama_embedding()
            
            # Initialize embedding model (now using patched version)
            self.embed_model = OllamaEmbedding(
                model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
                base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            )
            
            # Initialize LLM
            if self.valves.DEBUG_MODE:
                print(f"Initializing LLM with model: {self.valves.LLAMAINDEX_MODEL_NAME}")
            self.llm = Ollama(
                model=self.valves.LLAMAINDEX_MODEL_NAME,
                base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            )
            if self.valves.DEBUG_MODE:
                print("[SUCCESS] LLM initialized successfully")
            
            # Set global settings
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            
            # Initialize Chroma vector store from persistent directory
            if self.valves.DEBUG_MODE:
                print(f"Loading Chroma vector store from {self.valves.CHROMA_PERSIST_DIR}")
            
            # Create persistent Chroma client (not HTTP client)
            chroma_client = chromadb.PersistentClient(path=self.valves.CHROMA_PERSIST_DIR)
            chroma_collection = chroma_client.get_or_create_collection("monsterhunter_fextralife_wiki")
            chroma_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Create index from existing vector store
            index = VectorStoreIndex.from_vector_store(
                vector_store=chroma_store,
                embed_model=self.embed_model
            )
            
            # Create query engine
            if self.valves.DEBUG_MODE:
                print(f"Creating query engine with similarity_top_k={self.valves.SIMILARITY_TOP_K}")
            self.query_engine = index.as_query_engine(
                similarity_top_k=self.valves.SIMILARITY_TOP_K,
                llm=self.llm,
                response_mode="compact"
            )
            
            print("LlamaIndex Chroma RAG Pipeline startup complete!")
            
        except Exception as e:
            import traceback
            print(f"Error during startup: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            print("Pipeline will not be functional without proper initialization")

    async def on_shutdown(self):
        print("LlamaIndex Chroma RAG Pipeline shutting down...")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[Dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if self.valves.DEBUG_MODE:
            print(f"\n=== LlamaIndex Chroma RAG Pipeline ===")
            print(f"User message: {user_message}")
        
        if not self.query_engine:
            print("Query engine not initialized - pipeline is not functional")
            return "Sorry, the RAG system is not properly initialized. Please check the logs and ensure Chroma database exists."
        
        try:
            # Query the Chroma vector store using LlamaIndex
            if self.valves.DEBUG_MODE:
                print("Querying Chroma vector store...")
                print(f"Using LLM model: {self.llm.model if hasattr(self.llm, 'model') else 'unknown'}")
            
            response = self.query_engine.query(user_message)
            
            if self.valves.DEBUG_MODE:
                print(f"Retrieved response type: {type(response)}")
                if hasattr(response, 'source_nodes'):
                    print(f"Number of source nodes retrieved: {len(response.source_nodes)}")
                    # Log source documents for evaluation purposes
                    for i, node in enumerate(response.source_nodes[:3]):  # Show first 3
                        print(f"  Source {i+1}: {node.node.text[:100]}...")
                        print(f"  Score: {node.score if hasattr(node, 'score') else 'N/A'}")
                else:
                    print("No source nodes found in response")
            
            response_str = str(response)
            
            # Store response metadata for potential evaluation use
            self._last_response = response
            self._last_query = user_message
            
            # Check for empty responses
            if not response_str.strip() or response_str.strip().lower() in ['empty response', 'none', ''] or 'Empty Response' in response_str:
                if self.valves.DEBUG_MODE:
                    print("[WARNING] Response appears to be empty, trying direct LLM call...")
                    try:
                        direct_response = self.llm.complete(f"Answer this question about Monster Hunter: {user_message}")
                        print(f"Direct LLM response: '{str(direct_response)[:100]}...'")
                    except Exception as llm_error:
                        print(f"Direct LLM call failed: {llm_error}")
                
                return "I found relevant information in the knowledge base, but couldn't generate a proper response. Please try rephrasing your question."
            
            if self.valves.DEBUG_MODE:
                print(f"Successfully retrieved response ({len(response_str)} characters)")
            
            return response_str
            
        except Exception as e:
            import traceback
            print(f"Error querying Chroma vector store: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            return f"Sorry, I encountered an error while searching the knowledge base: {str(e)}"
    
    def get_last_response_for_evaluation(self):
        """Get the last LlamaIndex response object for evaluation purposes"""
        return getattr(self, '_last_response', None)
    
    def get_last_query_for_evaluation(self):
        """Get the last query for evaluation purposes"""
        return getattr(self, '_last_query', None)

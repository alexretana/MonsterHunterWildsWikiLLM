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

    def __init__(self):
        # Initialize the valves - CRITICAL for Open WebUI integration!
        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3:8b"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR", "../../chroma_db"),
                "SIMILARITY_TOP_K": int(os.getenv("SIMILARITY_TOP_K", "10")),  # Increased from 5 to 10
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
            print(f"Current valves configuration:")
            print(f"  - LLM Model: {self.valves.LLAMAINDEX_MODEL_NAME}")
            print(f"  - Embedding Model: {self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME}")
            print(f"  - Ollama URL: {self.valves.LLAMAINDEX_OLLAMA_BASE_URL}")
            print(f"  - Chroma Dir: {self.valves.CHROMA_PERSIST_DIR}")
            print(f"  - Top K: {self.valves.SIMILARITY_TOP_K}")
            
            # Apply the embedding fix
            patch_ollama_embedding()
            
            # Initialize embedding model (now using patched version)
            self.embed_model = OllamaEmbedding(
                model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
                base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            )
            
            # Initialize LLM
            print(f"Initializing LLM with model: {self.valves.LLAMAINDEX_MODEL_NAME}")
            self.llm = Ollama(
                model=self.valves.LLAMAINDEX_MODEL_NAME,
                base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            )
            print("[SUCCESS] LLM initialized successfully")
            
            # Set global settings
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            
            # Initialize Chroma vector store from persistent directory
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
            
            # Create query engine with debug-friendly settings
            print(f"Creating query engine with similarity_top_k={self.valves.SIMILARITY_TOP_K}")
            self.query_engine = index.as_query_engine(
                similarity_top_k=self.valves.SIMILARITY_TOP_K,
                llm=self.llm,
                # Add explicit response_mode for debugging
                response_mode="compact"
            )
            
            # Test retriever directly to debug similarity thresholds
            print("Testing retriever directly...")
            test_retriever = index.as_retriever(similarity_top_k=self.valves.SIMILARITY_TOP_K)
            test_nodes = test_retriever.retrieve("How does the stun mechanic work?")
            print(f"Direct retriever test found {len(test_nodes)} nodes")
            for i, node in enumerate(test_nodes[:2]):
                print(f"  Node {i+1}: score={getattr(node, 'score', 'N/A')}, text_len={len(getattr(node, 'text', ''))}")
            
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
            print(f"Using LLM model: {self.llm.model if hasattr(self.llm, 'model') else 'unknown'}")
            
            # First test the retriever directly
            print("\n--- Testing retriever directly ---")
            retriever = self.query_engine.retriever
            direct_nodes = retriever.retrieve(user_message)
            print(f"Direct retriever found {len(direct_nodes)} nodes")
            for i, node in enumerate(direct_nodes[:2]):
                score = getattr(node, 'score', 'N/A')
                text_len = len(getattr(node, 'text', ''))
                print(f"Direct node {i+1}: score={score}, text_len={text_len}")
            
            # Now run the full query engine
            print("\n--- Running full query engine ---")
            response = self.query_engine.query(user_message)
            
            print(f"Retrieved response type: {type(response)}")
            
            # Debug: Check what's in the response object
            if hasattr(response, 'source_nodes'):
                print(f"Number of source nodes retrieved: {len(response.source_nodes)}")
                for i, node in enumerate(response.source_nodes[:3]):  # Show first 3 nodes
                    print(f"Source node {i+1}: {str(node.text)[:200]}...")
                    print(f"Source node {i+1} score: {getattr(node, 'score', 'N/A')}")
            else:
                print("No source nodes found in response")
                
                # If no source nodes, check if it's a retriever filtering issue
                if len(direct_nodes) > 0:
                    print("\n[FAILURE] ISSUE: Retriever finds nodes but query engine has 0 source nodes!")
                    print("This suggests a filtering or threshold issue in the query engine.")
                    
                    # Test with a more relaxed similarity threshold
                    print("\nTesting with relaxed similarity settings...")
                    relaxed_engine = self.query_engine.retriever._index.as_query_engine(
                        similarity_top_k=10,  # More results
                        llm=self.llm,
                        response_mode="compact"
                    )
                    relaxed_response = relaxed_engine.query(user_message)
                    if hasattr(relaxed_response, 'source_nodes'):
                        print(f"Relaxed query found {len(relaxed_response.source_nodes)} source nodes")
                        if len(relaxed_response.source_nodes) > 0:
                            print("[SUCCESS] SOLUTION: Increasing similarity_top_k helps!")
                            # Use the relaxed response
                            response = relaxed_response
                
            if hasattr(response, 'response'):
                actual_response = response.response
                print(f"Actual response attribute: '{actual_response}'")
                print(f"Actual response type: {type(actual_response)}")
            else:
                print("No response attribute found")
            
            response_str = str(response)
            print(f"Response string length: {len(response_str)} characters")
            print(f"Response preview: '{response_str[:200]}'")
            print(f"Response exact match checks:")
            print(f"  - Empty after strip: {not response_str.strip()}")
            print(f"  - Equals 'Empty Response': {response_str.strip() == 'Empty Response'}")
            print(f"  - Contains 'Empty Response': {'Empty Response' in response_str}")
            
            if not response_str.strip() or response_str.strip().lower() in ['empty response', 'none', ''] or 'Empty Response' in response_str:
                print("[WARNING] Response appears to be empty or contains 'Empty Response'!")
                print("Attempting direct LLM call to debug...")
                
                # Try a direct LLM call to see if the model is working
                try:
                    direct_response = self.llm.complete(f"Answer this question about Monster Hunter: {user_message}")
                    print(f"Direct LLM response: '{str(direct_response)[:200]}'")
                except Exception as llm_error:
                    print(f"Direct LLM call failed: {llm_error}")
                
                return "I found relevant information in the knowledge base, but couldn't generate a proper response. Please try rephrasing your question."
            
            print("Successfully retrieved response from Chroma RAG")
            return response_str
            
        except Exception as e:
            import traceback
            print(f"Error querying Chroma vector store: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            return f"Sorry, I encountered an error while searching the knowledge base: {str(e)}"

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
from llama_index.core.prompts import PromptTemplate

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
        USE_CUSTOM_PROMPTS: bool
        USE_CONVERSATION_CONTEXT: bool
        RESPONSE_MODE: str

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
                "USE_CUSTOM_PROMPTS": os.getenv("USE_CUSTOM_PROMPTS", "true").lower() in ["true", "1", "yes"],
                "USE_CONVERSATION_CONTEXT": os.getenv("USE_CONVERSATION_CONTEXT", "false").lower() in ["true", "1", "yes"],
                "RESPONSE_MODE": os.getenv("RESPONSE_MODE", "compact"),
            }
        )
        
        # LlamaIndex components
        self.embed_model = None
        self.llm = None
        self.query_engine = None
        
        # Custom prompts
        self.custom_prompts = self._initialize_custom_prompts()
        
        print("LlamaIndex Chroma RAG Pipeline initialized")

    def _initialize_custom_prompts(self) -> Dict[str, PromptTemplate]:
        """Initialize custom prompt templates for Monster Hunter knowledge base."""
        
        # Enhanced QA prompt with domain-specific instructions
        mh_qa_template = PromptTemplate(
            """
            You are an expert Monster Hunter guide and wiki assistant. You have access to comprehensive Monster Hunter knowledge including monsters, weapons, armor, quests, mechanics, and strategies.

            Context Information:
            ---------------------
            {context_str}
            ---------------------

            Instructions:
            - Use ONLY the provided context to answer the question
            - If information is not in the context, clearly state "I don't have information about [specific topic] in my current knowledge base"
            - For Monster Hunter terms, provide clear definitions and explanations
            - When discussing strategies, weapons, or mechanics, be specific and actionable
            - Include relevant stats, locations, or requirements when available in the context
            - Use Monster Hunter terminology correctly (e.g., "Great Sword" not "Greatsword")
            - If asked about multiple items, organize the response with clear sections
            - For weapon recommendations, consider context like monster weaknesses and playstyle

            Question: {query_str}

            Answer (based solely on the provided context):
            """
        )
        
        # Enhanced refine prompt for multi-chunk responses
        mh_refine_template = PromptTemplate(
            """
            You are an expert Monster Hunter guide refining an answer with additional context.

            Original Question: {query_str}
            
            Existing Answer:
            {existing_answer}
            
            Additional Context:
            ------------
            {context_msg}
            ------------
            
            Instructions for refinement:
            - Integrate new information smoothly with the existing answer
            - Remove any contradictory information, prioritizing more specific/recent context
            - Maintain Monster Hunter terminology and accuracy
            - Keep the response organized and easy to follow
            - If new context doesn't add value, return the original answer unchanged
            - Ensure all information is supported by the provided contexts
            
            Refined Answer:
            """
        )
        
        # Summary template for multiple sources
        mh_summary_template = PromptTemplate(
            """
            You are an expert Monster Hunter guide synthesizing information from multiple sources.

            Context from Multiple Sources:
            ---------------------
            {context_str}
            ---------------------

            Instructions:
            - Synthesize information from all sources into a comprehensive answer
            - Resolve any contradictions by noting different perspectives or conditions
            - Organize information logically (e.g., basic info first, then strategies, then advanced tips)
            - Use Monster Hunter terminology accurately
            - Include specific details like stats, locations, requirements when available
            - If sources cover different aspects, address each relevant aspect
            - Cite when information comes from specific contexts (e.g., "According to one source...")

            Question: {query_str}

            Comprehensive Answer:
            """
        )
        
        return {
            "text_qa_template": mh_qa_template,
            "refine_template": mh_refine_template,
            "summary_template": mh_summary_template
        }

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
                response_mode=self.valves.RESPONSE_MODE
            )
            
            # Apply custom prompts if enabled
            if self.valves.USE_CUSTOM_PROMPTS:
                if self.valves.DEBUG_MODE:
                    print("Applying custom Monster Hunter prompts...")
                try:
                    self.query_engine.update_prompts({
                        "response_synthesizer:text_qa_template": self.custom_prompts["text_qa_template"],
                        "response_synthesizer:refine_template": self.custom_prompts["refine_template"],
                        "response_synthesizer:summary_template": self.custom_prompts["summary_template"]
                    })
                    if self.valves.DEBUG_MODE:
                        print("[SUCCESS] Custom prompts applied to query engine")
                        # Display applied prompts for verification
                        applied_prompts = self.query_engine.get_prompts()
                        print(f"Applied {len(applied_prompts)} custom prompt templates")
                except Exception as prompt_error:
                    print(f"[WARNING] Failed to apply custom prompts: {prompt_error}")
                    print("Continuing with default prompts...")
            else:
                if self.valves.DEBUG_MODE:
                    print("Using default LlamaIndex prompts (custom prompts disabled)")
            
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
            # Enhance query with conversation context if enabled
            query_to_use = user_message
            if self.valves.USE_CONVERSATION_CONTEXT:
                query_to_use = self._enhance_query_with_context(user_message, messages)
            
            # Query the Chroma vector store using LlamaIndex
            if self.valves.DEBUG_MODE:
                print("Querying Chroma vector store...")
                print(f"Using LLM model: {self.llm.model if hasattr(self.llm, 'model') else 'unknown'}")
                print(f"Response mode: {self.valves.RESPONSE_MODE}")
                if query_to_use != user_message:
                    print("Using enhanced query with conversation context")
            
            response = self.query_engine.query(query_to_use)
            
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
    
    def get_current_prompts(self) -> Dict[str, str]:
        """Get the currently active prompts from the query engine."""
        if not self.query_engine:
            return {"error": "Query engine not initialized"}
        
        try:
            prompts_dict = self.query_engine.get_prompts()
            return {key: str(template) for key, template in prompts_dict.items()}
        except Exception as e:
            return {"error": f"Failed to retrieve prompts: {e}"}
    
    def update_prompt(self, prompt_key: str, new_template: str) -> bool:
        """Update a specific prompt template."""
        if not self.query_engine:
            print("Query engine not initialized")
            return False
        
        try:
            template = PromptTemplate(new_template)
            self.query_engine.update_prompts({prompt_key: template})
            if self.valves.DEBUG_MODE:
                print(f"[SUCCESS] Updated prompt: {prompt_key}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to update prompt {prompt_key}: {e}")
            return False
    
    def _enhance_query_with_context(self, user_message: str, messages: List[Dict]) -> str:
        """Enhance the user query with conversation context if available."""
        if not messages or len(messages) <= 1:
            return user_message
        
        # Get recent conversation context (last 3 exchanges)
        recent_context = []
        for msg in messages[-6:]:  # Last 6 messages (3 exchanges)
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if role in ['user', 'assistant'] and content:
                recent_context.append(f"{role.title()}: {content[:200]}...")  # Truncate long messages
        
        if recent_context:
            context_str = "\n".join(recent_context)
            enhanced_query = f"""
            Previous conversation context:
            {context_str}
            
            Current question: {user_message}
            
            (Please answer the current question while being aware of the previous context)
            """.strip()
            
            if self.valves.DEBUG_MODE:
                print(f"Enhanced query with conversation context ({len(recent_context)} previous messages)")
            
            return enhanced_query
        
        return user_message

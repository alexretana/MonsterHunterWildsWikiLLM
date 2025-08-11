#!/usr/bin/env python3
"""
Dataset Generation Utility
Automatically generates evaluation questions from your Chroma vector database
Using LlamaIndex's question generation capabilities
"""

import asyncio
import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from datetime import datetime

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# LlamaIndex imports
from llama_index.core.evaluation import DatasetGenerator
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, Settings

# Import pipeline for Ollama patching
sys.path.append('custom_pipeline')

class EvaluationDatasetGenerator:
    """Generate evaluation datasets from existing Chroma vector database"""
    
    def __init__(self,
                 ollama_base_url: str = "http://localhost:11434",
                 llm_model: str = "llama3:8b",
                 embedding_model: str = "nomic-embed-text",
                 chroma_persist_dir: str = "chroma_db"):
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.chroma_persist_dir = chroma_persist_dir
        
        self.llm = None
        self.embed_model = None
        self.index = None
        self.dataset_generator = None
        
    async def setup(self):
        """Setup LlamaIndex components"""
        print("🔧 Setting up LlamaIndex components...")
        
        # Apply the Ollama embedding patch
        from custom_pipeline.llamaindex_chroma_rag import patch_ollama_embedding
        patch_ollama_embedding()
        
        # Initialize models
        self.llm = Ollama(
            model=self.llm_model,
            base_url=self.ollama_base_url
        )
        
        self.embed_model = OllamaEmbedding(
            model_name=self.embedding_model,
            base_url=self.ollama_base_url
        )
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # Connect to existing Chroma database
        print(f"📂 Connecting to Chroma database: {self.chroma_persist_dir}")
        chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)
        chroma_collection = chroma_client.get_or_create_collection("monsterhunter_fextralife_wiki")
        chroma_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create index from existing vector store
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=chroma_store,
            embed_model=self.embed_model
        )
        
        print(f"✅ Setup complete. Index has {len(chroma_collection.get()['ids'])} documents")
        
    def get_sample_nodes(self, num_nodes: int = 50) -> List:
        """Get a sample of nodes from the vector store for question generation"""
        print(f"📄 Sampling {num_nodes} nodes for question generation...")
        
        try:
            # Get retriever
            retriever = self.index.as_retriever(similarity_top_k=num_nodes)
            
            # Use a broad query to get diverse content
            sample_nodes = retriever.retrieve("Monster Hunter Wilds gameplay weapons monsters crafting")
            
            print(f"✅ Retrieved {len(sample_nodes)} nodes")
            return sample_nodes
        
        except Exception as e:
            print(f"❌ Error retrieving nodes: {e}")
            return []
    
    async def generate_questions(self, 
                               num_questions: int = 20,
                               num_nodes_sample: int = 50) -> List[Dict[str, Any]]:
        """Generate evaluation questions from the vector database"""
        print(f"🤔 Generating {num_questions} evaluation questions...")
        
        # Get sample nodes
        nodes = self.get_sample_nodes(num_nodes_sample)
        if not nodes:
            print("❌ No nodes retrieved, cannot generate questions")
            return []
        
        # Initialize dataset generator
        self.dataset_generator = DatasetGenerator.from_documents(
            [node.node for node in nodes],  # Extract the actual document nodes
            llm=self.llm,
            num_questions_per_chunk=max(1, num_questions // len(nodes))
        )
        
        try:
            # Generate questions
            eval_questions = await asyncio.to_thread(
                self.dataset_generator.generate_questions_from_nodes,
                num=num_questions
            )
            
            print(f"✅ Generated {len(eval_questions)} questions")
            return eval_questions
        
        except Exception as e:
            print(f"❌ Error generating questions: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def categorize_question(self, question: str) -> tuple:
        """Categorize question based on content keywords"""
        question_lower = question.lower()
        
        # Define category keywords
        categories = {
            "weapons": ["weapon", "sword", "bow", "hammer", "lance", "blade", "gun", "horn", "axe"],
            "monsters": ["monster", "creature", "beast", "dragon", "wyvern", "fanged", "bird"],
            "gameplay_mechanics": ["combat", "mechanics", "system", "skill", "ability", "mode", "focus"],
            "crafting": ["craft", "forge", "material", "recipe", "upgrade", "enhancement"],
            "world": ["area", "biome", "location", "environment", "region", "map"],
            "general": ["release", "platform", "requirement", "multiplayer", "solo"]
        }
        
        # Determine difficulty based on question complexity
        difficulty = "medium"  # default
        if any(word in question_lower for word in ["how", "what", "when", "where"]):
            if any(word in question_lower for word in ["complex", "advanced", "detailed", "mechanics"]):
                difficulty = "hard"
            elif any(word in question_lower for word in ["basic", "simple", "main", "general"]):
                difficulty = "easy"
        
        # Find best matching category
        best_category = "general"
        max_matches = 0
        
        for category, keywords in categories.items():
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            if matches > max_matches:
                max_matches = matches
                best_category = category
        
        return best_category, difficulty
    
    def format_for_evaluation(self, generated_questions: List) -> List[Dict[str, Any]]:
        """Format generated questions for evaluation dataset"""
        print("📋 Formatting questions for evaluation dataset...")
        
        formatted_questions = []
        
        for i, eq in enumerate(generated_questions):
            # Extract question text (LlamaIndex structure may vary)
            question_text = str(eq.query) if hasattr(eq, 'query') else str(eq)
            
            # Categorize the question
            category, difficulty = self.categorize_question(question_text)
            
            formatted_question = {
                "query": question_text,
                "expected_answer": None,  # We don't have ground truth answers
                "expected_context_ids": None,  # We could potentially add this
                "category": category,
                "difficulty": difficulty
            }
            
            formatted_questions.append(formatted_question)
        
        print(f"✅ Formatted {len(formatted_questions)} questions")
        return formatted_questions
    
    def save_dataset(self, questions: List[Dict[str, Any]], output_file: str):
        """Save the generated dataset to a JSON file"""
        print(f"💾 Saving dataset to {output_file}...")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Add metadata
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "EvaluationDatasetGenerator",
                "llm_model": self.llm_model,
                "embedding_model": self.embedding_model,
                "total_questions": len(questions),
                "chroma_source": self.chroma_persist_dir
            },
            "questions": questions
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dataset saved successfully!")
        
        # Print summary
        categories = {}
        difficulties = {}
        
        for q in questions:
            cat = q.get('category', 'unknown')
            diff = q.get('difficulty', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total Questions: {len(questions)}")
        print(f"   Categories: {dict(categories)}")
        print(f"   Difficulties: {dict(difficulties)}")
    
    async def generate_dataset(self, 
                             output_file: str,
                             num_questions: int = 20,
                             num_nodes_sample: int = 50):
        """Main workflow to generate evaluation dataset"""
        print("🚀 Starting evaluation dataset generation...")
        
        # Setup
        await self.setup()
        
        # Generate questions
        raw_questions = await self.generate_questions(num_questions, num_nodes_sample)
        if not raw_questions:
            print("❌ Failed to generate questions")
            return False
        
        # Format questions
        formatted_questions = self.format_for_evaluation(raw_questions)
        
        # Save dataset
        self.save_dataset(formatted_questions, output_file)
        
        print("✅ Dataset generation complete!")
        return True

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation dataset from Chroma database")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/datasets/generated_queries.json",
        help="Output file for generated dataset"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Number of questions to generate"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=50,
        help="Number of nodes to sample for generation"
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default="chroma_db",
        help="Chroma database directory"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3:8b",
        help="LLM model for question generation"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-embed-text",
        help="Embedding model"
    )
    
    args = parser.parse_args()
    
    # Check if Chroma database exists
    if not Path(args.chroma_dir).exists():
        print(f"❌ Error: Chroma database directory '{args.chroma_dir}' not found!")
        print("Please run your scraper first to populate the database.")
        return 1
    
    try:
        generator = EvaluationDatasetGenerator(
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            chroma_persist_dir=args.chroma_dir
        )
        
        success = await generator.generate_dataset(
            output_file=args.output,
            num_questions=args.num_questions,
            num_nodes_sample=args.num_nodes
        )
        
        return 0 if success else 1
    
    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

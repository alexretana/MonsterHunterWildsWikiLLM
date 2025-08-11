"""
RAG System Evaluation Framework
Based on LlamaIndex evaluation best practices

This module provides comprehensive evaluation for the Monster Hunter Wilds RAG system,
implementing both end-to-end and component-wise evaluation strategies.
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import nest_asyncio

# Enable nested asyncio to fix compatibility issues with LlamaIndex evaluators
nest_asyncio.apply()

# LlamaIndex imports for evaluation
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    RetrieverEvaluator,
    BatchEvalRunner
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Import our custom pipeline
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'custom_pipeline'))
from llamaindex_chroma_rag import Pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation/logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reduce noise from HTTP requests
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

@dataclass
class EvaluationQuery:
    """Represents a query for evaluation with expected answer and context"""
    query: str
    expected_answer: Optional[str] = None
    expected_context_ids: Optional[List[str]] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None  # easy, medium, hard

@dataclass
class EvaluationResult:
    """Stores evaluation results for a single query"""
    query: str
    response: str
    category: Optional[str]
    difficulty: Optional[str]
    
    # End-to-end metrics
    faithfulness_score: Optional[float] = None
    faithfulness_passing: Optional[bool] = None
    relevancy_score: Optional[float] = None
    relevancy_passing: Optional[bool] = None
    correctness_score: Optional[float] = None
    correctness_passing: Optional[bool] = None
    semantic_similarity_score: Optional[float] = None
    
    # Retrieval metrics (if component-wise evaluation)
    hit_rate: Optional[float] = None
    mrr: Optional[float] = None
    
    # Response metadata
    response_time: Optional[float] = None
    source_nodes_count: Optional[int] = None
    
    # Timestamp
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class RAGEvaluator:
    """
    Comprehensive RAG evaluation system implementing LlamaIndex best practices
    """
    
    def __init__(self, 
                 ollama_base_url: str = "http://localhost:11434",
                 llm_model: str = "llama3:8b",
                 embedding_model: str = "nomic-embed-text",
                 evaluation_llm_model: str = "llama3:8b"):
        """
        Initialize the RAG evaluator
        
        Args:
            ollama_base_url: Base URL for Ollama API
            llm_model: Model name for the RAG pipeline
            embedding_model: Embedding model name
            evaluation_llm_model: Model to use for evaluation (can be different/better)
        """
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.evaluation_llm_model = evaluation_llm_model
        
        # Initialize RAG pipeline
        self.rag_pipeline = None
        
        # Initialize evaluation LLM (can be different from pipeline LLM)
        self.evaluation_llm = Ollama(
            model=evaluation_llm_model,
            base_url=ollama_base_url
        )
        
        # Initialize evaluators
        self.faithfulness_evaluator = None
        self.relevancy_evaluator = None
        self.correctness_evaluator = None
        self.semantic_similarity_evaluator = None
        self.retriever_evaluator = None
        
        # Results storage
        self.evaluation_results: List[EvaluationResult] = []
        
        # Create directories
        os.makedirs("evaluation/logs", exist_ok=True)
        os.makedirs("evaluation/results", exist_ok=True)
        os.makedirs("evaluation/datasets", exist_ok=True)
        
        logger.info("RAGEvaluator initialized")
    
    async def setup(self):
        """Setup the RAG pipeline and evaluators"""
        logger.info("Setting up RAG pipeline and evaluators...")
        
        # Initialize RAG pipeline
        self.rag_pipeline = Pipeline()
        # Set the valves to match our configuration
        self.rag_pipeline.valves.LLAMAINDEX_OLLAMA_BASE_URL = self.ollama_base_url
        self.rag_pipeline.valves.LLAMAINDEX_MODEL_NAME = self.llm_model
        self.rag_pipeline.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME = self.embedding_model
        # Fix the Chroma database path to be absolute
        chroma_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chroma_db'))
        self.rag_pipeline.valves.CHROMA_PERSIST_DIR = chroma_db_path
        self.rag_pipeline.valves.DEBUG_MODE = True  # Enable debug mode to see what's happening
        
        # Run pipeline startup
        await self.rag_pipeline.on_startup()
        
        # Initialize evaluators
        self.faithfulness_evaluator = FaithfulnessEvaluator(llm=self.evaluation_llm)
        self.relevancy_evaluator = RelevancyEvaluator(llm=self.evaluation_llm)
        self.correctness_evaluator = CorrectnessEvaluator(llm=self.evaluation_llm)
        self.semantic_similarity_evaluator = SemanticSimilarityEvaluator()
        
        # Setup retriever evaluator if we can access the retriever
        if hasattr(self.rag_pipeline, 'query_engine') and self.rag_pipeline.query_engine:
            try:
                retriever = self.rag_pipeline.query_engine.retriever
                self.retriever_evaluator = RetrieverEvaluator.from_metric_names(
                    ["mrr", "hit_rate"], retriever=retriever
                )
                logger.info("Retriever evaluator set up successfully")
            except Exception as e:
                logger.warning(f"Could not set up retriever evaluator: {e}")
        
        logger.info("Setup complete")
    
    def load_evaluation_dataset(self, dataset_path: str) -> List[EvaluationQuery]:
        """Load evaluation queries from a JSON file"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different dataset formats
            # Format 1: Dict with metadata and questions keys (new generated format)
            # Format 2: Direct list of queries (legacy manual format)
            if isinstance(data, dict) and 'questions' in data:
                questions_data = data['questions']
                logger.debug(f"Loading dataset in new format with metadata from {dataset_path}")
            else:
                questions_data = data
                logger.debug(f"Loading dataset in legacy format from {dataset_path}")
            
            queries = []
            for item in questions_data:
                query = EvaluationQuery(
                    query=item['query'],
                    expected_answer=item.get('expected_answer'),
                    expected_context_ids=item.get('expected_context_ids'),
                    category=item.get('category'),
                    difficulty=item.get('difficulty', 'medium')
                )
                queries.append(query)
            
            logger.info(f"Loaded {len(queries)} queries from {dataset_path}")
            return queries
        
        except Exception as e:
            logger.error(f"Error loading dataset from {dataset_path}: {e}")
            return []
    
    async def evaluate_query(self, eval_query: EvaluationQuery) -> EvaluationResult:
        """Evaluate a single query end-to-end"""
        logger.info(f"Evaluating query: {eval_query.query[:100]}...")
        
        start_time = datetime.now()
        
        # Get response from RAG pipeline
        try:
            response = self.rag_pipeline.pipe(
                user_message=eval_query.query,
                model_id=self.llm_model,
                messages=[],
                body={}
            )
            response_str = str(response)
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            response_str = f"ERROR: {str(e)}"
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        # Initialize result
        result = EvaluationResult(
            query=eval_query.query,
            response=response_str,
            category=eval_query.category,
            difficulty=eval_query.difficulty,
            response_time=response_time
        )
        
        # Skip evaluation if response is an error
        if response_str.startswith("ERROR:"):
            logger.warning("Skipping evaluation due to pipeline error")
            return result
        
        # Get the actual response object for evaluation (need to query again)
        try:
            if self.rag_pipeline.query_engine:
                llama_response = self.rag_pipeline.query_engine.query(eval_query.query)
                result.source_nodes_count = len(llama_response.source_nodes) if hasattr(llama_response, 'source_nodes') else 0
                
                # Run evaluations
                await self._run_response_evaluations(llama_response, eval_query, result)
                
                # Run retrieval evaluation if we have expected context
                if eval_query.expected_context_ids and self.retriever_evaluator:
                    await self._run_retrieval_evaluation(eval_query, result)
                    
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
        
        logger.info(f"Evaluation complete. Faithfulness: {result.faithfulness_passing}, "
                   f"Relevancy: {result.relevancy_passing}, "
                   f"Response time: {result.response_time:.2f}s")
        
        return result
    
    async def _run_response_evaluations(self, response, eval_query: EvaluationQuery, result: EvaluationResult):
        """Run response-level evaluations with improved async handling"""
        
        # Faithfulness evaluation
        try:
            logger.debug(f"Running faithfulness evaluation for query: {eval_query.query[:50]}...")
            # Try using the native async method if available, otherwise fall back to thread
            if hasattr(self.faithfulness_evaluator, 'aevaluate_response'):
                faithfulness_result = await self.faithfulness_evaluator.aevaluate_response(response=response)
            else:
                faithfulness_result = await asyncio.to_thread(
                    self.faithfulness_evaluator.evaluate_response, response=response
                )
            result.faithfulness_score = getattr(faithfulness_result, 'score', None)
            result.faithfulness_passing = getattr(faithfulness_result, 'passing', None)
            logger.debug(f"Faithfulness evaluation complete: score={result.faithfulness_score}, passing={result.faithfulness_passing}")
        except Exception as e:
            logger.warning(f"Faithfulness evaluation error: {e}")
            result.faithfulness_score = None
            result.faithfulness_passing = None
        
        # Relevancy evaluation
        try:
            logger.debug(f"Running relevancy evaluation for query: {eval_query.query[:50]}...")
            if hasattr(self.relevancy_evaluator, 'aevaluate_response'):
                relevancy_result = await self.relevancy_evaluator.aevaluate_response(
                    query=eval_query.query, response=response
                )
            else:
                relevancy_result = await asyncio.to_thread(
                    self.relevancy_evaluator.evaluate_response, 
                    query=eval_query.query, 
                    response=response
                )
            result.relevancy_score = getattr(relevancy_result, 'score', None)
            result.relevancy_passing = getattr(relevancy_result, 'passing', None)
            logger.debug(f"Relevancy evaluation complete: score={result.relevancy_score}, passing={result.relevancy_passing}")
        except Exception as e:
            logger.warning(f"Relevancy evaluation error: {e}")
            result.relevancy_score = None
            result.relevancy_passing = None
        
        # Correctness evaluation (if we have expected answer)
        if eval_query.expected_answer:
            try:
                logger.debug(f"Running correctness evaluation for query: {eval_query.query[:50]}...")
                if hasattr(self.correctness_evaluator, 'aevaluate_response'):
                    correctness_result = await self.correctness_evaluator.aevaluate_response(
                        query=eval_query.query,
                        response=response,
                        reference=eval_query.expected_answer
                    )
                else:
                    correctness_result = await asyncio.to_thread(
                        self.correctness_evaluator.evaluate_response,
                        query=eval_query.query,
                        response=response,
                        reference=eval_query.expected_answer
                    )
                result.correctness_score = getattr(correctness_result, 'score', None)
                result.correctness_passing = getattr(correctness_result, 'passing', None)
                logger.debug(f"Correctness evaluation complete: score={result.correctness_score}, passing={result.correctness_passing}")
            except Exception as e:
                logger.warning(f"Correctness evaluation error: {e}")
                result.correctness_score = None
                result.correctness_passing = None
        
        # Semantic similarity evaluation
        if eval_query.expected_answer:
            try:
                logger.debug(f"Running semantic similarity evaluation for query: {eval_query.query[:50]}...")
                if hasattr(self.semantic_similarity_evaluator, 'aevaluate_response'):
                    similarity_result = await self.semantic_similarity_evaluator.aevaluate_response(
                        query=eval_query.query,
                        response=response,
                        reference=eval_query.expected_answer
                    )
                else:
                    similarity_result = await asyncio.to_thread(
                        self.semantic_similarity_evaluator.evaluate_response,
                        query=eval_query.query,
                        response=response,
                        reference=eval_query.expected_answer
                    )
                result.semantic_similarity_score = getattr(similarity_result, 'score', None)
                logger.debug(f"Semantic similarity evaluation complete: score={result.semantic_similarity_score}")
            except Exception as e:
                logger.warning(f"Semantic similarity evaluation error: {e}")
                result.semantic_similarity_score = None
    
    async def _run_retrieval_evaluation(self, eval_query: EvaluationQuery, result: EvaluationResult):
        """Run retrieval-level evaluations"""
        try:
            retrieval_result = await asyncio.to_thread(
                self.retriever_evaluator.evaluate,
                query=eval_query.query,
                expected_ids=eval_query.expected_context_ids
            )
            
            # Extract metrics from retrieval result
            if hasattr(retrieval_result, 'dict'):
                metrics = retrieval_result.dict()
                result.hit_rate = metrics.get('hit_rate')
                result.mrr = metrics.get('mrr')
            
        except Exception as e:
            logger.error(f"Retrieval evaluation error: {e}")
    
    async def evaluate_dataset(self, queries: List[EvaluationQuery]) -> List[EvaluationResult]:
        """Evaluate a full dataset of queries"""
        logger.info(f"Starting evaluation of {len(queries)} queries...")
        
        self.evaluation_results = []
        
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            result = await self.evaluate_query(query)
            self.evaluation_results.append(result)
            
            # Save intermediate results
            if (i + 1) % 10 == 0:
                await self.save_results(f"evaluation/results/intermediate_results_{i+1}.json")
        
        logger.info("Dataset evaluation complete")
        return self.evaluation_results
    
    async def save_results(self, filepath: str):
        """Save evaluation results to JSON file"""
        try:
            results_data = [asdict(result) for result in self.evaluation_results]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of evaluation results"""
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}
        
        total_queries = len(self.evaluation_results)
        
        # Overall metrics
        faithfulness_scores = [r.faithfulness_score for r in self.evaluation_results if r.faithfulness_score is not None]
        relevancy_scores = [r.relevancy_score for r in self.evaluation_results if r.relevancy_score is not None]
        correctness_scores = [r.correctness_score for r in self.evaluation_results if r.correctness_score is not None]
        similarity_scores = [r.semantic_similarity_score for r in self.evaluation_results if r.semantic_similarity_score is not None]
        
        # Passing rates
        faithfulness_passing = sum(1 for r in self.evaluation_results if r.faithfulness_passing) / total_queries * 100
        relevancy_passing = sum(1 for r in self.evaluation_results if r.relevancy_passing) / total_queries * 100
        correctness_passing = sum(1 for r in self.evaluation_results if r.correctness_passing) / total_queries * 100
        
        # Performance metrics
        response_times = [r.response_time for r in self.evaluation_results if r.response_time is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Category breakdown
        categories = {}
        for result in self.evaluation_results:
            cat = result.category or "uncategorized"
            if cat not in categories:
                categories[cat] = {"count": 0, "faithfulness_passing": 0, "relevancy_passing": 0}
            categories[cat]["count"] += 1
            if result.faithfulness_passing:
                categories[cat]["faithfulness_passing"] += 1
            if result.relevancy_passing:
                categories[cat]["relevancy_passing"] += 1
        
        # Convert to percentages
        for cat_data in categories.values():
            count = cat_data["count"]
            cat_data["faithfulness_passing_rate"] = (cat_data["faithfulness_passing"] / count) * 100
            cat_data["relevancy_passing_rate"] = (cat_data["relevancy_passing"] / count) * 100
        
        summary = {
            "evaluation_summary": {
                "total_queries": total_queries,
                "evaluation_date": datetime.now().isoformat(),
                "overall_metrics": {
                    "faithfulness_passing_rate": round(faithfulness_passing, 2),
                    "relevancy_passing_rate": round(relevancy_passing, 2),
                    "correctness_passing_rate": round(correctness_passing, 2),
                    "average_response_time": round(avg_response_time, 3)
                },
                "score_averages": {
                    "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 3) if faithfulness_scores else None,
                    "relevancy": round(sum(relevancy_scores) / len(relevancy_scores), 3) if relevancy_scores else None,
                    "correctness": round(sum(correctness_scores) / len(correctness_scores), 3) if correctness_scores else None,
                    "semantic_similarity": round(sum(similarity_scores) / len(similarity_scores), 3) if similarity_scores else None
                },
                "category_breakdown": categories
            }
        }
        
        return summary

    async def run_evaluation(self, dataset_path: str, output_dir: str = "evaluation/results") -> str:
        """Main evaluation workflow"""
        logger.info("Starting RAG evaluation workflow...")
        
        # Setup
        await self.setup()
        
        # Load dataset
        queries = self.load_evaluation_dataset(dataset_path)
        if not queries:
            raise ValueError(f"No queries loaded from {dataset_path}")
        
        # Run evaluation
        await self.evaluate_dataset(queries)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{output_dir}/evaluation_results_{timestamp}.json"
        summary_file = f"{output_dir}/evaluation_summary_{timestamp}.json"
        
        await self.save_results(results_file)
        
        # Generate and save summary
        summary = self.generate_summary_report()
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation complete! Results saved to {results_file}")
        logger.info(f"Summary saved to {summary_file}")
        
        return results_file

# Example usage
if __name__ == "__main__":
    async def main():
        evaluator = RAGEvaluator()
        
        # Run evaluation with sample dataset
        try:
            results_file = await evaluator.run_evaluation("evaluation/datasets/sample_queries.json")
            print(f"Evaluation completed. Results: {results_file}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
    
    asyncio.run(main())

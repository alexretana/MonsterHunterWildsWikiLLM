#!/usr/bin/env python3
"""
Simple script to run RAG evaluation
Usage: python run_evaluation.py [dataset_file]
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from rag_evaluator import RAGEvaluator

async def main():
    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="evaluation/datasets/sample_queries.json",
        help="Path to evaluation dataset JSON file"
    )
    parser.add_argument(
        "--ollama-url", 
        type=str, 
        default="http://localhost:11434",
        help="Ollama base URL"
    )
    parser.add_argument(
        "--llm-model", 
        type=str, 
        default="llama3:8b",
        help="LLM model for RAG pipeline"
    )
    parser.add_argument(
        "--eval-model", 
        type=str, 
        default="llama3:8b",
        help="LLM model for evaluation (can be different/better)"
    )
    parser.add_argument(
        "--embedding-model", 
        type=str, 
        default="nomic-embed-text",
        help="Embedding model"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="evaluation/results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Check if dataset file exists
    if not Path(args.dataset).exists():
        print(f"Error: Dataset file '{args.dataset}' not found!")
        print(f"Please create a dataset file or use the default: evaluation/datasets/sample_queries.json")
        return 1
    
    print("🔍 RAG Evaluation Starting...")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🤖 LLM Model: {args.llm_model}")
    print(f"🧮 Evaluation Model: {args.eval_model}")
    print(f"📝 Embedding Model: {args.embedding_model}")
    print(f"💾 Output Directory: {args.output_dir}")
    print("-" * 60)
    
    try:
        # Initialize evaluator
        evaluator = RAGEvaluator(
            ollama_base_url=args.ollama_url,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            evaluation_llm_model=args.eval_model
        )
        
        # Run evaluation
        results_file = await evaluator.run_evaluation(args.dataset, args.output_dir)
        
        # Print summary
        summary = evaluator.generate_summary_report()
        print("\n" + "="*60)
        print("📋 EVALUATION SUMMARY")
        print("="*60)
        
        if "evaluation_summary" in summary:
            s = summary["evaluation_summary"]
            print(f"📊 Total Queries: {s['total_queries']}")
            print(f"⏱️  Average Response Time: {s['overall_metrics']['average_response_time']:.3f}s")
            print(f"✅ Faithfulness Pass Rate: {s['overall_metrics']['faithfulness_passing_rate']:.1f}%")
            print(f"🎯 Relevancy Pass Rate: {s['overall_metrics']['relevancy_passing_rate']:.1f}%")
            print(f"✔️  Correctness Pass Rate: {s['overall_metrics']['correctness_passing_rate']:.1f}%")
            
            print(f"\n📈 SCORE AVERAGES:")
            scores = s["score_averages"]
            for metric, score in scores.items():
                if score is not None:
                    print(f"   {metric.title()}: {score:.3f}")
            
            print(f"\n📂 CATEGORY BREAKDOWN:")
            for category, data in s["category_breakdown"].items():
                print(f"   {category.title()}: {data['count']} queries")
                print(f"      Faithfulness: {data['faithfulness_passing_rate']:.1f}%")
                print(f"      Relevancy: {data['relevancy_passing_rate']:.1f}%")
        
        print(f"\n💾 Results saved to: {results_file}")
        print(f"📊 Summary saved to: {results_file.replace('results', 'summary')}")
        print("\n✅ Evaluation complete!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

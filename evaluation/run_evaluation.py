#!/usr/bin/env python3
"""
Automatic RAG evaluation script
Usage: python run_evaluation.py [options]

This script automatically discovers and evaluates all JSON datasets
in the evaluation/datasets/ directory.
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path
from typing import List

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from rag_evaluator import RAGEvaluator

def discover_datasets(datasets_dir: str = "evaluation/datasets") -> List[Path]:
    """Discover all JSON dataset files in the datasets directory"""
    datasets_path = Path(datasets_dir)
    if not datasets_path.exists():
        print(f"❌ Datasets directory '{datasets_dir}' not found!")
        return []
    
    json_files = list(datasets_path.glob("*.json"))
    print(f"🔍 Found {len(json_files)} dataset files:")
    for file in json_files:
        print(f"   📄 {file.name}")
    return json_files

async def evaluate_dataset(evaluator: RAGEvaluator, dataset_file: Path, output_dir: str) -> tuple:
    """Evaluate a single dataset and return results"""
    print(f"\n🔍 Evaluating: {dataset_file.name}")
    print("="*50)
    
    try:
        # Run evaluation
        results_file = await evaluator.run_evaluation(str(dataset_file), output_dir)
        
        # Generate summary
        summary = evaluator.generate_summary_report()
        
        # Print brief results
        if "evaluation_summary" in summary:
            s = summary["evaluation_summary"]
            print(f"   📊 Queries: {s['total_queries']}")
            print(f"   ⏱️  Avg Time: {s['overall_metrics']['average_response_time']:.2f}s")
            print(f"   ✅ Faithfulness: {s['overall_metrics']['faithfulness_passing_rate']:.1f}%")
            print(f"   🎯 Relevancy: {s['overall_metrics']['relevancy_passing_rate']:.1f}%")
            print(f"   ✔️  Correctness: {s['overall_metrics']['correctness_passing_rate']:.1f}%")
        
        return dataset_file.name, True, summary, results_file
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return dataset_file.name, False, None, None

async def main():
    parser = argparse.ArgumentParser(description="Run RAG system evaluation on all datasets")
    parser.add_argument(
        "--datasets-dir", 
        type=str, 
        default="evaluation/datasets",
        help="Directory containing dataset JSON files"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=None,
        help="Specific dataset file to evaluate (overrides auto-discovery)"
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
    
    print("🔍 RAG Multi-Dataset Evaluation Starting...")
    print(f"🤖 LLM Model: {args.llm_model}")
    print(f"🧮 Evaluation Model: {args.eval_model}")
    print(f"📝 Embedding Model: {args.embedding_model}")
    print(f"💾 Output Directory: {args.output_dir}")
    print("-" * 70)
    
    try:
        # Discover datasets or use specific one
        if args.dataset:
            if not Path(args.dataset).exists():
                print(f"❌ Dataset file '{args.dataset}' not found!")
                return 1
            dataset_files = [Path(args.dataset)]
        else:
            dataset_files = discover_datasets(args.datasets_dir)
            
        if not dataset_files:
            print("❌ No dataset files found!")
            return 1
        
        # Initialize evaluator once
        print(f"\n🔧 Initializing RAG evaluator...")
        evaluator = RAGEvaluator(
            ollama_base_url=args.ollama_url,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            evaluation_llm_model=args.eval_model
        )
        
        # Evaluate all datasets
        all_results = []
        successful_evals = 0
        total_queries = 0
        
        for dataset_file in dataset_files:
            result = await evaluate_dataset(evaluator, dataset_file, args.output_dir)
            all_results.append(result)
            
            if result[1]:  # Success flag
                successful_evals += 1
                if result[2] and "evaluation_summary" in result[2]:
                    total_queries += result[2]["evaluation_summary"]["total_queries"]
        
        # Print final summary
        print(f"\n" + "="*70)
        print("🏆 MULTI-DATASET EVALUATION COMPLETE")
        print("="*70)
        print(f"📊 Datasets Evaluated: {successful_evals}/{len(dataset_files)}")
        print(f"📝 Total Queries Processed: {total_queries}")
        
        print(f"\n📋 INDIVIDUAL DATASET RESULTS:")
        for dataset_name, success, summary, results_file in all_results:
            status = "✅" if success else "❌"
            print(f"   {status} {dataset_name}")
            
            if success and summary and "evaluation_summary" in summary:
                s = summary["evaluation_summary"]
                print(f"      📊 {s['total_queries']} queries, "
                      f"⏱️  {s['overall_metrics']['average_response_time']:.2f}s avg, "
                      f"✅ {s['overall_metrics']['faithfulness_passing_rate']:.0f}% faithful, "
                      f"🎯 {s['overall_metrics']['relevancy_passing_rate']:.0f}% relevant")
                if results_file:
                    print(f"      💾 {Path(results_file).name}")
        
        print(f"\n💾 All results saved to: {args.output_dir}/")
        print(f"\n🎉 Evaluation complete! {successful_evals} dataset(s) processed successfully.")
        
        return 0 if successful_evals > 0 else 1
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

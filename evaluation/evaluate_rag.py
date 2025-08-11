#!/usr/bin/env python3
"""
Complete RAG Evaluation Workflow
This script guides you through the entire evaluation process
"""

import asyncio
import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from rag_evaluator import RAGEvaluator
from generate_dataset import EvaluationDatasetGenerator

class RAGEvaluationWorkflow:
    """Complete workflow for RAG evaluation"""
    
    def __init__(self):
        self.evaluator = None
        self.dataset_generator = None
        
    async def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("🔍 Checking prerequisites...")
        
        issues = []
        
        # Check if Chroma database exists
        if not Path("chroma_db").exists():
            issues.append("❌ Chroma database not found. Run your scraper first to populate data.")
        else:
            print("✅ Chroma database found")
        
        # Check if sample dataset exists
        if not Path("evaluation/datasets/sample_queries.json").exists():
            issues.append("❌ Sample dataset not found")
        else:
            print("✅ Sample dataset found")
        
        # Check Ollama (we can't actually test connection here, but we can suggest)
        print("ℹ️  Make sure Ollama is running: ollama serve")
        print("ℹ️  Make sure you have models available: ollama list")
        
        if issues:
            print("\n⚠️  Issues found:")
            for issue in issues:
                print(f"   {issue}")
            return False
        
        print("✅ All prerequisites met!")
        return True
    
    async def menu(self):
        """Display interactive menu"""
        while True:
            print("\n" + "="*60)
            print("🧪 RAG EVALUATION WORKFLOW")
            print("="*60)
            print("1. 🏃 Quick Start - Run evaluation with sample dataset")
            print("2. 🎯 Generate custom dataset from your Chroma database")
            print("3. 📊 Run comprehensive evaluation")
            print("4. 🔧 Advanced evaluation options")
            print("5. 📈 Compare previous results")
            print("6. ❓ Help & Documentation")
            print("0. Exit")
            
            choice = input("\nSelect an option (0-6): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                await self.quick_start()
            elif choice == "2":
                await self.generate_dataset_interactive()
            elif choice == "3":
                await self.comprehensive_evaluation()
            elif choice == "4":
                await self.advanced_options()
            elif choice == "5":
                await self.compare_results()
            elif choice == "6":
                self.show_help()
            else:
                print("❌ Invalid option. Please choose 0-6.")
    
    async def quick_start(self):
        """Quick start evaluation with sample dataset"""
        print("\n🏃 QUICK START EVALUATION")
        print("-" * 40)
        
        if not await self.check_prerequisites():
            input("Press Enter to continue...")
            return
        
        print("Running evaluation with sample dataset...")
        
        try:
            self.evaluator = RAGEvaluator()
            results_file = await self.evaluator.run_evaluation(
                "evaluation/datasets/sample_queries.json"
            )
            
            # Show summary
            await self.show_results_summary(self.evaluator)
            
            print(f"\n💾 Detailed results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Quick start failed: {e}")
        
        input("\nPress Enter to continue...")
    
    async def generate_dataset_interactive(self):
        """Interactive dataset generation"""
        print("\n🎯 GENERATE CUSTOM DATASET")
        print("-" * 40)
        
        # Get parameters from user
        print("Configure dataset generation:")
        
        try:
            num_questions = int(input("Number of questions to generate (default: 20): ") or "20")
            num_nodes = int(input("Number of document nodes to sample (default: 50): ") or "50")
            output_file = input("Output filename (default: generated_queries.json): ") or "generated_queries.json"
            
            if not output_file.startswith("evaluation/datasets/"):
                output_file = f"evaluation/datasets/{output_file}"
            
            print(f"\n🤔 Generating {num_questions} questions from {num_nodes} document samples...")
            
            self.dataset_generator = EvaluationDatasetGenerator()
            success = await self.dataset_generator.generate_dataset(
                output_file=output_file,
                num_questions=num_questions,
                num_nodes_sample=num_nodes
            )
            
            if success:
                print(f"✅ Dataset generated successfully: {output_file}")
                
                # Ask if user wants to run evaluation immediately
                run_now = input("\nRun evaluation with this dataset now? (y/N): ").lower().strip()
                if run_now in ['y', 'yes']:
                    await self.run_evaluation_with_dataset(output_file)
            else:
                print("❌ Dataset generation failed")
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue...")
    
    async def comprehensive_evaluation(self):
        """Run comprehensive evaluation with options"""
        print("\n📊 COMPREHENSIVE EVALUATION")
        print("-" * 40)
        
        # List available datasets
        datasets_dir = Path("evaluation/datasets")
        if not datasets_dir.exists():
            print("❌ No datasets directory found")
            input("Press Enter to continue...")
            return
        
        datasets = list(datasets_dir.glob("*.json"))
        if not datasets:
            print("❌ No datasets found. Generate one first (option 2)")
            input("Press Enter to continue...")
            return
        
        print("Available datasets:")
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset.name}")
        
        try:
            choice = int(input(f"\nSelect dataset (1-{len(datasets)}): "))
            if 1 <= choice <= len(datasets):
                selected_dataset = str(datasets[choice - 1])
                await self.run_evaluation_with_dataset(selected_dataset)
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Invalid input")
        
        input("\nPress Enter to continue...")
    
    async def run_evaluation_with_dataset(self, dataset_path: str):
        """Run evaluation with specified dataset"""
        print(f"\n🔬 Running evaluation with {Path(dataset_path).name}")
        
        try:
            # Get evaluation parameters
            print("\nEvaluation configuration:")
            llm_model = input("LLM model (default: llama3:8b): ") or "llama3:8b"
            eval_model = input("Evaluation model (default: llama3:8b): ") or "llama3:8b"
            embedding_model = input("Embedding model (default: nomic-embed-text): ") or "nomic-embed-text"
            
            print(f"\n🚀 Starting evaluation...")
            print(f"   Dataset: {dataset_path}")
            print(f"   LLM: {llm_model}")
            print(f"   Evaluator: {eval_model}")
            print(f"   Embeddings: {embedding_model}")
            
            self.evaluator = RAGEvaluator(
                llm_model=llm_model,
                evaluation_llm_model=eval_model,
                embedding_model=embedding_model
            )
            
            results_file = await self.evaluator.run_evaluation(dataset_path)
            
            # Show summary
            await self.show_results_summary(self.evaluator)
            
            print(f"\n💾 Results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
    
    async def show_results_summary(self, evaluator):
        """Display evaluation results summary"""
        summary = evaluator.generate_summary_report()
        
        if "evaluation_summary" not in summary:
            print("❌ No results available")
            return
        
        s = summary["evaluation_summary"]
        
        print("\n" + "="*60)
        print("📋 EVALUATION SUMMARY")
        print("="*60)
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
        
        # Performance interpretation
        faith_rate = s['overall_metrics']['faithfulness_passing_rate']
        rel_rate = s['overall_metrics']['relevancy_passing_rate']
        
        print(f"\n🎯 INTERPRETATION:")
        if faith_rate >= 80 and rel_rate >= 80:
            print("   🟢 Excellent performance! Your RAG system is working well.")
        elif faith_rate >= 60 and rel_rate >= 60:
            print("   🟡 Good performance, but there's room for improvement.")
        else:
            print("   🔴 Performance needs attention. Consider improving your RAG pipeline.")
        
        if faith_rate < 70:
            print("   💡 Focus on reducing hallucinations (improve faithfulness)")
        if rel_rate < 70:
            print("   💡 Focus on improving response relevancy")
    
    async def advanced_options(self):
        """Advanced evaluation options"""
        print("\n🔧 ADVANCED OPTIONS")
        print("-" * 40)
        print("1. Batch evaluation with multiple models")
        print("2. Component-wise evaluation (retrieval only)")
        print("3. Custom evaluation metrics")
        print("4. Performance benchmarking")
        print("0. Back to main menu")
        
        choice = input("\nSelect option (0-4): ").strip()
        
        if choice == "1":
            await self.batch_evaluation()
        elif choice == "2":
            print("🚧 Component-wise evaluation coming soon!")
        elif choice == "3":
            print("🚧 Custom metrics coming soon!")
        elif choice == "4":
            print("🚧 Performance benchmarking coming soon!")
        
        if choice != "0":
            input("\nPress Enter to continue...")
    
    async def batch_evaluation(self):
        """Run evaluation with multiple models for comparison"""
        print("\n🔄 BATCH EVALUATION")
        print("-" * 40)
        
        models = [
            "llama3.2:1b",
            "llama3:8b", 
            "llama3.1:8b"
        ]
        
        print("Available models for comparison:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        dataset_path = "evaluation/datasets/sample_queries.json"
        
        print(f"\nRunning batch evaluation with dataset: {Path(dataset_path).name}")
        
        results = {}
        
        for model in models:
            print(f"\n🤖 Evaluating with {model}...")
            try:
                evaluator = RAGEvaluator(
                    llm_model=model,
                    evaluation_llm_model="llama3:8b"  # Use consistent evaluator
                )
                
                results_file = await evaluator.run_evaluation(
                    dataset_path, 
                    output_dir=f"evaluation/results/batch_{model.replace(':', '_')}"
                )
                
                summary = evaluator.generate_summary_report()
                if "evaluation_summary" in summary:
                    s = summary["evaluation_summary"]
                    results[model] = {
                        "faithfulness": s['overall_metrics']['faithfulness_passing_rate'],
                        "relevancy": s['overall_metrics']['relevancy_passing_rate'],
                        "response_time": s['overall_metrics']['average_response_time']
                    }
                
            except Exception as e:
                print(f"❌ Failed to evaluate {model}: {e}")
                results[model] = None
        
        # Display comparison
        print("\n📊 BATCH EVALUATION RESULTS")
        print("="*60)
        print(f"{'Model':<15} {'Faith%':<8} {'Rel%':<8} {'Time(s)':<8}")
        print("-"*40)
        
        for model, result in results.items():
            if result:
                print(f"{model:<15} {result['faithfulness']:<8.1f} {result['relevancy']:<8.1f} {result['response_time']:<8.3f}")
            else:
                print(f"{model:<15} {'FAILED':<25}")
    
    async def compare_results(self):
        """Compare previous evaluation results"""
        print("\n📈 COMPARE RESULTS")
        print("-" * 40)
        
        results_dir = Path("evaluation/results")
        if not results_dir.exists():
            print("❌ No results directory found. Run evaluations first.")
            input("Press Enter to continue...")
            return
        
        # Find summary files
        summary_files = list(results_dir.glob("*summary*.json"))
        if not summary_files:
            print("❌ No evaluation summaries found")
            input("Press Enter to continue...")
            return
        
        # Sort by timestamp (newest first)
        summary_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"Found {len(summary_files)} evaluation results:")
        for i, file in enumerate(summary_files[:10], 1):  # Show last 10
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            print(f"  {i}. {file.name} ({mtime.strftime('%Y-%m-%d %H:%M')})")
        
        if len(summary_files) < 2:
            print("❌ Need at least 2 evaluation results to compare")
            input("Press Enter to continue...")
            return
        
        try:
            first = int(input(f"\nSelect first result (1-{min(10, len(summary_files))}): "))
            second = int(input(f"Select second result (1-{min(10, len(summary_files))}): "))
            
            if 1 <= first <= len(summary_files) and 1 <= second <= len(summary_files):
                await self.display_comparison(summary_files[first-1], summary_files[second-1])
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Invalid input")
        
        input("\nPress Enter to continue...")
    
    async def display_comparison(self, file1: Path, file2: Path):
        """Display comparison between two evaluation results"""
        try:
            with open(file1) as f:
                data1 = json.load(f)
            with open(file2) as f:
                data2 = json.load(f)
            
            s1 = data1["evaluation_summary"]
            s2 = data2["evaluation_summary"]
            
            print(f"\n📊 COMPARISON")
            print("="*60)
            print(f"{'Metric':<25} {'Result 1':<15} {'Result 2':<15} {'Change':<10}")
            print("-"*65)
            
            metrics = [
                ("Faithfulness %", "faithfulness_passing_rate"),
                ("Relevancy %", "relevancy_passing_rate"),
                ("Correctness %", "correctness_passing_rate"),
                ("Avg Response Time", "average_response_time")
            ]
            
            for display_name, key in metrics:
                val1 = s1['overall_metrics'][key]
                val2 = s2['overall_metrics'][key]
                change = val2 - val1
                change_str = f"{change:+.2f}"
                if key.endswith("_rate"):
                    change_str += "%"
                else:
                    change_str += "s"
                
                print(f"{display_name:<25} {val1:<15.2f} {val2:<15.2f} {change_str:<10}")
            
            print(f"\nFiles compared:")
            print(f"  Result 1: {file1.name}")
            print(f"  Result 2: {file2.name}")
            
        except Exception as e:
            print(f"❌ Error comparing results: {e}")
    
    def show_help(self):
        """Show help and documentation"""
        print("\n❓ HELP & DOCUMENTATION")
        print("="*60)
        print("""
📚 EVALUATION METRICS EXPLAINED:

• Faithfulness (0-1): Measures if responses are faithful to retrieved context
  Higher = fewer hallucinations, more grounded responses
  
• Relevancy (0-1): Measures if responses are relevant to the query  
  Higher = more on-topic, useful responses
  
• Correctness (0-1): Measures factual accuracy against ground truth
  Only available when expected answers are provided
  
• Semantic Similarity (0-1): Measures similarity to expected answers
  Only available when expected answers are provided

🎯 PERFORMANCE TARGETS:

• Faithfulness > 0.8: Excellent (minimal hallucinations)
• Relevancy > 0.8: Excellent (highly relevant responses)
• Response Time < 3s: Good performance for interactive use

🔧 IMPROVEMENT TIPS:

• Low Faithfulness: Improve retrieval quality, adjust prompts
• Low Relevancy: Improve embedding model, query preprocessing  
• Slow Response: Use smaller models, optimize vector database
• Low Correctness: Improve data quality, better context retrieval

📁 FILES:

• evaluation/datasets/: Query datasets for testing
• evaluation/results/: Evaluation results and summaries
• evaluation/logs/: Detailed evaluation logs

🚀 GETTING STARTED:

1. Run "Quick Start" to baseline your system
2. Generate custom datasets from your data  
3. Iterate on improvements and re-evaluate
4. Track progress over time with comparisons
        """)
        
        input("\nPress Enter to continue...")

async def main():
    """Main entry point"""
    print("🧪 RAG Evaluation Workflow")
    print("Comprehensive evaluation for your Monster Hunter Wilds RAG system")
    
    workflow = RAGEvaluationWorkflow()
    await workflow.menu()

if __name__ == "__main__":
    asyncio.run(main())

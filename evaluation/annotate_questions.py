#!/usr/bin/env python3
"""
Interactive Question Annotation Tool for RAG Evaluation

This tool provides an interactive interface to review and annotate questions in the 
Monster Hunter Wilds evaluation dataset. It uses the RAG pipeline to generate answers
and allows the user to accept, reject, or manually provide correct answers.

Features:
- Interactive review of questions with null expected_answer
- RAG model integration for automatic answer generation  
- User options to accept, reject, skip, or manually enter answers
- Progress tracking and automatic saving
- Backup creation and metadata recording
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import shutil
from dataclasses import dataclass

# Add the custom pipeline to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'custom_pipeline'))
from llamaindex_chroma_rag import Pipeline

@dataclass
class AnnotationSession:
    """Tracks the current annotation session state"""
    total_questions: int = 0
    processed_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    manual_count: int = 0
    skipped_count: int = 0
    skipped_questions: List[Dict] = None
    start_time: datetime = None
    
    def __post_init__(self):
        if self.skipped_questions is None:
            self.skipped_questions = []

class QuestionAnnotator:
    """Interactive tool for annotating questions with expected answers"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = None
        self.rag_pipeline = None
        self.session = AnnotationSession()
        
        # Configuration
        self.save_interval = 5  # Save every N questions
        self.stats_interval = 10  # Show stats every N questions
        
        # Backup original file
        self.backup_path = self._create_backup()
        
    def _create_backup(self) -> str:
        """Create a backup of the original dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.dataset_path}.backup_{timestamp}"
        try:
            shutil.copy2(self.dataset_path, backup_path)
            print(f"📁 Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"⚠️  Warning: Could not create backup: {e}")
            return None
    
    async def setup_rag_pipeline(self):
        """Initialize and setup the RAG pipeline"""
        print("🔧 Setting up RAG pipeline...")
        
        try:
            self.rag_pipeline = Pipeline()
            
            # Configure the pipeline with default settings
            self.rag_pipeline.valves.LLAMAINDEX_OLLAMA_BASE_URL = "http://localhost:11434"
            self.rag_pipeline.valves.LLAMAINDEX_MODEL_NAME = "llama3:8b"
            self.rag_pipeline.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME = "nomic-embed-text"
            
            # Set Chroma database path
            chroma_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chroma_db'))
            self.rag_pipeline.valves.CHROMA_PERSIST_DIR = chroma_db_path
            
            # Start the pipeline
            await self.rag_pipeline.on_startup()
            print("✅ RAG pipeline ready!")
            
        except Exception as e:
            print(f"❌ Failed to setup RAG pipeline: {e}")
            print("⚠️  You can still manually annotate questions without RAG generation.")
            self.rag_pipeline = None
    
    def load_dataset(self):
        """Load the dataset from JSON file"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            
            questions = self.dataset.get('questions', [])
            self.session.total_questions = len([q for q in questions if q.get('expected_answer') is None])
            
            print(f"📊 Loaded dataset with {len(questions)} total questions")
            print(f"🎯 Found {self.session.total_questions} questions needing annotation")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            sys.exit(1)
    
    async def generate_answer(self, query: str) -> Optional[str]:
        """Generate an answer using the RAG pipeline"""
        if not self.rag_pipeline:
            return None
            
        try:
            response = self.rag_pipeline.pipe(
                user_message=query,
                model_id=self.rag_pipeline.valves.LLAMAINDEX_MODEL_NAME,
                messages=[],
                body={}
            )
            return str(response)
        except Exception as e:
            print(f"⚠️  Error generating answer: {e}")
            return None
    
    def display_question(self, question: Dict, index: int) -> None:
        """Display question information in a formatted way"""
        print("\n" + "="*80)
        print(f"📝 Question {index + 1} of {self.session.total_questions}")
        print(f"Category: {question.get('category', 'Unknown')}")
        print(f"Difficulty: {question.get('difficulty', 'Unknown')}")
        print("-"*80)
        print(f"Query: {question['query']}")
        print("-"*80)
    
    def display_generated_answer(self, answer: str) -> None:
        """Display the generated answer"""
        print(f"🤖 Generated Answer:")
        print("-"*40)
        print(answer)
        print("-"*40)
    
    def get_user_choice(self, has_generated_answer: bool = True) -> str:
        """Get user choice for the current question"""
        if has_generated_answer:
            print("\nOptions:")
            print("  [a] Accept generated answer")
            print("  [r] Reject and provide correct answer")
            print("  [s] Skip this question")
            print("  [q] Quit and save progress")
            
            while True:
                choice = input("\nYour choice (a/r/s/q): ").lower().strip()
                if choice in ['a', 'r', 's', 'q']:
                    return choice
                print("❌ Invalid choice. Please enter 'a', 'r', 's', or 'q'")
        else:
            print("\nOptions (no generated answer available):")
            print("  [m] Manually provide answer")
            print("  [s] Skip this question")  
            print("  [q] Quit and save progress")
            
            while True:
                choice = input("\nYour choice (m/s/q): ").lower().strip()
                if choice in ['m', 's', 'q']:
                    return choice
                print("❌ Invalid choice. Please enter 'm', 's', or 'q'")
    
    def get_manual_answer(self, original_answer: Optional[str] = None) -> Optional[str]:
        """Get manual answer from user"""
        print("\n📝 Enter the correct answer:")
        if original_answer:
            print(f"(Original generated answer: {original_answer[:100]}...)")
        print("(Press Enter on empty line to finish, or type 'CANCEL' to cancel)")
        
        lines = []
        while True:
            line = input()
            if line == '':
                break
            if line.upper() == 'CANCEL':
                return None
            lines.append(line)
        
        answer = '\n'.join(lines).strip()
        return answer if answer else None
    
    def update_question(self, question: Dict, answer: str, source: str, original_generated: Optional[str] = None) -> None:
        """Update question with the annotated answer"""
        question['expected_answer'] = {
            'answer': answer,
            'source': source,  # 'generated_accepted', 'generated_rejected', 'manual'
            'timestamp': datetime.now().isoformat(),
            'original_generated': original_generated
        }
    
    def save_dataset(self) -> None:
        """Save the current dataset state"""
        try:
            with open(self.dataset_path, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, indent=2, ensure_ascii=False)
            print(f"💾 Progress saved to {self.dataset_path}")
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
    
    def show_session_stats(self) -> None:
        """Display current session statistics"""
        if self.session.processed_count == 0:
            return
            
        print(f"\n📊 Session Statistics:")
        print(f"   Processed: {self.session.processed_count}/{self.session.total_questions}")
        print(f"   Accepted: {self.session.accepted_count}")
        print(f"   Rejected: {self.session.rejected_count}")
        print(f"   Manual: {self.session.manual_count}")
        print(f"   Skipped: {self.session.skipped_count}")
        
        if self.session.accepted_count > 0:
            acceptance_rate = (self.session.accepted_count / 
                             (self.session.accepted_count + self.session.rejected_count + self.session.manual_count)) * 100
            print(f"   Acceptance Rate: {acceptance_rate:.1f}%")
        
        if self.session.start_time:
            elapsed = datetime.now() - self.session.start_time
            print(f"   Elapsed Time: {elapsed}")
    
    def _remove_skipped_questions(self) -> None:
        """Remove skipped questions from the dataset"""
        if not self.session.skipped_questions:
            return
        
        print(f"\n🗑️  Removing {len(self.session.skipped_questions)} skipped questions from dataset...")
        
        # Create a set of skipped question queries for fast lookup
        skipped_queries = {q['query'] for q in self.session.skipped_questions}
        
        # Filter out skipped questions
        original_count = len(self.dataset.get('questions', []))
        self.dataset['questions'] = [
            q for q in self.dataset.get('questions', [])
            if q['query'] not in skipped_queries
        ]
        new_count = len(self.dataset['questions'])
        
        print(f"📊 Dataset size: {original_count} → {new_count} questions")
        
        # Update metadata if it exists
        if 'metadata' in self.dataset:
            self.dataset['metadata']['total_questions'] = new_count
            if 'last_filtered' not in self.dataset['metadata']:
                self.dataset['metadata']['filtered_history'] = []
            self.dataset['metadata']['filtered_history'].append({
                'timestamp': datetime.now().isoformat(),
                'removed_count': len(self.session.skipped_questions),
                'reason': 'annotation_skip',
                'description': 'Questions skipped during annotation as unrealistic'
            })
    
    def _report_skipped_questions(self) -> None:
        """Report the skipped questions"""
        if not self.session.skipped_questions:
            return
        
        print(f"\n📋 All the following {len(self.session.skipped_questions)} questions were skipped, and thus are removed from the dataset:")
        print("" + "="*80)
        
        for i, question in enumerate(self.session.skipped_questions, 1):
            print(f"\n{i}. [{question.get('category', 'Unknown')}] [{question.get('difficulty', 'Unknown')}]")
            print(f"   {question['query']}")
        
        print("" + "="*80)
        print(f"✅ These questions have been removed from the dataset as they were deemed not realistically something someone would ask.")
    
    async def annotate_questions(self):
        """Main annotation loop"""
        print("\n🚀 Starting annotation session...")
        self.session.start_time = datetime.now()
        
        questions = self.dataset.get('questions', [])
        unannotated_questions = [(i, q) for i, q in enumerate(questions) 
                               if q.get('expected_answer') is None]
        
        if not unannotated_questions:
            print("✅ All questions are already annotated!")
            return
        
        print(f"📋 Found {len(unannotated_questions)} questions to annotate")
        
        try:
            for question_index, (original_index, question) in enumerate(unannotated_questions):
                self.display_question(question, question_index)
                
                # Try to generate answer
                generated_answer = None
                if self.rag_pipeline:
                    print("🔄 Generating answer...")
                    generated_answer = await self.generate_answer(question['query'])
                
                if generated_answer:
                    self.display_generated_answer(generated_answer)
                    choice = self.get_user_choice(has_generated_answer=True)
                else:
                    print("⚠️  No generated answer available")
                    choice = self.get_user_choice(has_generated_answer=False)
                
                # Process user choice
                if choice == 'q':
                    print("\n🛑 Quitting and saving progress...")
                    break
                elif choice == 's':
                    print("⏭️  Skipped (will be removed from dataset)")
                    self.session.skipped_count += 1
                    self.session.skipped_questions.append(question.copy())
                elif choice == 'a' and generated_answer:
                    self.update_question(question, generated_answer, 'generated_accepted')
                    self.session.accepted_count += 1
                    print("✅ Answer accepted")
                elif choice == 'r' and generated_answer:
                    manual_answer = self.get_manual_answer(generated_answer)
                    if manual_answer:
                        self.update_question(question, manual_answer, 'generated_rejected', generated_answer)
                        self.session.rejected_count += 1
                        print("✅ Answer rejected and replaced")
                    else:
                        print("❌ Cancelled - question skipped (will be removed from dataset)")
                        self.session.skipped_count += 1
                        self.session.skipped_questions.append(question.copy())
                elif choice == 'm':
                    manual_answer = self.get_manual_answer()
                    if manual_answer:
                        self.update_question(question, manual_answer, 'manual')
                        self.session.manual_count += 1
                        print("✅ Manual answer added")
                    else:
                        print("❌ Cancelled - question skipped (will be removed from dataset)")
                        self.session.skipped_count += 1
                        self.session.skipped_questions.append(question.copy())
                
                self.session.processed_count += 1
                
                # Save periodically
                if self.session.processed_count % self.save_interval == 0:
                    self.save_dataset()
                
                # Show stats periodically
                if self.session.processed_count % self.stats_interval == 0:
                    self.show_session_stats()
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Interrupted by user. Saving progress...")
        
        # Remove skipped questions from dataset
        if self.session.skipped_questions:
            self._remove_skipped_questions()
        
        # Final save and stats
        self.save_dataset()
        self.show_session_stats()
        
        # Report skipped questions
        if self.session.skipped_questions:
            self._report_skipped_questions()
        
        print(f"\n🎉 Annotation session complete!")
        print(f"💾 Dataset saved to: {self.dataset_path}")
        if self.backup_path:
            print(f"📁 Original backup: {self.backup_path}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactive annotation tool for RAG evaluation questions"
    )
    parser.add_argument(
        "dataset_path", 
        help="Path to the questions dataset JSON file (required)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset file not found: {args.dataset_path}")
        sys.exit(1)
    
    print(f"🎯 Monster Hunter Wilds Question Annotator")
    print(f"📄 Dataset: {args.dataset_path}")
    
    async def run_annotation():
        annotator = QuestionAnnotator(args.dataset_path)
        
        # Setup
        annotator.load_dataset()
        await annotator.setup_rag_pipeline()
        
        # Run annotation
        await annotator.annotate_questions()
    
    try:
        asyncio.run(run_annotation())
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

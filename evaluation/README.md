# 🧪 RAG System Evaluation Framework

A comprehensive evaluation system for your Monster Hunter Wilds RAG pipeline based on LlamaIndex evaluation best practices.

## 📋 Overview

This evaluation framework implements both **End-to-End** and **Component-Wise** evaluation strategies as recommended by LlamaIndex documentation:

### 🎯 Evaluation Metrics

**End-to-End Metrics:**
- **Faithfulness**: Are responses faithful to retrieved context? (No hallucinations)
- **Relevancy**: Are responses relevant to the query?
- **Correctness**: Are responses factually correct? (when ground truth available)
- **Semantic Similarity**: How similar are responses to expected answers?

**Component-Wise Metrics:**
- **Hit Rate**: Percentage of queries where relevant documents are retrieved
- **MRR (Mean Reciprocal Rank)**: Quality of retrieval ranking
- **Response Time**: Performance measurement

**Quality Categories:**
- Weapons, Monsters, Gameplay Mechanics, Crafting, World/Environment, Technical, General

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r evaluation/requirements.txt
```

### 2. Run Basic Evaluation

```bash
cd evaluation
python run_evaluation.py
```

This will:
- Use the sample dataset (`evaluation/datasets/sample_queries.json`)
- Run comprehensive evaluation on all queries
- Generate detailed results and summary reports
- Save results to `evaluation/results/`

### 3. Generate Your Own Dataset

Create evaluation questions from your existing Chroma database:

```bash
python generate_dataset.py --num-questions 50 --output evaluation/datasets/my_questions.json
```

### 4. Run Custom Evaluation

```bash
python run_evaluation.py --dataset evaluation/datasets/my_questions.json --eval-model llama3.1:8b
```

## 📂 Directory Structure

```
evaluation/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── rag_evaluator.py      # Main evaluation framework
├── run_evaluation.py     # Simple evaluation runner
├── generate_dataset.py   # Dataset generation utility
├── datasets/             # Evaluation datasets
│   ├── sample_queries.json    # Hand-crafted sample queries
│   └── generated_queries.json # Auto-generated questions
├── results/              # Evaluation results
│   ├── evaluation_results_TIMESTAMP.json
│   └── evaluation_summary_TIMESTAMP.json
└── logs/                 # Evaluation logs
    └── evaluation.log
```

## 🔧 Configuration Options

### RAG Evaluator Parameters

```python
evaluator = RAGEvaluator(
    ollama_base_url="http://localhost:11434",  # Ollama API endpoint
    llm_model="llama3:8b",                     # Model for RAG pipeline
    embedding_model="nomic-embed-text",        # Embedding model
    evaluation_llm_model="llama3.1:8b"        # Model for evaluation (can be better)
)
```

### Command Line Options

```bash
python run_evaluation.py \
    --dataset evaluation/datasets/sample_queries.json \
    --ollama-url http://localhost:11434 \
    --llm-model llama3:8b \
    --eval-model llama3.1:8b \
    --embedding-model nomic-embed-text \
    --output-dir evaluation/results
```

## 📊 Understanding Results

### Summary Report Structure

```json
{
  "evaluation_summary": {
    "total_queries": 15,
    "evaluation_date": "2025-01-11T12:00:00",
    "overall_metrics": {
      "faithfulness_passing_rate": 85.2,
      "relevancy_passing_rate": 91.3,
      "correctness_passing_rate": 76.8,
      "average_response_time": 2.145
    },
    "score_averages": {
      "faithfulness": 0.852,
      "relevancy": 0.913,
      "correctness": 0.768,
      "semantic_similarity": 0.721
    },
    "category_breakdown": {
      "weapons": {
        "count": 3,
        "faithfulness_passing_rate": 100.0,
        "relevancy_passing_rate": 100.0
      }
    }
  }
}
```

### Key Metrics Explained

- **Faithfulness (0-1)**: Higher = fewer hallucinations
- **Relevancy (0-1)**: Higher = more relevant responses
- **Correctness (0-1)**: Higher = more factually correct (requires ground truth)
- **Response Time**: Lower = faster responses

### Interpreting Results

- **🟢 Good Performance**: Faithfulness > 0.8, Relevancy > 0.8
- **🟡 Needs Improvement**: Faithfulness 0.6-0.8, Relevancy 0.6-0.8
- **🔴 Poor Performance**: Faithfulness < 0.6, Relevancy < 0.6

## 📈 Evaluation Strategies

### 1. End-to-End Evaluation (Recommended Start)

Best for getting an overall system health check:

```bash
python run_evaluation.py --dataset evaluation/datasets/sample_queries.json
```

**Use when:**
- You want to assess overall system performance
- You're iterating on the complete pipeline
- You want to catch integration issues

### 2. Component-Wise Evaluation

For debugging specific components:

```python
# Focus on retrieval quality
evaluator.retriever_evaluator.evaluate(
    query="What weapons are in Monster Hunter Wilds?",
    expected_ids=["doc_123", "doc_456"]
)
```

**Use when:**
- You've identified issues in end-to-end evaluation
- You're optimizing specific components
- You want to understand failure modes

## 🛠️ Creating Custom Datasets

### Manual Dataset Creation

Create a JSON file with this structure:

```json
[
  {
    "query": "What are the weapon types in Monster Hunter Wilds?",
    "expected_answer": "Monster Hunter Wilds features 14 weapon types...",
    "category": "weapons",
    "difficulty": "easy"
  }
]
```

### Automated Dataset Generation

```bash
python generate_dataset.py \
    --num-questions 100 \
    --num-nodes 200 \
    --output evaluation/datasets/large_dataset.json
```

## 📋 Best Practices

### 1. Start with End-to-End
- Use overall metrics as your north star
- Focus on faithfulness and relevancy first
- Track performance over time

### 2. Use Diverse Datasets
- Include queries of different difficulties
- Cover all main categories (weapons, monsters, etc.)
- Test edge cases and uncommon queries

### 3. Iterative Improvement
- Baseline with current system
- Make targeted improvements
- Re-evaluate to measure impact
- Compare results over time

### 4. Evaluation Model Selection
- Use a better model for evaluation than your RAG pipeline
- Consider using GPT-4 or Claude for gold-standard evaluation
- Balance cost vs. accuracy

## 🔍 Troubleshooting

### Common Issues

**1. "No queries loaded from dataset"**
- Check dataset file path and format
- Ensure JSON is valid

**2. "Ollama connection failed"**
- Verify Ollama is running: `ollama serve`
- Check if models are available: `ollama list`

**3. "Chroma database not found"**
- Run your scraper first to populate data
- Check `chroma_db/` directory exists

**4. "Evaluation taking too long"**
- Reduce dataset size for testing
- Use faster evaluation models
- Consider async batch evaluation

### Performance Tips

- Use smaller models for testing (llama3.2:1b)
- Start with small datasets (10-20 queries)
- Enable DEBUG_MODE for detailed logging
- Monitor Ollama resource usage

## 🚀 Next Steps

1. **Run Initial Baseline**: Use sample dataset to establish current performance
2. **Generate Custom Dataset**: Create questions from your specific wiki content
3. **Identify Improvement Areas**: Focus on lowest-scoring categories
4. **Iterate and Improve**: Make targeted improvements to your RAG pipeline
5. **Track Progress**: Compare evaluation results over time

## 📚 Additional Resources

- [LlamaIndex Evaluation Guide](https://docs.llamaindex.ai/en/stable/optimizing/evaluation/evaluation/)
- [End-to-End Evaluation](https://docs.llamaindex.ai/en/stable/optimizing/evaluation/e2e_evaluation/)
- [Component-Wise Evaluation](https://docs.llamaindex.ai/en/stable/optimizing/evaluation/component_wise_evaluation/)

## 🤝 Contributing

Found issues or have improvements? Please update the evaluation framework to better serve your RAG development needs!

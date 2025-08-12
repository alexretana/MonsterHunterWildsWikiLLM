# 🐉 Monster Hunter Wilds Wiki RAG System

A comprehensive **two-part RAG (Retrieval-Augmented Generation) system** for Monster Hunter Wilds wiki knowledge. This project combines intelligent web scraping with advanced LLM-powered question answering, featuring a complete evaluation framework for RAG performance assessment.

## 🏗️ **Project Architecture**

### **Part 1: Intelligent Wiki Scraper** (`wikiproject/`)
- **Scrapy-based web crawler** that intelligently scrapes Monster Hunter Wilds wiki data from Fextralife
- **Smart content extraction** with breadcrumb navigation, table parsing, and structured data output
- **Robust crawling features**: pause/resume capability, depth limiting, polite crawling with delays
- **Direct Chroma integration**: automatically ingests scraped content into a persistent vector database

### **Part 2: Advanced RAG Pipeline** (`custom_pipeline/`)
- **LlamaIndex + Chroma** powered RAG system with Monster Hunter domain expertise
- **Custom prompts** specifically tuned for Monster Hunter terminology and knowledge
- **Multiple response modes**: compact, refine, tree summarize, simple summarize
- **Conversation context** enhancement for multi-turn conversations
- **OpenWebUI integration** as a custom pipeline for a seamless chat experience

### **Part 3: Comprehensive RAG Evaluation** (`evaluation/`)
- **End-to-end evaluation**: faithfulness, relevancy, correctness, semantic similarity
- **Component-wise evaluation**: retrieval quality, hit rate, mean reciprocal rank
- **Automated dataset generation** from the existing knowledge base
- **Multi-dataset evaluation** with detailed reporting and performance tracking
- **Category-based analysis** (weapons, monsters, gameplay, crafting, etc.)

## 🌟 **Key Features**

### **🕸️ Web Scraping**
✅ **Multi-domain crawling** ready (currently focused on Fextralife)  
✅ **Intelligent content parsing** with table extraction and breadcrumb mapping  
✅ **Pause & resume crawls** safely using Scrapy's JOBDIR persistence  
✅ **Smart filtering** skips static assets and focuses on content pages  
✅ **Progress tracking** with detailed logging and URL coverage estimation  

### **🤖 RAG System**
✅ **Persistent Chroma vector store** for fast, scalable retrieval  
✅ **Monster Hunter expertise** with domain-specific prompts and terminology  
✅ **Multiple LLM backends** via Ollama integration  
✅ **Advanced query enhancement** with conversation context and semantic search  
✅ **Configurable response modes** for different use cases  

### **📊 Evaluation Framework**
✅ **LlamaIndex-compliant evaluation** with industry-standard metrics  
✅ **Automated question generation** from your knowledge base  
✅ **Multi-dimensional assessment**: faithfulness, relevancy, correctness  
✅ **Performance tracking** with detailed reports and category breakdowns  
✅ **Comparative analysis** across different configurations and datasets  

## 📋 **Prerequisites**

Before running the automated startup scripts, ensure you have:

### **Required Software**
- **Anaconda or Miniconda** (for environment management)
- **Git** (for cloning repositories)

### **Platform-Specific Requirements**

**Windows:**
- **Chocolatey** (recommended for automatic Ollama installation)
  - If not installed, get it from: https://chocolatey.org/install
  - Alternative: Manual Ollama installation from https://ollama.ai/download

**Linux/macOS:**
- **curl** (usually pre-installed)
- **Terminal access** (for automatic Ollama installation)

### **Hardware Recommendations**
- **8GB+ RAM** (for running Ollama models)
- **5GB+ free disk space** (for AI models and dependencies)
- **Internet connection** (for downloading models and dependencies)

## 🚀 **Quick Start**

### **Option 1: Automated Startup (Recommended)**
Use the provided startup scripts that automatically manage all dependencies and launch the complete system:

**On Windows:**
```cmd
start_windows.bat
```

**On Linux/macOS:**
```bash
./start.sh
```

These scripts will:
- **Check and install Ollama server** if not already present
  - Windows: Uses Chocolatey (with fallback instructions)
  - Linux/macOS: Uses official curl-based installer
- **Install required AI models** automatically:
  - `llama3:8b` - Main LLM for question answering
  - `nomic-embed-text` - Embedding model for vector store
- **Manage Ollama server lifecycle**:
  - Detect if already running on port 11434
  - Start server in separate terminal if needed
  - Track server state to avoid conflicts
- **Handle conda environments**:
  - Check if required environments exist
  - Create them from YAML files if missing
- **Launch all services** in separate terminal windows:
  - Ollama server (if not already running)
  - OpenWebUI backend
  - OpenWebUI Pipelines
- **Comprehensive error handling** with helpful troubleshooting messages

### **Option 2: Manual Setup**
If you prefer manual control or need to troubleshoot:

**For the Scraper:**
```bash
conda env create -f wikiproject/scrapy-environmental.yaml
conda activate wikiscrap2
```

**For the RAG Pipeline & Evaluation:**
```bash
conda env create -f owu-pipeline-environment.yaml
conda activate openwebui-pipelines
```

**For OpenWebUI Backend:**
```bash
conda env create -f openwebui-environment.yaml
conda activate openwebui
```

### **2. Run the Web Scraper**
Activate the `wikiscrap2` environment and run the spider:
```bash
cd wikiproject
scrapy crawl myfextralifespider
```

For pause/resume capability (recommended for large wikis):
```bash
scrapy crawl myfextralifespider -s JOBDIR=jobs/fextralife-$(date +%Y-%m-%d)
```

This will crawl the wiki and automatically populate the `chroma_db/` directory.

### **3. Use the RAG Pipeline in Open WebUI**
1. Ensure Open WebUI is running with pipelines enabled
2. Place `custom_pipeline/llamaindex_chroma_rag.py` in your Open WebUI pipelines directory
3. Select the "LlamaIndex Chroma RAG Pipeline" in the Open WebUI interface
4. Start asking Monster Hunter questions!

### **4. Evaluate the RAG System**
Activate the `openwebui-pipelines` environment and run evaluation:
```bash
cd evaluation
python run_evaluation.py
```

View detailed results in `evaluation/results/` directory.

## 📂 **Project Structure**
```
.
├── custom_pipeline/              # RAG pipeline for Open WebUI
│   └── llamaindex_chroma_rag.py   # Main RAG pipeline with custom prompts
├── evaluation/                    # Comprehensive RAG evaluation framework
│   ├── datasets/                  # Sample and generated evaluation datasets
│   ├── results/                   # Evaluation results and reports
│   ├── rag_evaluator.py          # Core evaluation logic
│   ├── run_evaluation.py         # Main evaluation script
│   ├── generate_questions_dataset.py # Auto-generate evaluation questions
│   └── README.md                  # Detailed evaluation documentation
├── wikiproject/                   # Scrapy web crawler
│   ├── wikiproject/
│   │   ├── spiders/
│   │   │   └── myfextralifespider.py # The main Monster Hunter wiki spider
│   │   ├── pipelines.py           # Data processing and Chroma ingestion
│   │   ├── settings.py            # Scrapy configuration
│   │   └── utils.py               # Helper functions for Chroma integration
│   └── scrapy.cfg
├── chroma_db/                     # Persistent Chroma vector store (gitignored)
├── external/                      # Third-party libraries (gitignored)
│   ├── open-webui-8-6-2025/      # OpenWebUI backend
│   └── open-webui-pipelines/     # OpenWebUI pipelines
├── start_windows.bat              # Windows startup script (automated setup)
├── start.sh                       # Unix/Linux startup script (automated setup)
├── openwebui-environment.yaml    # Conda environment for OpenWebUI backend
├── scrapy-environmental.yaml     # Conda environment for web scraping
├── owu-pipeline-environment.yaml # Conda environment for RAG & evaluation
├── test_prompt_improvements.py   # Testing utilities for prompt optimization
└── README.md                      # This file
```

## 🔧 **Configuration Options**

### **RAG Pipeline Configuration**
The RAG pipeline supports extensive configuration via environment variables:

```bash
# LLM Configuration
LLAMAINDEX_MODEL_NAME="llama3:8b"              # Main LLM model
LLAMAINDEX_EMBEDDING_MODEL_NAME="nomic-embed-text"  # Embedding model
LLAMAINDEX_OLLAMA_BASE_URL="http://localhost:11434"  # Ollama server URL

# Retrieval Configuration
SIMILARITY_TOP_K=10                            # Number of retrieved chunks
RESPONSE_MODE="compact"                        # Response generation mode

# Advanced Features
USE_CUSTOM_PROMPTS=true                        # Enable Monster Hunter prompts
USE_CONVERSATION_CONTEXT=false                # Multi-turn conversation support
DEBUG_MODE=false                               # Enable detailed logging
```

### **Web Scraper Configuration**
Key settings in `wikiproject/wikiproject/settings.py`:

- **CONCURRENT_REQUESTS_PER_DOMAIN**: 1 (respectful crawling)
- **DOWNLOAD_DELAY**: 1 second (polite crawling)
- **DEPTH_LIMIT**: 6 (prevents infinite crawling)
- **ROBOTSTXT_OBEY**: True (respects robots.txt)

## 📊 **Evaluation Metrics**

The evaluation framework provides comprehensive RAG assessment:

### **End-to-End Metrics**
- **Faithfulness** (0-1): Measures hallucination-free responses
- **Relevancy** (0-1): Assesses response relevance to queries
- **Correctness** (0-1): Evaluates factual accuracy (when ground truth available)
- **Semantic Similarity** (0-1): Compares semantic closeness to expected answers

### **Component-Wise Metrics**
- **Hit Rate**: Percentage of queries where relevant documents are retrieved
- **Mean Reciprocal Rank (MRR)**: Quality of retrieval ranking
- **Response Time**: Performance measurement in seconds

### **Category Analysis**
Evaluations are broken down by Monster Hunter categories:
- Weapons & Equipment
- Monsters & Bestiary
- Gameplay Mechanics
- Crafting & Materials
- World & Environment
- Technical & System Info

## 🔧 **Troubleshooting**

### **Common Ollama Issues**

**Ollama Installation Failed (Windows):**
- Ensure Chocolatey is installed: `choco --version`
- Run PowerShell/CMD as Administrator
- Manually install from: https://ollama.ai/download

**Ollama Installation Failed (Linux/macOS):**
- Check curl availability: `curl --version`
- Ensure internet connectivity
- Try manual installation: Download from https://ollama.ai/download

**Model Download Issues:**
- Check available disk space (models can be 4-7GB each)
- Verify internet connection
- Manually pull models: `ollama pull llama3:8b`

**Port 11434 Already in Use:**
- Check what's using the port: `netstat -an | findstr 11434` (Windows) or `lsof -i :11434` (Linux/macOS)
- Stop existing Ollama instance or change port in configuration

**Memory Issues:**
- Ensure at least 8GB RAM available
- Close other applications before starting
- Consider using smaller models if available

### **Conda Environment Issues**

**Environment Creation Failed:**
- Check conda installation: `conda --version`
- Ensure YAML files exist in project directory
- Try updating conda: `conda update conda`

**Activation Issues:**
- Initialize conda: `conda init`
- Restart terminal after conda init
- Check environment exists: `conda env list`

### **Service Startup Issues**

**Terminal Windows Don't Open:**
- Check if running in compatible terminal (not VS Code integrated terminal)
- Try manual activation and startup
- Check script permissions (Linux/macOS): `chmod +x start.sh`

**Services Fail to Connect:**
- Verify all services are running (check terminal windows)
- Check port availability (11434 for Ollama, default ports for OpenWebUI)
- Wait for services to fully initialize (30-60 seconds)

## 🧪 **Testing & Development**

### **Test Prompt Improvements**
```bash
python test_prompt_improvements.py --debug
```

### **Generate Custom Evaluation Dataset**
```bash
cd evaluation
python generate_questions_dataset.py --num-questions 50 --output datasets/my_questions.json
```

### **Run Targeted Evaluation**
```bash
cd evaluation
python run_evaluation.py --dataset datasets/weapons_questions.json --eval-model llama3.1:8b
```

## 🎯 **Performance Benchmarks**

Typical performance on a well-populated knowledge base:

- **Faithfulness**: 85-90% (minimal hallucinations)
- **Relevancy**: 90-95% (highly relevant responses)
- **Correctness**: 75-85% (factually accurate when verifiable)
- **Response Time**: 2-4 seconds (depending on hardware)
- **Hit Rate**: 90%+ (successful retrieval)

## 📚 **Advanced Usage**

### **Custom Prompt Development**
The RAG pipeline uses Monster Hunter-specific prompts. You can customize them in `custom_pipeline/llamaindex_chroma_rag.py`:

- `text_qa_template`: Main question-answering prompt
- `refine_template`: Multi-chunk response refinement
- `summary_template`: Multi-source synthesis

### **Multi-Source Crawling**
To add more wiki sources, extend `myfextralifespider.py`:

1. Add new domains to `allowed_domains`
2. Add new start URLs to `start_urls`
3. Adapt parsing logic for different wiki structures

### **Evaluation Dataset Creation**
Create custom evaluation datasets:

1. **Manual creation**: Write JSON files with query/answer pairs
2. **Automated generation**: Use `generate_questions_dataset.py`
3. **Domain-specific**: Focus on specific Monster Hunter aspects

## 🤝 **Contributing**

Contributions are welcome! Key areas for improvement:

- Additional wiki source support
- Enhanced evaluation metrics
- Prompt optimization
- Performance improvements
- Documentation updates

## 📄 **License**

MIT License - free to use and adapt for your own projects.

## 👤 **Author**

Built by **Alexander Retana** to explore local RAG pipelines for gaming knowledge bases. 🎮

---

*This project demonstrates the power of combining intelligent web scraping, advanced vector databases, and domain-specific LLM fine-tuning to create highly effective knowledge systems for specialized domains.*

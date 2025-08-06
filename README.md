# 🐉 Monster Hunter Wilds Wiki Scraper & RAG Pipeline

This project is a **Scrapy-based web crawler** designed to scrape data from multiple wiki sites about **Monster Hunter Wilds**.  
The scraped data is then processed and uploaded to an **Open WebUI** knowledge base, enabling a local LLM chatbot to answer questions grounded in real wiki content — and link users directly to relevant pages.


## 📦 Features

✅ Crawls multiple Monster Hunter Wilds wiki sources (initially starting with Fextralife).  
✅ Periodic estimation of total pages & unique URLs using the **Chao1 estimator**, to track crawl coverage.  
✅ Pause & resume crawls safely (using Scrapy’s `JOBDIR` and persistent `self.state`).  
✅ Skips static assets (images, etc.) to focus on useful content.  
✅ Logs crawl progress & estimation to file and console.  
✅ Designed for integration into a **Retrieval-Augmented Generation (RAG)** system via Open WebUI.


## 🛠 Project Structure (planned)
```

.
├── spiders/
│ └── myfextralifespider.py # Main Scrapy spider
├── pipelines/ # Processing pipelines
├── logs/ # Logs (gitignored)
├── jobs/ # JOBDIR for paused crawls (gitignored)
├── environment_wikiscrap.yml # Conda env for crawler & data processing
├── environment_openwebui.yml # Conda env for running Open WebUI
├── README.md
└── .gitignore
```

## 🚀 How to run

Clone this repo:
```bash
git clone https://github.com/alexretana/MonsterHunterWildsWikiLLM.git
cd MonsterHunterWildsWikiLLM
git submodule update --init --recursive
```

Create the Conda environment:

```bash
conda env create -f environment_wikiscrap.yml
conda activate wikiscrap
```

Run the spider:

```bash
scrapy crawl myfextralifespider
```
To pause and resume (recommended for large wikis):

```bash
scrapy crawl myfextralifespider -s JOBDIR=jobs/fextralife-2025-08-02
```

📝 Crawled pages, logs, and estimation stats will be saved in logs/ and jobs/ (both ignored by git).

## 📊 How it works
Each crawled URL is tracked in url_counter.

Every 10 pages (configurable), the spider estimates:

total unique pages still undiscovered (via Chao1)

% coverage of the wiki
These stats are logged & stored in Scrapy stats.

The final dataset will later be cleaned and uploaded to Open WebUI as a vectorized knowledge base for question answering.


## 🧰 Environments
environment_wikiscrap.yml: main environment (Python, Scrapy, pandas, JupyterLab, etc.)

environment_openwebui.yml: runs Open WebUI for chatbot interface.

Recreate them anytime:

```bash
conda env create -f environment_wikiscrap.yml
```

📚 Next steps
✅ Extend spider to other wiki sites (3 total planned).
✅ Clean & normalize HTML → Markdown / JSON.
✅ Create an ingest script to upload data to Open WebUI knowledge base.
✅ Build a web interface for querying.

## 📝 License
MIT (add if you want) – free to use & adapt.

Built by Alexander Retana to explore local RAG pipelines for game wikis. 🎮

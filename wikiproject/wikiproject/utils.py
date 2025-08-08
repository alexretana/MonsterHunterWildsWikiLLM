import pandas as pd
from tqdm import tqdm
import requests
import json
import os
from rich.tree import Tree

# Chroma/LlamaIndex imports
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.ollama import OllamaEmbedding

OPEN_WEBUI_DOMAIN_NAME = 'http://localhost'
_API_KEY = None

DOCUMENTS_DIR = "./output/documents"

def read_dot_api_key():
    global _API_KEY
    if _API_KEY is None:
        # Assuming your API key is in a file named api_key.txt
        try:
            with open('.open_webui_api_key', 'r') as f:
                _API_KEY = f.read().strip()
            if _API_KEY is None:
                raise ValueError("API key not found in file or environment.")
        except FileNotFoundError:
            raise FileNotFoundError("'.open_webui_api_key' file not found.")
    return _API_KEY

def make_headers(extra_headers: dict = None):
    api_key = read_dot_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers

def print_response_error(response):
    error_message = f"Recieved Non-Successful Status Code({response.status_code}), and message :{response.text}"
    print(error_message)
    return error_message

def purge_openwebui():
    """Optional function to purge all files from OpenWebUI"""
    full_url = OPEN_WEBUI_DOMAIN_NAME + ":8080/api/v1/files/all"
    headers = make_headers()
    response = requests.delete(url=full_url, headers=headers)
    if response.status_code not in range(200, 299):
        return print_response_error(response)
    print("Successfully purged all files from OpenWebUI")
    return True

def upsert_into_chroma(df):
    """
    Upserts DataFrame content into Chroma vector store.
    Returns nothing; the store is now persistent.
    """
    print("Starting Chroma ingestion...")
    
    # Initialize embedding model and vector store
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
    chroma_store = ChromaVectorStore(persist_dir="./chroma_db")
    
    # Convert DataFrame rows to LlamaIndex Documents
    documents = []
    for _, row in df.iterrows():
        doc = Document(
            text=row["page_text"], 
            metadata={"url": row["url"]}
        )
        documents.append(doc)
    
    print(f"Creating vector index with {len(documents)} documents...")
    
    # Create index from documents
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        vector_store=chroma_store,
        show_progress=True,
    )
    
    # Persist the storage context
    index.storage_context.persist()
    print(f"Successfully ingested {len(documents)} documents into Chroma vector store")
    
def get_knowledge_list():
    # Check if collections already exists
    full_url = OPEN_WEBUI_DOMAIN_NAME + ":8080/api/v1/knowledge/list"
    headers = make_headers()
    response = requests.get(url=full_url, headers=headers)
    # Exit early if api call fails
    if response.status_code not in range(200,299):
        return print_response_error(response)

    # Parse response for list of exiting knowledges. See which are missing
    response_json = response.json()
    return response_json


def dedupe_and_build_breadcrumb_map():
    # read in data, dedupe, rewrite
    filename = './output/fextralife-monsterhunterwildswiki.jsonl'
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    outputDf = pd.DataFrame(data).drop_duplicates(subset=['url'], keep='last')

    # Ingest into Chroma instead of uploading to OpenWebUI
    upsert_into_chroma(outputDf)

    # Rewrite after deduping 
    outputDf.to_json(filename, orient='records', lines=True)

    # Build tree and total page count
    breadCrumbs = outputDf['breadcrumb&title']
    tree = Tree("monsterhunterwilds.wiki.fextralife.com:root")
    nodes = {}
    for breadCrumb in breadCrumbs:
        parts = [p for p in breadCrumb.split('/') if p]
        parent = tree
        partial = ""
        for part in parts:
            partial += "/" + part
            if partial not in nodes:
                nodes[partial] = parent.add(part)
            parent = nodes[partial]
    total_page_count = len(outputDf)
    return tree, total_page_count

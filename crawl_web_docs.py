import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_openai import ChatOpenAI
from openai import AsyncAzureOpenAI
from langchain_qdrant import QdrantVectorStore
from uuid import uuid4
from langchain_core.documents import Document
from qdrant_client import QdrantClient


load_dotenv()
pyansys_module = os.getenv("PYANSYS_MODULE")

# NVIDIA NIM LLM Model
# chat_model = ChatOpenAI(
#     model=os.getenv("LLM_MODEL"),
#     api_key=os.getenv("NGC_API_KEY"),
#     base_url="https://integrate.api.nvidia.com/v1",
#     temperature=0.0,
# )

# Azure LLM
openai_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
    api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# NVIDIA Embedding model
embed_model = NVIDIAEmbeddings(
    base_url="https://integrate.api.nvidia.com/v1", 
    model=os.getenv('EMBEDDING_MODEL'),
    api_key=os.getenv("NGC_API_KEY"),
)

# Qdrant Client
qdrant_client = QdrantClient(path="langchain_qdrant")

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]

def chunk_text(text: str, chunk_size: int = 8000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using gpt."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "o4-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}


# Not required for qdrant
# async def get_embedding(text: str) -> List[float]:
#     """Get embedding vector from OpenAI."""
#     try:
        
#         # Nvidia NIM Embedding Client
#         response = embed_model.embed_query(
#             text=text
#         )
        
#         return response.data[0].embedding
#     except Exception as e:
#         print(f"Error getting embedding: {e}")
#         return [0] * 2048

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding - not required for qdrant
    #embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": f"{pyansys_module}_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        document = [Document(
            page_content= chunk.content,
            metadata={
                "url": chunk.url,
                "chunk_number": chunk.chunk_number,
                "title": chunk.title,
                "summary": chunk.summary,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
        )]
        uuid = [str(uuid4())]
        
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=f"{pyansys_module}",
            embedding=embed_model,
        )
        
        result = vector_store.add_documents(documents=document, ids=uuid)
        
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

PROCESSED_URLS_FILE = "processed_urls.json"

def load_processed_urls() -> set:
    if os.path.exists(PROCESSED_URLS_FILE):
        with open(PROCESSED_URLS_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_url(url: str):
    processed = load_processed_urls()
    processed.add(url)
    with open(PROCESSED_URLS_FILE, "w") as f:
        json.dump(list(processed), f)


async def crawl_parallel(urls: List[str], max_concurrent: int = 15):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    processed_urls = load_processed_urls()
    urls_to_process = [url for url in urls if url not in processed_urls]
    
    if not urls_to_process:
        print("All URLs have already been processed.")
        return
    
    # Replace with your actual cookies obtained from the browser after logging in
    github_sso_cookies = [
        {"name": "__Host-gh_pages_id", "value": "31249389", "domain": "heart.docs.pyansys.com", "path": "/", "httpOnly": True, "secure": True},
        {"name": "__Host-gh_pages_session", "value": "373e7a73-dfb8-4911-baae-ca3e0b4b52d8", "domain": "heart.docs.pyansys.com", "path": "/", "httpOnly": True, "secure": True},
        {"name": "__Host-gh_pages_token", "value": "GHSAT0AAAAAAC43JF6G4SNWNUW3JCMDY2ZUZ5CNJEA", "domain": "heart.docs.pyansys.com", "path": "/", "httpOnly": True, "secure": True},
        {"name": "_ga", "value": "GA1.1.188032431.1737560099", "domain": ".pyansys.com", "path": "/"},
        {"name": "_ga_JQJKPV6ZVB", "value": "GS1.1.1738049374.4.0.1738049374.0.0.0", "domain": ".pyansys.com", "path": "/"},
        # Add all necessary cookies here
    ]
    
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        #extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        extra_args=["--no-sandbox"],
        cookies=github_sso_cookies
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                    save_processed_url(url)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_pyansys_docs_urls() -> List[str]:
    """Get URLs from PyAnsys Module sitemap."""
    sitemap_url = os.getenv("MODULE_SITEMAP_URL")  # Update with the actual URL to your sitemap.xml
    local_sitemap_path = os.getenv("LOCAL_SITEMAP_FILE")  # Update with the actual path to your local sitemap.xml

    # Replace with your actual cookies obtained from the browser after logging in
    github_sso_cookies = {
        "__Host-gh_pages_id": "31249389",
        "__Host-gh_pages_session": "373e7a73-dfb8-4911-baae-ca3e0b4b52d8",
        "__Host-gh_pages_token": "GHSAT0AAAAAAC43JF6G4SNWNUW3JCMDY2ZUZ5CNJEA",
        "_ga": "GA1.1.188032431.1737560099",
        "_ga_JQJKPV6ZVB": "GS1.1.1738049374.4.0.1738049374.0.0.0",
        # Add all necessary cookies here
    }

    try:
        response = requests.get(sitemap_url, cookies=github_sso_cookies)
        response.raise_for_status()
        
        # Parse the XML from the online source
        root = ElementTree.fromstring(response.content)
    except Exception as e:
        print(f"Error fetching sitemap from online source: {e}\nTrying to load local sitemap...")
        try:
            if os.path.exists(local_sitemap_path):
                with open(local_sitemap_path, 'r') as file:
                    root = ElementTree.parse(file).getroot()
            else:
                print(f"Local sitemap file not found: {local_sitemap_path}")
                return []
        except Exception as e:
            print(f"Error loading local sitemap: {e}")
            return []

    # Extract all URLs from the sitemap
    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
    
    return urls

async def main():
    # Get URLs from PyAnsys Module docs
    urls = get_pyansys_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())

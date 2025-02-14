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
from langchain_qdrant import QdrantVectorStore
from uuid import uuid4
from langchain_core.documents import Document
from qdrant_client import QdrantClient


load_dotenv()
collection_module = os.getenv("COLLECTION_MODULE")

# LLM Model
chat_model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("NGC_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.0,
)

# Embedding model
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
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        # Nvidia NIM CLient
        messages=[
                ("system", system_prompt),
                ("user", f"URL: {url}\n\nContent:\n{chunk[:1000]}..."),  # Send first 1000 chars for context
            ]
        json_model = chat_model.bind(response_format={ "type": "json_object" })
        # Async call - currently not supported by Nvidia NIM endpoint, try it with self-hosted endpoint later
        # response = await json_model.ainvoke(messages)
        response = json_model.invoke(messages)
        
        return json.loads(response.content)
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
        "source": f"{collection_module}_docs",
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
            collection_name=f"{collection_module}",
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

async def crawl_parallel(urls: List[str], max_concurrent: int = 15):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    
    # Replace with your actual cookies obtained from the browser after logging in
    github_sso_cookies = [
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
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_module_docs_urls() -> List[str]:
    """Get URLs from Module sitemap."""
    sitemap_url = os.getenv("MODULE_SITEMAP_URL")  # Update with the actual URL to your sitemap.xml
    local_sitemap_path = os.getenv("LOCAL_SITEMAP_FILE")  # Update with the actual path to your local sitemap.xml

    # Replace with your actual cookies obtained from the browser after logging in
    github_sso_cookies = {
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
    # Get URLs from Module docs
    urls = get_module_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())

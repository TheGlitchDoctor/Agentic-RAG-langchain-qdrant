import os
import json
import asyncio

from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import asyncio
from typing import List
import aiofiles
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from uuid import uuid4
from langchain_core.documents import Document
from qdrant_client import QdrantClient

load_dotenv()
pyansys_module = os.getenv("PYANSYS_MODULE")

# LLM Model
chat_model = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("NGC_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.5,
    max_retries=2,
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
    filename: str
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

async def get_title_and_summary(chunk: str, filename: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from file chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title from the heading or QUESTION if available. If it's a middle chunk, either extract heading/QUESTION or derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative. Also remove any special characters from the title.
    Treat patient_name as dynamic, change it in the title and summary to be generic."""
    
    try:
        # Nvidia NIM CLient
        messages=[
                ("system", system_prompt),
                ("user", f"Filename: {filename}\n\nContent:\n{chunk[:1000]}..."),  # Send first 1000 chars for context
            ]
        json_model = chat_model.bind(response_format={ "type": "json_object" })
        # Async call - currently not supported by Nvidia NIM endpoint, try it with self-hosted endpoint later
        # response = await json_model.ainvoke(messages)
        response = json_model.invoke(messages)
        print(response)
        
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

async def process_chunk(chunk: str, chunk_number: int, file_path: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    filename = file_path.split("/")[-1]
    extracted = await get_title_and_summary(chunk, filename)
    
    # Get embedding - not required for qdrant
    #embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": f"{pyansys_module}_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "filename": filename
    }
    
    return ProcessedChunk(
        filename=filename,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        #print(chunk.content)
        document = [Document(
            page_content= chunk.content,
            metadata={
                "filename": chunk.filename,
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
        
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.filename}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None
    
async def process_file(file_path: str):
    """Process a single file."""
    async with aiofiles.open(file_path, 'r') as f:
        content = await f.read()
    chunks = chunk_text(content)
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, file_path) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(files: str, max_concurrent: int = 15):
    """Process multiple files in parallel with a concurrency limit."""
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(file_path):
        async with semaphore:
            return await process_file(file_path)
    
    tasks = [process_with_semaphore(file) for file in files]
    await asyncio.gather(*tasks)
    

async def list_files_in_folder(folder_path: str) -> List[str]:
    """List all files in the given folder path."""
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(('.py', '.pdf', '.json', '.toml', '.rst', '.html')):
                files.append(os.path.join(root, filename))
    return files

async def main():
    # Get files from PyAnsys local path
    folder_path = os.getenv("LOCAL_DATA_PATH")
    files = await list_files_in_folder(folder_path)
    if not files:
        print("No files found to embed")
        return
    
    print(f"Found {len(files)} files to embed..")
    await crawl_parallel(files, max_concurrent=15)

if __name__ == "__main__":
    asyncio.run(main())

from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
import httpx
import os

from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent
from langchain_community.llms import OpenAI as LangChainOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchText
from typing import List


load_dotenv()
collection_module = os.getenv("COLLECTION_MODULE")


# Embedding model
embed_model = NVIDIAEmbeddings(
    base_url="https://integrate.api.nvidia.com/v1", 
    model=os.getenv('EMBEDDING_MODEL'),
    #model="baai/bge-m3",
    api_key=os.getenv("NGC_API_KEY"),
)

# Qdrant Client
qdrant_client = QdrantClient(path="langchain_qdrant")
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=f"{collection_module}",
    embedding=embed_model,
)

async def list_documentation_pages() -> List[str]:
    """
    Retrieve a list of all available module documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # result = supabase.from_(f'{collection_module}_site_pages') \
        #     .select('url') \
        #     .eq('metadata->>source', f'{collection_module}_docs') \
        #     .execute()
        
        url_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.url",
                    match=MatchText(text="{url-here}")
                )
            ]
        )
        result = qdrant_client.scroll(collection_name=f"{collection_module}", scroll_filter=url_filter, with_payload=True)
        
        #print(result)
        # if not result.data:
        if len(result.points) == 0:
            return []
            
        urls = sorted(set(doc['url'] for doc in result.points))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


async def get_page_content(url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # result = supabase.from_(f'{collection_module}_site_pages') \
        #     .select('title, content, chunk_number') \
        #     .eq('url', url) \
        #     .eq('metadata->>source', f'{collection_module}_docs') \
        #     .order('chunk_number') \
        #     .execute()
        
        content_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.url",
                    match=MatchText(text=f"{url}")
                )
            ]
        )
        result = qdrant_client.scroll(collection_name=f"{collection_module}", scroll_filter=content_filter, order_by="metadata.chunk_number", with_payload=True)
        #print(result)
        # if not result.data:
        if len(result.points) == 0:
            return f"No content found for URL: {url}"
            
        page_title = result.points[0]['metadata']['title']
        formatted_content = [f"# {page_title}\n"]
        
        for chunk in result.points:
            formatted_content.append(chunk['page_content'])
            
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

tools = [
    StructuredTool.from_function( 
        coroutine=list_documentation_pages, 
        description="Retrieve a list of all available module documentation pages."
    ),
    StructuredTool.from_function(
        coroutine=get_page_content, 
        description="Retrieve the full content of a specific documentation page by combining all its chunks."
    )
]

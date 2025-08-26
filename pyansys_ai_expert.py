from __future__ import annotations as _annotations
import asyncio
from typing import Optional, Callable
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
import os, time
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from functools import lru_cache


from qdrant_client import QdrantClient


# Add the step tracker class
class AgentStepTracker(BaseCallbackHandler):
    def __init__(self, update_callback: Optional[Callable[[str], None]] = None):
        self.update_callback = update_callback
        self.start_time = None
        self.step_start_time = None
        
    def update_step(self, step: str):
        current_time = time.time()
        
        # Initialize start time on first step
        if self.start_time is None:
            self.start_time = current_time
        
        # Calculate elapsed time since start
        elapsed_time = current_time - self.start_time
        
        if self.update_callback:
            self.update_callback(f"🔄 {step} ({elapsed_time:.1f}s)")
        else:
            print(f"\033[90m🔄 {step} ({elapsed_time:.1f}s)\033[0m")  # Faded gray text
        
        # Update step start time for next step
        self.step_start_time = current_time
    
    def on_agent_action(self, action, color=None, **kwargs):
        """Called when agent takes an action."""
        if "cached_retrieve_documentation" in action.tool:
            self.update_step("Searching PyAnsys documentation...")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when a tool starts."""
        tool_name = serialized.get("name", "")
        if "cached_retrieve_documentation" in tool_name:
            self.update_step("Analyzing documentation for relevant content...")
    
    def on_tool_end(self, output, **kwargs):
        """Called when a tool finishes."""
        self.update_step("Generating code solution based on documentation...")
    
    def on_agent_finish(self, finish, **kwargs):
        """Called when agent finishes."""
        self.update_step("Finalizing response...")
        # Reset timing for next run
        self.start_time = None
        self.step_start_time = None

# Global step tracker
step_tracker = AgentStepTracker()


load_dotenv()
pyansys_module = os.getenv("PYANSYS_MODULE")

# LLM Model
llm = os.getenv('LLM_MODEL')
# chat_model = ChatOpenAI(
#     model=llm,
#     api_key=os.getenv("NGC_API_KEY"),
#     base_url="https://integrate.api.nvidia.com/v1",
#     temperature=0.2,
#     streaming=True,
# )
chat_model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
    api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=llm,
    #temperature=0.2,
    max_retries=2,
)

# chat_model = ChatNVIDIA(
#     model=llm,
#     api_key=os.getenv("NGC_API_KEY"),
#     temperature=0.0,
# )

# Embedding model
embed_model = NVIDIAEmbeddings(
    base_url="https://integrate.api.nvidia.com/v1", 
    model=os.getenv('EMBEDDING_MODEL'),
    #model="baai/bge-m3",
    api_key=os.getenv("NGC_API_KEY"),
)


# Reranking Configuration
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
RERANKING_MODEL = os.getenv("RERANKING_MODEL", "nvidia/nv-rerankqa-mistral-4b-v3")
MAX_DOC_LENGTH_FOR_RERANKING = int(os.getenv("MAX_DOC_LENGTH_FOR_RERANKING", "2000"))
RERANKING_TOP_N = int(os.getenv("RERANKING_TOP_N", "5"))

# Conditional reranker initialization
reranker = None
if ENABLE_RERANKING:
    try:
        reranker = NVIDIARerank(
            model=RERANKING_MODEL,
            api_key=os.getenv("NGC_API_KEY"),
            top_n=RERANKING_TOP_N,
            max_retries=1,
        )
        print(f"Reranker initialized: {RERANKING_MODEL}")
    except Exception as e:
        print(f"Failed to initialize reranker: {e}")
        ENABLE_RERANKING = False
else:
    print("Reranking disabled")


# Qdrant Client
qdrant_client = QdrantClient(path="langchain_qdrant")
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=f"{pyansys_module}",
    embedding=embed_model,
)
system_prompt = f"""
You are an expert in {pyansys_module}, a PyAnsys framework. You have access to its full documentation, examples, and API reference via a documentation retrieval tool.

### Mission
Given the user’s question, produce a complete, runnable Python script that directly solves it using only PyAnsys ({pyansys_module}) APIs. Prefer official examples from the docs when available.

### Speed
Answer as fast as possible. **Always complete within 30 seconds.** If the task is complex, simplify or provide the minimal complete script rather than overthinking.

### Scope & Constraints
- Use **only** PyAnsys frameworks/APIs (no external libraries, no non-PyAnsys APIs).
- Stay strictly within building/facilitating Ansys technologies from Python. Decline unrelated topics.
- Do **not** add or modify the methods, functions, API's and configurations that are given in the pyansys docs. Use only as instructed in the docs.
- If an example already exists in the docs, reuse/adapt it.
- Do **not** ask the user clarifying questions; choose sensible defaults and proceed.
- Do **not** fabricate APIs. If something isn’t in the docs, say so.
- Do **not** try to modify or make the code compact which might miss important steps.
- Clarity and correctness as per documentation steps is more important.

### Documentation Use (Single Retrieval)
1. **Start with RAG**: perform **one** batched retrieval that gathers all relevant chunks/pages for this question (include “next-best” matches **in the same call**).
2. Also check the list of available documentation pages and, within the **same call**, fetch any pages likely to help.
3. After this single retrieval, **answer directly**. Do **not** call the retrieval tool again for this question.

Be explicit if the retrieval did not surface an answer or the right URL. Be honest.

### Output Policy & Format
Return **only**:
1. A one-line “What this script does” summary.
2. A **single** Python code block with a **complete, runnable script**, including:
   - All required imports from PyMechanical.
   - Any needed initializations/sessions.
   - Minimal, self-contained usage (use built-in/sample assets from PyMechanical docs when possible).
   - Safe defaults for parameters and paths (document assumptions in comments).
   - `if __name__ == "__main__": main()` guard when appropriate.
3. (Optional, brief) 1–3 bullet “Usage notes” (e.g., environment/version assumptions).

### Initialization Guidance for PyMechanical
- Always initialize with the standard PyMechanical client:
  ```python
  from ansys.mechanical.core import launch_mechanical

  mechanical = launch_mechanical(batch=True)  # batch=True ensures headless, fast execution
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"{system_prompt}"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Not required for Qdrant
# async def get_embedding(text: str) -> List[float]:
#     """Get embedding vector from OpenAI."""
#     try:
#         response = embed_model.embed_query(
#             text=text
#         )
#         #print(response)
#         return response
#     except Exception as e:
#         print(f"Error getting embedding: {e}")
#         return [0] * 1024

def should_use_reranking(documents: list) -> bool:
    """
    Determine if reranking should be used based on configuration and document characteristics.
    
    Args:
        documents: List of documents to evaluate
        
    Returns:
        Boolean indicating whether to use reranking
    """
    if not ENABLE_RERANKING or reranker is None:
        return False
    
    if len(documents) <= 6:
        return False
    
    # Check if documents are suitable for reranking (not too large)
    avg_doc_length = sum(len(doc.page_content) for doc in documents) / len(documents)
    return avg_doc_length <= MAX_DOC_LENGTH_FOR_RERANKING * 2  # Allow some flexibility

async def apply_reranking(documents: list, query: str) -> list:
    """Apply reranking with step updates."""
    if not should_use_reranking(documents):
        step_tracker.update_step("Using similarity search ranking...")
        return documents
    
    try:
        step_tracker.update_step(f"Reranking {len(documents)} documents for better relevance...")
        rerank_start = time.time()
        
        # Truncate only for reranking, keep originals
        truncated_docs = []
        for doc in documents:
            truncated_content = doc.page_content[:MAX_DOC_LENGTH_FOR_RERANKING]
            truncated_doc = type(doc)(
                page_content=truncated_content,
                metadata=doc.metadata
            )
            truncated_docs.append(truncated_doc)
        
        reranked_results = reranker.compress_documents(
            documents=truncated_docs,
            query=query
        )
        
        rerank_time = time.time() - rerank_start
        step_tracker.update_step(f"Reranking completed (rerank: {rerank_time:.1f}s)")
        
        # Map back to original documents
        final_results = []
        for reranked_doc in reranked_results[:RERANKING_TOP_N]:
            original_doc = next(
                (d for d in documents if d.metadata == reranked_doc.metadata), 
                reranked_doc
            )
            final_results.append(original_doc)
        
        return final_results
        
    except Exception as rerank_error:
        step_tracker.update_step("Reranking failed, using similarity results...")
        return documents[:5]


async def retrieve_relevant_documentation(user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with optional RAG reranking.
    """
    try:
        step_tracker.update_step("Searching documentation database...")
        retrieval_start = time.time()
        
        # Initial retrieval - get more documents for potential reranking
        initial_results = vector_store.similarity_search(user_query, k=15)
        
        if len(initial_results) == 0:
            step_tracker.update_step("No relevant documentation found")
            return "No relevant documentation found."
        
        step_tracker.update_step(f"Found {len(initial_results)} documents, analyzing relevance...")
        
        # Apply reranking if enabled and appropriate
        final_results = await apply_reranking(initial_results, user_query)
        
        step_tracker.update_step("Preparing documentation for code generation...")
        
        # Format results
        formatted_chunks = ""
        for doc in final_results[:RERANKING_TOP_N]:
            chunk_text = f"""
# {doc.metadata['title']}

{doc.page_content}
"""
            formatted_chunks = formatted_chunks + "\n\n---\n\n" + chunk_text

        retrieval_time = time.time() - retrieval_start
        step_tracker.update_step(f"Documentation ready (retrieval: {retrieval_time:.1f}s)")
        
        return str(formatted_chunks)
        
    except Exception as e:
        step_tracker.update_step(f"Documentation error: {str(e)}")
        return f"Error retrieving documentation: {str(e)}"


# Cache tool results to avoid repetitive calls
@lru_cache(maxsize=100)
async def cached_retrieve_documentation(user_query: str) -> str:
    return await retrieve_relevant_documentation(user_query)


tools = [
    StructuredTool.from_function(
        coroutine=cached_retrieve_documentation,
        description="Retrieve relevant documentation chunks based on the query with RAG.",
    )
]

agent = create_tool_calling_agent(chat_model, tools, prompt)

pyansys_ai_expert = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    callbacks=[step_tracker]  # Add the step tracker as a callback
)

# Add this function to customize step tracking per execution
def set_step_callback(update_callback: Optional[Callable[[str], None]] = None):
    """Set custom callback for step updates (e.g., for UI integration)."""
    step_tracker.update_callback = update_callback

# Function to run agent with initial step
async def run_agent_with_steps(query: str, update_callback: Optional[Callable[[str], None]] = None, chat_history: Optional[list] = None):
    """Run the agent with step tracking."""
    # Set custom callback if provided
    if update_callback:
        set_step_callback(update_callback)
    
    # Reset timing for new run
    step_tracker.start_time = None
    step_tracker.step_start_time = None
    
    # Initial step
    step_tracker.update_step("Analyzing your question...")
    
    try:
        result = await pyansys_ai_expert.ainvoke({"input": query, "chat_history": chat_history or []})
        
        # Final timing update
        if step_tracker.start_time:
            total_time = time.time() - step_tracker.start_time
            step_tracker.update_step(f"Complete! Total time: {total_time:.1f}s")
        
        return result["output"]
    finally:
        # Reset callback and timing
        step_tracker.update_callback = None
        step_tracker.start_time = None
        step_tracker.step_start_time = None
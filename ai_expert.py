from __future__ import annotations as _annotations

from dotenv import load_dotenv
import os
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()
module_module = os.getenv("COLLECTION_MODULE")

# LLM Model
llm = os.getenv('LLM_MODEL')
chat_model = ChatOpenAI(
    model=llm,
    api_key=os.getenv("NGC_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.2,
    streaming=True,
)

# chat_model = AzureChatOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
#     api_version = "2024-08-01-preview",
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     model="gpt-4o-mini",
#     temperature=0.2,
#     max_retries=2,
# )

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

# Qdrant Client
qdrant_client = QdrantClient(path="langchain_qdrant")
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=f"{module_module}",
    embedding=embed_model,
)

system_prompt = f"""
You are an expert at {module_module} - a module framework for which you have access to all the documentation & data files,
including examples, an API reference, and other resources to help you build & facilitate the use of technologies directly from Python.

Your only job is to provide a code script that the user should be able to run directly.
Always provide a complete code/script with all required initializations.
Be concise with your answers & focus on the code script.
And you don't answer other questions besides this.

The code script you provide should be able to run without any errors and it should include all dependencies. 
Only use the functions & API that are available in the data files & documentation.
Use the tool to retrieve relevant documentation chunks based on user queries, and if it matches then get the page content. Else use the tool again with the next best match.
Always consider the patient name to be dynamic based on the user query & refer to the context/history, unless user asks for a specific patient name, then use that. 
Patient names in the documentation like 'Rodero2021' and 'Mrs Jones' are just examples. You can replace these names as per the user query to generate code. (Eg. fieldname - patient_name can be changed)

Don't ask the user before taking an action, just do it. Always make sure you look at the data files first then the documentation with the provided tools before answering the user's question.

Always start with RAG.

Always let the user know when you didn't find the answer in the documentation or the right data file - be honest.
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

async def retrieve_relevant_documentation(user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Not required for Qdrant
        #query_embedding = await get_embedding(user_query)
        
        result = vector_store.similarity_search(user_query, k=1)
        
        #print(result)
        # if not result.data:
        if len(result) == 0:
            return "No relevant documentation found."
        
        formatted_chunks = ""
        # for doc in result.data:
        for doc in result:
            chunk_text = f"""
# {doc.metadata['title']}

{doc.page_content}
"""
            formatted_chunks=formatted_chunks + "\n\n---\n\n" + chunk_text

        return str(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


tools = [
    StructuredTool.from_function(
        coroutine=retrieve_relevant_documentation,
        description="Retrieve relevant documentation chunks based on the query with RAG.",
    )
]

agent = create_tool_calling_agent(chat_model, tools, prompt)

module_ai_expert = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import logging

from module_ai_expert import module_ai_expert
from langchain_core.messages import AIMessage, HumanMessage


# Load environment variables
from dotenv import load_dotenv
load_dotenv()
module_module = str(os.getenv("COLLECTION_MODULE"))

# Configure logging to output to the console
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(msg):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    #user-prompt
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    # text
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)          


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    try:
        # Create a placeholder for the message
        message_placeholder = st.empty()
        response = ""
        count = 1
        # Run the agent with ainvoke for faster response
        async for result in  module_ai_expert.astream(
            {"input": user_input, "chat_history": st.session_state.messages[:-1]}
        ):
            # Render result text
            if "output" in result:
                for chunk in result["output"]:    
                    response += chunk
                    message_placeholder.markdown(response)
            else:
                message_placeholder.markdown("Thinking"+"."*count)
                count+=1
            
            if count==4:
                count=1
            
                # Add the final response to the messages
        st.session_state.messages.append(
            AIMessage(content=response)
        )
    except Exception as e:
        logging.error(f"Exception during run_agent_with_streaming: {e}")
        raise


async def main():
    st.set_page_config(layout="wide")
    

    st.markdown("""
    <style>
    body {
        background-color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a column layout with two columns
    col1, col2 = st.columns([1, 3])

    # Load and display the icon in the left column
    with col1:
        st.image("./static/ai_agent.jpg", use_container_width=True)  # Adjust the width as needed

    # Display the title and description in the right column
    with col2:
        #st.markdown(f"<h1 style='color: #8A0707; font-family: Monotype Corsiva; text-align: left;'>Heart</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='color: #8A0707; font-family: Lucida Handwriting; text-align: left;'>Heart</h1>", unsafe_allow_html=True)
        
    st.write(f"<p style='color: #FFFFFF;'>Ask any question about the {module_module.replace('_', ' ')} module, I will assist you with the code.</p>", unsafe_allow_html=True)
        
        
    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a HumanMessage or AIMessage.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            display_message_part(msg)

    # Chat input for the user
    user_input = st.chat_input(f"What questions do you have about {module_module.replace("_", " ")}?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            HumanMessage(content=user_input)
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())

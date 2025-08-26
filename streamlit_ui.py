from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import logging

from pyansys_ai_expert import pyansys_ai_expert, run_agent_with_steps
from langchain_core.messages import AIMessage, HumanMessage


# Load environment variables
from dotenv import load_dotenv
load_dotenv()
pyansys_module = str(os.getenv("PYANSYS_MODULE"))

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

def streamlit_update(step_text):
    """Update function for step tracking in Streamlit."""
    # This will be set dynamically in the streaming function
    if hasattr(st.session_state, 'step_placeholder') and st.session_state.step_placeholder:
        st.session_state.step_placeholder.markdown(f":gray[{step_text}]")


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    try:
        # Create placeholders for step tracking and response
        step_placeholder = st.empty()
        message_placeholder = st.empty()
        
        # Store step placeholder in session state for the callback
        st.session_state.step_placeholder = step_placeholder
        
        response = ""
        
        # Use the new run_agent_with_steps function with step tracking
        response = await run_agent_with_steps(
            user_input, 
            update_callback=streamlit_update,
            chat_history=st.session_state.messages[:-1]
        )
        
        # Clear the step tracking placeholder and show final response
        step_placeholder.empty()
        message_placeholder.markdown(response)
                
        # Add the final response to the messages
        st.session_state.messages.append(
            AIMessage(content=response)
        )
        
        # Clean up
        if hasattr(st.session_state, 'step_placeholder'):
            del st.session_state.step_placeholder
            
    except Exception as e:
        logging.error(f"Exception during run_agent_with_streaming: {e}")
        # Clear step placeholder on error
        if hasattr(st.session_state, 'step_placeholder'):
            st.session_state.step_placeholder.empty()
            del st.session_state.step_placeholder
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
        st.image("./static/pyansys_dark.png", use_container_width=True)  # Adjust the width as needed

    # Display the title and description in the right column
    with col2:
        #st.markdown(f"<h1 style='color: #8A0707; font-family: Monotype Corsiva; text-align: left;'>Heart</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='color: #FFFFFF; font-family: Lucida Handwriting; text-align: left;'>Mechanical</h1>", unsafe_allow_html=True)
        
    st.write(f"<p style='color: #FFFFFF;'>Ask any question about the {pyansys_module.replace('_', ' ')} module, I will assist you with the code.</p>", unsafe_allow_html=True)
        
        
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
    user_input = st.chat_input(f"What questions do you have about {pyansys_module.replace('_', ' ')}?")

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

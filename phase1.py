import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.runnable import Runnable
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class StrOutputParser(Runnable):
    def invoke(self, input_str, config=None):
        # Handle dictionary input
        if isinstance(input_str, dict):
            input_str = str(list(input_str.values())[0])
        
        # Convert the input to string and extract the content
        input_str = str(input_str)
        
        # Clean up the response
        if 'content=' in input_str:
            # Extract content between content=" and the next quote
            start = input_str.find('content="') + 9
            end = input_str.find('"', start)
            if end != -1:
                input_str = input_str[start:end]
        
        # Remove the thinking process if present
        if '<think>' in input_str:
            start = input_str.find('</think>') + 8
            input_str = input_str[start:].strip()
        
        # Get the actual content from AIMessage if present
        if hasattr(input_str, 'content'):
            input_str = input_str.content
        
        # Clean up newlines, ### markers, and convert escaped newlines to actual newlines
        input_str = input_str.replace('\\n', '\n')
        lines = input_str.splitlines()
        cleaned_lines = [line.lstrip('###').strip() for line in lines if line.strip()]
        input_str = ' '.join(cleaned_lines)
            
        return input_str.strip()
    
    def parse(self, input_str):
        return str(input_str)

st.title("RAG Chatbot!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Pass your prompt here!")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best, the most accurate and the most
                                                      precise answers. Answer the following Question: {user_input}.
                                                      Start the answer directly. No small talk please.""")

    model = "deepseek-r1-distill-llama-70b"  # Using the deepseek model as requested
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
        st.stop()
        
    groq_chat = ChatGroq(
        groq_api_key=api_key,
        model_name=model
    )

    chain = groq_sys_prompt | groq_chat | StrOutputParser()
    response = chain.invoke({"user_input": prompt})

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

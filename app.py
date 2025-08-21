import streamlit as st
import asyncio
import nest_asyncio

# Event loop muammosini hal qilish
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()

from langchain_helper import get_qa_chain, create_vector_db

st.title("Codebasics Q&A 🌱")

btn = st.button("Create Knowledgebase")

if btn:
    create_vector_db()
    
messages = st.container(height=200)

if prompt := st.chat_input("Ask your question"):
    messages.chat_message("user").write(prompt)
    chain = get_qa_chain()
    response = chain(prompt)
    messages.chat_message("assistant").write(response["result"])
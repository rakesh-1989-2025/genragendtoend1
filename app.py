import streamlit as st

from pinecone import Pinecone

from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

import requests
import os

load_dotenv()

st.title("Bihar GenAI RAG Chatbot

Welcome to the Bihar GenAI RAG Chatbot!

This chatbot is designed to answer questions only about Bihar. You can ask about:

Bihar's history
Culture and traditions
Tourist attractions
Districts and cities
Education and universities
Government schemes
Economy and industries
Festivals and famous personalities
Geography and demographics
Important Note

✅ The chatbot will provide answers only for Bihar-related questions.")

pc=Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index=pc.Index("textgenai")

embed_model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

question=st.text_input(
    "Ask Question"
)

if st.button("Submit"):

    q_emb=embed_model.encode(question).tolist()

    result=index.query(
        vector=q_emb,
        top_k=3,
        include_metadata=True
    )

    context=""

    for match in result["matches"]:
        context += match["metadata"]["text"] + "\n"

    headers={
        "Authorization":f"Bearer {os.getenv('AI21_API_KEY')}",
        "Content-Type":"application/json"
    }

    payload={
        "model":"jamba-large",
        "messages":[
            {
                "role":"user",
                "content":f"""
                Context:
                {context}

                Question:
                {question}

                Answer:
                """
            }
        ]
    }

    response=requests.post(
        "https://api.ai21.com/studio/v1/chat/completions",
        headers=headers,
        json=payload
    )

    answer=response.json()["choices"][0]["message"]["content"]

    st.write(answer)
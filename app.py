import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
import os

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("Bihar GenAI RAG Chatbot")

st.markdown("""
### Welcome to the Bihar GenAI RAG Chatbot!

This chatbot is designed to answer questions **only about Bihar**.

You can ask about:

- Bihar's History
- Culture and Traditions
- Tourist Attractions
- Districts and Cities
- Education and Universities
- Government Schemes
- Economy and Industries
- Festivals and Famous Personalities
- Geography and Demographics

### Important Note
✅ The chatbot will provide answers only for Bihar-related questions.
❌ Questions unrelated to Bihar will not be answered.
""")

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index = pc.Index("textgenai")

# Load embedding model
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# User Input
question = st.text_input("Ask a Question about Bihar")

if st.button("Submit"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Create embedding
    q_emb = embed_model.encode(question).tolist()

    # Search Pinecone
    result = index.query(
        vector=q_emb,
        top_k=3,
        include_metadata=True
    )

    # Build context
    context = ""

    if result.get("matches"):
        for match in result["matches"]:
            context += match["metadata"]["text"] + "\n"

    # AI21 API Headers
    headers = {
        "Authorization": f"Bearer {os.getenv('AI21_API_KEY')}",
        "Content-Type": "application/json"
    }

    # Prompt
    prompt = f"""
You are a Bihar State Expert Assistant.

Rules:
1. Answer ONLY questions related to Bihar.
2. If the question is not related to Bihar, reply:
   "Sorry, I can answer only Bihar-related questions."
3. Use the provided context to answer.
4. If the answer is not available in the context, say:
   "I could not find information about that in my Bihar knowledge base."

Context:
{context}

Question:
{question}
"""

    payload = {
        "model": "jamba-large",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(
            "https://api.ai21.com/studio/v1/chat/completions",
            headers=headers,
            json=payload
        )

        response.raise_for_status()

        answer = response.json()["choices"][0]["message"]["content"]

        st.subheader("Answer")
        st.write(answer)

    except Exception as e:
        st.error(f"Error: {e}")
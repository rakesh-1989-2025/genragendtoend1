from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

from pinecone import Pinecone

from dotenv import load_dotenv

import os
import uuid

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

pc=Pinecone(api_key=PINECONE_API_KEY)

index=pc.Index("textgenai")

loader=TextLoader("data\knowledge.txt")

documents=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs=splitter.split_documents(documents)

model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

vectors=[]

for doc in docs:

    emb=model.encode(doc.page_content).tolist()

    vectors.append(
        (
            str(uuid.uuid4()),
            emb,
            {"text":doc.page_content}
        )
    )

index.upsert(vectors=vectors)

print("Data Uploaded Successfully")
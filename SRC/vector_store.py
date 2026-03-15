import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

load_dotenv()

def build_vectorstore(texts):
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts).tolist()

    pc = Pinecone(api_key=os.getenv("PINE_API_KEY"))
    index_name = "xml-app"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)

    vector_data = [
        {"id": str(i), "values": emb, "metadata": {"text": texts[i]}}
        for i, emb in enumerate(embeddings)
    ]
    index.upsert(vectors=vector_data)

    embeddings_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings_model, text_key="text")

    return vectorstore
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def rag_query(query, vectorstore):
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a strict RAG assistant.

RULES:
- Fully understand all 3 retrieved chunks carefully.
- Generate the best, properly detailed answer (up to 500 words).
- If any coding lines or syntax appear in context, show them clearly in a separate formatted block.
- If the answer is not found inside CONTEXT, reply exactly:
  "Answer is not found"

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a strict RAG model."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
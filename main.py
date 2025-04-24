import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import pinecone
import requests

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")



pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone.Index(os.getenv("PINECONE_INDEX"))


app = FastAPI()

class ChatRequest(BaseModel):
    question: str


def embed_text(text: str):
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0]["embedding"]


def retrieve_context(question: str):
    vector = embed_text(question)
    result = index.query(vector=vector, top_k=5, include_metadata=True)
    return "\n".join([match["metadata"]["text"] for match in result["matches"]])

@app.post("/ask")
async def ask_question(data: ChatRequest):
    context = retrieve_context(data.question)

    messages = [
    {"role": "system", "content": (
        "You are a professional and friendly virtual assistant for Al-Rafidain Bank's Cards and Electronic Payment Department, powered by Qi Card. "
        "You fully understand and communicate fluently in Iraqi Arabic, including all dialects from different regions of Iraq. "
        "You only respond in Iraqi Arabic dialect. "
        "Only answer questions related to banking, cards, financial transactions, and technical issues under the department’s scope. "
        "If the question is personal or requires user credentials, respond with: 'Your inquiry involves personal credentials and will be handled later.' "
        "If the question is outside your scope, respond politely with: 'Your question is not related to our services. Please feel free to ask anything related to Qi Card or Al-Rafidain Bank.' "
        "Always be friendly, respectful, and helpful to the user. "
        "Always stay compliant with Central Bank of Iraq regulations and PCI DSS security standards. "
        "Use the following CONTEXT only to generate your response."
    )},
    {"role": "user", "content": (
        f"You are now chatting with a customer of Al-Rafidain Bank. Use the CONTEXT provided below to understand your role, "
        "and then answer the user's question in Iraqi Arabic only. "
        "Only respond if the question relates to financial services, cards, or transactions. "
        "If not, follow the system instructions.\n\n"
        f"CONTEXT:\n{context.strip()}\n\n"
        f"USER QUESTION:\n{data.question.strip()}"
    )}
    ]

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/llama-4-scout",
            "messages": messages
        }
    )

    if response.status_code != 200:
        return {"error": response.json()}

    reply = response.json()["choices"][0]["message"]["content"]
    return {"reply": reply}

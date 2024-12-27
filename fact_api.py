from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import random

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#google_api_key = os.getenv('GOOGLE_API_KEY')
#groq_api_key = os.getenv('GROQ_API_KEY')

app = FastAPI()

llm = ChatGroq(groq_api_key= "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", model_name="Llama3-8b-8192")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

loader = PyPDFDirectoryLoader("/home/ishan/Projects/QA_System/Data/")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
final_documents = text_splitter.split_documents(docs)
vectors = FAISS.from_documents(final_documents, embeddings)

prompt = ChatPromptTemplate.from_template(
    """
    Please provide the most accurate response based on the question and make it super easy to understand like you are explaining to a 5 years old and make it sound like a story.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

class FactRequest(BaseModel):
    question: str

@app.post("/get-fact")
async def get_fact(request: FactRequest):
    try:
        random_page = random.choice(final_documents)
        random_fact_prompt = f"Tell a random fact about exoplanet from page {random_page}."
        
        response = retrieval_chain.invoke({'input': random_fact_prompt})
        return {"fact": response['answer']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

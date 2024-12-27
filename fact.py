import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from dotenv import load_dotenv
import os
import random
load_dotenv()


google_api_key = os.getenv('GOOGLE_API_KEY')
#groq_api_key = os.getenv('GROG_API_KEY')


# Initialize the LLM and embeddings
llm = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load PDF documents
loader = PyPDFDirectoryLoader("/home/ishan/Projects/QA_System/Data/")
docs = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
final_documents = text_splitter.split_documents(docs)

# Initialize Qdrant client and create a collection
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "exoplanet_qa_system"

# Check if collection exists, if not create it
if not qdrant_client.has_collection(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.embed_size, distance=Distance.COSINE)
    )

# Create Qdrant vector store and add documents
vectors = Qdrant.from_documents(
    documents=final_documents, 
    embedding=embeddings, 
    collection_name=collection_name, 
    client=qdrant_client
)

#vectors=FAISS.from_documents(final_documents,embeddings)

prompt=ChatPromptTemplate.from_template(
"""
Please generate a random page from the given pdf.
Please provide the most accurate response based on the question and make it super easy to understand like you are explaining to a 5 years old and make it sound like a story.
<context>
{context}
<context>
Questions:{input}

""")

document_chain=create_stuff_documents_chain(llm,prompt)
retriever=vectors.as_retriever()
retrieval_chain=create_retrieval_chain(retriever,document_chain)

random_page = random.choice(final_documents)
random_fact_prompt = f"Tell a random fact about exoplanet from page {random_page}."

response = retrieval_chain.invoke({'input': random_fact_prompt})
print(response['answer'])
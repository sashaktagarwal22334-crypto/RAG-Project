import os
import argparse
import warnings
warnings.filterwarnings("ignore")

if hasattr(os.sys.stdout, "reconfigure"):
    os.sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise SystemExit(
        "Configuration error: set GROQ_API_KEY in the .env file."
    )

# 2. Document Loading
project_dir = os.path.dirname(os.path.abspath(__file__))
downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
pdf_path = os.environ.get("DSA_PDF_PATH")
if not pdf_path:
    pdf_path = os.path.join(project_dir, "DSA_PYQ_Practice_Book_100pages.pdf")
if not os.path.isfile(pdf_path):
    pdf_path = os.path.join(downloads_dir, "DSA_PYQ_Practice_Book_100pages.pdf")
if not os.path.isfile(pdf_path):
    raise FileNotFoundError(
        "DSA practice PDF not found. Set DSA_PDF_PATH to its location."
    )
print(f"Loading document from: {pdf_path}")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} document pages.")

# 3. Document Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} text chunks.")

# 4. Embeddings & In-Memory Vector Store (Qdrant + FastEmbed)
print("Creating embeddings and indexing into Qdrant vector store...")
embeddings = FastEmbedEmbeddings()

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="rag_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="rag_collection",
    embedding=embeddings,
)
vectorstore.add_documents(chunks)
print("Indexing complete!")

# 5. Initialize NVIDIA LLM
# Note: nemotron uses reasoning tokens, so max_completion_tokens should be >= 1024
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    groq_api_key=api_key,
    temperature=0.2,
    max_tokens=2500,
)

# 6. Retriever & RAG Prompt
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert tutor in Data Structures and Algorithms. "
        "Answer the question accurately based ONLY on the provided context. "
        "Show the reasoning, formula or traversal needed, and state the answer clearly. "
        "If the answer cannot be found in the context, say that you don't know.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}"),
])

rag_chain = prompt | llm | StrOutputParser()

# 7. Test RAG with a question
def query_rag(question: str):
    print("\n" + "="*60)
    print(f"Query: {question}")
    print("="*60)
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant context chunk(s):")
    for i, doc in enumerate(retrieved_docs, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        preview = doc.page_content.strip().replace("\n", " ")[:150]
        print(f" [{i}] (Page {page_num}): {preview.encode('ascii', 'replace').decode('ascii')}...")

    # Build context string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate response
    print("\nGenerating response from Groq...")
    try:
        response = rag_chain.invoke({"context": context, "question": question})
    except Exception as exc:
        error_text = str(exc)
        if "model_not_found" in error_text:
            raise SystemExit(
                "Groq model error: set GROQ_MODEL in .env to a model listed by "
                "your Groq account (for example: openai/gpt-oss-20b)."
            ) from exc
        if "invalid_api_key" in error_text or "401" in error_text:
            raise SystemExit(
                "Groq API key error: replace GROQ_API_KEY in .env with a valid key."
            ) from exc
        raise SystemExit(f"RAG request failed: {error_text}") from exc
    print("\nAnswer:")
    print(response)
    print("="*60)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ask questions about the DSA PYQ practice book."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="Explain the concept of minimum and maximum height in a binary tree.",
    )
    args = parser.parse_args()
    query_rag(args.question)
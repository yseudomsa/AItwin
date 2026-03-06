"""
RAG Chatbot using LangChain + ChromaDB + Google Gemini
Embeddings: HuggingFace all-MiniLM-L6-v2 (local, via sentence-transformers)
Data source: Plain text notepad file
Interface: Terminal / CLI
"""

import json
import redis
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# ── Load .env file automatically ───────────────────────────────────────────────
load_dotenv()

# ── LangChain imports ──────────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate


# ── Configuration ──────────────────────────────────────────────────────────────
NOTEPAD_FILE    = "knowledge_base.txt"                 # Path to your notepad/text file
CHROMA_DB_DIR   = "./chroma_db"                        # Local folder for ChromaDB persistence
FORCE_REINDEX   = True
GEMINI_MODEL    = "gemini-2.5-flash"                   # Free-tier Gemini model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Local HuggingFace model
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_TTL = int(os.getenv("REDIS_TTL", "86400"))
REDIS_KEY = "chatbot:history"


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """Fetch API key from .env file, environment, or prompt the user."""
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if key:
        print("✅  GOOGLE_API_KEY loaded from .env")
        return key
    # Fallback: prompt the user
    print("\n🔑  GOOGLE_API_KEY not found in .env or environment.")
    print("   Tip: create a .env file with:  GOOGLE_API_KEY=your_key_here")
    key = input("   Paste your Gemini API key: ").strip()
    if not key:
        print("❌  API key is required. Exiting.")
        sys.exit(1)
    os.environ["GOOGLE_API_KEY"] = key
    return key

def get_redis_client():
    """Connect to Redis and verify connection."""
    try:
        client = redis.Redis(host = REDIS_HOST, port=REDIS_PORT, decode_responses = True)
        client.ping()
        print("Redis connected! - chat history will now persist between sessions.")
        return client
    except redis.exceptions.ConnectionError:
        print("Redis not available! - chat history will only last this session")
        return None

def load_history(client) -> list:
    """Load Chat History from Redis."""
    if not client:
        return[]
    raw = client.get(REDIS_KEY)
    if raw:
        data = json.loads(raw)
        print(f"Loadaed {len(data)} previous messages(s) from Reddis.")
        return data
    return []

def save_history(client, history: list):
    """Save chat history with TTL."""
    if not client:
        return
    client.set(REDIS_KEY, json.dumps(history), ex=REDIS_TTL)


def load_and_split_notepad(filepath: str):
    """Load the text file and split it into chunks."""
    path = Path(filepath)
    if not path.exists():
        print(f"❌  Notepad file not found: {filepath}")
        print("   Please create the file and add your notes, then re-run.")
        sys.exit(1)

    print(f"📄  Loading notepad: {filepath}")
    loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️   Split into {len(chunks)} chunks.")
    return chunks


def build_vector_store(chunks, embeddings, persist_dir: str) -> Chroma:
    """Create or load a persistent ChromaDB vector store."""
    db_path = Path(persist_dir)

    if db_path.exists() and any(db_path.iterdir()) and not FORCE_REINDEX:
        print("🗄️   Loading existing ChromaDB vector store …")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
    else:
        print("🔨  Building ChromaDB vector store (first run – may take a moment) …")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        print("✅  Vector store created and persisted.")

    return vectorstore


def build_chain(vectorstore: Chroma, llm, redis_client):
    """Build the conversational retrieval chain with Redis backed-memory."""

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant that answers questions strictly based on the provided context (notes).
If the answer is not in the context, say "I don't have information about that in my notes."
Be concise and clear. Context: {context}, Question: {question}
Answer:""",
    )

    chat_history = load_history(redis_client)  # stores (human, ai) tuples in memory
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def chat(user_input: str) -> str:
        # Retrieve relevant docs
        docs = retriever.invoke(user_input)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Build prompt
        prompt = qa_prompt.format(context=context, question=user_input)

        # Build messages with history
        messages = []
        for entry in chat_history:
            messages.append(HumanMessage(content=entry["human"]))
            messages.append(AIMessage(content=entry["ai"]))
        messages.append(HumanMessage(content=prompt))

        # Call LLM
        response = llm.invoke(messages)
        answer = response.content.strip()

        # Save to history
        chat_history.append({"human": user_input, "ai": answer})
        save_history(redis_client, chat_history)
        return answer

    return chat


def run_chatbot(chain):
    """Main REPL loop for the terminal chatbot."""
    print("\n" + "═" * 60)
    print("  🤖  RAG Chatbot  |  Gemini LLM · MiniLM Embeddings · ChromaDB")
    print("  📒  Knowledge source: your notepad file")
    print("  Type  'quit' or 'exit' to stop.")
    print("  Type  'reload' to re-index the notepad file.")
    print("═" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("👋  Goodbye!")
            break

        if user_input.lower() == "reload":
            print("🔄  Reload triggered. Please restart the app to re-index.")
            continue

        try:
            answer = chain(user_input)
            print(f"\n🤖 Bot: {answer}\n")
        except Exception as exc:
            print(f"\n⚠️  Error: {exc}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("\n🚀  Starting RAG Chatbot …\n")

    # 1. API key
    api_key = get_api_key()

    # 2. LLM + Embeddings
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.2,
        convert_system_message_to_human=True,
    )
    print(f"🔍  Loading embedding model: {EMBEDDING_MODEL} (downloading on first run) …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    # 3. Load notepad & build vector store
    chunks = load_and_split_notepad(NOTEPAD_FILE)
    vectorstore = build_vector_store(chunks, embeddings, CHROMA_DB_DIR)

    # 4. Build conversational chain
    redis_client = get_redis_client()

    # 5. Run CLI chatbot
    chain = build_chain(vectorstore, llm, redis_client)

    run_chatbot(chain)

if __name__ == "__main__":
    main()

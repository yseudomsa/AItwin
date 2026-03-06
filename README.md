# RAG Chatbot — LangChain + ChromaDB + Gemini + Redis

A terminal-based RAG (Retrieval-Augmented Generation) chatbot that answers questions
from a plain `.txt` notepad file using Google Gemini (free API), ChromaDB, and Redis for persistent chat history.

---

## 📁 Project Structure

```
.
├── chatbot.py           # Main application
├── knowledge_base.txt   # Your notepad (edit this with your notes!)
├── requirements.txt     # Python dependencies
├── .env                 # Your API keys and config (never commit this!)
├── .env.example         # Template for .env
└── chroma_db/           # Auto-created: persisted vector store
```

---

## ⚙️ Setup

### 1. Prerequisites
- Python 3.11 (recommended — 3.14 may cause LangChain compatibility issues)
- A free Google Gemini API key → https://aistudio.google.com/app/apikey
- Redis for Windows → https://github.com/microsoftarchive/redis/releases (`Redis-x64-3.0.504.msi`)

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure your `.env` file
Copy `.env.example` to `.env` and fill in your values:
```
GOOGLE_API_KEY=your_gemini_api_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TTL=86400
```
`REDIS_TTL` is how long chat history is kept in seconds. `86400` = 24 hours.

### 4. Edit your notepad
Open `knowledge_base.txt` and add your own notes.
The chatbot will only answer from what's in this file.

### 5. Run the chatbot
```bash
python chatbot.py
```

---

## 💬 Usage

```
You: What is LangChain?
🤖 Bot: LangChain is a framework for developing applications powered by large language models...

You: quit
👋  Goodbye!
```

**Special commands:**
| Command  | Action                         |
|----------|--------------------------------|
| `quit`   | Exit the chatbot               |
| `exit`   | Exit the chatbot               |
| `reload` | Reminder to restart & re-index |

---

## 🔄 Updating Your Notes

1. Edit `knowledge_base.txt` with new content.
2. Set `FORCE_REINDEX = True` in `chatbot.py`.
3. Run `python chatbot.py` — it will re-index automatically.
4. Once done editing, set `FORCE_REINDEX = False` for faster startups.

---

## ⏱️ Startup Time

| Condition | Startup Time |
|---|---|
| `FORCE_REINDEX = True` | ~30 seconds (re-embeds everything) |
| `FORCE_REINDEX = False` | ~5 seconds (loads existing vectors) |

Set `FORCE_REINDEX = False` when your notes are finalized for much faster startups.

---

## 🧠 How It Works

```
knowledge_base.txt
       │
       ▼
  Text Splitter  (chunks of ~500 chars)
       │
       ▼
  MiniLM Embeddings  (local, via sentence-transformers)
       │
       ▼
   ChromaDB  (local vector store, persisted to disk)
       │
  User Question ──► Similarity Search ──► Top 4 Chunks
                                               │
                                               ▼
                                      Gemini 2.0 Flash
                                      (answers from context)
                                               │
                                               ▼
                                    Answer printed to terminal
                                               │
                                               ▼
                                Redis  (saves chat history, TTL 24hrs)
```

---

## 🗄️ Redis — Persistent Chat History

Redis stores your conversation history so the chatbot remembers past sessions.

- Chat history is saved as a JSON string under the key `chatbot:history`
- It auto-expires after `REDIS_TTL` seconds (default 24 hours)
- If Redis is unavailable, the chatbot still works — history just resets on exit

**To view stored history:**
```bash
redis-cli
> GET chatbot:history
```

**To clear history and start fresh:**
```bash
redis-cli del chatbot:history
```

**To check Redis is running:**
```bash
redis-cli ping
# PONG = running
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core RAG orchestration |
| `langchain-google-genai` | Gemini LLM |
| `langchain-huggingface` | MiniLM embeddings |
| `langchain-chroma` | ChromaDB integration |
| `langchain-text-splitters` | Text chunking |
| `langchain-core` | Prompts and messages |
| `chromadb` | Local vector database |
| `sentence-transformers` | all-MiniLM-L6-v2 model |
| `google-generativeai` | Google AI SDK |
| `redis` | Persistent chat history |
| `python-dotenv` | Load `.env` file |

---

## 🔒 Security

- Never commit your `.env` file to Git
- Add these to your `.gitignore`:
```
.env
chroma_db/
```

# RAG Chatbot — LangChain + ChromaDB + Gemini

A terminal-based RAG (Retrieval-Augmented Generation) chatbot that answers questions
from a plain `.txt` notepad file using Google Gemini (free API) and ChromaDB.

---

## 📁 Project Structure

```
.
├── chatbot.py           # Main application
├── knowledge_base.txt   # Your notepad (edit this with your notes!)
├── requirements.txt     # Python dependencies
└── chroma_db/           # Auto-created: persisted vector store
```

---

## ⚙️ Setup

### 1. Prerequisites
- Python 3.9+
- A free Google Gemini API key → https://aistudio.google.com/app/apikey

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key
Either set it as an environment variable (recommended):
```bash
# Linux / macOS
export GOOGLE_API_KEY="your_key_here"

# Windows CMD
set GOOGLE_API_KEY=your_key_here

# Windows PowerShell
$env:GOOGLE_API_KEY="your_key_here"
```
Or just run the app — it will prompt you to paste it.

### 4. Edit your notepad
Open `knowledge_base.txt` and replace or add your own notes.
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
| Command  | Action                        |
|----------|-------------------------------|
| `quit`   | Exit the chatbot              |
| `exit`   | Exit the chatbot              |
| `reload` | Reminder to restart & re-index |

---

## 🔄 Updating Your Notes

1. Edit `knowledge_base.txt` with new content.
2. Delete the `chroma_db/` folder to force re-indexing:
   ```bash
   rm -rf chroma_db/
   ```
3. Re-run `python chatbot.py` — it will rebuild the vector store automatically.

---

## 🧠 How It Works

```
knowledge_base.txt
       │
       ▼
  Text Splitter  (chunks of ~500 chars)
       │
       ▼
  Google Embeddings  (semantic vectors)
       │
       ▼
   ChromaDB  (local vector store)
       │
  User Question ──► Similarity Search ──► Top 4 Chunks
                                               │
                                               ▼
                                    Gemini 1.5 Flash
                                    (answers from context)
                                               │
                                               ▼
                                         Terminal Output
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core RAG orchestration |
| `langchain-google-genai` | Gemini LLM + Embeddings |
| `langchain-chroma` | ChromaDB integration |
| `chromadb` | Local vector database |
| `google-generativeai` | Google AI SDK |

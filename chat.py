"""
chat.py — Interactive chat with Qwen + RAG Wiki

Sends your prompts to a local Qwen model (via Ollama) and automatically
enriches each query with context retrieved from the RAG wiki system.
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from rag_wiki import RagWikiRetriever, RagWikiRetrieverConfig
from rag_wiki.storage.sqlite import SQLiteStateStore

import os

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL      = "qwen3:8b"
EMBED_MODEL       = "mxbai-embed-large:latest"
CHROMA_DIR        = "./chroma_db"
COLLECTION_NAME   = "my_docs"
WIKI_SAVE_DIR     = "wiki/documents"
USER_ID           = "chat-user"

# ── Build components ──────────────────────────────────────────────────────────
# Create wiki/ and wiki/documents/ before anything tries to open the DB
os.makedirs(WIKI_SAVE_DIR, exist_ok=True)

DB_PATH = f"sqlite:///{os.path.abspath('wiki/rag_wiki_state.db')}"

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
    persist_directory=CHROMA_DIR,
)

pending_suggestions: list = []

def on_suggestion(event):
    pending_suggestions.append(event)

retriever = RagWikiRetriever(
    user_id=USER_ID,
    global_retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    state_store=SQLiteStateStore(DB_PATH),
    config=RagWikiRetrieverConfig(
        fetch_threshold=3,
        reset_threshold=3,
        wiki_save_dir=WIKI_SAVE_DIR,
    ),
    on_suggestion=on_suggestion,
)

llm = OllamaLLM(model=OLLAMA_MODEL)

# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant with access to a personal knowledge base.
Read ALL of the context carefully before answering. The answer is likely in the context even if the question is in a different language.

Context:
{context}

Question: {question}

Instructions:
- If the context contains relevant information, use it to answer fully.
- The question may be in Turkish — still search the context for the answer.
- Only say the context lacks information if you truly cannot find anything relevant.

Answer:""",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def format_docs(docs) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    if not docs:
        return "(no relevant documents found)"
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('doc_title', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

def handle_suggestions():
    """Prompt the user to accept or decline wiki save suggestions."""
    for event in pending_suggestions:
        print(
            f"\n💡 '{event.doc_title}' has come up {event.fetch_count}× — "
            "save to personal library?"
        )
        answer = input("   [y/n]: ").strip().lower()
        if answer == "y":
            retriever.accept_suggestion(event.doc_id)
            print("   ✅ Saved — next query will load directly from wiki")
        else:
            retriever.decline_suggestion(event.doc_id)
            print("   ⏭  Skipped — will suggest again after threshold resets")
    pending_suggestions.clear()

def ask(question: str) -> str:
    """Retrieve context via RAG wiki, then query Qwen."""
    docs = retriever.invoke(question)

    # Show provenance (which docs were used and from where)
    provenance = retriever.last_provenance.render()
    if provenance:
        print(f"\n📚 Provenance:\n{provenance}")

    context = format_docs(docs)
    prompt_text = PROMPT.format(context=context, question=question)
    return llm.invoke(prompt_text)

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  RAG Wiki + Qwen Chat  (model: {OLLAMA_MODEL})")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        print("\nThinking…")
        try:
            answer = ask(user_input)
            print(f"\nQwen: {answer}")
        except Exception as e:
            print(f"\n⚠️  Error: {e}")

        # Handle any wiki save suggestions after each turn
        handle_suggestions()

if __name__ == "__main__":
    main()

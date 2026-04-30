from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rag_wiki import RagWikiRetriever, RagWikiRetrieverConfig, MemoryStateStore
from rag_wiki.storage.sqlite import SQLiteStateStore

import os

os.makedirs("wiki/documents", exist_ok=True)

vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"),
    persist_directory="./chroma_db",
)

pending_suggestions = []

def on_suggestion(event):
    pending_suggestions.append(event)

retriever = RagWikiRetriever(
    user_id="test-user",
    global_retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    state_store=SQLiteStateStore("sqlite:///./wiki/rag_wiki_state.db"),
    config=RagWikiRetrieverConfig(
        fetch_threshold=2,
        reset_threshold=3,
        wiki_save_dir="wiki/documents",
    ),
    on_suggestion=on_suggestion,
)

queries = [
    "?",
    "What are the university locations?",
    "What are the university scholarships?",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"Query: {q}")
    print('='*60)

    docs = retriever.invoke(q)
    print(retriever.last_provenance.render())

    for event in pending_suggestions:
        print(f"\n💡 '{event.doc_title}' has come up {event.fetch_count}× — save to library?")
        answer = input("   [y/n]: ").strip().lower()
        if answer == "y":
            retriever.accept_suggestion(event.doc_id)
            print(f"   ✅ Saved to personal KB — next query will skip vector search")
        else:
            retriever.decline_suggestion(event.doc_id)
            print(f"   ⏭  Skipped — will suggest again after threshold resets")

    pending_suggestions.clear()
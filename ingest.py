from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load docs from a /docs folder
loader = DirectoryLoader("/Users/sstprk/Desktop/sample_docs", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

# Split with origin metadata preserved
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

from collections import defaultdict
counts = defaultdict(int)

# Inject doc_id metadata on each chunk
for chunk in chunks:
    doc_id = chunk.metadata.get("source", "unknown")
    chunk.metadata["doc_id"] = doc_id
    chunk.metadata["doc_title"] = doc_id.split("/")[-1]
    chunk.metadata["chunk_index"] = counts[doc_id]
    counts[doc_id] += 1

# Embed and store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    persist_directory="./chroma_db",
    collection_name="my_docs",
)
print(f"Ingested {len(chunks)} chunks from {len(docs)} documents")
"""
benchmark.py — RAG vs RAG-Wiki realistic performance comparison.

No pre-warmup. The cache builds naturally during the benchmark run via
auto-save, reflecting real-world usage patterns.

Supports both local Chroma and remote Pinecone as the vector DB backend.
Set PINECONE_API_KEY env var to use Pinecone; otherwise falls back to Chroma.

Metrics collected per query:
  - response_time_s      : wall-clock seconds for retrieval
  - chunks_retrieved     : number of Document objects returned
  - total_chars          : total characters across all retrieved chunks
  - estimated_tokens     : total_chars / 4  (rough GPT-style estimate)
  - estimated_cost_usd   : estimated_tokens / 1000 * COST_PER_1K_TOKENS
  - from_cache           : (rag-wiki only) whether any doc came from personal KB
  - cache_hit_rate       : fraction of returned docs that were cache hits
  - context_relevance    : cosine similarity between query embedding and
                           concatenated retrieved text embedding (0–1)
  - auto_saved_this_query: (rag-wiki only) whether an auto-save fired

Usage:
    # Local Chroma (default):
    python benchmark.py

    # Remote Pinecone:
    PINECONE_API_KEY=pk-... PINECONE_INDEX_NAME=my-index python benchmark.py
"""

import os
import time
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rag_wiki import RagWikiRetriever, RagWikiRetrieverConfig
from rag_wiki.storage.sqlite import SQLiteStateStore
from rag_wiki.lifecycle.fetch_counter import AutoSaveEvent

# ─── Configuration ────────────────────────────────────────────────────────────

EMBED_MODEL        = os.getenv("EMBED_MODEL", "mxbai-embed-large:latest")
CHROMA_DIR         = "./chroma_db"
COLLECTION_NAME    = "my_docs"
WIKI_STATE_DB      = "sqlite:///./wiki/rag_wiki_state.db"
WIKI_SAVE_DIR      = "wiki/documents"
OUTPUT_DIR         = Path("benchmark_results")
COST_PER_1K_TOKENS = float(os.getenv("COST_PER_1K_TOKENS", "0.0001"))
RAG_K              = 5
RAGWIKI_K          = 5

# Pinecone config (set env vars to enable)
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-wiki-bench")
PINECONE_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "")

# ─── 20 benchmark questions ───────────────────────────────────────────────────

QUESTIONS = [
    # Personal profile
    "What is Salih Toprak's current role and location?",
    "What university is Salih studying at and when does he graduate?",
    "What is Salih's GPA and degree program?",
    "What Erasmus exchange did Salih complete and where?",
    "What IEEE paper did Salih co-author and what was it about?",
    "What TÜBİTAK project did Salih work on?",
    "What are Salih's main technical skills?",
    "What internships has Salih completed?",
    "What is Salih's English proficiency level?",
    "What are Salih's strengths for applying to European universities?",
    # Turkey opportunities
    "What career paths are available in Turkey for a software engineer?",
    "What are the top tech companies hiring in Turkey?",
    "What is the average salary for a data scientist in Turkey?",
    "What are the best cities in Turkey for tech jobs?",
    "What Turkish government programs support tech startups?",
    # Cross-document / synthesis
    "How does Salih's background align with European master's programs?",
    "What scholarships could Salih apply for given his profile?",
    "What programming languages and frameworks does Salih know?",
    "What research experience does Salih have?",
    "What is Salih's experience with machine learning and AI?",
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


import math
from sentence_transformers import CrossEncoder

def context_relevance(query: str, docs: list, cross_encoder: CrossEncoder) -> float:
    """
    Cross-encoder relevance score for retrieved chunks.
    Returns mean score across all chunks, normalised to [0, 1].
    Returns 0.0 if no docs.
    
    Cross-encoder scores are logits (unbounded). Normalise using sigmoid:
      score = 1 / (1 + exp(-logit))
    This maps any real value to (0, 1) without clipping.
    """
    if not docs:
        return 0.0
    pairs  = [(query, doc.page_content[:500]) for doc in docs]
    logits = cross_encoder.predict(pairs)
    scores = [1 / (1 + math.exp(-float(l))) for l in logits]
    return round(float(np.mean(scores)), 4)


def run_query(retriever, query, cross_encoder, *, is_wiki=False) -> dict:
    """Run one query and return all metrics."""
    t0   = time.perf_counter()
    docs = retriever.invoke(query)
    elapsed = time.perf_counter() - t0

    full_text  = " ".join(d.page_content for d in docs)
    tokens     = estimate_tokens(full_text)
    cache_hits = [d for d in docs if d.metadata.get("from_cache", False)]
    relevance  = context_relevance(query, docs, cross_encoder)

    return {
        "response_time_s":    round(elapsed, 3),
        "chunks_retrieved":   len(docs),
        "total_chars":        len(full_text),
        "estimated_tokens":   tokens,
        "estimated_cost_usd": round(tokens / 1000 * COST_PER_1K_TOKENS, 6),
        "from_cache":         len(cache_hits) > 0,
        "cache_hit_rate":     round(len(cache_hits) / max(len(docs), 1), 3),
        "context_relevance":  round(relevance, 4),
    }


def build_retrievers_and_embed_model():
    """Build plain + wiki base retrievers and embed model based on env config."""
    from langchain_ollama import OllamaEmbeddings
    embed_model = OllamaEmbeddings(model=EMBED_MODEL)

    if PINECONE_API_KEY:
        print("  → Using Pinecone remote vector DB (native SDK)")
        from rag_wiki.adapters.pinecone_retriever import PineconeRetriever

        plain_retriever = PineconeRetriever(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME,
            embed_model=embed_model,
            top_k=RAG_K,
            namespace=PINECONE_NAMESPACE or None,
        )
        wiki_base_retriever = PineconeRetriever(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME,
            embed_model=embed_model,
            top_k=RAGWIKI_K,
            namespace=PINECONE_NAMESPACE or None,
        )
    else:
        print("  → Using local Chroma DB")
        from langchain_chroma import Chroma
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
            persist_directory=CHROMA_DIR,
        )
        plain_retriever = vectorstore.as_retriever(search_kwargs={"k": RAG_K})
        wiki_base_retriever = vectorstore.as_retriever(search_kwargs={"k": RAGWIKI_K})

    return plain_retriever, wiki_base_retriever, embed_model


# ─── Plotting ─────────────────────────────────────────────────────────────────

COLORS = {"rag": "#4C72B0", "ragwiki": "#DD8452"}


def bar_comparison(questions, rag_vals, wiki_vals, title, ylabel, filename, higher_is_better=True):
    x     = np.arange(len(questions))
    width = 0.38
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - width/2, rag_vals,  width, label="Plain RAG",  color=COLORS["rag"],     alpha=0.85)
    ax.bar(x + width/2, wiki_vals, width, label="RAG-Wiki",   color=COLORS["ragwiki"], alpha=0.85)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(questions))], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for i, (rv, wv) in enumerate(zip(rag_vals, wiki_vals)):
        winner = "ragwiki" if (wv > rv if higher_is_better else wv < rv) else "rag"
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color=COLORS[winner])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


def summary_radar(rag_summary, wiki_summary, filename):
    metrics = [
        ("Response\nTime ↓",    "response_time_s",    False),
        ("Tokens\nUsed ↓",      "estimated_tokens",   False),
        ("Cost ↓",              "estimated_cost_usd", False),
        ("Context\nRelevance ↑","context_relevance",  True),
        ("Cache\nHit Rate ↑",   "cache_hit_rate",     True),
        ("Chunks\nRetrieved",   "chunks_retrieved",   True),
    ]
    labels = [m[0] for m in metrics]
    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    def normalise(key, hib, rv, wv):
        mx = max(rv, wv, 1e-9)
        mn = min(rv, wv)
        if mx == mn:
            return 0.5, 0.5
        r = (rv - mn) / (mx - mn)
        w = (wv - mn) / (mx - mn)
        if not hib:
            r, w = 1 - r, 1 - w
        return r, w

    rv, wv = [], []
    for _, key, hib in metrics:
        r, w = normalise(key, hib, rag_summary.get(key, 0), wiki_summary.get(key, 0))
        rv.append(r); wv.append(w)
    rv += rv[:1]; wv += wv[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, rv, "o-", linewidth=2, color=COLORS["rag"],     label="Plain RAG")
    ax.fill(angles, rv, alpha=0.15, color=COLORS["rag"])
    ax.plot(angles, wv, "o-", linewidth=2, color=COLORS["ragwiki"], label="RAG-Wiki")
    ax.fill(angles, wv, alpha=0.15, color=COLORS["ragwiki"])
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Overall Performance Comparison\n(normalised, higher = better)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


def summary_table_chart(rag_summary, wiki_summary, filename):
    rows = [
        ("Avg Response Time (s)",    "response_time_s",    False, ".3f"),
        ("Avg Chunks Retrieved",     "chunks_retrieved",   True,  ".1f"),
        ("Avg Tokens Used",          "estimated_tokens",   False, ".0f"),
        ("Avg Est. Cost (USD)",      "estimated_cost_usd", False, ".6f"),
        ("Avg Context Relevance",    "context_relevance",  True,  ".4f"),
        ("Avg Cache Hit Rate",       "cache_hit_rate",     True,  ".3f"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    col_labels = ["Metric", "Plain RAG", "RAG-Wiki", "Δ (Wiki − RAG)", "Winner"]
    table_data, cell_colors = [], []
    for label, key, hib, fmt in rows:
        rv, wv = rag_summary.get(key, 0), wiki_summary.get(key, 0)
        delta = wv - rv
        if hib:
            winner = "RAG-Wiki ✓" if wv > rv else ("Plain RAG ✓" if rv > wv else "Tie")
            wc = "#c8e6c9" if wv > rv else ("#ffcdd2" if rv > wv else "#fff9c4")
        else:
            winner = "RAG-Wiki ✓" if wv < rv else ("Plain RAG ✓" if rv < wv else "Tie")
            wc = "#c8e6c9" if wv < rv else ("#ffcdd2" if rv < wv else "#fff9c4")
        table_data.append([label, format(rv, fmt), format(wv, fmt), f"{delta:+.4f}", winner])
        cell_colors.append(["#f5f5f5", "#e3f2fd", "#fff3e0", "#fafafa", wc])
    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center",
                   loc="center", cellColours=cell_colors)
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474f")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Summary: Plain RAG vs RAG-Wiki", fontsize=13, fontweight="bold", pad=15, y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def cache_hit_timeline(wiki_results, auto_save_log, filename):
    """Show cache hits and auto-save events over time."""
    n = len(wiki_results)
    hits = [1 if r["from_cache"] else 0 for r in wiki_results]
    cumulative = np.cumsum(hits)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Top: per-query cache hit + auto-save markers
    bar_colors = [COLORS["ragwiki"] if h else "#ccc" for h in hits]
    ax1.bar(range(1, n+1), hits, color=bar_colors)
    # Mark auto-save events
    for qi in auto_save_log:
        ax1.axvline(x=qi, color="#2e7d32", linewidth=2, linestyle="--", alpha=0.7)
    ax1.set_ylabel("Cache Hit (1=yes)", fontsize=10)
    ax1.set_title("RAG-Wiki: Cache Hits & Auto-Save Events (natural, no warmup)", fontsize=12, fontweight="bold")
    ax1.set_yticks([0, 1])
    if auto_save_log:
        ax1.legend([mpatches.Patch(color="#2e7d32", alpha=0.7)], ["Auto-save fired"],
                   loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Bottom: cumulative
    ax2.plot(range(1, n+1), cumulative, "o-", color=COLORS["ragwiki"], linewidth=2)
    ax2.fill_between(range(1, n+1), cumulative, alpha=0.15, color=COLORS["ragwiki"])
    ax2.set_ylabel("Cumulative Cache Hits", fontsize=10)
    ax2.set_xlabel("Query Number", fontsize=10)
    ax2.set_xticks(range(1, n+1))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


def response_time_distribution(rag_results, wiki_results, filename):
    rag_times  = [r["response_time_s"] for r in rag_results]
    wiki_times = [r["response_time_s"] for r in wiki_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([rag_times, wiki_times], labels=["Plain RAG", "RAG-Wiki"],
                    patch_artist=True, medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor(COLORS["rag"])
    bp["boxes"][1].set_facecolor(COLORS["ragwiki"])
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    for i, (times, color) in enumerate([(rag_times, COLORS["rag"]), (wiki_times, COLORS["ragwiki"])], 1):
        jitter = np.random.uniform(-0.1, 0.1, len(times))
        ax.scatter([i + j for j in jitter], times, color=color, alpha=0.6, s=30, zorder=5)
    ax.set_title("Response Time Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Seconds", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    os.makedirs(WIKI_SAVE_DIR, exist_ok=True)

    n = len(QUESTIONS)
    print("=" * 60)
    print("RAG-Wiki Benchmark (realistic — no warmup)")
    print("=" * 60)

    # ── Shared components ─────────────────────────────────────────────────────
    print(f"\n[1/4] Initialising retrievers and embedding model...")
    plain_retriever, wiki_base_retriever, embed_model = build_retrievers_and_embed_model()
    
    print("  → Loading CrossEncoder for scoring...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  → Scoring: cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Track auto-save events for the timeline chart
    auto_save_log: list[int] = []       # query indices where auto-save fired
    current_query_idx = {"val": 0}

    def on_auto_save(event: AutoSaveEvent):
        auto_save_log.append(current_query_idx["val"])
        print(f"       ✅ Auto-saved: {event.doc_title[:50]} (fetched {event.fetch_count}×)")

    wiki_retriever = RagWikiRetriever(
        user_id          = "benchmark-user",
        global_retriever = wiki_base_retriever,
        state_store      = SQLiteStateStore(WIKI_STATE_DB),
        config           = RagWikiRetrieverConfig(
            fetch_threshold      = 3,
            reset_threshold      = 10,
            similarity_threshold = 0.5,
            local_top_k          = 5,
            wiki_save_dir        = WIKI_SAVE_DIR,
        ),
        on_auto_save     = on_auto_save,
    )

    # ── Run Plain RAG ─────────────────────────────────────────────────────────
    print(f"\n[2/4] Running Plain RAG ({n} queries)...")
    rag_results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"  Q{i:02d}: {q[:60]}...")
        metrics = run_query(plain_retriever, q, cross_encoder)
        rag_results.append(metrics)
        print(f"       {metrics['response_time_s']:.2f}s | "
              f"{metrics['chunks_retrieved']} chunks | "
              f"relevance={metrics['context_relevance']:.3f}")

    # ── Run RAG-Wiki (cold start, natural cache buildup) ──────────────────────
    print(f"\n[3/4] Running RAG-Wiki ({n} queries, cold start)...")
    wiki_results = []
    for i, q in enumerate(QUESTIONS, 1):
        current_query_idx["val"] = i
        print(f"  Q{i:02d}: {q[:60]}...")
        metrics = run_query(wiki_retriever, q, cross_encoder, is_wiki=True)
        wiki_results.append(metrics)
        
        cache_tag = " [CACHE HIT]" if metrics["from_cache"] else ""
        keyword_tag = ""
        if getattr(wiki_retriever, "last_provenance", None):
            if not wiki_retriever.last_provenance.embedding_model_resolved:
                keyword_tag = " [KEYWORD FALLBACK]"
                
        print(f"       {metrics['response_time_s']:.2f}s | "
              f"{metrics['chunks_retrieved']} chunks | "
              f"relevance={metrics['context_relevance']:.3f}{cache_tag}{keyword_tag}")

    # ── Compute summaries ─────────────────────────────────────────────────────
    def mean(results, key):
        return statistics.mean(r[key] for r in results)

    rag_summary = {
        "response_time_s":    mean(rag_results,  "response_time_s"),
        "chunks_retrieved":   mean(rag_results,  "chunks_retrieved"),
        "estimated_tokens":   mean(rag_results,  "estimated_tokens"),
        "estimated_cost_usd": mean(rag_results,  "estimated_cost_usd"),
        "context_relevance":  mean(rag_results,  "context_relevance"),
        "cache_hit_rate":     0.0,
    }
    wiki_summary = {
        "response_time_s":    mean(wiki_results, "response_time_s"),
        "chunks_retrieved":   mean(wiki_results, "chunks_retrieved"),
        "estimated_tokens":   mean(wiki_results, "estimated_tokens"),
        "estimated_cost_usd": mean(wiki_results, "estimated_cost_usd"),
        "context_relevance":  mean(wiki_results, "context_relevance"),
        "cache_hit_rate":     mean(wiki_results, "cache_hit_rate"),
    }

    # ── Save raw results ──────────────────────────────────────────────────────
    raw = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "backend":        "pinecone" if PINECONE_API_KEY else "chroma",
        "questions":      QUESTIONS,
        "rag_results":    rag_results,
        "wiki_results":   wiki_results,
        "rag_summary":    rag_summary,
        "wiki_summary":   wiki_summary,
        "auto_save_log":  auto_save_log,
    }
    with open(OUTPUT_DIR / "raw_results.json", "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\n  Raw results saved to benchmark_results/raw_results.json")

    # ── Generate plots ────────────────────────────────────────────────────────
    print(f"\n[4/4] Generating charts...")

    bar_comparison(QUESTIONS,
        [r["response_time_s"] for r in rag_results],
        [r["response_time_s"] for r in wiki_results],
        "Response Time per Query (lower is better)", "Seconds",
        "01_response_time.png", higher_is_better=False)

    bar_comparison(QUESTIONS,
        [r["estimated_tokens"] for r in rag_results],
        [r["estimated_tokens"] for r in wiki_results],
        "Estimated Tokens per Query (lower = cheaper context)", "Tokens (estimated)",
        "02_tokens.png", higher_is_better=False)

    bar_comparison(QUESTIONS,
        [r["context_relevance"] for r in rag_results],
        [r["context_relevance"] for r in wiki_results],
        "Context Relevance per Query (higher is better)", "Cosine Similarity",
        "03_context_relevance.png", higher_is_better=True)

    bar_comparison(QUESTIONS,
        [r["chunks_retrieved"] for r in rag_results],
        [r["chunks_retrieved"] for r in wiki_results],
        "Chunks Retrieved per Query", "Number of Document Chunks",
        "04_chunks_retrieved.png", higher_is_better=True)

    cache_hit_timeline(wiki_results, auto_save_log, "05_cache_hit_timeline.png")
    response_time_distribution(rag_results, wiki_results, "06_response_time_distribution.png")
    summary_radar(rag_summary, wiki_summary, "07_radar_summary.png")
    summary_table_chart(rag_summary, wiki_summary, "08_summary_table.png")

    # ── Print console summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Backend: {'Pinecone (remote)' if PINECONE_API_KEY else 'Chroma (local)'}")
    print(f"Warmup: None (cold start)\n")
    print(f"{'Metric':<30} {'Plain RAG':>12} {'RAG-Wiki':>12} {'Delta':>12}")
    print("-" * 66)

    rows = [
        ("Avg Response Time (s)",  "response_time_s",    False, ".3f"),
        ("Avg Chunks Retrieved",   "chunks_retrieved",   True,  ".1f"),
        ("Avg Tokens Used",        "estimated_tokens",   False, ".0f"),
        ("Avg Est. Cost (USD)",    "estimated_cost_usd", False, ".6f"),
        ("Avg Context Relevance",  "context_relevance",  True,  ".4f"),
        ("Avg Cache Hit Rate",     "cache_hit_rate",     True,  ".3f"),
    ]
    for label, key, hib, fmt in rows:
        rv, wv = rag_summary[key], wiki_summary[key]
        delta = wv - rv
        sign = "+" if delta >= 0 else ""
        tag = ("✓ wiki" if wv > rv else ("✓ rag" if rv > wv else "tie")) if hib \
              else ("✓ wiki" if wv < rv else ("✓ rag" if rv < wv else "tie"))
        print(f"{label:<30} {format(rv, fmt):>12} {format(wv, fmt):>12} "
              f"{sign}{delta:>+.4f}  {tag}")

    total_cache_hits = sum(1 for r in wiki_results if r["from_cache"])
    print(f"\nRAG-Wiki cache hits: {total_cache_hits}/{n} queries "
          f"({total_cache_hits/n*100:.0f}%)")
    print(f"Auto-save events: {len(auto_save_log)} (at queries: {auto_save_log})")
    print(f"\nAll charts saved to: {OUTPUT_DIR.resolve()}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

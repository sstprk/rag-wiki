class HybridRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        # 1. inject PINNED docs
        # 2. check CLAIMED cache (semantic match)
        # 3. fallback to global_retriever
        # 4. increment fetch counters
        # 5. check thresholds → emit suggestions
        # 6. return docs + provenance metadata
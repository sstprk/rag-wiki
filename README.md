# Hybrid RAG + User-Curated Knowledge Base

## Overview

This project proposes a hybrid knowledge retrieval architecture that combines traditional Retrieval-Augmented Generation (RAG) with a user-driven, persistent knowledge layer.

The system is designed to evolve from a purely probabilistic retrieval model into a semi-deterministic, user-informed knowledge system. It enables the model to learn which documents are consistently useful and prioritize them over time, reducing noise, improving relevance, and increasing transparency.

---

## Motivation

Standard RAG systems suffer from several limitations:

* **Noisy retrieval** due to vector similarity approximations
* **Loss of document context** caused by chunking
* **Stateless behavior**, where useful documents are not remembered
* **Hallucination risk** when combining unrelated chunks
* **Lack of user control** over what knowledge is considered important

This architecture addresses these issues by introducing:

* Persistent user-level knowledge memory
* Document-level awareness
* Retrieval feedback loops
* Transparent context provenance

---

## Core Architecture

The system consists of three main layers:

### 1. Global RAG Layer

* Standard vector database
* Documents are:

  * Split into chunks
  * Embedded
  * Stored with metadata

**Metadata includes:**

* `document_id`
* `document_title`
* `section`
* `source_path`
* `chunk_index`

This layer is shared across all users and acts as the primary discovery mechanism.

---

### 2. User-Local Knowledge Layer

A personalized knowledge base built from documents the user has explicitly or implicitly selected.

**Key properties:**

* Stores documents (or partial documents) promoted from RAG results
* Acts as a high-priority retrieval source
* Evolves based on user interaction

---

### 3. Transparency / Wiki Layer

A user-facing layer that:

* Displays which documents were used in responses
* Shows origin and provenance of retrieved chunks
* Enables users to:

  * Understand context sources
  * Decide whether to persist documents

---

## Retrieval Flow

### Step-by-step pipeline:

1. **User query received**

2. **Local Knowledge Check**

   * Search user-local documents first
   * If relevant matches found → prioritize them

3. **Global RAG Retrieval**

   * Perform semantic search in vector database
   * Retrieve top-k chunks

4. **Merge & Rerank**

   * Combine local and global results
   * Rerank based on:

     * Relevance
     * Source priority (local > global)

5. **Context Assembly**

   * Construct final prompt context
   * Maintain document boundaries where possible

6. **Response Generation**

7. **Transparency Output**

   * Show:

     * Which documents were used
     * Whether they came from:

       * Global RAG
       * User-local storage

---

## Document Promotion Mechanism

A key innovation in this system is **promotion from retrieval to persistent knowledge**.

### Trigger Conditions

Instead of prompting users every time, promotion is based on repeated retrieval patterns.

A document becomes eligible for promotion when:

* Multiple chunks from the same document are retrieved frequently
* The document consistently ranks high in relevance

### Example Threshold Logic

```python
promotion_score =
    frequency_weight * retrieval_count
  + relevance_weight * avg_similarity
  + recency_weight * recent_usage
```

When `promotion_score` exceeds a threshold:

* The user is prompted:

  > "This document has been frequently used. Would you like to save it?"

---

## Promotion Outcomes

If user accepts:

* Document is added to **User-Local Knowledge Layer**
* Retrieval priority increases
* Future queries check this document first

If user declines:

* No action taken
* System continues tracking usage

---

## Storage Strategy

Three possible approaches:

### 1. Full Document Storage

* Entire document saved locally
* Pros: complete context
* Cons: higher storage cost

### 2. Partial Storage (Chunk + Neighbors)

* Only relevant sections stored
* Pros: efficient
* Cons: may miss broader context

### 3. Lazy Reference Model

* Store pointer to original document
* Fetch full content only when needed

---

## Decay & Feedback Loop

To prevent overfitting:

* Documents are not permanently prioritized
* Relevance is continuously evaluated

### Decay Factors:

* Lack of recent usage
* Lower relevance scores over time

### Result:

* Frequently used documents → stay prioritized
* Irrelevant ones → gradually demoted

---

## Retrieval Prioritization Strategy

The system uses a tiered retrieval approach:

1. **User-Local Knowledge**
2. **Global RAG**
3. **Merged + Reranked Results**

Local knowledge is prioritized but **never exclusively trusted** to avoid bias and staleness.

---

## Benefits

### 1. Reduced Hallucination

* Stronger document grounding
* Less cross-document mixing

### 2. Improved Retrieval Accuracy

* Learns from user behavior
* Reduces reliance on similarity alone

### 3. Personalized Knowledge Base

* Adapts to individual users
* Builds long-term context

### 4. Transparency & Trust

* Clear source attribution
* Inspectable reasoning context

### 5. Efficiency Gains

* Fewer repeated vector searches
* Faster responses for known topics

---

## Challenges & Considerations

### Threshold Tuning

* Too low → user fatigue
* Too high → missed opportunities

### Storage Management

* Handling large documents
* Version control for updated sources

### Query Routing Complexity

* Balancing local vs global retrieval
* Avoiding bias toward old data

### Cold Start Problem

* System behaves like standard RAG initially
* Improves over time with usage

---

## Future Improvements

* Automatic topic/domain clustering
* Smarter promotion scoring models
* Cross-user shared learning (optional)
* UI enhancements for document exploration
* Version-aware document tracking

---

## Use Cases

* Developer documentation assistants
* Research tools
* Enterprise knowledge systems
* Legal and academic workflows
* Personal knowledge management systems

---

## Summary

This architecture extends traditional RAG by introducing a user-driven memory layer that transforms retrieval from a stateless process into a continuously learning system.

Instead of repeatedly searching for relevant information, the system learns what matters to the user and adapts accordingly—balancing global knowledge discovery with personalized, persistent context.

---

## License

MIT License (or specify your preferred license)

---

## Contributing

Contributions are welcome. Please open issues or submit pull requests for improvements, ideas, or bug fixes.

---

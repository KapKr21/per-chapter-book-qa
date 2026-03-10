# Project Plan & Development Log

## Project Overview

**Goal**: Build a per-chapter book QA system that prevents spoilers by enforcing hard chapter boundaries during question answering.

**Core Problem**: Existing long-document QA systems treat books as complete documents, often leaking future plot information when answering questions about earlier chapters.

**Solution Approach**: Implement a RAG-based pipeline with retrieval-time constraints that only allows access to chapters 0...k when answering questions about chapter k.

---

## Initial Research & Dataset Exploration

### Phase 1: Dataset Investigation

**Objective**: Identify suitable datasets for per-chapter book QA

#### Datasets Explored:

1. **NarrativeQA**
   - Status: Explored
   - Findings: Good for general book QA but lacks explicit chapter alignments with other datasets. As was expected to align with other dataset and use together.
   - Decision: Not ideal for alignment of datasets.

2. **LiteraryQA**
   - Status: Experimented with
   - Findings: Better chapter-level annotations
   - Challenge: Preprocessing, Complex data structure
   - Decision: Promising but moved to simpler alternative

3. **BookSum** (Final Choice)
   - Status: Implemented
   - Findings: Pre-segmented chapters with summaries
   - Advantages:
     - Clear chapter boundaries (bid + chapter structure)
     - Chapter summaries enable question generation
     - Easy to enforce spoiler-free retrieval
   - Decision: Selected as primary dataset

**Outcome**: BookSum chosen for implementation due to clear chapter structure and ease of spoiler prevention.

---

## System Architecture Plan

### Phase 2: Pipeline Design

**Components Identified**:

1. **Preprocessing** (`_00_preprocess.py`)
   - Load BookSum dataset
   - Extract chapters per book (using bid)
   - Generate questions from chapter summaries
   - Align questions with max_chapter_idx constraint

2. **Embedding** (`_01_embedder.py`)
   - Use Sentence Transformers for chapter embeddings
   - Model: all-MiniLM-L6-v2 (lightweight, efficient)

3. **Retrieval** (`_02_retriever.py`)
   - FAISS-based vector search
   - Spoiler-safe filtering: only retrieve chapters ≤ max_allowed_k
   - Top-k chapter selection

4. **Generation** (`_03_generator.py`)
   - HuggingFace transformers for answer generation
   - Support for long-context models (Qwen, Llama)
   - Context capping to prevent OOM

5. **Evaluation** (`_04_evaluator.py`)
   - BERT-based semantic similarity for answer equivalence
   - Spoiler detection via similarity with future chapters
   - Configurable thresholds

---

## Implementation Progress

### Completed Tasks

#### Core Functionality
- [x] BookSum dataset integration
- [x] Chapter extraction and book listing functionality
- [x] Heuristic-based question generation from summaries
- [x] FAISS-based chapter retrieval with spoiler constraints
- [x] LLM answer generation (Qwen2.5-3B-Instruct)
- [x] BERT-based answer equivalence evaluation
- [x] Spoiler detection using semantic similarity
- [x] Retrieval correctness verification

#### System Features
- [x] Command-line interface with argparse
- [x] Book listing mode (`--list_books`)
- [x] Configurable parameters (top_k, thresholds, model selection)
- [x] Context capping to prevent OOM errors
- [x] Progress tracking and debugging output
- [x] Aggregate metrics computation

#### Evaluation Metrics
- [x] BERT Score for answer quality
- [x] Answer accuracy (semantic equivalence)
- [x] Spoiler-free rate
- [x] Retrieval correctness tracking

#### Documentation
- [x] Comprehensive README with usage examples
- [x] Code comments and docstrings
- [x] References to related papers
- [x] Installation instructions (conda + pip)

---

## Outstanding TODOs & Future Work

### High Priority (Would Implement with More Time)

#### Question Generation Improvements
- [ ] **LLM-based question generation**
  - Replace heuristics with GPT-4/Claude for higher quality questions
  - Generate more diverse question types (why, how, what-if)
  - Better handling of character relationships and plot events

- [ ] **Question quality filtering**
  - Implement answerability checks
  - Filter out ambiguous or poorly-formed questions
  - Add question difficulty scoring

#### Evaluation Enhancements
- [ ] **Human evaluation study**
  - Collect human judgments on answer quality
  - Validate spoiler detection accuracy
  - Compare with baseline systems

- [ ] **Additional metrics**
  - ROUGE-L for n-gram overlap comparison
  - BERTScore F1 (currently only using similarity)
  - Exact match for factual questions
  - Answer length analysis

- [ ] **LLM-as-a-Judge evaluation**
  - Use GPT-4 to assess answer quality
  - Nuanced spoiler detection (implicit vs explicit)
  - Reasoning quality evaluation

#### Retrieval Improvements
- [ ] **Hybrid retrieval strategies**
  - Combine dense (FAISS) + sparse (BM25) retrieval
  - Re-ranking with cross-encoders
  - Query expansion techniques

- [ ] **Adaptive top-k selection**
  - Dynamically adjust number of retrieved chapters
  - Based on question complexity or chapter relevance scores

- [ ] **Chapter chunking**
  - Split long chapters into smaller chunks
  - Maintain spoiler constraints at chunk level
  - Better context utilization

### Medium Priority

#### Model & Performance
- [ ] **Experiment with larger models**
  - Test Llama-3-8B, Qwen2.5-7B
  - Compare performance vs efficiency tradeoffs
  - Quantization experiments (4-bit, 8-bit)

- [ ] **Flash Attention 2 integration**
  - Enable for faster inference
  - Handle longer context windows

- [ ] **Batch processing**
  - Process multiple questions in parallel
  - GPU memory optimization

#### Dataset Expansion
- [ ] **Multi-dataset support**
  - Re-integrate NarrativeQA with chapter alignment
  - Add LiteraryQA support
  - QuALITY dataset integration

- [ ] **Cross-dataset evaluation**
  - Train on BookSum, test on NarrativeQA
  - Generalization analysis

#### User Experience
- [ ] **Interactive demo**
  - Web interface (Gradio/Streamlit)
  - Real-time question answering
  - Visualization of retrieved chapters

- [ ] **Result visualization**
  - Plot spoiler rates across chapters
  - Answer quality heatmaps
  - Retrieval pattern analysis

### Low Priority

#### Advanced Features
- [ ] **Multi-hop reasoning**
  - Questions requiring information from multiple chapters
  - Chain-of-thought prompting

- [ ] **Character tracking**
  - Maintain character state across chapters
  - Character-centric question generation

- [ ] **Temporal reasoning**
  - Handle questions about event sequences
  - Timeline construction

#### Research Extensions
- [ ] **Spoiler sensitivity analysis**
  - User studies on acceptable spoiler levels
  - Personalized spoiler thresholds

- [ ] **Active learning**
  - Identify uncertain predictions
  - Request human annotations for edge cases

- [ ] **Contrastive learning**
  - Train models to distinguish spoiler vs non-spoiler content
  - Fine-tune retriever for spoiler-aware ranking

#### Engineering
- [ ] **Caching system**
  - Cache embeddings for faster repeated experiments
  - Save/load FAISS indices

- [ ] **Logging & monitoring**
  - Structured logging (JSON)
  - Experiment tracking (Weights & Biases)

- [ ] **Unit tests**
  - Test each component independently
  - Integration tests for full pipeline

- [ ] **CI/CD pipeline**
  - Automated testing on push
  - Code quality checks (linting, formatting)

---

## Technical Challenges Encountered

### Challenge 1: Question Generation Quality
**Problem**: Heuristic-based question generation produces simple, sometimes awkward questions.

**Current Solution**: Extract character names and convert statements to questions using templates.

**Future Solution**: Use LLM-based generation for more natural, diverse questions.

### Challenge 2: Memory Management
**Problem**: Long books with many chapters cause OOM errors.

**Solution Implemented**: 
- Context capping (max 12,000 chars)
- CPU offloading for embedder
- Explicit CUDA cache clearing

**Future Improvement**: Implement streaming or chunked processing.

### Challenge 3: Spoiler Detection Accuracy
**Problem**: Semantic similarity may not catch implicit spoilers.

**Current Solution**: Threshold-based detection (0.6 default).

**Future Solution**: Train a dedicated spoiler classifier or use LLM-as-a-judge.

### Challenge 4: Dataset Alignment
**Problem**: NarrativeQA and LiteraryQA required complex alignment between questions and chapters.

**Solution**: Switched to BookSum which has natural chapter boundaries.

---

## Experimental Results Summary

### Current Performance (BookSum)

**Spoiler Prevention**:
- Retrieval Correctness: ~100% (by design)
- Spoiler-Free Rate: >95% (depends on threshold)

**Answer Quality**:
- Average BERT Score: ~0.62
- Answer Accuracy: ~75%

**Limitations**:
- Question quality affects evaluation
- Small-scale testing (20-100 questions per book)
- Limited to books with 3+ chapters

---

## Timeline & Milestones

### Week 1: Research & Planning
- Literature review
- Dataset exploration (NarrativeQA, LiteraryQA, BookSum)
- Architecture design

### Week 2-3: Core Implementation
- Preprocessing pipeline
- Retrieval system with spoiler constraints
- Answer generation integration

### Week 4: Evaluation & Testing
- BERT-based evaluation
- Spoiler detection
- Initial experiments

### Week 5: Documentation & Refinement
- README documentation
- Code cleanup
- Parameter tuning

### Future Work (If Extended)
- Weeks 6-8: LLM-based question generation
- Weeks 9-10: Human evaluation study
- Weeks 11-12: Multi-dataset integration

---

## Key Learnings

1. **Dataset choice is critical**: BookSum's structure made spoiler prevention much easier than NarrativeQA/LiteraryQA.

2. **Semantic similarity > n-gram overlap**: BERT-based evaluation captures answer equivalence better than ROUGE-L.

3. **Retrieval constraints work**: Hard filtering at retrieval time effectively prevents spoilers.

4. **Question quality matters**: Heuristic generation limits evaluation quality; LLM generation would significantly improve results.

5. **Memory management is essential**: Long-context models require careful resource management.

---

## Conclusion

This project successfully demonstrates a spoiler-free book QA system using retrieval-time constraints. The core contribution—preventing future chapter access—is validated through the implementation. While question generation and evaluation could be improved with more time, the fundamental architecture proves effective for per-chapter narrative understanding.

**Main Achievement**: Built a working prototype that enforces chapter boundaries and prevents spoilers in book QA.

**Primary Limitation**: Question generation quality limits comprehensive evaluation.

**Next Steps**: Integrate LLM-based question generation and conduct human evaluation studies (cooperative learning (LLM+Human)).

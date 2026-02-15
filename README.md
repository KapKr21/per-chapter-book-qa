# Per-Chapter-BookQA: Per-Chapter LLM Understanding

This project addresses the "spoiler" problem in long-document Question Answering (QA). While many existing systems perform QA over entire books, they often accidentally leak future plot information. Per-Chapter-BookQA enforces hard chapter boundaries to ensure the AI only "knows" what a reader would know up to a specific point.

## The Core Research Goal
Can we build a per-chapter book QA system that prevents spoilers?

* **Current Gap**: Existing approaches treat books as full documents (e.g., NarrativeQA) without a notion of reading progress.
* **Our Solution**: Implement a RAG-style pipeline with restricted retrieval—querying only Chapters $1 \dots k$ for a question about Chapter $k$.

## System Architecture
The pipeline is built using **PyTorch** and the **Hugging Face** ecosystem:

1. **Preprocessing**: Splits books into chapters using **BOOKSUM** segmentation and aligns **NarrativeQA** questions to specific chapters.
2. **Restricted Retrieval**: Uses **Sentence Transformers** and **FAISS** to index chapters individually, physically preventing the retrieval of "future" context.
3. **Answer Generation**: Passes retrieved passages into a modern, long-context LLM to handle free-form answering.
4. **Evaluation**: Combines **ROUGE-L** with **LLM-as-a-Judge** and a custom **Spoiler-Violation Rate** metric.

## Setup & Installation

### 1. Cloning the Repository

```bash
git clone [https://github.com/your-username/BookQA.git](https://github.com/KapKr21/per-chapter-book-qa.git)
cd per-chapter-book-qa
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate per-chapter-book-qa
```

## Datasets

This project anchors its research on the following primary datasets:

- NarrativeQA: For the core QA tasks.

- BOOKSUM: For chapter-level segmentation and summarization data.

- LiteraryQA: Used to justify evaluation choices and LLM-as-a-judge grounding.
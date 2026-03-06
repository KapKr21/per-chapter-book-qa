# Per-Chapter-BookQA: Per-Chapter LLM Understanding

This project addresses the "spoiler" problem in long-document Question Answering (QA). While many existing systems perform QA over entire books, they often accidentally leak future plot information. Per-Chapter-BookQA enforces hard chapter boundaries to ensure the AI only "knows" what a reader would know up to a specific point.

## The Core Research Goal
Can we build a per-chapter book QA system that prevents spoilers?

* **Current Gap**: Existing approaches treat books as full documents without a notion of reading progress.
* **Our Solution**: Implement a RAG-style pipeline with restricted retrieval—querying only Chapters $1 \dots k$ for a question about Chapter $k$.

## System Architecture
The pipeline is built using **PyTorch** and the **Hugging Face** ecosystem:

1. **Preprocessing**: Uses **BookSum** dataset for chapter segmentation. Questions are generated from chapter summaries.
2. **Restricted Retrieval**: Uses **Sentence Transformers** and **FAISS** to index chapters individually, physically preventing the retrieval of "future" context.
3. **Answer Generation**: Passes retrieved passages into a modern, long-context LLM to handle free-form answering.
4. **Evaluation**: Uses **BERT-based semantic similarity** for answer equivalence and spoiler detection (instead of ROUGE-L).

## Key Innovation: BERT Answer Equivalence

Following best practices in QA evaluation, we use **semantic similarity** instead of n-gram overlap:

- **BERT Score**: Measures semantic equivalence between predicted and ground truth answers
- **Spoiler Detection**: Uses semantic similarity to detect if answers contain future chapter information
- **LLM-as-a-Judge**: Optional integration for nuanced answer quality assessment

This approach better captures answer correctness compared to traditional ROUGE-L metrics.

## Setup & Installation

### 1. Cloning the Repository

```bash
git clone https://github.com/KapKr21/per-chapter-book-qa.git
cd per-chapter-book-qa
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate per-chapter-book-qa
```

Or install with pip:

```bash
pip install -r requirements.txt
```

## Datasets

This project uses:

- **BookSum**: Chapter-level book segmentation with summaries (primary dataset)

## Usage

### Quick Start

1. **List available books**:

```bash
python main_booksum.py --list_books --booksum_split "train[:5000]"
```

This shows books with their BID (Book ID) and chapter count.

2. **Run experiment on a specific book**:

```bash
# Example with book BID 145 (266 chapters)
python main_booksum.py --book_bid 145 --use_retriever --max_questions_per_chapter 2
```

### Command Line Arguments

```bash
python main_booksum.py \
  --book_bid 145 \                      # BookSum book ID
  --booksum_split "train[:5000]" \      # Dataset split to load
  --max_questions_per_chapter 3 \       # Questions per chapter
  --use_retriever \                     # Enable FAISS retrieval (recommended)
  --top_k 2 \                           # Retrieve top-k chapters
  --model_id "Qwen/Qwen2.5-3B-Instruct" # LLM for answer generation
```

### Example Output

```
EXPERIMENT SUMMARY
============================================================
Book BID: 145
Total Chapters: 266
Total Questions: 798

Answer Quality Metrics:
  Average BERT Score: 0.6234
  Answer Accuracy: 0.7500 (600/800)

Spoiler Detection Metrics:
  Spoiler Rate: 0.0523 (42/798 answerable)
  Spoiler-Free Rate: 0.9477
  Avg Spoiler Score: 0.1234
============================================================
```

## How It Works

1. **Chapter Segmentation**: BookSum provides pre-segmented chapters
2. **Question Generation**: Simple heuristics convert chapter summaries into questions
3. **Spoiler Prevention**: For a question about chapter k, only chapters 0...k are retrievable
4. **Answer Generation**: LLM generates answers using only "safe" (non-spoiler) context
5. **Evaluation**: Checks if generated answers contain information from future chapters

## Key Features

- **Spoiler-Free**: Hard constraint prevents accessing future chapters
- **Scalable**: FAISS-based retrieval handles long books efficiently
- **Simple**: Uses only BookSum dataset (no complex data alignment needed)
- **Extensible**: Easy to add better question generation (e.g., using LLMs)
- **Modern Evaluation**: BERT-based semantic similarity for answer equivalence
- **Research-Grade**: Follows best practices from recent QA literature

## Project Structure

```
per-chapter-book-qa/
├── main_booksum.py              # Main entry point for BookSum-only experiments
├── src/
│   ├── preprocess.py            # BookSum preprocessing & question generation
│   ├── embedder.py              # Chapter embedding with Sentence Transformers
│   ├── retriever.py             # Spoiler-safe chapter retrieval with FAISS
│   ├── generator.py             # Answer generation with LLMs
│   └── evaluator.py             # BERT-based answer equivalence + spoiler detection
├── requirements.txt             # Python dependencies
└── environment.yml              # Conda environment specification
```

## Improving Question Quality

The current implementation uses simple heuristics to generate questions from chapter summaries. For better quality, you can:

1. **Use an LLM for question generation**:
```python
# In src/preprocess_booksum.py, modify _generate_questions_from_summary()
# to use GPT-4, Claude, or another LLM to generate questions
```

2. **Manual annotation**: Create a small set of high-quality questions for evaluation

3. **Hybrid approach**: Combine generated questions with human-written ones

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{per-chapter-bookqa,
  title={Per-Chapter Book QA: Preventing Spoilers in Long-Document Understanding},
  author={Your Name},
  year={2026},
  url={https://github.com/KapKr21/per-chapter-book-qa}
}
```

## License

MIT License - see LICENSE file for details.

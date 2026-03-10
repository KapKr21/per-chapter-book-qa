# Per-Chapter-BookQA: Per-Chapter LLM Understanding

This project addresses the "spoiler" problem in long-document Question Answering (QA). While many existing systems perform QA over entire books, they often accidentally leak future plot information. Per-Chapter-BookQA enforces hard chapter boundaries to ensure the AI only "knows" what a reader would know up to a specific point.

## The Core Research Goal
Can we build a per-chapter book QA system that prevents spoilers?

* **Current Gap**: Existing approaches treat books as full documents without a notion of reading progress.
* **The Solution**: Implementing a RAG-style pipeline with restricted retrieval—querying only Chapters $1 \dots k$ for a question about Chapter $k$.

## System Architecture
The pipeline is built using **PyTorch** and the **Hugging Face** ecosystem:

1. **Preprocessing**: Uses **BookSum** dataset for chapter segmentation. Questions are generated from chapter summaries.
2. **Restricted Retrieval**: Uses **Sentence Transformers** and **FAISS** to index chapters individually, physically preventing the retrieval of "future" context.
3. **Answer Generation**: Passes retrieved passages into a modern, long-context LLM to handle free-form answering.
4. **Evaluation**: Uses **BERT-based semantic similarity** for answer equivalence and spoiler detection.

## Key Innovation: BERT Answer Equivalence

Following best practices in QA evaluation, the project uses **semantic similarity** instead of n-gram overlap:

- **BERT Score**: Measures semantic equivalence between predicted and ground truth answers using all-MiniLM-L6-v2
- **Spoiler Detection**: Uses semantic similarity to detect if answers contain future chapter information
- **Configurable Thresholds**: Adjustable similarity thresholds for answer equivalence (default: 0.5) and spoiler detection (default: 0.6)

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
python main.py --list_books --booksum_split "train[:5000]"
```

This shows books with their BID (Book ID) and chapter count.

2. **Run experiment on a specific book**:

```bash
#Quick test with 5 questions (recommended for first run)
python main.py --book_bid 145 --use_retriever --max_total_questions 05

#Medium test with 11 questions
python main.py --book_bid 145 --use_retriever --max_total_questions 11

#Full experiment with 20 questions
python main.py --book_bid 145 --use_retriever --max_total_questions 20
```

### Command Line Arguments

```bash
python main.py \
  --book_bid 145 \                      # BookSum book ID
  --booksum_split "train[:5000]" \      # Dataset split to load
  --max_questions_per_chapter 1 \       # Questions per chapter (default: 1)
  --max_total_questions 5 \             # Total questions to evaluate (default: 20)
  --use_retriever \                     # Enable FAISS retrieval
  --top_k 2 \                           # Retrieve top-k chapters (default: 2)
  --model_id "Qwen/Qwen2.5-3B-Instruct" \ # LLM for answer generation
  --max_new_tokens 64 \                 # Max tokens to generate (default: 64)
  --max_context_chars 12000 \           # Max context size (default: 12000)
  --spoiler_threshold 0.6               # Spoiler detection threshold (default: 0.6)
```

**Key Parameters:**
- `--max_total_questions`: Limits evaluation for faster testing (20-100 recommended)
- `--max_questions_per_chapter`: How many questions to generate per chapter (1-2 recommended)
- `--use_retriever`: Always use this for proper spoiler prevention
- `--spoiler_threshold`: Higher values = less sensitive spoiler detection (0-1 range)

### Example Output

```
Experiment Summary

Book ID: 145
Book Title: [Book Title]
Total Chapters: 266
Total Questions: 20

Spoiler Prevention (Core Contribution):
  Retrieval Correctness: 20/20 (100%)
  Spoiler-Free Rate: 1.0000
  System Design: Only chapters 0...k accessible for chapter k
  Result: PERFECT

Answer Quality Metrics:
  Average BERT Score: 0.6234
  Answer Accuracy: 0.7500 (15/20)
```

## How It Works

1. **Chapter Segmentation**: BookSum provides pre-segmented chapters with summaries
2. **Question Generation**: Heuristic-based conversion of chapter summaries into questions (extracts character names, converts statements to questions)
3. **Spoiler Prevention**: For a question about chapter k, only chapters 0...k are retrievable via FAISS index
4. **Context Capping**: Limits context to 12,000 characters by default to prevent OOM errors
5. **Answer Generation**: LLM (Qwen2.5-3B-Instruct by default) generates answers using only "safe" (non-spoiler) context
6. **Evaluation**: 
   - BERT-based semantic similarity for answer equivalence (threshold: 0.5)
   - Spoiler detection via semantic similarity with future chapters (threshold: 0.6)
   - Retrieval correctness verification

## Key Features

- **Spoiler-Free**: Hard constraint prevents accessing future chapters through retrieval-time filtering
- **Scalable**: FAISS-based retrieval handles long books efficiently
- **Simple**: Uses only BookSum dataset (no complex data alignment needed)
- **Extensible**: Easy to add better question generation (e.g., using LLMs)
- **Modern Evaluation**: BERT-based semantic similarity (all-MiniLM-L6-v2) for answer equivalence
- **Research-Grade**: Follows best practices from recent QA literature
- **Memory Efficient**: Context capping and 4-bit quantization support to prevent OOM
- **Flexible Models**: Supports any HuggingFace causal LM (Qwen, Llama, etc.)

## Project Structure

```
per-chapter-book-qa/
├── main.py                      # Main entry point for experiments
├── src/
│   ├── _00_preprocess.py        # BookSum preprocessing & question generation
│   ├── _01_embedder.py          # Chapter embedding with Sentence Transformers
│   ├── _02_retriever.py         # Spoiler-safe chapter retrieval with FAISS
│   ├── _03_generator.py         # Answer generation with LLMs (Qwen/Llama support)
│   └── _04_evaluator.py         # BERT-based answer equivalence + spoiler detection
├── requirements.txt             # Python dependencies
└── environment.yml              # Conda environment specification
```

## Improving Question Quality

The current implementation uses heuristic-based question generation from chapter summaries:
- Extracts character names (capitalized words)
- Converts statements to questions
- Generates character-focused and generic questions

For better quality, we can:

1. **Use an LLM for question generation**:
```python
# In src/_00_preprocess.py, modify _generate_questions_from_summary()
# to use GPT-4, Claude, or another LLM to generate questions
```

2. **Manual annotation**: Create a small set of high-quality questions for evaluation

3. **Hybrid approach**: Combine generated questions with human-written ones

## References & Related Work

This project builds upon several key papers in long-document QA and narrative understanding:

### Primary Datasets Explored

1. **NarrativeQA: Reading Comprehension Challenge**
   - Kočiský, T., et al. (2018). "The NarrativeQA Reading Comprehension Challenge"
   - *Transactions of the Association for Computational Linguistics*, 6, 317-328
   - [Paper](https://aclanthology.org/Q18-1023/) | [Dataset](https://github.com/deepmind/narrativeqa)
   - Explored during initial research phase

2. **LiteraryQA: Effective Evaluation of Long-document Narrative QA**
   - Efrat, A., et al. (2024). "LiteraryQA: Towards Effective Evaluation of Long-document Narrative QA"
   - *arXiv preprint arXiv:2403.10552*
   - [Paper](https://arxiv.org/abs/2403.10552)
   - Experimented with during development

3. **BookSum: Long-form Narrative Summarization** ⭐ (Final Implementation)
   - Kryscinski, W., et al. (2021). "BOOKSUM: A Collection of Datasets for Long-form Narrative Summarization"
   - *Findings of EMNLP 2021*
   - [Paper](https://aclanthology.org/2021.findings-emnlp.195/) | [Dataset](https://huggingface.co/datasets/kmfoda/booksum)
   - Primary dataset used in this implementation

### Long-Context Architectures

4. **Longformer: The Long-Document Transformer**
   - Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer"
   - *arXiv preprint arXiv:2004.05150*
   - [Paper](https://arxiv.org/abs/2004.05150)

5. **Big Bird: Transformers for Longer Sequences**
   - Zaheer, M., et al. (2020). "Big Bird: Transformers for Longer Sequences"
   - *NeurIPS 2020*
   - [Paper](https://arxiv.org/abs/2007.14062)

### Retrieval-Augmented Generation

6. **RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP**
   - Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
   - *NeurIPS 2020*
   - [Paper](https://arxiv.org/abs/2005.11401)

### Additional Long-Document QA Benchmarks

7. **QuALITY: Question Answering with Long Input Texts**
   - Pang, R. Y., et al. (2022). "QuALITY: Question Answering with Long Input Texts, Yes!"
   - *NAACL 2022*
   - [Paper](https://aclanthology.org/2022.naacl-main.391/) | [Dataset](https://github.com/nyu-mll/quality)

## Acknowledgments

This project was developed with AI assistance for brainstorming. The core research direction and spoiler-prevention methodology are original contributions.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{per-chapter-bookqa,
  title={Per-Chapter Book QA: Preventing Spoilers in Long-Document Understanding},
  author={Kapil Kumar Khatri},
  year={2026},
  url={https://github.com/KapKr21/per-chapter-book-qa}
}
```

## License

MIT License - see LICENSE file for details.
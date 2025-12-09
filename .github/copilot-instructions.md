# GitHub Copilot Instructions for IWAC-Hugging-Face

## Project Overview

This repository contains Python scripts to manage and update the **Islam West Africa Collection (IWAC)** dataset on Hugging Face Hub:
- **Dataset URL**: https://huggingface.co/datasets/fmadore/islam-west-africa-collection
- **Source**: Omeka S API at https://islam.zmo.de/api

The dataset contains multiple configurations (subsets):
- `articles` - Newspaper articles (resource_class_id = 36)
- `audiovisual` - Audiovisual documents (resource_class_id = 38)
- `documents` - General documents
- `publications` - Islamic publications
- `references` - Academic references
- `index` - Index entries

## Architecture & Key Components

### Upload Scripts
Each subset has its own upload script that:
1. Fetches items from Omeka S API with caching
2. Maps Omeka fields to flat dataset columns
3. Merges with existing HF dataset (preserving computed columns)
4. Pushes updated dataset to Hugging Face Hub

Main upload scripts:
- `upload_newspaper_hf.py` - Articles (main template with Rich console)
- `audiovisual/upload_audiovisual_hf.py`
- `document/upload_documents_hf.py`
- `index/upload_index_hf.py`
- `islamic-publications/upload_Islamic_publications_hf.py`
- `reference/upload_reference_hf.py`

### Post-Processing Scripts
Scripts that enrich the dataset with computed columns:
- `lemmatize_update_hf.py` - French lemmatization with spaCy
- `post-processing/calculate_sentiment_AI.py` - AI sentiment analysis (Gemini/ChatGPT)
- `post-processing/calculate_sentiment.py` - Rule-based sentiment
- `post-processing/calculate_lexical_richness.py` - Text statistics
- `post-processing/calculate_word_count.py` - Word counts
- `post-processing/semantic_embedding.py` - Sentence embeddings
- `post-processing/topic_modeling/` - BERTopic modeling

### Utilities
- `country_mapper.py` - Maps newspaper names to countries (Benin, Burkina Faso, Côte d'Ivoire, Niger, Togo)
- `data/fetch_datasets.py` - Download datasets locally

## Hardware Constraints

- Development is done on **CPU only** (no GPU available)
- Prefer CPU-optimized models and libraries:
  - spaCy: Use `fr_core_news_lg` instead of transformer models (`fr_dep_news_trf`)
  - Processing: Consider batch sizes and model complexity for CPU performance
  - Embeddings: Use lightweight models optimized for CPU
- Mention CPU limitations when suggesting computationally intensive operations

## Code Style & Conventions

### Console Output - Use Rich Library
**Always use Rich for console output** instead of plain print statements or basic logging:

```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich import box

console = Console()

# Logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)

# Progress bars
with Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    console=console,
) as progress:
    task = progress.add_task("[cyan]Processing...", total=100)
    # work here
    progress.update(task, advance=1)

# Status spinners for indeterminate operations
with console.status("[bold green]Loading...", spinner="dots"):
    # work here

# Panels for important information
console.print(Panel(content, title="Title", border_style="blue"))

# Tables for structured data
table = Table(title="Summary", box=box.ROUNDED)
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")
console.print(table)

# Status icons
console.print("[green]✓[/green] Success message")
console.print("[yellow]⚠[/yellow] Warning message")  
console.print("[red]✗[/red] Error message")
console.print("[blue]→[/blue] Process flow")
console.print("[yellow]ℹ[/yellow] Info message")
```

### Async Patterns
Upload scripts use async/await with aiohttp for efficient API calls:

```python
import asyncio
import aiohttp
import aiofiles

class ConnectionManager:
    """Singleton for managing HTTP session"""
    async def get(self) -> aiohttp.ClientSession:
        # Returns shared session
        
# Retry decorator for resilient API calls
def async_retry(max_tries: int = 5, exceptions=(aiohttp.ClientError, asyncio.TimeoutError)):
    # Exponential backoff retry logic
```

### Caching Strategy
API responses are cached locally to avoid redundant calls:
- Cache directory: `.cache_omk*` (gitignored)
- Format: Gzipped JSON files
- Duration: 24 hours by default

### Dataset Merge Logic
When updating datasets, preserve columns computed by post-processing:
1. Fetch fresh data from Omeka API
2. Load existing dataset from HF Hub
3. Identify columns only in existing dataset (computed columns)
4. Merge on `o:id` field, keeping computed columns for existing items

### Environment Variables
Required in `.env`:
```
OMEKA_BASE_URL=https://islam.zmo.de/api
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential
HF_TOKEN=your_huggingface_token
GOOGLE_API_KEY=your_gemini_key  # For AI sentiment
CHATGPT=your_openai_key         # For AI sentiment
```

## Common Field Mappings

Omeka fields to dataset columns:
- `o:id` → `o:id` (primary key)
- `dcterms:title` → `title`
- `dcterms:creator` → `author`
- `dcterms:date` → `pub_date`
- `dcterms:publisher` → `newspaper`
- `dcterms:subject` → `subject` (pipe-separated)
- `dcterms:language` → `language`
- `dcterms:spatial` → `spatial`
- `bibo:content` → `OCR`
- `fabio:hasURL` → `URL`

## Dependencies

Key libraries:
- `datasets`, `huggingface_hub` - HF Hub interaction
- `aiohttp`, `aiofiles` - Async HTTP/file operations
- `rich` - Beautiful console output (REQUIRED)
- `pandas`, `pyarrow` - Data manipulation
- `spacy` - French NLP (lemmatization)
- `bertopic`, `sentence-transformers` - Topic modeling
- `google.genai`, `openai` - AI sentiment analysis

## Testing Changes

Before pushing to HF Hub:
1. Test with a small subset of items
2. Verify merge logic preserves existing computed columns
3. Check that `o:id` is present and non-null
4. Review the dataset summary table output

## Digital Humanities Best Practices

This is a **research dataset for Digital Humanities** studying Islam in West Africa. Code suggestions should follow DH principles:

### Domain Relevance
- **Never filter out domain-specific terms** as stopwords (Islamic organizations like COSIM, FAIB, UIB; religious events like Ramadan, Tabaski, Maouloud; religious figures and titles)
- These terms are **core to the research** - they should appear in topic labels, analyses, and visualizations
- Only remove truly non-informative noise: OCR artifacts, English stopwords (from non-French docs), generic boilerplate

### Topic Modeling Guidelines
When working with `post-processing/topic_modeling/`:
- **Target 60-100 meaningful topics** for ~12K documents (not 500+ fragmented clusters)
- Prioritize **topic coherence** over topic count - use coherence metrics (C_v ≥ 0.4)
- Reduce outliers aggressively (35%+ outliers indicates poor clustering)
- Keep Islamic/religious terminology in topic labels - they're semantically meaningful
- Use `--nr-topics` to merge similar topics if too fragmented

### Reproducibility
- Always save parameters in `training_parameters.json`
- Document coherence metrics for academic accountability
- Use fixed random seeds (42) for reproducible results

### Stopwords Philosophy
```python
# BAD - removes domain-relevant terms:
stopwords = {"ramadan", "tabaski", "cosim", "imam", "mosquée"}

# GOOD - only removes noise:
stopwords = {"the", "of", "and", "lp", "bf", "monsieur", "fcfa"}
```

## Language

- Code comments: English or French (existing codebase uses both)
- Console messages: English preferred for new code
- Documentation: English

# Transformers & LangChain Examples

A collection of Python scripts for working with LLMs, LangChain chains, Hugging Face models, and sentence embeddings.

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy the example environment file and add your API keys:

```powershell
copy .env_example .env
```

## Environment Variables

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | OpenAI chat and chain scripts |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI scripts |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Azure OpenAI scripts |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI scripts |
| `GOOGLE_API_KEY` | Gemini scripts |
| `GEMINI_MODEL` | Gemini scripts (default: `gemini-2.5-flash`) |
| `HUGGINGFACEHUB_API_TOKEN` | Hugging Face model downloads |

## Scripts

### Chat Models

| Script | Description |
|---|---|
| `chat_model_basic_openai.py` | Basic OpenAI chat using `init_chat_model` |
| `chat_model_basic_openai_azure.py` | Azure OpenAI chat via the v1 API |
| `chat_model_basic_gemini.py` | Google Gemini chat via the Gemini Developer API |

### LangChain Chains (LCEL)

| Script | Description |
|---|---|
| `chains_openai.py` | Simple prompt → model chain |
| `chains_under_the_hood.py` | How LCEL chains work internally |
| `chains_parallel.py` | Parallel runnables with `RunnableParallel` |
| `chains_branching_openai.py` | Conditional routing with `RunnableBranch` |

### Hugging Face & Transformers

| Script | Description |
|---|---|
| `simple_transformer.py` | Sentiment analysis with the `transformers` pipeline |
| `langchain_hug.py` | Local Q&A with `google/flan-t5-large` via LangChain |
| `create_embeddings.py` | Text embeddings with `sentence-transformers` |

## Usage

Run any script from the project root:

```powershell
python chat_model_basic_openai.py
python chains_parallel.py
python langchain_hug.py
python create_embeddings.py
```

## Notes

- **OpenAI / Azure / Gemini scripts** require the corresponding API keys in `.env`.
- **`langchain_hug.py`** downloads `google/flan-t5-large` (~3 GB) on first run and runs locally on CPU by default.
- **`create_embeddings.py`** downloads `all-mpnet-base-v2` on first run.
- **`simple_transformer.py`** downloads a default text-classification model on first run.
- For GPU support with PyTorch, install the appropriate CUDA build from [pytorch.org](https://pytorch.org) before installing other dependencies.

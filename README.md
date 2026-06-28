# Transformers & LangChain Examples

A collection of Python scripts for working with LLMs, LangChain chains, chat history, Hugging Face models, and sentence embeddings.

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

Scripts are numbered by topic: `1x` = chat models, `2x` = chains.

### `11_chat_model_basic_openai.py`

A minimal introduction to calling OpenAI chat models with LangChain. Loads `OPENAI_API_KEY` from `.env`, initializes a model via `init_chat_model("gpt-5.4-nano")`, sends a single `HumanMessage`, and prints the text response. Use this as the starting point for OpenAI integration.

**Requires:** `OPENAI_API_KEY`

---

### `12_chat_model_basic_gemini.py`

Calls Google Gemini through LangChain's Google GenAI integration. Loads `GOOGLE_API_KEY` from `.env` and optionally `GEMINI_MODEL` (defaults to `gemini-2.5-flash`). Initializes the model with `model_provider="google_genai"` and sends a simple math question to verify the connection.

**Requires:** `GOOGLE_API_KEY`

---

### `13_chat_model_basic_openai_azure.py`

Connects to Azure OpenAI using the v1 API. Reads `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_DEPLOYMENT_NAME` from `.env`, then passes them to `init_chat_model` with `model_provider="openai"` and a custom `base_url`. Demonstrates how to use your Azure deployment name instead of a public OpenAI model name.

**Requires:** `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME`

---

### `14_chat_model_history.py`

An interactive chat loop with in-memory conversation history. Maintains a Python list of `SystemMessage`, `HumanMessage`, and `AIMessage` objects and passes the full history to the model on each turn so the AI remembers prior context within the session. History is lost when the script exits.

**Requires:** `OPENAI_API_KEY`

---

### `15_chat_model_history_persistant_memory.py`

An interactive chat loop with persistent conversation history stored in SQLite. Uses LangChain's standard `SQLChatMessageHistory` (from `langchain-community`) and `RunnableWithMessageHistory` to automatically load and save messages to `chat_history.db`. Re-running the script resumes the same session. A system prompt is seeded once for new sessions.

**Requires:** `OPENAI_API_KEY`

---

### `21_chains_openai.py`

Introduces LangChain Expression Language (LCEL) by building a prompt → model chain with the pipe operator (`|`). A `ChatPromptTemplate` with system and human messages is combined with an OpenAI chat model to generate jokes about a given topic. Shows the simplest form of chaining without extra output parsing.

**Requires:** `OPENAI_API_KEY`

---

### `22_chains_under_the_hood.py`

Explains what happens behind the scenes when you write `prompt | model`. Instead of using the pipe operator, it builds the same chain manually with `RunnableLambda` and `RunnableSequence`: format the prompt, invoke the model, then extract the text content. Useful for understanding how LCEL chains are composed step by step.

**Requires:** `OPENAI_API_KEY`

---

### `23_chains_parallel.py`

Demonstrates running multiple LLM calls in parallel with `RunnableParallel`. The chain first lists product features, then simultaneously generates pros and cons analyses from those features, and finally merges both results into a single review. Shows how to fan out work across branches and combine outputs in a later step.

**Requires:** `OPENAI_API_KEY`

---

### `24_chains_branching_openai.py`

Shows conditional routing with `RunnableBranch`. A classification chain first determines whether customer feedback is positive, negative, neutral, or needs escalation. Based on that label, the chain routes to a different prompt template and response strategy. Mimics a simple customer-support workflow where the response depends on sentiment.

**Requires:** `OPENAI_API_KEY`

---

### `simple_transformer.py`

A standalone Hugging Face `transformers` example with no LangChain. Uses the high-level `pipeline("text-classification")` API to run sentiment analysis on two product review texts — one negative and one positive — and prints the results as pandas DataFrames. No API keys needed; the default classification model is downloaded automatically on first run.

**Requires:** None (downloads model on first run)

---

### `langchain_hug.py`

Runs a local LLM without any cloud API. Loads `google/flan-t5-large` from Hugging Face, wraps it in a custom `Seq2SeqPipeline` (needed because newer `transformers` versions removed the `text2text-generation` task), and chains it with a `PromptTemplate` via LangChain's `HuggingFacePipeline`. Answers two open-ended questions step by step. Runs on CPU by default; first run downloads ~3 GB.

**Requires:** `HUGGINGFACEHUB_API_TOKEN` (optional, for gated models or faster downloads)

---

### `create_embeddings.py`

Generates dense vector embeddings for text using `sentence-transformers`. Loads the `all-mpnet-base-v2` model, encodes a sample sentence about access permissions, and prints the embedding vector along with its dimension (768). Useful for semantic search, clustering, or similarity matching. No API keys needed; the model is downloaded on first run.

**Requires:** None (downloads model on first run)

## Usage

Run any script from the project root:

```powershell
python 11_chat_model_basic_openai.py
python 14_chat_model_history.py
python 15_chat_model_history_persistant_memory.py
python 23_chains_parallel.py
python langchain_hug.py
python create_embeddings.py
```

Interactive chat scripts (`14_*`, `15_*`) accept input until you type `exit`.

## Notes

- **OpenAI / Azure / Gemini scripts** require the corresponding API keys in `.env`.
- **`15_chat_model_history_persistant_memory.py`** creates `chat_history.db` in the project directory. Delete this file to start a fresh conversation.
- **`langchain_hug.py`** downloads `google/flan-t5-large` (~3 GB) on first run and runs locally on CPU by default.
- **`create_embeddings.py`** downloads `all-mpnet-base-v2` on first run.
- **`simple_transformer.py`** downloads a default text-classification model on first run.
- For GPU support with PyTorch, install the appropriate CUDA build from [pytorch.org](https://pytorch.org) before installing other dependencies.

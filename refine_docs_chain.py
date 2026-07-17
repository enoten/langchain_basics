"""End-to-end example: RefineDocumentsChain.

RefineDocumentsChain summarizes the first document, then iteratively refines
that answer with each remaining document — better for longer corpora that
do not fit in a single prompt.
"""

from dotenv import load_dotenv
from langchain_classic.chains import LLMChain, RefineDocumentsChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

# How each Document is formatted before being passed into a prompt.
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}",
)
document_variable_name = "context"
initial_response_name = "prev_response"

llm = OpenAI(temperature=0)

# First pass: summarize the first document alone.
initial_prompt = PromptTemplate.from_template(
    "Write a short summary of the following content:\n\n{context}\n\nSummary:"
)
initial_llm_chain = LLMChain(llm=llm, prompt=initial_prompt)

# Refine pass: update the running summary with each next document.
refine_prompt = PromptTemplate.from_template(
    "Here is your current summary:\n{prev_response}\n\n"
    "Refine and improve it using this additional content "
    "(keep it concise; keep important facts):\n\n{context}\n\n"
    "Updated summary:"
)
refine_llm_chain = LLMChain(llm=llm, prompt=refine_prompt)

chain = RefineDocumentsChain(
    initial_llm_chain=initial_llm_chain,
    refine_llm_chain=refine_llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
    initial_response_name=initial_response_name,
    return_intermediate_steps=True,  # include each step's answer in the result
)

# Sample documents refined one-by-one.
docs = [
    Document(
        page_content=(
            "LangChain is a framework for building applications powered by "
            "large language models. It provides abstractions for prompts, "
            "chains, agents, and retrieval."
        ),
        metadata={"source": "intro"},
    ),
    Document(
        page_content=(
            "A RefineDocumentsChain first summarizes the initial document, "
            "then loops over every remaining document and asks the LLM to "
            "update the running answer."
        ),
        metadata={"source": "refine-chain"},
    ),
    Document(
        page_content=(
            "Refine is useful when documents are too long to stuff into one "
            "prompt. The tradeoff is more LLM calls — one per document — "
            "so it is slower and more expensive than stuffing."
        ),
        metadata={"source": "when-to-use"},
    ),
]

if __name__ == "__main__":
    print(f"Refining across {len(docs)} documents "
          f"(1 initial + {len(docs) - 1} refine calls)\n")

    result = chain.invoke({"input_documents": docs})
    steps = result["intermediate_steps"]

    # Step 0 = initial summary; later steps = each refine pass.
    for i, (doc, step_text) in enumerate(zip(docs, steps, strict=True)):
        source = doc.metadata.get("source", f"doc-{i}")
        label = "INITIAL" if i == 0 else f"REFINE #{i}"
        print("=" * 60)
        print(f"{label}  |  source={source}")
        print("-" * 60)
        print("Document fed into this step:")
        print(doc.page_content)
        print("-" * 60)
        print("LLM answer after this step:")
        print(step_text.strip())
        print()

    print("=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(result["output_text"].strip())

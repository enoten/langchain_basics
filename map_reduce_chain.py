"""End-to-end example: MapReduceDocumentsChain.

MapReduceDocumentsChain:
  1. MAP   — summarize each document independently (parallelizable)
  2. REDUCE — stuff those per-doc summaries into one final LLM call
"""

from dotenv import load_dotenv
from langchain_classic.chains import (
    LLMChain,
    MapReduceDocumentsChain,
    ReduceDocumentsChain,
    StuffDocumentsChain,
)
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}",
)
document_variable_name = "context"

llm = OpenAI(temperature=0)

# --- MAP: one LLM call per document ---
map_prompt = PromptTemplate.from_template(
    "Write a one-sentence summary of this content:\n\n{context}\n\nSummary:"
)
map_llm_chain = LLMChain(llm=llm, prompt=map_prompt)

# --- REDUCE: combine the map summaries into one answer ---
reduce_prompt = PromptTemplate.from_template(
    "Combine these document summaries into one short paragraph:\n\n"
    "{context}\n\nCombined summary:"
)
reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Stuff the map outputs together for the reduce step.
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
    document_separator="\n\n *** \n\n",
)

# If there are too many map outputs to stuff at once, ReduceDocumentsChain
# can collapse them recursively. Here we only need the combine step.
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
)

chain = MapReduceDocumentsChain(
    llm_chain=map_llm_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name=document_variable_name,
    return_intermediate_steps=True,  # expose per-document map summaries
)

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
            "MapReduceDocumentsChain first maps an LLM over each document "
            "independently, producing one summary per document."
        ),
        metadata={"source": "map-step"},
    ),
    Document(
        page_content=(
            "Those per-document summaries are then reduced — typically stuffed "
            "into a single prompt — to produce one final combined answer."
        ),
        metadata={"source": "reduce-step"},
    ),
    Document(
        page_content=(
            "Map-reduce shines for large corpora: map work can run in parallel, "
            "and reduce only sees the shorter summaries, not the original text."
        ),
        metadata={"source": "when-to-use"},
    ),
]

if __name__ == "__main__":
    print(
        f"Map-reduce across {len(docs)} documents "
        f"({len(docs)} map calls + 1 reduce call)\n"
    )

    result = chain.invoke({"input_documents": docs})
    map_summaries = result["intermediate_steps"]

    print("=" * 60)
    print("MAP STEP  (one summary per document)")
    print("=" * 60)
    for i, (doc, summary) in enumerate(zip(docs, map_summaries, strict=True)):
        source = doc.metadata.get("source", f"doc-{i}")
        print("-" * 60)
        print(f"MAP #{i + 1}  |  source={source}")
        print("-" * 60)
        print("Original document:")
        print(doc.page_content)
        print()
        print("Mapped summary:")
        print(summary.strip())
        print()

    # What the reduce chain actually receives (map outputs joined).
    reduce_context = ("\n\n *** \n\n").join(s.strip() for s in map_summaries)
    print("=" * 60)
    print("REDUCE INPUT  (stuffed map summaries)")
    print("=" * 60)
    print(reduce_context)
    print()

    print("=" * 60)
    print("FINAL OUTPUT  (after reduce)")
    print("=" * 60)
    print(result["output_text"].strip())

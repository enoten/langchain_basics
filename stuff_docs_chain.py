"""End-to-end example: StuffDocumentsChain.

StuffDocumentsChain concatenates (\"stuffs\") all input documents into a single
prompt context, then runs one LLM call — best for short docs that fit in context.
"""

from dotenv import load_dotenv
from langchain_classic.chains import LLMChain, StuffDocumentsChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

# How each Document is formatted before being joined into the context string.
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}",
)
document_variable_name = "context"

# Prompt must include `document_variable_name` (here: {context}).
prompt = PromptTemplate.from_template(
    "Write a short summary of the following content:\n\n{context}\n\nSummary:"
)

llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, 
                     prompt=prompt)

chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
    document_separator="\n\n *** \n\n",
)

# Sample documents to stuff into one prompt.
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
            "A StuffDocumentsChain takes multiple documents, formats each one, "
            "joins them into a single context string, and passes that string "
            "to an LLM in one call."
        ),
        metadata={"source": "stuff-chain"},
    ),
    Document(
        page_content=(
            "Stuffing works well when the combined documents are short enough "
            "to fit in the model context window. For longer corpora, prefer "
            "map-reduce or refine strategies instead."
        ),
        metadata={"source": "when-to-use"},
    ),
]

if __name__ == "__main__":
    # Show the stuffed context exactly as the LLM receives it in {context}.
    stuffed_context = chain._get_inputs(docs)[document_variable_name]
    print("----- Merged context sent to the LLM -----")
    print(stuffed_context)
    print("------------------------------------------")

    result = chain.invoke({"input_documents": docs})
    print()
    print(result["output_text"])
    #print(result)
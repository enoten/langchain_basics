"""End-to-end example: SimpleSequentialChain.

SimpleSequentialChain runs single-input / single-output chains in order.
The string output of each step is fed directly as the input to the next.
"""

from dotenv import load_dotenv
from langchain_classic.chains import LLMChain, SimpleSequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI(temperature=0.7)

# Step 1: topic -> play synopsis
synopsis_prompt = PromptTemplate.from_template(
    "You are a playwright. Given the title of a play, write a short "
    "2-3 sentence synopsis.\n\n"
    "Title: {title}\n"
    "Synopsis:"
)

synopsis_chain = LLMChain(llm=llm, 
                          prompt=synopsis_prompt,
                          output_key="synopsis")

# Step 2: synopsis -> critic review
review_prompt = PromptTemplate.from_template(
    "You are a theater critic. Given this play synopsis, write a short "
    "2-3 sentence review.\n\n"
    "Synopsis: {synopsis}\n"
    "Review:"
)
review_chain = LLMChain(llm=llm, 
                        prompt=review_prompt,
                        output_key="review")

# Output of synopsis_chain becomes input of review_chain automatically.
overall_chain = SimpleSequentialChain(
    chains=[synopsis_chain, review_chain],
    verbose=True,  # prints each step's output as the chain runs
)

if __name__ == "__main__":
    topic = "Fairy Tale: Sunset on the beach"

    print("=" * 60)
    print("INPUT")
    print("=" * 60)
    print(topic)
    print()

    # Run steps manually first so the handoff is obvious.
    synopsis = synopsis_chain.invoke({"title": topic})["synopsis"].strip()
    print("=" * 60)
    print("STEP 1 — synopsis_chain  (title -> synopsis)")
    print("=" * 60)
    print(synopsis)
    print()

    review = review_chain.invoke({"synopsis": synopsis})["review"].strip()
    print("=" * 60)
    print("STEP 2 — review_chain  (synopsis -> review)")
    print("=" * 60)
    print(review)
    print()

    # Same pipeline via SimpleSequentialChain (one call).
    print("=" * 60)
    print("SimpleSequentialChain.invoke  (same pipeline)")
    print("=" * 60)
    result = overall_chain.invoke({"input": topic})
    print()
    print("FINAL OUTPUT")
    print("-" * 60)
    print(result["output"].strip())

import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()

MODEL_ID = "google/flan-t5-large"
GENERATION_KWARGS = {"max_new_tokens": 512}


class Seq2SeqPipeline:
    """Minimal text2text pipeline compatible with HuggingFacePipeline."""

    task = "text2text-generation"

    def __init__(self, model, tokenizer, generation_kwargs=None):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs or {}

    def __call__(self, prompts, **kwargs):
        merged_kwargs = {**self.generation_kwargs, **kwargs}
        if not isinstance(prompts, list):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, **merged_kwargs)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({"generated_text": text})
        return results


model_kwargs = {}
if token := os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN"):
    model_kwargs["token"] = token

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **model_kwargs)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, **model_kwargs)

prompt = PromptTemplate.from_template(
    "Question: {question} Answer: Let's think step by step."
)

llm = HuggingFacePipeline(
    pipeline=Seq2SeqPipeline(model, tokenizer, GENERATION_KWARGS),
    model_id=MODEL_ID,
    pipeline_kwargs=GENERATION_KWARGS,
)

chain = prompt | llm

print("")

question1 = "Explain the concept of black holes in simple terms."
response = chain.invoke({"question": question1})
print(response)

print("--------------------------------")

question2 = "Provide a brief overview of the history of artificial intelligence."
response = chain.invoke({"question": question2})
print(response)

#from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


# Create a chat model (LangChain v1 unified initialization)
model = init_chat_model("gpt-5.4-nano")

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
#chain = prompt_template | model | StrOutputParser()
chain = prompt_template | model

# Run the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(result)
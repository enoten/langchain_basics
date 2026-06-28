from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# Load environment variables from .env
load_dotenv()

# Create a chat model (LangChain v1 unified initialization)
model = init_chat_model("gpt-5.4-nano")

# Invoke the model with a chat message (preferred for chat models)
messages = [HumanMessage(content="What is the capital of the United States of America?")]
result = model.invoke(messages)

#print("Full result:")
#print(result)
print("Content only:")
print(result.text)

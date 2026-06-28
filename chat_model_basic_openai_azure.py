# Azure OpenAI (v1 API): https://docs.langchain.com/oss/python/integrations/chat/openai/#using-with-azure-openai
#
# Required .env variables:
#   AZURE_OPENAI_API_KEY=your-api-key
#   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
#   AZURE_OPENAI_DEPLOYMENT=your-gpt-deployment-name

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# Load environment variables from .env
load_dotenv()

azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

# model is your Azure deployment name; base_url targets the Azure OpenAI v1 API
model = init_chat_model(
    deployment_name,
    model_provider="openai",
    base_url=azure_endpoint,
    api_key=api_key,
)

# Invoke the model with a chat message (preferred for chat models)
messages = [HumanMessage(content="What is 81 divided by 9?")]
result = model.invoke(messages)

#print("Full result:")
#print(result)
print("Content only:")
print(result.text)

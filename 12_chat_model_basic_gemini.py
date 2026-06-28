# Google Gemini: https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai
#
# Required .env variables:
#   GOOGLE_API_KEY=your-api-key
#
# Optional:
#   GEMINI_MODEL=gemini-2.5-flash

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# Load environment variables from .env
load_dotenv()

model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# model_provider="google_genai" uses the Gemini Developer API (Google AI Studio)
model = init_chat_model(
    model_name,
    model_provider="google_genai",
    api_key=os.environ["GOOGLE_API_KEY"],
)

# Invoke the model with a chat message (preferred for chat models)
messages = [HumanMessage(content="What is 81 divided by 9?")]
result = model.invoke(messages)

#print("Full result:")
#print(result)
print("Content only:")
print(result.text)

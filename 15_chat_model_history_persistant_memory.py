from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

SESSION_ID = "default-chat"
DB_PATH = "sqlite:///chat_history.db"

model = init_chat_model("gpt-5.4-nano")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return persistent chat history backed by SQLite."""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=DB_PATH,
    )


# Seed the system prompt once for new sessions
history = get_session_history(SESSION_ID)
if not history.messages:
    history.add_message(SystemMessage(content="You are a helpful AI assistant."))

with_history = RunnableWithMessageHistory(
    model,
    get_session_history,
)

config = {"configurable": {"session_id": SESSION_ID}}

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    result = with_history.invoke(
        [HumanMessage(content=query)],
        config=config,
    )
    response = result.content if hasattr(result, "content") else str(result)
    print(f"AI: {response}")

print("---- Message History (from DB) ----")
for message in get_session_history(SESSION_ID).messages:
    print(message)

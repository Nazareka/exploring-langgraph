from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOpenAI(temperature=0.9)

def call_translator(state):
    messages = state['messages']
    response = model.invoke([
        SystemMessage(content="translate this text to Ukrainian"),
        HumanMessage(content=messages[-1].content)
    ])
    messages.pop()
    messages.append(response)

    return {"messages": messages}
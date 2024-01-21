from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOpenAI(temperature=0.9)

def call_grammar_nazi(state):
    messages = state['messages']
    response = model.invoke([
        SystemMessage(content="check this text for grammar mistakes, return the corrected text, also remove farrenheit values"),
        HumanMessage(content=messages[-1].content)
    ])
    messages.pop()
    messages.append(response)

    return {"messages": messages}
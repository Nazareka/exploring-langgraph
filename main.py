import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
from roles.grammar_nazi import call_grammar_nazi
from roles.translator import call_translator
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from icecream import ic

tools = [TavilySearchResults(max_results=1)]

tool_executor = ToolExecutor(tools)

model = ChatOpenAI(temperature=0, streaming=True)

functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the function that calls the model
def call_model(state):
    messages = state['messages']
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "Search the information about the topic, if after searching you still don't know the answer or you haven't found the acceptable answer, "
                    "search again with different query up to 3 times."
                )
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | model
    response = chain.invoke({"messages": messages})

    return {"messages": messages + [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state['messages']
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)

    return {"messages": messages + [function_message]}

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.add_node("translator", call_translator)
workflow.add_node("grammar_nazi", call_grammar_nazi)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we call the translator node.
        "end": "translator"
    }
)

# We now add edges from the `action` node to the `translator` and then `grammar_nazi` nodes.
workflow.add_edge('action', 'agent')
workflow.add_edge('translator', 'grammar_nazi')
workflow.add_edge('grammar_nazi', END)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

question = "what interesting is happening in the Europe?"

inputs = {"messages": [HumanMessage(content=question)]}
ic(app.invoke(inputs))
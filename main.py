import os
from dotenv import load_dotenv
from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generate_response_chain, review_dispute_chain

# Load environment variables from .env file
load_dotenv()

# Ensure the API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key not found in environment variables")

REVIEW = "review"
RESPOND = "respond"

def response_node(state: Sequence[BaseMessage]):
    return generate_response_chain.invoke({"messages": state})

def review_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = review_dispute_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(RESPOND, response_node)
builder.add_node(REVIEW, review_node)
builder.set_entry_point(REVIEW)

def should_continue(state: List[BaseMessage]):
    if len(state) > 4:  # Adjust this number as needed
        return END
    return RESPOND

builder.add_conditional_edges(REVIEW, should_continue)
builder.add_edge(RESPOND, REVIEW)

graph = builder.compile()

if __name__ == "__main__":
    print("Credit Card Dispute Response Generator")
    inputs = HumanMessage(content="""
    Customer Dispute:
    I don't recognize a charge of $299.99 from 'TechGadget Store' on my credit card statement dated July 15, 2023. 
    I've never shopped at this store and I believe this charge is fraudulent. 
    Please investigate and remove this charge from my account.
    """)
    response = graph.invoke(inputs)
    print(response)
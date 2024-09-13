import os
from dotenv import load_dotenv
from typing import List, Sequence, Union, Dict, Any, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, MessageGraph
from chains import (
    classify_dispute_chain, 
    review_dispute_chain, 
    generate_response_chain, 
    fraud_detection_chain, 
    customer_history_chain, 
)

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key not found in environment variables")

CLASSIFY = "classify"
FRAUD_CHECK = "fraud_check"
HISTORY_CHECK = "history_check"
REVIEW = "review"
RESPOND = "respond"
QUALITY_CHECK = "quality_check"

def classification_node(state: List[BaseMessage]) -> List[BaseMessage]:
    classification = classify_dispute_chain.invoke({"messages": state})
    return state + [SystemMessage(content=f"Dispute Classification: {classification.content}")]

def fraud_check_node(state: List[BaseMessage]) -> List[BaseMessage]:
    last_message = state[-1].content
    classification = last_message.split(": ")[-1]
    fraud_result = fraud_detection_chain.invoke({"messages": state[:-1], "classification": classification})
    return state + [SystemMessage(content=f"Fraud Check Result: {fraud_result.content}")]

def history_check_node(state: List[BaseMessage]) -> List[BaseMessage]:
    history_result = customer_history_chain.invoke({"messages": state})
    return state + [SystemMessage(content=f"Customer History: {history_result.content}")]

def review_node(state: List[BaseMessage]) -> List[BaseMessage]:
    classification = state[-3].content.split(": ")[-1]
    fraud_result = state[-2].content.split(": ")[-1]
    history_result = state[-1].content.split(": ")[-1]
    review = review_dispute_chain(
        messages=state[:-3], 
        classification=classification, 
        fraud_result=fraud_result, 
        history_result=history_result
    )
    return state + [SystemMessage(content=f"Dispute Review: {review.content}")]

def respond_node(state: List[BaseMessage]) -> List[BaseMessage]:
    classification = state[-4].content.split(": ")[-1]
    review = state[-1].content.split(": ")[-1]
    response = generate_response_chain(messages=state, classification=classification, review=review)
    return state + [HumanMessage(content=response.content)]

def quality_check_node(state: List[BaseMessage]) -> Tuple[List[BaseMessage], bool]:
    response = state[-1].content
    quality_result = quality_check_chain.invoke({"response": response})
    passed = quality_result.content.strip().upper() == "PASS"
    return state + [SystemMessage(content=f"Quality Check: {'Passed' if passed else 'Failed'}")], passed

builder = MessageGraph()

builder.add_node(CLASSIFY, classification_node)
builder.add_node(FRAUD_CHECK, fraud_check_node)
builder.add_node(HISTORY_CHECK, history_check_node)
builder.add_node(REVIEW, review_node)
builder.add_node(RESPOND, respond_node)

builder.set_entry_point(CLASSIFY)

builder.add_edge(CLASSIFY, FRAUD_CHECK)
builder.add_edge(FRAUD_CHECK, HISTORY_CHECK)
builder.add_edge(HISTORY_CHECK, REVIEW)
builder.add_edge(REVIEW, RESPOND)


def should_continue(state: Tuple[List[BaseMessage], bool]) -> Union[str, Dict[str, Any]]:
    messages, passed = state
    if passed:
        return END
    else:
        return {RESPOND: messages[:-2]}  # Remove the last response and quality check message



graph = builder.compile()

if __name__ == "__main__":
    print("Complex Credit Card Dispute Response Generator")
    inputs = HumanMessage(content="""
    Customer Dispute:
    I don't recognize a charge of $299.99 from 'TechGadget Store' on my credit card statement dated July 15, 2023. 
    I've never shopped at this store and I believe this charge is fraudulent. 
    Please investigate and remove this charge from my account.
    """)
    response = graph.invoke(inputs)
    for message in response:
        if isinstance(message, SystemMessage):
            print(f"[System] {message.content}")
        elif isinstance(message, HumanMessage):
            print(f"[Response] {message.content}")
        else:
            print(f"[Other] {message.content}")
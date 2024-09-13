import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

llm = ChatOpenAI(model=openai_model, api_key=openai_api_key)

classification_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a credit card company AI assistant tasked with classifying customer disputes. "
               "Categorize the dispute into one of the following categories: "
               "Fraudulent Charge, Billing Error, Quality Dispute, Service Not Received, Other. "
               "Provide only the category name as the response."),
    MessagesPlaceholder(variable_name="messages"),
])

fraud_detection_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a fraud detection AI. Given a dispute classification and details, assess the likelihood of fraud."),
    MessagesPlaceholder(variable_name="messages"),
])

customer_history_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer history analysis AI. Review the customer's history and provide relevant insights."),
    MessagesPlaceholder(variable_name="messages"),
])

review_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a credit card company representative reviewing a customer's dispute. "
               "The dispute has been classified as: {classification}. "
               "Fraud check result: {fraud_result}. "
               "Customer history: {history_result}. "
               "Analyze the dispute details and provide a thorough assessment. "
               "Consider factors such as transaction history, merchant information, and company policies. "
               "Provide detailed recommendations on how to proceed with the dispute."),
    MessagesPlaceholder(variable_name="messages"),
])

response_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a credit card company AI assistant tasked with drafting responses to customer disputes. "
               "The dispute has been classified as: {classification}. "
               "Review of the dispute: {review}. "
               "Generate a professional and empathetic response to the customer's dispute. "
               "Ensure the response adheres to company policies and legal requirements."),
    MessagesPlaceholder(variable_name="messages"),
])


classify_dispute_chain = classification_prompt | llm
fraud_detection_chain = fraud_detection_prompt | llm
customer_history_chain = customer_history_prompt | llm


def review_dispute_chain(messages, classification, fraud_result, history_result):
    chain = review_prompt | llm
    return chain.invoke({"messages": messages, "classification": classification, "fraud_result": fraud_result, "history_result": history_result})

def generate_response_chain(messages, classification, review):
    chain = response_prompt | llm
    return chain.invoke({"messages": messages, "classification": classification, "review": review})
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Get the API key and model name from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # Default to gpt-3.5-turbo if not specified

# Ensure the API key is available
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Initialize your llm model
llm = ChatOpenAI(model=openai_model, api_key=openai_api_key)

review_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a credit card company representative reviewing a customer's dispute. "
            "Analyze the dispute details and provide a thorough assessment. "
            "Consider factors such as transaction history, merchant information, and company policies. "
            "Provide detailed recommendations on how to proceed with the dispute.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a credit card company AI assistant tasked with drafting responses to customer disputes. "
            "Generate a professional and empathetic response to the customer's dispute. "
            "If provided with a review of the dispute, incorporate that information into your response. "
            "Ensure the response adheres to company policies and legal requirements.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate_response_chain = response_prompt | llm
review_dispute_chain = review_prompt | llm
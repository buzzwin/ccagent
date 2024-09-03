# Credit Card Dispute Response Generator

This project is a Credit Card Dispute Response Generator that uses OpenAI's language models to review customer disputes and generate appropriate responses.

## Prerequisites

- Python 3.8 or higher
- Poetry (Python package manager)

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/credit-card-dispute-system.git
   cd credit-card-dispute-system
   ```

2. Install dependencies using Poetry:

   ```
   poetry install
   ```

3. Create a `.env` file in the project root and add your OpenAI API key and preferred model:

   ```
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo
   ```

   Note: Make sure to keep your `.env` file private and never commit it to version control.

## Running the Application

To run the application, use the following command:

```
poetry run python main.py
```

This will process the example dispute in `main.py` and generate a response.

## Customizing Inputs

To process different disputes, you can modify the `inputs` in the `main.py` file. For example:

```python
inputs = HumanMessage(content="""
Customer Dispute:
I was charged twice for my hotel stay at Luxury Inn on August 3, 2023.
The correct charge of $450 appears once, but there's a duplicate charge for the same amount.
Please remove the duplicate charge.
""")
```

## Project Structure

- `main.py`: The entry point of the application. It sets up the processing graph and handles the main logic flow.
- `chains.py`: Contains the LangChain setup, including the prompts and chain definitions.
- `.env`: Contains environment variables (not in version control).
- `pyproject.toml`: Poetry's configuration file, specifying project dependencies.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

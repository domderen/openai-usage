from openai import OpenAI
from src.openai_usage.usage import OpenAIUsage

# Make sure you have your OPENAI_API_KEY set in your environment

if __name__ == "__main__":
    client = OpenAI()
    with OpenAIUsage(client) as usage:
        # Adjust the model and prompt as needed for your account and API version
        response = client.responses.create(
            model="gpt-3.5-turbo",
            input="Say hello!"
        )
        print("Response object:", response)
        print("Usage from response:", getattr(response, "usage", None))
        print("Usage from context manager:", usage)
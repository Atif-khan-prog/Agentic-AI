from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

while True:

    user = input('You: ')

    if user.lower() == 'exit':
        break

    response = client.responses.create(
        model='gpt-5.6-luna',
        input = user
    )

    print("AI: ",response.output_text)
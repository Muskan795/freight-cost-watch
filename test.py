import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)
client = Groq(api_key=os.environ["GROQ_API_KEY"])

reply = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Explain what an API is in one line"}],
)
print(reply.choices[0].message.content)
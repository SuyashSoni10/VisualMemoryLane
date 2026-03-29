import os
from groq import Groq

client = Groq(api_key= GROQ_API_KEY)

completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a smart desk assistant."},
        {"role": "user", "content": "Objects on desk: bottle (absent 40 mins), book (present 2 hrs). Give one reminder."}
    ]
)

print(completion.choices[0].message.content)
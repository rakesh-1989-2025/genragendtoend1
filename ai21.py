import requests
import os

def ai21_response(question, context):

    api_key=os.getenv("AI21_API_KEY")

    url="https://api.ai21.com/studio/v1/chat/completions"

    headers={
        "Authorization":f"Bearer {api_key}",
        "Content-Type":"application/json"
    }

    prompt=f"""
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    payload={
        "model":"jamba-large",
        "messages":[
            {
                "role":"user",
                "content":prompt
            }
        ]
    }

    response=requests.post(
        url,
        headers=headers,
        json=payload
    )

    return response.json()["choices"][0]["message"]["content"]
import requests

API_KEY = ""
url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "openai/gpt-oss-120b",
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "temperature": 1,
    "max_completion_tokens": 8192,
    "top_p": 1
}

response = requests.post(url, headers=headers, json=payload)
result = response.json()

print(result["choices"][0]["message"]["content"])
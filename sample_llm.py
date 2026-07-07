import requests

BASE_URL = "https://radeon-global.anruicloud.com/instances/hf-180-1a071167/proxy/8000/v1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "Explain what a large language model is in one sentence."
        }
    ],
    "temperature": 0.7,
    "max_tokens": 128,
}

response = requests.post(
    f"{BASE_URL}/chat/completions",
    json=payload,
)

print("Status:", response.status_code)
print(response.json())

# Print just the assistant's reply
if response.status_code == 200:
    print("\nAssistant:")
    print(response.json()["choices"][0]["message"]["content"])
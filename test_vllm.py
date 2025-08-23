import requests
import json

api_url = "http://localhost:8000/v1/completions"

headers = {
    "Content-Type": "application/json"
}

payload = {
        "model": "aisingapore/Gemma-SEA-LION-v3-9B-IT",
        "prompt": "What is the capital of France?",
        "max_tokens": 128
}

try:
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    response.raise_for_status() # Raises an HTTPError if the response status is 4xx or 5xx

    result = response.json()
    print("API response successful!")
    print(json.dumps(result, indent=2))

    # Print just the generated text
    if result.get("choices"):
        print("\nGenerated Text:")
        print(result["choices"][0]["text"])

except requests.exceptions.RequestException as e:
    print(f"Error calling the vLLM API: {e}")

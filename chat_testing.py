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
root@901e594409bb:~# ls
chat_testing.py  test_vllm.py  timing_tests.py  vllm_benchmark_results.csv
root@901e594409bb:~# cat chat_testing.py
import requests
import json

# =============================================================================
# VLLM API Client Script
#
# This script provides a basic function to interact with a vLLM server
# via its RESTful API. It sends a prompt and prints the generated response.
#
# Prerequisites:
# - A vLLM server running and accessible at the specified URL.
# - The 'requests' library installed (`pip install requests`).
#
# =============================================================================

def chat_with_vllm(prompt: str, api_url: str = "http://localhost:8000/v1/completions"):
    """
    Sends a prompt to a vLLM server and returns the generated text.

    Args:
        prompt (str): The text prompt to send to the model.
        api_url (str): The URL of the vLLM API endpoint.
                       NOTE: We've updated this to the standard '/v1/completions'
                       endpoint used by the OpenAI-compatible server.

    Returns:
        str: The generated text from the model, or an error message if the
             request fails.
    """
    # Define the headers for the API request.
    headers = {"Content-Type": "application/json"}

    # Define the payload with the prompt and other generation parameters.
    # The 'n' parameter specifies the number of sequences to generate.
    # The 'use_beam_search' and 'temperature' are optional parameters.
    payload = {
        "model": "aisingapore/Gemma-SEA-LION-v3-9B-IT",  # Updated model name as requested
        "prompt": prompt,
        "n": 1,
        "use_beam_search": False,
        "temperature": 0.7,
        "max_tokens": 512,
    }

    print("Sending request to vLLM server...")
    print(f"Prompt: '{prompt}'")

    try:
        # Make a POST request to the vLLM API.
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx).

        # Parse the JSON response.
        data = response.json()

        # The OpenAI-compatible API for completions returns a 'choices' array
        # with 'text' inside.
        if "choices" in data and len(data["choices"]) > 0:
            generated_text = data["choices"][0]["text"]
            return generated_text
        else:
            return "Error: No text was generated."

    except requests.exceptions.RequestException as e:
        return f"Error connecting to the vLLM server: {e}"
    except (json.JSONDecodeError, KeyError) as e:
        return f"Error parsing the server response: {e}"

# --- Example Usage ---
if __name__ == "__main__":
    # Example prompt to send to the model.
    user_prompt = "What are the three most important things for a healthy lifestyle?"

    # Get the response from the model.
    model_response = chat_with_vllm(user_prompt)

    # Print the model's response.
    print("\n" + "="*50)
    print("Model Response:")
    print(model_response)
    print("="*50)

    # Updated second example prompt
    user_prompt = "What is the largest ship (container, cargo, military, and cruise) in the world. Tell me 10 things about it"
    model_response = chat_with_vllm(user_prompt)
    print("\n" + "="*50)
    print("Model Response:")
    print(model_response)
    print("="*50)

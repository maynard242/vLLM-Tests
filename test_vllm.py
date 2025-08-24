"""
test_vllm_chat.py

Send a chat-style request to a local vLLM server and validate the response.
"""

import argparse
import json
import time
import requests
import sys

API_BASE = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_BASE}/v1/chat/completions"
MODELS_ENDPOINT = f"{API_BASE}/v1/models"
HEADERS = {"Content-Type": "application/json"}


def test_vllm_chat(model_name: str, user_message: str, max_tokens: int = 128,
                   temperature: float = 0.7, top_p: float = 0.9, timeout: int = 30) -> bool:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    print("Testing vLLM chat API...")
    print(f"Endpoint: {CHAT_ENDPOINT}")
    print(f"Model: {model_name}")
    print(f"User message: {user_message}")
    print("-" * 60)

    resp = None  # Initialize resp variable
    
    try:
        start = time.time()
        resp = requests.post(CHAT_ENDPOINT, headers=HEADERS, data=json.dumps(payload), timeout=timeout)
        latency = time.time() - start

        resp.raise_for_status()
        print(f"✅ API reachable (latency: {latency:.2f}s)")

        result = resp.json()
        print("✅ Response parsed as JSON")

        # Basic structure checks
        if not isinstance(result, dict):
            print("❌ Unexpected response shape (expected JSON object).")
            print(json.dumps(result, indent=2))
            return False

        choices = result.get("choices")
        if not choices:
            print("❌ No 'choices' in response.")
            print(json.dumps(result, indent=2))
            return False

        print("✅ 'choices' field present")

        # Extract content from the response
        first = choices[0]
        content = None
        
        if "message" in first and isinstance(first["message"], dict):
            content = first["message"].get("content")
            print("✅ Found message content in chat format")
        elif "text" in first:  # fallback for servers that use completions format
            content = first.get("text")
            print("✅ Found text content in completions format")
        elif "delta" in first and isinstance(first["delta"], dict):
            # streaming-style partial, try to collect text if present
            content = first["delta"].get("content") or first["delta"].get("text")
            print("✅ Found delta content in streaming format")

        if content is None:
            print("⚠️  Could not locate message content in first choice.")
            print("Available keys in first choice:", list(first.keys()))
            print("First choice structure:")
            print(json.dumps(first, indent=2))
            return False
        else:
            content = content.strip()
            if content:
                print("✅ Generated content is not empty")
                print("\n" + "="*50)
                print("CHAT RESPONSE:")
                print("="*50)
                print(content)
                print("="*50)
            else:
                print("⚠️  Generated content is empty")

        # Print usage if available
        usage = result.get("usage")
        if usage:
            print(f"\n📊 Token Usage:")
            print(f"   Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"   Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"   Total tokens: {usage.get('total_tokens', 'N/A')}")
        else:
            print("\n⚠️  No usage information provided")

        # Print full raw response optionally (commented out by default to reduce noise)
        # print("\nFull response (raw JSON):")
        # print(json.dumps(result, indent=2))

        return True

    except requests.exceptions.ConnectionError:
        print("❌ Connection error: cannot connect to vLLM server.")
        print("   - Ensure vLLM is running on localhost:8000")
        print("   - Check if firewall is blocking the connection")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Timeout: request took longer than {timeout}s")
        print("   - Try increasing --timeout")
        print("   - Ensure the model is loaded and not overloaded")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
        if resp is not None:
            print(f"   Status code: {resp.status_code}")
            try:
                error_detail = resp.json()
                print("   Server error details:")
                print(json.dumps(error_detail, indent=4))
            except:
                print("   Server response text:")
                print(resp.text[:500])  # Limit output length
        return False
    except json.JSONDecodeError:
        print("❌ Invalid JSON returned by server")
        if resp is not None:
            print("Raw response (first 500 chars):")
            print(resp.text[:500])
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return False


def list_models():
    """List available models from the vLLM server"""
    print("Fetching available models...")
    try:
        resp = requests.get(MODELS_ENDPOINT, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        models = data.get("data", [])
        if not models:
            print("⚠️  No models found in response")
            print("Raw response:")
            print(json.dumps(data, indent=2))
            return []
            
        print("✅ Available models:")
        model_ids = []
        for i, model in enumerate(models, 1):
            model_id = model.get("id") or model.get("model") or f"model_{i}"
            print(f"   {i}. {model_id}")
            if model.get("id"):
                model_ids.append(model.get("id"))
                
        return model_ids
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to models endpoint")
        return []
    except Exception as e:
        print(f"⚠️  Could not fetch models list: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM chat completions API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_vllm_chat.py --model "aisingapore/Gemma-SEA-LION-v3-9B-IT"
  python test_vllm_chat.py -m "meta-llama/Llama-2-7b-chat-hf" --message "Hello, who are you?"
  python test_vllm_chat.py --list-models
        """
    )
    
    parser.add_argument("--model", "-m", 
                       help="Model name to test (required unless --list-models)")
    parser.add_argument("--message", "-M", 
                       default="What is the capital of France?",
                       help="User message to send (default: 'What is the capital of France?')")
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="Maximum tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for sampling (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p for nucleus sampling (default: 0.9)")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Request timeout in seconds (default: 30)")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()

    # Handle list-models flag
    if args.list_models:
        list_models()
        return

    # Validate model argument
    if not args.model:
        print("❌ No model specified!")
        print("\nAttempting to list available models...")
        available = list_models()
        
        if available:
            print(f"\nPlease specify a model using --model. Example:")
            print(f"  python {sys.argv[0]} --model \"{available[0]}\"")
        else:
            print(f"\nPlease specify a model using --model. Example:")
            print(f"  python {sys.argv[0]} --model \"aisingapore/Gemma-SEA-LION-v3-9B-IT\"")
        
        print("\nUse --help for more options.")
        return

    # Validate numeric arguments
    if args.max_tokens <= 0:
        print("❌ max-tokens must be positive")
        return
    if not (0.0 <= args.temperature <= 2.0):
        print("❌ temperature should be between 0.0 and 2.0")
        return
    if not (0.0 <= args.top_p <= 1.0):
        print("❌ top-p should be between 0.0 and 1.0")
        return

    # Run the test
    success = test_vllm_chat(
        model_name=args.model,
        user_message=args.message,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout
    )

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 vLLM chat API test PASSED!")
    else:
        print("🔧 vLLM chat API test FAILED - check errors above")


if __name__ == "__main__":
    main()

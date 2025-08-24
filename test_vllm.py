import argparse
import json
import time
import requests
import sys

# =============================================================================
# VLLM Chat API Testing Script
#
# This script tests the chat completions API of a vLLM server by sending
# requests and validating responses. It can also list available models.
#
# Features:
# - Chat completions API testing
# - Model listing functionality
# - Response validation and parsing
# - Comprehensive error handling
# - Configurable parameters (temperature, top_p, max_tokens)
#
# Prerequisites:
# - A vLLM server running and accessible on localhost:8000
# - The 'requests' library installed (`pip install requests`)
#
# Usage:
# python test_vllm_chat.py --model MODEL_NAME [--message MESSAGE] [options]
# python test_vllm_chat.py --list-models
#
# =============================================================================

# Configuration constants
API_BASE = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_BASE}/v1/chat/completions"
MODELS_ENDPOINT = f"{API_BASE}/v1/models"
HEADERS = {"Content-Type": "application/json"}


def test_vllm_chat(model_name: str, user_message: str, max_tokens: int = 128,
                   temperature: float = 0.7, top_p: float = 0.9, timeout: int = 30) -> bool:
    """
    Test the vLLM chat completions API with the specified parameters.

    Args:
        model_name (str): The name of the model to test.
        user_message (str): The user message to send.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Temperature for sampling.
        top_p (float): Top-p for nucleus sampling.
        timeout (int): Request timeout in seconds.

    Returns:
        bool: True if the test was successful, False otherwise.
    """
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

    resp = None  # Initialize response variable
    
    try:
        start_time = time.time()
        resp = requests.post(CHAT_ENDPOINT, headers=HEADERS, data=json.dumps(payload), timeout=timeout)
        latency = time.time() - start_time

        resp.raise_for_status()
        print(f"✅ API reachable (latency: {latency:.2f}s)")

        result = resp.json()
        print("✅ Response parsed as JSON")

        # Basic structure validation
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
        first_choice = choices[0]
        content = None
        
        # Try different response formats
        if "message" in first_choice and isinstance(first_choice["message"], dict):
            content = first_choice["message"].get("content")
            print("✅ Found message content in chat format")
        elif "text" in first_choice:  # Fallback for servers using completions format
            content = first_choice.get("text")
            print("✅ Found text content in completions format")
        elif "delta" in first_choice and isinstance(first_choice["delta"], dict):
            # Streaming-style partial response
            content = first_choice["delta"].get("content") or first_choice["delta"].get("text")
            print("✅ Found delta content in streaming format")

        if content is None:
            print("⚠️  Could not locate message content in first choice.")
            print("Available keys in first choice:", list(first_choice.keys()))
            print("First choice structure:")
            print(json.dumps(first_choice, indent=2))
            return False
        else:
            content = content.strip()
            if content:
                print("✅ Generated content is not empty")
                print("\n" + "=" * 50)
                print("CHAT RESPONSE:")
                print("=" * 50)
                print(content)
                print("=" * 50)
            else:
                print("⚠️  Generated content is empty")

        # Display usage statistics if available
        usage = result.get("usage")
        if usage:
            print(f"\n📊 Token Usage:")
            print(f"   Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"   Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"   Total tokens: {usage.get('total_tokens', 'N/A')}")
        else:
            print("\n⚠️  No usage information provided")

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
    """
    List available models from the vLLM server.
    
    Returns:
        list: List of model IDs if successful, empty list otherwise.
    """
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


def parse_arguments_parser():
    """Create and return the argument parser (for help display)."""
    parser = argparse.ArgumentParser(
        description="Test vLLM chat completions API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_vllm_chat.py --model "aisingapore/Gemma-SEA-LION-v3-9B-IT"
  python test_vllm_chat.py -m "meta-llama/Llama-2-7b-chat-hf" --message "Hello, who are you?"
  python test_vllm_chat.py --list-models
  python test_vllm_chat.py --model "mistralai/Mistral-7B-Instruct-v0.1" --max-tokens 256 --temperature 0.8
        """
    )
    
    parser.add_argument(
        "--model", "-m", 
        type=str,
        help="Model name to test (required unless --list-models)"
    )
    
    parser.add_argument(
        "--message", "-M", 
        type=str,
        default="What is the capital of France?",
        help="User message to send (default: 'What is the capital of France?')"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=128,
        help="Maximum tokens to generate (default: 128)"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for sampling (default: 0.7)"
    )
    
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.9,
        help="Top-p for nucleus sampling (default: 0.9)"
    )
    
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--list-models", 
        action="store_true",
        help="List available models and exit"
    )
    
    return parser


def parse_arguments():
    """Parse command line arguments."""
    parser = parse_arguments_parser()
    return parser.parse_args()


def validate_arguments(args):
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        bool: True if arguments are valid, False otherwise.
    """
    # Skip validation if just listing models
    if args.list_models:
        return True
        
    # Validate model argument (only if not listing models)
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
        return False

    # Validate numeric arguments
    if args.max_tokens <= 0:
        print("❌ max-tokens must be positive")
        return False
        
    if not (0.0 <= args.temperature <= 2.0):
        print("❌ temperature should be between 0.0 and 2.0")
        return False
        
    if not (0.0 <= args.top_p <= 1.0):
        print("❌ top-p should be between 0.0 and 1.0")
        return False

    return True


def main():
    """Main function to orchestrate the chat API testing."""
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser = parse_arguments_parser()
        parser.print_help()
        return
    
    # Parse command line arguments
    args = parse_arguments()

    # Handle list-models flag
    if args.list_models:
        list_models()
        return

    # Validate arguments
    if not validate_arguments(args):
        return

    print(f"Starting chat API test for model: {args.model}")
    print(f"Configuration: {args.max_tokens} max tokens, temp={args.temperature}, top_p={args.top_p}")
    print("=" * 60)

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
        print("✅ All validation checks completed successfully")
    else:
        print("🔧 vLLM chat API test FAILED - check errors above")
        print("❌ One or more validation checks failed")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/bin/bash
# =============================================================================
# vLLM Server Launcher Script
#
# This script launches a vLLM inference server with configurable parameters
# including model name, maximum model length, and other key options.
#
# Features:
# - Easy model specification via command line arguments
# - Configurable server parameters (port, host, max-model-len, etc.)
# - GPU memory management options
# - Enhanced visual output with status indicators
# - Process management (start/stop/status)
# - Automatic dependency checking
#
# Prerequisites:
# - vLLM installed (`pip install vllm`)
# - CUDA-compatible GPU (recommended)
# - Sufficient GPU memory for the specified model
#
# Usage:
# ./launch_vllm.sh --model MODEL_NAME [OPTIONS]
# ./launch_vllm.sh --help
#
# =============================================================================

# Exit on any error
set -e

# Color codes for enhanced output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Unicode symbols
CHECKMARK="✅"
ARROW="➤"
GEAR="⚙️"
ROCKET="🚀"
WARNING="⚠️"
INFO="ℹ️"
STOP="🛑"

# Default configuration
DEFAULT_MODEL=""
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
DEFAULT_MAX_MODEL_LEN=""
DEFAULT_GPU_MEMORY_UTILIZATION="0.9"
DEFAULT_TENSOR_PARALLEL_SIZE="1"
DEFAULT_DTYPE="auto"
DEFAULT_TRUST_REMOTE_CODE="false"

# PID file for process management
PID_FILE="/tmp/vllm_server.pid"
LOG_FILE="/tmp/vllm_server.log"

# Function to print section headers
print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${WHITE}  $1${NC}"
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Function to print step information
print_step() {
    echo -e "${BOLD}${CYAN}${ARROW} $1${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${BOLD}${GREEN}${CHECKMARK} $1${NC}"
}

# Function to print warnings
print_warning() {
    echo -e "${BOLD}${YELLOW}${WARNING} $1${NC}"
}

# Function to print info messages
print_info() {
    echo -e "${BOLD}${PURPLE}${INFO} $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${BOLD}${RED}❌ $1${NC}"
}

# Function to show help
show_help() {
    echo -e "${BOLD}${WHITE}vLLM Server Launcher${NC}"
    echo ""
    echo -e "${BOLD}USAGE:${NC}"
    echo "  $0 --model MODEL_NAME [OPTIONS]"
    echo "  $0 --stop"
    echo "  $0 --status"
    echo "  $0 --help"
    echo ""
    echo -e "${BOLD}REQUIRED:${NC}"
    echo "  --model MODEL_NAME           Model name or path (e.g., 'meta-llama/Llama-2-7b-chat-hf')"
    echo ""
    echo -e "${BOLD}OPTIONS:${NC}"
    echo "  --host HOST                  Server host (default: $DEFAULT_HOST)"
    echo "  --port PORT                  Server port (default: $DEFAULT_PORT)"
    echo "  --max-model-len LENGTH       Maximum model context length (default: model's max)"
    echo "  --gpu-memory-utilization PCT GPU memory utilization (0.0-1.0, default: $DEFAULT_GPU_MEMORY_UTILIZATION)"
    echo "  --tensor-parallel-size SIZE  Number of GPUs for tensor parallelism (default: $DEFAULT_TENSOR_PARALLEL_SIZE)"
    echo "  --dtype DTYPE                Model dtype (auto/half/float16/bfloat16/float32, default: $DEFAULT_DTYPE)"
    echo "  --trust-remote-code          Allow remote code execution (use with caution)"
    echo ""
    echo -e "${BOLD}MANAGEMENT:${NC}"
    echo "  --stop                       Stop running vLLM server"
    echo "  --status                     Check server status"
    echo "  --logs                       Show server logs"
    echo ""
    echo -e "${BOLD}EXAMPLES:${NC}"
    echo "  # Basic usage"
    echo "  $0 --model aisingapore/Gemma-SEA-LION-v3-9B-IT"
    echo ""
    echo "  # With custom context length and port"
    echo "  $0 --model meta-llama/Llama-2-7b-chat-hf --max-model-len 4096 --port 8001"
    echo ""
    echo "  # Multi-GPU setup"
    echo "  $0 --model meta-llama/Llama-2-13b-chat-hf --tensor-parallel-size 2 --gpu-memory-utilization 0.95"
    echo ""
    echo "  # Check status and stop server"
    echo "  $0 --status"
    echo "  $0 --stop"
}

# Function to check dependencies
check_dependencies() {
    print_step "Checking dependencies..."
    
    # Check if vLLM is installed
    if ! python3 -c "import vllm" 2>/dev/null; then
        print_error "vLLM is not installed. Please install it first:"
        echo "  pip install vllm"
        exit 1
    fi
    
    # Check if CUDA is available (optional but recommended)
    if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        print_success "CUDA available with $GPU_COUNT GPU(s)"
    else
        print_warning "CUDA not available - using CPU mode (will be slow)"
    fi
    
    print_success "Dependencies checked"
}

# Function to validate arguments
validate_arguments() {
    # Check if model is specified
    if [[ -z "$MODEL" ]]; then
        print_error "Model name is required"
        echo ""
        show_help
        exit 1
    fi
    
    # Validate port number
    if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [[ "$PORT" -lt 1024 ]] || [[ "$PORT" -gt 65535 ]]; then
        print_error "Invalid port number: $PORT (must be between 1024-65535)"
        exit 1
    fi
    
    # Validate GPU memory utilization
    if ! command -v bc > /dev/null 2>&1; then
        # Fallback validation without bc
        if ! python3 -c "
val = float('$GPU_MEMORY_UTILIZATION')
if val <= 0.0 or val > 1.0:
    exit(1)
" 2>/dev/null; then
            print_error "GPU memory utilization must be between 0.0 and 1.0, got: $GPU_MEMORY_UTILIZATION"
            exit 1
        fi
    else
        if ! python3 -c "float('$GPU_MEMORY_UTILIZATION')" 2>/dev/null; then
            print_error "Invalid GPU memory utilization: $GPU_MEMORY_UTILIZATION"
            exit 1
        fi
        
        if (( $(echo "$GPU_MEMORY_UTILIZATION > 1.0" | bc -l) )) || (( $(echo "$GPU_MEMORY_UTILIZATION <= 0.0" | bc -l) )); then
            print_error "GPU memory utilization must be between 0.0 and 1.0, got: $GPU_MEMORY_UTILIZATION"
            exit 1
        fi
    fi
    
    # Validate tensor parallel size
    if ! [[ "$TENSOR_PARALLEL_SIZE" =~ ^[0-9]+$ ]] || [[ "$TENSOR_PARALLEL_SIZE" -lt 1 ]]; then
        print_error "Invalid tensor parallel size: $TENSOR_PARALLEL_SIZE"
        exit 1
    fi
    
    print_success "Arguments validated"
}

# Function to check if server is running
is_server_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Function to stop the server
stop_server() {
    print_header "${STOP} STOPPING VLLM SERVER"
    
    if is_server_running; then
        local pid=$(cat "$PID_FILE")
        print_step "Stopping vLLM server (PID: $pid)..."
        
        if kill "$pid" 2>/dev/null; then
            # Wait for process to stop
            local count=0
            while ps -p "$pid" > /dev/null 2>&1 && [[ $count -lt 30 ]]; do
                sleep 1
                ((count++))
            done
            
            if ps -p "$pid" > /dev/null 2>&1; then
                print_warning "Force killing server..."
                kill -9 "$pid" 2>/dev/null
            fi
            
            rm -f "$PID_FILE"
            print_success "vLLM server stopped"
        else
            print_error "Failed to stop server"
            exit 1
        fi
    else
        print_info "vLLM server is not running"
    fi
}

# Function to show server status
show_status() {
    print_header "${INFO} VLLM SERVER STATUS"
    
    if is_server_running; then
        local pid=$(cat "$PID_FILE")
        print_success "vLLM server is running (PID: $pid)"
        
        # Try to get server info
        if command -v curl > /dev/null 2>&1; then
            print_step "Checking server health..."
            if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
                print_success "Server is responding to health checks"
            else
                print_warning "Server is not responding (may still be starting up)"
            fi
            
            print_step "Available models:"
            if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null; then
                true # Output already displayed
            else
                print_warning "Could not retrieve model information"
            fi
        fi
    else
        print_info "vLLM server is not running"
    fi
}

# Function to show logs
show_logs() {
    print_header "📋 VLLM SERVER LOGS"
    
    if [[ -f "$LOG_FILE" ]]; then
        print_info "Showing last 50 lines of server logs:"
        echo ""
        tail -n 50 "$LOG_FILE"
    else
        print_warning "No log file found at $LOG_FILE"
    fi
}

# Function to start the server
start_server() {
    print_header "${ROCKET} LAUNCHING VLLM SERVER"
    
    # Check if server is already running
    if is_server_running; then
        print_warning "vLLM server is already running"
        show_status
        exit 1
    fi
    
    # Display configuration
    print_info "Server Configuration:"
    echo -e "  ${BOLD}Model:${NC} $MODEL"
    echo -e "  ${BOLD}Host:${NC} $HOST"
    echo -e "  ${BOLD}Port:${NC} $PORT"
    [[ -n "$MAX_MODEL_LEN" ]] && echo -e "  ${BOLD}Max Model Length:${NC} $MAX_MODEL_LEN"
    echo -e "  ${BOLD}GPU Memory Utilization:${NC} $GPU_MEMORY_UTILIZATION"
    echo -e "  ${BOLD}Tensor Parallel Size:${NC} $TENSOR_PARALLEL_SIZE"
    echo -e "  ${BOLD}Data Type:${NC} $DTYPE"
    echo -e "  ${BOLD}Trust Remote Code:${NC} $TRUST_REMOTE_CODE"
    echo ""
    
    # Build command
    local cmd="python3 -m vllm.entrypoints.openai.api_server"
    cmd+=" --model '$MODEL'"
    cmd+=" --host $HOST"
    cmd+=" --port $PORT"
    [[ -n "$MAX_MODEL_LEN" ]] && cmd+=" --max-model-len $MAX_MODEL_LEN"
    cmd+=" --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
    cmd+=" --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
    cmd+=" --dtype $DTYPE"
    [[ "$TRUST_REMOTE_CODE" == "true" ]] && cmd+=" --trust-remote-code"
    
    print_step "Starting vLLM server..."
    print_info "Command: $cmd"
    print_info "Logs will be saved to: $LOG_FILE"
    echo ""
    
    # Start server in background
    nohup bash -c "$cmd" > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    
    print_success "vLLM server started with PID: $pid"
    
    # Wait a moment and check if it's still running
    sleep 3
    if ps -p "$pid" > /dev/null 2>&1; then
        print_success "Server is running successfully"
        print_info "Server will be available at: http://$HOST:$PORT"
        print_info "API endpoint: http://$HOST:$PORT/v1"
        print_info "Use '$0 --logs' to view server logs"
        print_info "Use '$0 --status' to check server status"
        print_info "Use '$0 --stop' to stop the server"
    else
        print_error "Server failed to start. Check logs:"
        show_logs
        rm -f "$PID_FILE"
        exit 1
    fi
}

# Parse command line arguments
MODEL="$DEFAULT_MODEL"
HOST="$DEFAULT_HOST"
PORT="$DEFAULT_PORT"
MAX_MODEL_LEN="$DEFAULT_MAX_MODEL_LEN"
GPU_MEMORY_UTILIZATION="$DEFAULT_GPU_MEMORY_UTILIZATION"
TENSOR_PARALLEL_SIZE="$DEFAULT_TENSOR_PARALLEL_SIZE"
DTYPE="$DEFAULT_DTYPE"
TRUST_REMOTE_CODE="$DEFAULT_TRUST_REMOTE_CODE"

ACTION="start"

# Show help if no arguments provided
if [[ $# -eq 0 ]]; then
    show_help
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --trust-remote-code)
            TRUST_REMOTE_CODE="true"
            shift
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        --logs)
            ACTION="logs"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

# Execute action
case $ACTION in
    "start")
        check_dependencies
        validate_arguments
        start_server
        ;;
    "stop")
        stop_server
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    *)
        print_error "Unknown action: $ACTION"
        show_help
        exit 1
        ;;
esac

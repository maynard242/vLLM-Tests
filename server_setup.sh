#!/bin/bash
# =============================================================================
# Ubuntu Server Setup Script for vLLM Environment
#
# This script sets up an Ubuntu server with the necessary tools and packages
# for running vLLM inference servers and related Python applications.
#
# Features:
# - System package updates and upgrades
# - Essential tools installation (vim)
# - Python package management (pip upgrade)
# - vLLM and related Python packages installation
# - Hugging Face CLI setup
# - Enhanced visual output with progress indicators
#
# Prerequisites:
# - Ubuntu/Debian-based system
# - Root or sudo privileges
# - Internet connection
#
# Usage:
# sudo ./setup_server.sh
#
# =============================================================================

# Exit immediately if any command fails
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
PACKAGE="📦"
ROCKET="🚀"
WARNING="⚠️"
INFO="ℹ️"

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

# Function to show progress
show_progress() {
    local duration=$1
    local sleep_interval=0.1
    local progress=0
    local bar_length=50
    
    while [ $progress -le $duration ]; do
        # Calculate percentage
        local percent=$((progress * 100 / duration))
        local num_chars=$((progress * bar_length / duration))
        
        # Build progress bar
        local bar="["
        for ((i=0; i<num_chars; i++)); do
            bar+="█"
        done
        for ((i=num_chars; i<bar_length; i++)); do
            bar+="░"
        done
        bar+="]"
        
        # Display progress
        printf "\r${CYAN}%s %3d%%${NC}" "$bar" "$percent"
        
        sleep $sleep_interval
        ((progress++))
    done
    echo ""
}

# Function to run command with status
run_with_status() {
    local description="$1"
    local command="$2"
    
    print_step "$description"
    if eval "$command" > /dev/null 2>&1; then
        print_success "Completed: $description"
    else
        echo -e "${BOLD}${RED}❌ Failed: $description${NC}"
        echo -e "${BOLD}${RED}Command: $command${NC}"
        exit 1
    fi
}

# Main script starts here
clear
print_header "${ROCKET} UBUNTU SERVER SETUP FOR VLLM ENVIRONMENT"

print_info "Starting automated server setup process..."
print_info "This script will install essential tools and Python packages for vLLM"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_warning "This script must be run as root (use sudo)"
   echo "Usage: sudo $0"
   exit 1
fi

print_success "Running with root privileges"
echo ""

# --- 1. System Package Updates ---
print_header "${PACKAGE} SYSTEM PACKAGE MANAGEMENT"

print_step "Updating package lists..."
if apt update > /dev/null 2>&1; then
    print_success "Package lists updated successfully"
else
    echo -e "${BOLD}${RED}❌ Failed to update package lists${NC}"
    exit 1
fi

print_step "Upgrading installed packages (this may take a while)..."
echo -e "${YELLOW}${INFO} Please wait while system packages are upgraded...${NC}"
if apt upgrade -y > /dev/null 2>&1; then
    print_success "System packages upgraded successfully"
else
    echo -e "${BOLD}${RED}❌ Failed to upgrade packages${NC}"
    exit 1
fi

# --- 2. Essential Tools Installation ---
print_header "${GEAR} ESSENTIAL TOOLS INSTALLATION"

run_with_status "Installing vim text editor" "apt install -y vim"
run_with_status "Installing curl (if not present)" "apt install -y curl"
run_with_status "Installing wget (if not present)" "apt install -y wget"

# --- 3. Python Environment Setup ---
print_header "🐍 PYTHON ENVIRONMENT SETUP"

print_step "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Found: $PYTHON_VERSION"
else
    print_warning "Python3 not found, installing..."
    run_with_status "Installing Python3" "apt install -y python3 python3-pip"
fi

print_step "Upgrading pip package manager..."
if python3 -m pip install --upgrade pip > /dev/null 2>&1; then
    PIP_VERSION=$(pip3 --version | cut -d' ' -f2)
    print_success "Pip upgraded to version $PIP_VERSION"
else
    echo -e "${BOLD}${RED}❌ Failed to upgrade pip${NC}"
    exit 1
fi

# --- 4. Python Package Installation ---
print_header "${PACKAGE} PYTHON PACKAGES INSTALLATION"

declare -a packages=("requests" "openai" "pynvml" "vllm")
declare -a package_descriptions=(
    "HTTP library for API requests"
    "OpenAI API client library"
    "NVIDIA GPU monitoring library"
    "vLLM inference server framework"
)

for i in "${!packages[@]}"; do
    package="${packages[$i]}"
    description="${package_descriptions[$i]}"
    
    print_step "Installing $package ($description)..."
    if pip3 install "$package" > /dev/null 2>&1; then
        # Get installed version
        version=$(pip3 show "$package" 2>/dev/null | grep Version | cut -d' ' -f2)
        print_success "$package v$version installed successfully"
    else
        echo -e "${BOLD}${RED}❌ Failed to install $package${NC}"
        exit 1
    fi
done

# --- 5. Hugging Face CLI Installation ---
print_header "🤗 HUGGING FACE CLI INSTALLATION"

print_step "Installing Hugging Face CLI with all features..."
if pip3 install -U "huggingface_hub[cli]" > /dev/null 2>&1; then
    HF_VERSION=$(pip3 show huggingface-hub 2>/dev/null | grep Version | cut -d' ' -f2)
    print_success "Hugging Face CLI v$HF_VERSION installed successfully"
else
    echo -e "${BOLD}${RED}❌ Failed to install Hugging Face CLI${NC}"
    exit 1
fi

# --- 6. Installation Verification ---
print_header "${CHECKMARK} INSTALLATION VERIFICATION"

print_step "Verifying installations..."

# Check Python packages
declare -a verify_packages=("requests" "openai" "pynvml" "vllm" "huggingface_hub")
all_good=true

for package in "${verify_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        version=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null)
        print_success "$package (v$version) - OK"
    else
        echo -e "${BOLD}${RED}❌ $package - FAILED${NC}"
        all_good=false
    fi
done

# Check Hugging Face CLI
if command -v huggingface-cli &> /dev/null; then
    print_success "Hugging Face CLI command available"
else
    echo -e "${BOLD}${RED}❌ Hugging Face CLI command not found${NC}"
    all_good=false
fi

# --- 7. Final Summary ---
print_header "${ROCKET} SETUP COMPLETION SUMMARY"

if [ "$all_good" = true ]; then
    print_success "All components installed and verified successfully!"
    echo ""
    print_info "Installed Components:"
    echo -e "  ${BOLD}• System Tools:${NC} vim, curl, wget"
    echo -e "  ${BOLD}• Python Packages:${NC} requests, openai, pynvml, vllm"
    echo -e "  ${BOLD}• CLI Tools:${NC} Hugging Face CLI"
    echo ""
    print_info "Next Steps:"
    echo -e "  ${BOLD}1.${NC} Start a vLLM server: ${CYAN}python3 -m vllm.entrypoints.openai.api_server --model MODEL_NAME${NC}"
    echo -e "  ${BOLD}2.${NC} Test the setup with the provided benchmark scripts"
    echo -e "  ${BOLD}3.${NC} Login to Hugging Face: ${CYAN}huggingface-cli login${NC} (optional)"
    echo ""
    print_success "Server setup completed successfully! ${ROCKET}"
else
    echo -e "${BOLD}${RED}❌ Setup completed with errors. Please check the failed components above.${NC}"
    exit 1
fi

print_header "🎉 READY TO ROCK!"

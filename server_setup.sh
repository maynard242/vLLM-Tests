#!/bin/bash

# This script updates an Ubuntu server, installs vim, and sets up a Python environment with specific packages, including the Hugging Face CLI.
# The 'set -e' command ensures that the script will exit immediately if any command fails.
set -e

# --- 1. Update and Upgrade System Packages (as root) ---
echo "Updating package lists..."
apt update

echo "Upgrading installed packages..."
apt upgrade -y

# --- 2. Install Essential Tools (as root) ---
echo "Installing vim text editor..."
apt install -y vim

# --- 3. Upgrade Pip and Install Python Packages ---
echo "Upgrading pip package manager..."
# Use python3 to be explicit, as 'python' can sometimes point to python2 on older systems.
python3 -m pip install --upgrade pip

echo "Installing required Python packages: requests, openai, pynvml, and vllm..."
pip3 install requests openai pynvml vllm

# --- 4. Install Hugging Face CLI ---
echo "Installing Hugging Face CLI..."
pip3 install -U "huggingface_hub[cli]"

# --- 5. Completion Message ---
echo "✅ Server setup complete!"


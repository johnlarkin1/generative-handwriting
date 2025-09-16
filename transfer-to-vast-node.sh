#!/bin/bash

# Prompt user for SSH connection string
echo "Enter the SSH connection string (e.g., 'ssh -p 45234 root@74.15.83.102 -L 8080:localhost:8080'):"
read ssh_input

# Extract the port and host
port=$(echo "$ssh_input" | grep -oE '\-p [0-9]+' | awk '{print $2}')
host=$(echo "$ssh_input" | grep -oE 'root@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | cut -d'@' -f2)

# Verify the extracted port and host
if [[ -z "$port" || -z "$host" ]]; then
    echo "❌ Failed to extract port or host from the input."
    echo "Expected format: ssh -p PORT root@HOST -L 8080:localhost:8080"
    exit 1
fi

echo "✅ Extracted connection details:"
echo "   Port: $port"
echo "   Host: $host"
echo ""

# Define remote directory
remote_dir="~/code"

# Clean remote directory first
echo "🧹 Cleaning remote directory..."
ssh -i ~/.ssh/id_rsa_vast -p $port root@$host "rm -rf ~/code/* ~/code/.*"

# Copy only essential files
echo "📦 Transferring project files to remote server..."
rsync -avz -e "ssh -i ~/.ssh/id_rsa_vast -p $port" \
    --exclude='generative_handwriting/saved*' \
    --exclude='generative_handwriting/saved_models*' \
    --exclude='generative_handwriting/handwriting_visualizations*' \
    --exclude='saved_models*' \
    --exclude='saved*' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.venv' \
    --exclude='.mypy_cache' \
    --exclude='.claude' \
    --exclude='poetry.lock' \
    --exclude='.DS_Store' \
    --exclude='transfer-to-vast-node.sh' \
    --progress \
    generative_handwriting pyproject.toml README.md root@$host:$remote_dir/

if [ $? -ne 0 ]; then
    echo "❌ File transfer failed!"
    exit 1
fi

echo "✅ File transfer completed!"
echo ""

# Run setup commands remotely
echo "🔧 Setting up remote environment..."
ssh -i ~/.ssh/id_rsa_vast -p $port root@$host << 'EOF'
touch ~/.no_auto_tmux

# Update system and install dependencies
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y vim screen wget g++ git curl python3-pip python3-venv

# Install Poetry
echo "🎭 Installing Poetry..."
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH for current session and future sessions
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Download config files
echo "⚙️  Setting up config files..."
wget -q https://www.dropbox.com/s/cxws7jpt3nlxn2l/vimrc -O ~/.vimrc
wget -q https://www.dropbox.com/s/wbdlntxmujpat9o/screenrc -O ~/.screenrc

# Set up project
echo "🐍 Installing Python dependencies..."
cd ~/code
export PATH="$HOME/.local/bin:$PATH"
# Remove existing poetry environment and recreate
poetry env remove python3.11 2>/dev/null || true
poetry env use python3.11
# Clear cache and install exact versions
poetry cache clear --all pypi -n
poetry install --only=main --no-cache
# Add CUDA-enabled TensorFlow
poetry add "tensorflow[and-cuda]==2.18.0"

# Test installation
echo "🧪 Testing TensorFlow installation..."
poetry run python -c "import tensorflow as tf; print('✅ TensorFlow version:', tf.__version__)"

echo ""
echo "🚀 Setup complete!"
echo ""
echo "🔥 Your handwriting synthesis environment is ready!"
echo "📍 Project location: ~/code"
echo ""
echo "Commands to run:"
echo "  cd ~/code"
echo "  poetry shell"
echo "  python generative_handwriting/train_handwriting_prediction.py"
echo ""
EOF

# Print final connection info
echo ""
echo "🎯 Setup completed successfully!"
echo ""
echo "🔗 To connect to your machine:"
echo "   ssh -i ~/.ssh/id_rsa_vast -p $port root@$host"
echo ""
echo "🚀 Once connected, start training with our NaN-resistant fixes:"
echo "   cd ~/code"
echo "   poetry shell"
echo "   python generative_handwriting/train_handwriting_prediction.py"
echo ""
echo "✨ Features enabled:"
echo "   • Fixed gradient clipping bug"
echo "   • Enhanced MDN numerical stability"
echo "   • Real-time NaN detection & monitoring"
echo "   • Emergency checkpoint saving"
echo "   • Reduced learning rate (5e-5)"
echo ""

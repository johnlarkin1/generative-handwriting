#!/bin/bash

# Prompt user for input
# echo "Enter the SSH connection string (e.g., 'ssh -p 17097 root@5.158.194.150 -L 8080:localhost:8080'):"
# read ssh_input

# # Extract the port and host
# port=$(echo "$ssh_input" | grep -oE '-p [0-9]+' | awk '{print $2}')
# host=$(echo "$ssh_input" | grep -oE 'root@[^ ]+' | cut -d'@' -f2)

# # Verify the extracted port and host
# if [[ -z "$port" || -z "$host" ]]; then
#     echo "Failed to extract port or host from the input."
#     exit 1
# fi

port=to-do
host=to-do

# Define remote directory
remote_dir="~/code"

# Copy local code and requirements.txt
rsync -avz -e "ssh -i ~/.ssh/id_rsa_vast -p $port" --exclude='src/logs' --exclude='src/saved_models' --exclude='src/.git' --exclude='src/saved' src root@$host:$remote_dir/
scp -i ~/.ssh/id_rsa_vast -P $port requirements.txt root@$host:$remote_dir/

# Run setup commands remotely
ssh -i ~/.ssh/id_rsa_vast -p $port root@$host << 'EOF'
touch ~/.no_auto_tmux
apt-get update
apt-get install -y vim screen wget g++ git
wget https://www.dropbox.com/s/cxws7jpt3nlxn2l/vimrc -O ~/.vimrc
wget https://www.dropbox.com/s/wbdlntxmujpat9o/screenrc -O ~/.screenrc
python3 -m venv $HOME/env
source $HOME/env/bin/activate
EOF

# Print final SSH command for user
echo "To access your machine, use: ssh -i ~/.ssh/id_rsa_vast -p $port root@$host"

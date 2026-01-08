#!/bin/bash

# AWS EC2 GPU Instance Setup
setup_aws_gpu() {
    echo "Setting up AWS EC2 GPU instance..."
    
    # Update system
    sudo apt-get update
    sudo apt-get upgrade -y
    
    # Install NVIDIA drivers
    sudo apt-get install -y nvidia-driver-470
    
    # Install CUDA Toolkit
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
    
    # Install Python and pip
    sudo apt-get install -y python3-pip python3-dev
    
    # Install ML frameworks
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    pip3 install tensorflow
    pip3 install jupyter notebook
    
    echo "AWS GPU setup complete!"
}

# Google Colab Alternative Setup
setup_colab_environment() {
    echo "Setting up Colab-like environment..."
    
    # Check GPU availability
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    
    # Install additional packages
    pip install wandb tensorboard matplotlib seaborn
    pip install transformers datasets
    pip install accelerate
    
    # Mount Google Drive (if in Colab)
    if [ -d "/content" ]; then
        from google.colab import drive
        drive.mount('/content/drive')
    fi
}

# Setup monitoring tools
setup_monitoring() {
    echo "Setting up GPU monitoring tools..."
    
    # Install gpustat
    pip3 install gpustat
    
    # Install nvtop
    sudo apt-get install -y nvtop
    
    # Create monitoring script
    cat > ~/monitor_gpu.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
    echo ""
    echo "=== Process Info ==="
    nvidia-smi pmon -c 1
    sleep 2
done
EOF
    chmod +x ~/monitor_gpu.sh
    
    echo "Monitoring setup complete! Run ~/monitor_gpu.sh to monitor GPU"
}

# Setup Jupyter for remote access
setup_jupyter_remote() {
    echo "Setting up Jupyter for remote access..."
    
    # Generate Jupyter config
    jupyter notebook --generate-config
    
    # Create password
    python3 -c "from notebook.auth import passwd; print(passwd())" > ~/.jupyter/jupyter_password.txt
    
    # Configure Jupyter
    cat >> ~/.jupyter/jupyter_notebook_config.py << 'EOF'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.allow_remote_access = True
EOF
    
    echo "Jupyter remote setup complete!"
    echo "Start with: jupyter notebook --no-browser --port=8888"
}

# Docker GPU setup
setup_docker_gpu() {
    echo "Setting up Docker with GPU support..."
    
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    
    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    
    # Test GPU in Docker
    sudo docker run --rm --gpus all nvidia/cuda:11.7.0-base-ubuntu20.04 nvidia-smi
    
    echo "Docker GPU setup complete!"
}

# Main menu
echo "GPU Deep Learning Setup Scripts"
echo "1. Setup AWS EC2 GPU Instance"
echo "2. Setup Colab-like Environment"
echo "3. Setup GPU Monitoring Tools"
echo "4. Setup Jupyter for Remote Access"
echo "5. Setup Docker with GPU Support"
echo "6. Run all setups"

read -p "Enter your choice (1-6): " choice

case $choice in
    1) setup_aws_gpu ;;
    2) setup_colab_environment ;;
    3) setup_monitoring ;;
    4) setup_jupyter_remote ;;
    5) setup_docker_gpu ;;
    6) 
        setup_aws_gpu
        setup_colab_environment
        setup_monitoring
        setup_jupyter_remote
        setup_docker_gpu
        ;;
    *) echo "Invalid choice" ;;
esac
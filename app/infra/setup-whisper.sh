"""
Script initialisation EC2
"""
#!/bin/bash

# Log setup
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "=== Summora Whisper EC2 Setup ==="

# Mise à jour système
apt-get update -y
apt-get install -y python3-pip docker.io

# Installation Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Démarrage Docker
systemctl start docker
systemctl enable docker

# Installation NVIDIA drivers (pour g4dn)
apt-get install -y nvidia-driver-470
apt-get install -y nvidia-docker2

# Redémarrage pour drivers
reboot

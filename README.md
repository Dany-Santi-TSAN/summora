# Summora
Speech in, sense out

## 🚀 Workflow optimal pour AWS

### 1. Transfer script + code
#### Transfer tout en une fois
scp -i keys/summora-keypair-190625.pem -r . ubuntu@IP:~/summora/

#### Ou script seul
scp -i keys/summora-keypair-190625.pem deploy/setup_aws_gpu.sh ubuntu@IP:~/

### 2. Exécution

bash setup_aws_gpu.sh

### (reboot auto)
#### Après reboot:

bash setup_aws_gpu.sh --post-reboot

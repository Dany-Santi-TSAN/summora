# Summora
Speech in, sense out

## AWS Console
- Instance: g4dn.xlarge
- AMI: Ubuntu 22.04 LTS
- Key: summora-keypair-190625
- Security group: summora-security-group


## 🚀 Workflow optimal pour AWS

### Connexion + transfer
ssh -i keys/summora-keypair-190625.pem ubuntu@[NOUVELLE_IP]
scp -i keys/summora-keypair-190625.pem deploy/setup_aws_gpu.sh ubuntu@[NOUVELLE_IP]:~/

### Test setup phase 1
bash setup_aws_gpu.sh
#### (reboot auto)

### Test setup phase 2
ssh -i keys/summora-keypair-190625.pem ubuntu@[NOUVELLE_IP]
bash setup_aws_gpu.sh --post-reboot

### Validation
summora
gpu-status

## Test final benchmark

### Transfer code + test
scp -i keys/summora-keypair-190625.pem -r * ubuntu@[NOUVELLE_IP]:~/summora/
cd summora
python benchmark_whisper_gpu.py

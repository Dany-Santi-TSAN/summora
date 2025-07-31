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
# Transfer setup + requirements en une fois
scp -i keys/summora-keypair-190625.pem \
    {deploy/setup_aws_gpu.sh,requirements.txt} \
    ubuntu@[NOUVELLE_IP]:~/

### Test setup phase 1
bash setup_aws_gpu.sh
#### (reboot auto)

### Test setup phase 2
ssh -i keys/summora-keypair-190625.pem ubuntu@[NOUVELLE_IP]
bash setup_aws_gpu.sh --post-reboot

## Test final benchmark

### Transfer code + test
scp -i keys/summora-keypair-190625.pem -r * ubuntu@[NOUVELLE_IP]:~/summora/
### environnement
source summora-env/bin/activate
cd summora

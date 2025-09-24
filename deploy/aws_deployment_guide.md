# Summora
Speech in, sense out
Anlayse intelligente de réunion

## AWS Console
- Instance: g4dn.xlarge
- AMI: Ubuntu 22.04 LTS
- EBS : 50go gp2
- Key: summora-keypair-190625
- Security group: summora-security-group
- id compte aws : 788922688762
- id canonical : 04dd8c701a0859e4b9f1f1eb3a095014256b308718513e4de1e3cbe1a8218327


## 🚀 Workflow optimal pour AWS

### Connexion + transfer
ssh -i keys/summora-keypair-190625.pem ubuntu@[NOUVELLE_IP]
# Transfer setup + requirements en une fois
scp -i keys/summora-keypair-190625.pem \
    {deploy/deploy_summora_aws.sh,requirements.txt} \
    ubuntu@[NOUVELLE_IP]:~/

### Setup phase 1
bash deploy_summora_aws.sh
#### (reboot auto)

### Setup phase 2 (post reboot)
ssh -i keys/summora-keypair-190625.pem ubuntu@[NOUVELLE_IP]
bash deploy_summora_aws.sh --post-reboot

## Test final

### Transfer code + test
scp -i keys/summora-keypair-190625.pem -r * ubuntu@[NOUVELLE_IP]:~/summora/

### Environnement
source summora-env/bin/activate
cd summora

### Vérification installation des dépendances
pip install -r requirements.txt

### Créer le dot env
touch .env
echo "OPENROUTER_API_KEY=vjkj..." >> .env
echo "HF_TOKEN=flkdjfkl..." >> .env
cat .env

# Récupérer tous les fichiers du dossier ~/summora de l'instance
scp -i keys/summora-keypair-190625.pem -r ubuntu@[NOUVELLE_IP]:~/summora/* ./output/

### Penser à résilier l'instance

## /!\ Zsh interprète le wildcard avant SCP

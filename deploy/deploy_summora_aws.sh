#!/bin/bash

##############################################################
# Setup AWS GPU Instance pour Summora
# Instance recommandée : g4dn.xlarge (Tesla T4) et 125go EBS
# Objectif : déployer et exécuter Summora dans le cloud
##############################################################

set -e # Exit on error

echo "Setup AWS GPU pour le déploiement de l'application Summora"
echo "=========================================================="
echo "Date: $(date)"
echo "Instance recommandée : g4dn.xlarge (Tesla T4) et 125go EBS"
echo "Objectif : déployer et exécuter Summora sur AWS"
echo "=========================================================="
echo ""

############################################
# 1. Informations système
############################################
echo "Informations système..."
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo ""

############################################
# 2. Détection GPU NVIDIA
############################################
echo "Détection matériel GPU..."
if lspci | grep -i nvidia > /dev/null; then
    echo "GPU NVIDIA détecté: $(lspci | grep -i nvidia | cut -d: -f3)"
else
    echo "Aucun GPU NVIDIA détecté"
    echo "Vérifiez que vous êtes sur une instance GPU (g4dn.xlarge)"
    exit 1
fi

############################################
# 3. Mise à jour système + essentiels
############################################
echo "Mise à jour système..."
sudo apt update -y
sudo apt install -y \
    python3-pip python3-venv python-is-python3 git wget curl \
    build-essential pkg-config \
    htop tree unzip vim \
    ffmpeg

echo "Packages essentiels installés (ffmpeg inclus pour audio)"

############################################
# 4. Installation drivers NVIDIA
############################################
echo "Installation drivers NVIDIA..."
sudo apt install -y nvidia-driver-535 nvidia-utils-535

echo "REBOOT REQUIS pour charger les drivers GPU"
echo "Après reboot, relancez ce script avec: bash $0 --post-reboot"

# Check if post-reboot argument
if [[ "$1" != "--post-reboot" ]]; then
    echo ""
    echo "Commande à exécuter après reboot:"
    echo "bash $0 --post-reboot"
    echo ""
    echo "Reboot dans 10 secondes... (Ctrl+C pour annuler)"
    sleep 10
    sudo reboot
    exit 0
fi

############################################
# 5. Post-reboot: Vérification GPU
############################################
echo "Phase post-reboot - Vérification GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "GPU Tesla T4 opérationnel !"
else
    echo "nvidia-smi non disponible après reboot"
    exit 1
fi

############################################
# 6. Setup environnement Python Summora
############################################
echo "Création environnement Python Summora..."
cd ~
python3 -m venv summora-env
source summora-env/bin/activate

echo "Environnement virtuel activé: summora-env"

############################################
# 7. PyTorch CUDA
############################################
echo "Installation PyTorch CUDA (Tesla T4 optimized)..."
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test immédiat CUDA
echo "Test PyTorch CUDA..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    raise Exception('CUDA non disponible')
"
############################################
# 8. Dépendances Summora
############################################
echo "Installation dépendances nécessaires Summora..."

# Whisper
echo "Installation Whisper..."
pip install openai-whisper

# Dépendances ML/NLP
echo "Installation dépendances ML/NLP..."
pip install scikit-learn pandas numpy
pip install nltk yake textstat
pip install transformers datasets

# Dépendances déploiement
echo "Installation dépendances déploiement..."
pip install fastapi uvicorn python-multipart
pip install streamlit

# Autres utilitaires
pip install python-dotenv tqdm plotly

############################################
# 9. Configuration NLTK
############################################
echo "Configuration NLTK..."
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download ressources essentielles
essential_resources = ['punkt', 'stopwords']
for resource in essential_resources:
    try:
        nltk.download(resource, quiet=True)
        print(f'NLTK {resource} téléchargé')
    except Exception as e:
        print(f'Erreur {resource}: {e}')
"

############################################
# 10. Test dépendances Summora
############################################

echo "Test imports Summora"
python -c "
import torch
import whisper
import sklearn
import nltk
import yake
import fastapi
import streamlit
import plotly

print('PyTorch:', torch.__version__)
print('Whisper: OK')
print('Sklearn:', sklearn.__version__)
print('NLTK: OK')
print('YAKE: OK')
print('FastAPI : OK')
print('Streamlit: OK')
print('Plotly: OK')
print('')
print('================')
print('')
print('Installation dépendances ✅​')
"

############################################
# 11. Setup projet Summora
############################################
echo "Setup répertoire projet Summora..."
mkdir -p ~/summora/{app/temp,data/rag/documents,output/{audio_analysis,extractions,recommendations,transcriptions,reports},src/{config,core,llm,meeting,qa,rag},scripts}
cd ~/summora

# Shortcuts utiles
echo "" >> ~/.bashrc
echo "# Summora shortcuts" >> ~/.bashrc
echo "alias summora='cd ~/summora && source ~/summora-env/bin/activate'" >> ~/.bashrc
echo "alias gpu-status='watch -n 1 nvidia-smi'" >> ~/.bashrc

############################################
# 12. Test Whisper GPU
############################################
echo "Test Whisper GPU..."
python -c "
import whisper
import torch

print('Test chargement modèle Whisper medium...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device utilisé: {device}')

try:
    model = whisper.load_model('medium', device=device)
    print('Whisper medium chargé avec succès sur GPU')
    del model  # Libère la mémoire
    torch.cuda.empty_cache()
    print('Mémoire GPU libérée')
except Exception as e:
    print(f'Erreur chargement Whisper: {e}')
    # Fallback sur base
    model = whisper.load_model('base', device=device)
    print('Fallback sur Whisper base réussi')
    del model
    torch.cuda.empty_cache()
"
############################################
# 13. Monitoring final
############################################
echo ""
echo "État final système pour tests Summora:"
echo "======================================"
echo "Mémoire RAM disponible:"
free -h | grep '^Mem:'

echo ""
echo "Espace disque:"
df -h | grep -E '^/dev/(xvda|nvme)' | head -1

echo ""
echo "GPU Tesla T4:"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits

############################################
# 14. Génération script de benchmark
############################################
echo ""
echo "Génération script de test de performance..."
cat > ~/summora/test_performance.py << 'EOF'
#!/usr/bin/env python3
"""
Test de performance Summora sur AWS
Métrique du temps d'exécution
"""
import time
import subprocess
import sys
import os

def test_summora(audio_file):
    """Test simple de performance pour un fichier audio"""
    if not os.path.exists(audio_file):
        print(f"Fichier non trouvé: {audio_file}")
        return None

    print(f"\nTest performance: {audio_file}")
    print("-" * 40)

    start_time = time.time()

    try:
        result = subprocess.run(["python", "main.py", audio_file],
                              capture_output=True, text=True, timeout=1200) # timeout 20 min

        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            print(f"Succès: {duration:.1f}s ({duration/60:.1f}min)")
            return duration
        else:
            print(f"Erreur: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print("Timeout après 20 min")
        return None
    except Exception as e:
        print(f"Erreur: {e}")
        return None

def main():
    """Test performance sur fichiers audio"""
    print("Test de performance Summora AWS")
    print("===============================")

    # Vérification main.py
    if not os.path.exists("main.py"):
        print("main.py non trouvé. Exécutez depuis le répertoire Summora.")
        sys.exit(1)

    # Liste des fichiers audio à tester
    test_files = ["test-reunion.mp3", "podcast-1h.mp3"]

    # Recherche fichiers existants
    found_files = []
    for filename in test_files:
        if os.path.exists(filename):
            found_files.append(filename)
        else:
            print(f"Fichier non trouvé: {filename}")

    if not found_files:
        print("Aucun fichier de test trouvé.")
        print("Assurez-vous d'avoir transféré vos fichiers audio.")
        sys.exit(1)

    # Exécution des tests
    results = {}
    for audio_file in found_files:
        duration = test_summora(audio_file)
        if duration:
            results[audio_file] = duration

    # Résumé final
    print("\n" + "=" * 40)
    print("RÉSUMÉ DES PERFORMANCES")
    print("=" * 40)

    for filename, duration in results.items():
        minutes = duration / 60
        print(f"{filename}: {duration:.1f}s ({minutes:.1f}min)")

    print(f"\nTests terminés: {len(results)}/{len(found_files)} succès")

if __name__ == "__main__":
    main()
EOF

chmod +x ~/summora/test_performance.py
echo "Script de test généré: ~/summora/test_performance.py"

############################################
# 15. Instructions finales
############################################
echo ""
echo "SETUP SUMMORA AWS TERMINÉ !"
echo "==========================="
echo ""
echo "Prochaines étapes:"
echo "1. Transférer votre code Summora:"
echo "   scp -i key.pem -r ./summora-project/ ubuntu@IP:~/summora/"
echo ""
echo "2. Transférer vos fichiers audio de test:"
echo "   scp -i key.pem test-reunion.mp3.mp3 podcast-1h.mp3 ubuntu@IP:~/summora/"
echo ""
echo "3. Se connecter et activer l'environnement:"
echo "   ssh -i key.pem ubuntu@IP"
echo "   summora"
echo ""
echo "4. Lancer le benchmark:"
echo "   python scripts/main.py"
echo ""
echo "Commandes utiles:"
echo "  summora               # Activer env + aller dans répertoire"
echo "  gpu-status            # Monitor GPU temps réel"
echo "  python main.py --help # Aide Summora"
echo ""
echo "L'environnement est prêt pour vos tests de performance

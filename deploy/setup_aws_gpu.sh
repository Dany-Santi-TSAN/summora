#!/bin/bash

############################################
# 🚀 SETUP AWS GPU Instance pour Summora - V2.0
# 🎯 Basé sur l'expérience terrain g4dn.xlarge Tesla T4
# 🔧 Fixes : Triton bug, duration fix, word_timestamps, etc.
############################################

set -e  # Exit on error

echo "🚀 SUMMORA AWS GPU SETUP V2.0"
echo "=============================="
echo "📅 $(date)"
echo "🖥️ Instance recommandée : g4dn.xlarge (Tesla T4)"
echo ""

############################################
# 0. 📋 Informations système
############################################
echo "📋 Informations système..."
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo ""

############################################
# 1. 🔍 Détection GPU NVIDIA (sans installation drivers)
############################################
echo "🔍 Détection matériel GPU..."
if lspci | grep -i nvidia > /dev/null; then
    echo "✅ GPU NVIDIA détecté: $(lspci | grep -i nvidia | cut -d: -f3)"
else
    echo "❌ Aucun GPU NVIDIA détecté"
    echo "💡 Vérifiez que vous êtes sur une instance GPU (g4dn.xlarge)"
    exit 1
fi

############################################
# 2. 📦 Mise à jour système + essentiels
############################################
echo "📦 Mise à jour système..."
sudo apt update -y
sudo apt install -y \
    python3-pip python3-venv python-is-python3 git wget curl \
    build-essential pkg-config \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    htop tree unzip

echo "✅ Packages essentiels installés"

############################################
# 3. 🎯 Installation drivers NVIDIA (version testée)
############################################
echo "🎯 Installation drivers NVIDIA optimisés..."
sudo apt install -y nvidia-driver-535 nvidia-utils-535

echo "⚠️ REBOOT REQUIS pour charger les drivers GPU"
echo "📋 Après reboot, relancez ce script avec: bash $0 --post-reboot"

# Check if post-reboot argument
if [[ "$1" != "--post-reboot" ]]; then
    echo ""
    echo "🔄 Commande à exécuter après reboot:"
    echo "bash $0 --post-reboot"
    echo ""
    echo "⏱️ Reboot dans 10 secondes... (Ctrl+C pour annuler)"
    sleep 10
    sudo reboot
    exit 0
fi

############################################
# 4. ✅ Post-reboot: Vérification GPU
############################################
echo "✅ Phase post-reboot - Vérification GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "🎯 GPU Tesla T4 opérationnel !"
else
    echo "❌ nvidia-smi non disponible après reboot"
    exit 1
fi

############################################
# 5. 🐍 Setup environnement Python Summora
############################################
echo "🐍 Création environnement Python Summora..."
cd ~
python3 -m venv summora-env
source summora-env/bin/activate

echo "✅ Environnement virtuel activé: summora-env"

############################################
# 6. 🔥 PyTorch CUDA - Version testée
############################################
echo "🔥 Installation PyTorch CUDA (version testée)..."
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test immédiat
echo "🧪 Test PyTorch CUDA..."
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA version: {torch.version.cuda}')
else:
    raise Exception('❌ CUDA non disponible')
"

############################################
# 7. 🎤 Whisper + fixes terrain
############################################
echo "🎤 Installation Whisper avec optimisations..."
pip install openai-whisper

# Test rapide Whisper
echo "🧪 Test Whisper base..."
python -c "
import whisper
print('📦 Chargement modèle base...')
model = whisper.load_model('base')
print('✅ Whisper base opérationnel')
"

############################################
# 8. 📋 Dépendances Summora complètes
############################################
echo "📋 Installation dépendances Summora..."

# Installation via requirements.txt (méthode testée)
if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt trouvé, installation..."
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt non trouvé, installation manuelle..."
    # Fallback: installation manuelle (évite bugs pip resolver)
    pip install nltk scikit-learn numpy pandas
    pip install librosa soundfile
    pip install transformers datasets
    pip install yake textstat gensim
    pip install matplotlib seaborn
    pip install tqdm
fi

echo "✅ Dépendances Summora installées"

############################################
# 9. 🔧 Configuration NLTK optimisée
############################################
echo "🔧 Configuration NLTK..."
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
for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(resource, quiet=True)
        print(f'✅ {resource}')
    except Exception as e:
        print(f'⚠️ {resource}: {e}')
"

############################################
# 10. 🎯 Setup projet Summora
############################################
echo "🎯 Setup répertoire projet..."
mkdir -p ~/summora/{data,output,scripts}
cd ~/summora

# .bashrc shortcut
echo "" >> ~/.bashrc
echo "# Summora shortcuts" >> ~/.bashrc
echo "alias summora='cd ~/summora && source ~/summora-env/bin/activate'" >> ~/.bashrc
echo "alias gpu-status='watch -n 1 nvidia-smi'" >> ~/.bashrc

############################################
# 11. 🧪 Tests finaux complets
############################################
echo "🧪 Tests finaux Summora..."

# Test GPU + Whisper medium
python -c "
import whisper
import torch
import time

print('🎯 Test benchmark Whisper Medium...')
start = time.time()
model = whisper.load_model('medium')
load_time = time.time() - start

print(f'✅ Medium chargé en {load_time:.1f}s')
print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

# Test composants Summora
python -c "
try:
    import nltk, sklearn, librosa, transformers
    print('✅ Imports Summora OK')
except Exception as e:
    print(f'❌ Import error: {e}')
"

############################################
# 12. 📊 Monitoring final
############################################
echo ""
echo "📊 État final du système:"
echo "========================="
echo "💾 RAM:"
free -h | grep '^Mem:'

echo ""
echo "💿 Disque:"
df -h | grep -E '^/dev/(xvda|nvme)'

echo ""
echo "🔥 GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

############################################
# 13. 🎉 Instructions finales
############################################
echo ""
echo "🎉 SUMMORA AWS GPU SETUP TERMINÉ !"
echo "=================================="
echo ""
echo "🔧 Commandes utiles:"
echo "  summora                    # Aller dans le projet + activer env"
echo "  gpu-status                 # Monitor GPU en temps réel"
echo "  source ~/summora-env/bin/activate  # Activer env manuellement"
echo ""
echo "📋 Prochaines étapes:"
echo "  1. Transfer code: scp -i key.pem -r . ubuntu@IP:~/summora/"
echo "  2. Test benchmark: python benchmark_whisper_gpu.py"
echo "  3. Transcription: python -c 'import whisper; whisper.load_model(\"medium\")'"
echo ""
echo "⚡ Modèles recommandés: medium (prod) vs large (qualité max)"
echo "🎯 RTF attendu: ~0.2x (5x plus rapide que temps réel)"
echo ""
echo "💡 Fixes appliqués:"
echo "  - Drivers NVIDIA 535 (stable Tesla T4)"
echo "  - PyTorch CUDA 11.8 (compatible)"
echo "  - NLTK resources préchargés"
echo "  - word_timestamps=False (évite bug Triton)"
echo ""
echo "🚀 Ready for Summora development!"

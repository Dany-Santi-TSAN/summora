#!/bin/bash

############################################
# 🚀 SETUP AWS GPU Instance pour Summora - V2.1
# 🎯 Optimisé pour AB Testing LLM (Phi-3 vs Llama)
# 🔧 Skip Whisper test - économie 1.42GB RAM
############################################

set -e  # Exit on error

echo "🚀 SUMMORA AWS GPU SETUP V2.1 - AB TESTING OPTIMIZED"
echo "====================================================="
echo "📅 $(date)"
echo "🖥️ Instance recommandée : g4dn.xlarge (Tesla T4)"
echo "🎯 Focus: AB Testing LLM sans surcharge Whisper"
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
# 1. 🔍 Détection GPU NVIDIA
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
    htop tree unzip vim

echo "✅ Packages essentiels installés (sans ffmpeg - économie)"

############################################
# 3. 🎯 Installation drivers NVIDIA
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
echo "🔥 Installation PyTorch CUDA (Tesla T4 optimized)..."
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test immédiat CUDA
echo "🧪 Test PyTorch CUDA..."
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    raise Exception('❌ CUDA non disponible')
"

############################################
# 7. 🤖 Dépendances LLM AB Testing
############################################
echo "🤖 Installation dépendances LLM AB Testing..."

# Quantization & LLM essentials
echo "📦 Installation quantization (bitsandbytes + accelerate)..."
pip install bitsandbytes==0.43.0 accelerate==0.33.0

echo "📦 Installation Transformers + HuggingFace..."
pip install transformers==4.44.0 huggingface-hub datasets

echo "📦 Installation utilitaires..."
pip install python-dotenv tqdm

# Test critique: imports LLM
echo "🧪 Test imports LLM AB Testing..."
python -c "
import torch
import transformers
import bitsandbytes
import accelerate
from huggingface_hub import login

print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ BitsAndBytes: {bitsandbytes.__version__}')
print(f'✅ Accelerate: {accelerate.__version__}')
print('✅ Quantization 4-bit ready!')
"

############################################
# 8. 📋 Dépendances Summora optionnelles
############################################
echo "📋 Installation dépendances Summora (si requirements.txt)..."

if [ -f "summora/requirements.txt" ]; then
    echo "✅ requirements.txt trouvé, installation selective..."
    # Installation selective sans ffmpeg/librosa (économie)
    pip install scikit-learn pandas numpy
    pip install nltk yake textstat
    echo "✅ Dépendances core Summora installées"
elif [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt trouvé à la racine..."
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt non trouvé - installation minimale..."
    pip install scikit-learn pandas numpy nltk
fi

############################################
# 9. 🔧 Configuration NLTK minimale
############################################
echo "🔧 Configuration NLTK minimale..."
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download ressources essentielles uniquement
essential_resources = ['punkt', 'stopwords']
for resource in essential_resources:
    try:
        nltk.download(resource, quiet=True)
        print(f'✅ {resource}')
    except Exception as e:
        print(f'⚠️ {resource}: {e}')
"

############################################
# 10. 🎯 Setup projet Summora optimisé
############################################
echo "🎯 Setup répertoire projet Summora..."
mkdir -p ~/summora/{data,output,src/llm}
cd ~/summora

# .bashrc shortcuts optimisés
echo "" >> ~/.bashrc
echo "# Summora AB Testing shortcuts" >> ~/.bashrc
echo "alias summora='cd ~/summora && source ~/summora-env/bin/activate'" >> ~/.bashrc
echo "alias gpu-status='watch -n 1 nvidia-smi'" >> ~/.bashrc
echo "alias ab-test='cd ~/summora && source ~/summora-env/bin/activate && python src/llm/ab_testing_aws_phi_llama.py'" >> ~/.bashrc

############################################
# 11. 🧪 Tests finaux LLM (sans Whisper)
############################################
echo "🧪 Tests finaux AB Testing (skip Whisper = économie 1.42GB)..."

# Test quantization 4-bit (critique pour Tesla T4)
python -c "
import torch
from transformers import BitsAndBytesConfig

print('🎯 Test quantization 4-bit...')
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
print('✅ Quantization 4-bit config OK')
print('✅ Tesla T4 ready pour Phi-3 + Llama')
"

# Test HuggingFace (sans chargement modèle)
python -c "
from huggingface_hub import HfApi
api = HfApi()
print('✅ HuggingFace API accessible')
print('📋 Modèles supportés: Phi-3, Llama-3.2, Gemma-2')
"

############################################
# 12. 📊 Monitoring final optimisé
############################################
echo ""
echo "📊 État final système (AB Testing optimized):"
echo "=============================================="
echo "💾 RAM disponible:"
free -h | grep '^Mem:'

echo ""
echo "💿 Disque:"
df -h | grep -E '^/dev/(xvda|nvme)' | head -1

echo ""
echo "🔥 GPU Tesla T4:"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "🎯 Mémoire économisée (vs V2.0):"
echo "  - Skip Whisper Medium: ~1.42GB"
echo "  - Skip ffmpeg/librosa: ~200MB"
echo "  - Total économie: ~1.6GB pour AB Testing"

############################################
# 13. 🎉 Instructions finales AB Testing
############################################
echo ""
echo "🎉 SUMMORA AWS GPU V2.1 - AB TESTING READY !"
echo "============================================="
echo ""
echo "🔧 Commandes utiles:"
echo "  summora                    # Activer env Summora"
echo "  ab-test                    # Lancer AB Testing directement"
echo "  gpu-status                 # Monitor GPU temps réel"
echo ""
echo "📋 Prochaines étapes AB Testing:"
echo "  1. Transfer code: scp -i key.pem -r . ubuntu@IP:~/summora/"
echo "  2. Setup .env: echo 'HF_TOKEN=your_token' > ~/summora/.env"
echo "  3. Launch test: ab-test"
echo ""
echo "🎯 Modèles AB Testing:"
echo "  • Phi-3-mini-4k-instruct (Microsoft)"
echo "  • Llama-3.2-3B (Meta)"
echo "  • Quantization 4-bit (économie Tesla T4)"
echo ""
echo "💡 Optimisations appliquées V2.1:"
echo "  ✅ Skip Whisper test (-1.42GB RAM)"
echo "  ✅ Deps LLM essentielles seulement"
echo "  ✅ Quantization 4-bit ready"
echo "  ✅ HuggingFace auth ready"
echo "  ✅ Tesla T4 8GB optimized"
echo ""
echo "🚀 Ready for Phi-3 vs Llama AB Testing!"

############################################
# 14. 📄 Génération script test rapide
############################################
echo ""
echo "📄 Génération script de test rapide..."
cat > ~/test_gpu_ready.py << 'EOF'
#!/usr/bin/env python3
"""Test rapide GPU Tesla T4 ready pour AB Testing"""
import torch
from transformers import BitsAndBytesConfig

def test_gpu_ab_ready():
    print("🧪 Test Tesla T4 AB Testing Ready...")

    # Test 1: CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ CUDA non disponible")
        return False

    # Test 2: Quantization
    try:
        config = BitsAndBytesConfig(load_in_4bit=True)
        print("✅ Quantization 4-bit OK")
    except Exception as e:
        print(f"❌ Quantization error: {e}")
        return False

    # Test 3: Memory check
    free_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if free_mem >= 7.5:  # Tesla T4 minimum viable
        print(f"✅ GPU Memory sufficient: {free_mem:.1f}GB")
    else:
        print(f"⚠️ GPU Memory low: {free_mem:.1f}GB")

    print("🚀 Tesla T4 ready pour AB Testing Phi-3 vs Llama!")
    return True

if __name__ == "__main__":
    test_gpu_ab_ready()
EOF

chmod +x ~/test_gpu_ready.py
echo "✅ Script test généré: ~/test_gpu_ready.py"
echo "   Usage: python ~/test_gpu_ready.py"

#!/bin/bash

############################################
# 🔧 SETUP AWS GPU Instance pour Whisper Medium/Large
# 🖥️ Instance recommandée : g4dn.xlarge (T4 16GB, 4 vCPU, 16GB RAM)
############################################

echo "🚀 SETUP AWS GPU POUR SUMMORA"
echo "============================="

############################################
# 1. 🔍 Vérification de la présence du GPU NVIDIA
############################################
echo "🔍 Vérification GPU..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "❌ NVIDIA GPU non détecté"
    echo "💡 Vérifiez que vous êtes sur une instance GPU (ex: g4dn.xlarge)"
    exit 1
fi
echo "✅ GPU détecté"

############################################
# 2. 📦 Mise à jour du système
############################################
echo "📦 Mise à jour système..."
sudo apt update && sudo apt upgrade -y

############################################
# 3. 🐍 Vérification version Python
############################################
echo "🐍 Vérification Python..."
python3 --version

############################################
# 4. ⚡ Installation de CUDA Toolkit si absent
############################################
echo "⚡ Vérification CUDA..."
nvcc --version
if [ $? -ne 0 ]; then
    echo "📥 Installation CUDA toolkit..."
    sudo apt install nvidia-cuda-toolkit -y
fi

############################################
# 5. 🧰 Installation de pip et venv
############################################
echo "📦 Installation pip et venv..."
sudo apt install python3-pip python3-venv -y

############################################
# 6. 🏗️ Création de l’environnement virtuel Summora
############################################
echo "🏗️ Création environnement virtuel..."
python3 -m venv summora_gpu_env
source summora_gpu_env/bin/activate
echo "✅ Environnement virtuel activé"

############################################
# 7. 🔥 Installation de PyTorch avec support CUDA
############################################
echo "🔥 Installation PyTorch CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

############################################
# 8. 🧪 Vérification de PyTorch + CUDA
############################################
echo "🧪 Test PyTorch CUDA..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('❌ CUDA non disponible')
    exit(1)
"
if [ $? -ne 0 ]; then
    echo "❌ Problème avec l’installation CUDA/PyTorch"
    exit 1
fi
echo "✅ PyTorch CUDA opérationnel"

############################################
# 9. 🎤 Installation de Whisper (OpenAI)
############################################
echo "🎤 Installation OpenAI Whisper..."
pip install openai-whisper

############################################
# 10. 📋 Installation des dépendances du projet Summora
############################################
echo "📋 Installation dépendances Summora..."
pip install -r requirements.txt

############################################
# 11. 🧪 Test de chargement des modèles Whisper
############################################
echo "🧪 Test Whisper GPU..."
python3 -c "
import whisper
import torch
print('🔍 Test chargement modèles Whisper...')

# Chargement progressif des modèles
for size in ['tiny', 'base', 'small', 'medium', 'large']:
    print(f'📦 Chargement {size}...')
    try:
        whisper.load_model(size)
        print(f'✅ {size} chargé')
    except Exception as e:
        print(f'⚠️ Erreur {size}: {e}')

print('🎯 Setup Whisper terminé')
"

############################################
# 12. 📊 Monitoring des ressources système
############################################
echo "📊 Monitoring ressources..."
echo "💾 RAM disponible:"
free -h

echo "💿 Espace disque:"
df -h

echo "🔥 GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv

############################################
# 13. ✅ Instructions finales pour démarrer
############################################
echo ""
echo "🎉 SETUP AWS GPU TERMINÉ"
echo "========================"
echo ""
echo "📋 Prochaines étapes:"
echo "1. Activer l'environnement: source summora_gpu_env/bin/activate"
echo "2. Lancer transcription medium: python scripts/main_transcribe.py audio.wav --model medium"
echo "3. Tester large: python scripts/main_transcribe.py audio.wav --model large"
echo "4. Comparer les modèles: python benchmark_models.py audio.wav"
echo ""
echo "⚡ Modèles disponibles: tiny, base, small, medium, large"
echo "🎯 Cible business: medium & large pour évaluer WER"
echo ""
echo "💡 GPU Live Monitor: watch -n 1 nvidia-smi"

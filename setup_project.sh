#!/usr/bin/env bash
# =============================================================================
# ECO-REWIND: Project Setup Script
# Run once to install all dependencies and verify your environment.
# =============================================================================
set -e
 
echo "======================================================"
echo " ECO-REWIND: Environment Setup"
echo "======================================================"
 
# --- Detect environment ---
PYTHON=${PYTHON:-python3}
ENV_NAME="eco-rewind"
 
if [ "$1" = "--colab" ]; then
    echo "[+] Colab mode: installing packages directly"
    pip install -q \
        torch torchvision \
        rasterio \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        scikit-learn \
        tqdm \
        wandb \
        pyyaml \
        einops \
        scipy \
        Pillow \
        geopandas \
        shapely
    echo "[+] Colab install complete."
    exit 0
fi
 
if [ "$1" = "--runpod" ]; then
    echo "[+] RunPod mode: installing packages + rclone for Google Drive"
    pip install -q \
        torch torchvision \
        rasterio \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        scikit-learn \
        tqdm \
        wandb \
        pyyaml \
        einops \
        scipy \
        Pillow \
        geopandas \
        shapely
 
    # Install rclone (for Google Drive mounting on RunPod)
    if ! command -v rclone &> /dev/null; then
        echo "[+] Installing rclone..."
        curl https://rclone.org/install.sh | bash
        echo ""
        echo "======================================================="
        echo " IMPORTANT: Configure rclone for Google Drive access:"
        echo "   rclone config"
        echo "   → New remote → name: gdrive → type: drive"
        echo "   → Follow OAuth flow"
        echo ""
        echo " Then mount with:"
        echo "   mkdir -p /content/drive/MyDrive"
        echo "   rclone mount gdrive: /content/drive/MyDrive --daemon"
        echo "======================================================="
    fi
    echo "[+] RunPod install complete."
    exit 0
fi
 
# --- Local setup with venv ---
echo "[+] Local mode: creating virtual environment '$ENV_NAME'"
$PYTHON -m venv $ENV_NAME
source $ENV_NAME/bin/activate
 
echo "[+] Upgrading pip..."
pip install --upgrade pip -q
 
echo "[+] Installing dependencies..."
pip install -r requirements.txt -q
 
echo ""
echo "======================================================"
echo " Setup complete!"
echo ""
echo " Activate your environment:"
echo "   source $ENV_NAME/bin/activate"
echo ""
echo " Then run the pipeline:"
echo "   python scripts/01_build_tensors.py"
echo "   python scripts/02_train_convlstm.py"
echo "   python scripts/03_train_unet.py"
echo "   python scripts/04_generate_counterfactual.py"
echo "   python scripts/05_compute_metrics.py"
echo "   python scripts/06_ablation.py"
echo "======================================================"
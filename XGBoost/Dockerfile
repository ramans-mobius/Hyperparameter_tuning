FROM gurpreetgandhi/nesy-factory:v25

# Check PyTorch and CUDA versions
# RUN python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Version: {torch.version.cuda}')"

WORKDIR /app
COPY train_boost.py .

# STEP 1: Clean up existing problematic PyG installations
RUN pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric torch-geometric-temporal

#STEP 2: Install PyTorch and main dependencies first
RUN pip install --no-cache-dir \
    torchvision \
    tqdm \
    kubeflow \
    kubeflow-katib \
    pandas \
    numpy 

RUN pip install --no-cache-dir \
    scikit-learn \
    xgboost \
    lightgbm \
    catboost \
    requests \
    scipy

# Optional: Install additional utilities
RUN pip install --no-cache-dir \
    matplotlib \
    seaborn \
    plotly

# Install PyG the simple way
RUN pip install torch-geometric

ENTRYPOINT ["python", "train_boost.py"]

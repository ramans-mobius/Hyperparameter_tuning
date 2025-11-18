FROM nikhilv215/nesy-factory:v14

# Add this command to see the exact torch version and its CUDA support
RUN python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Version: {torch.version.cuda}')"

WORKDIR /app
COPY train_usecase1.py . #change to usecase2, usecase3 according to need
RUN pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric pandas pykeen
# STEP 2: Reinstall the PyG helper libraries from the official repository,
# specifically targeting the versions compiled for torch 2.4.1 and CPU.
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.1+cpu.html

# STEP 3: Now install the main torch-geometric library and your other dependencies.
# Pip will now use the correct, compatible helper libraries we just installed.
RUN pip install --no-cache-dir \
    torch-geometric \
    tqdm \
    kubernetes \
    kubeflow-katib \
    pandas

RUN mkdir tmp

ENTRYPOINT ["python", "train_usecase1.py"] #change to usecase2, usecase3 according to need

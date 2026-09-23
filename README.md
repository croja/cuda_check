# ONNX Runtime / CUDA Diagnostic Script

`check_cuda.py` is a self-contained diagnostic script that checks whether
ONNX Runtime can use your NVIDIA GPU (`CUDAExecutionProvider`) and, if so,
loads `test.onnx` and runs a single test inference on it.

It checks, in order: the Python environment, whether `nvidia-smi` is
reachable, whether `onnxruntime` is importable and reports CUDA as an
available provider, whether `test.onnx` is present next to the script, and
finally whether a real `InferenceSession` can be created and run on the GPU.

Files expected in this folder:

- `check_cuda.py` — the script
- `test.onnx` — the model it loads (must stay next to the script)

## 1. Prerequisites

- An NVIDIA GPU with a recent driver installed.
- **CUDA Toolkit 12.x** and **cuDNN 9.x** installed on the machine. This is
  the combination required by `onnxruntime-gpu==1.18.1`, the version this
  script is built against (see [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
  for the full compatibility matrix). The pip package does **not** bundle
  CUDA/cuDNN — they must be installed separately and visible on `PATH`
  (Windows) / `LD_LIBRARY_PATH` (Linux).
- Python 3.12 (the script and its dependencies were tested against 3.12;
  any recent 3.9–3.12 build should work, but 3.12 is recommended to match
  the existing environment).

Check your GPU/driver before doing anything else:

```
nvidia-smi
```

If this fails, install/update the NVIDIA driver first — nothing else in
this guide will work without it.

## 2. Install Python

### Windows

1. Download the Python 3.12 installer from
   [python.org/downloads](https://www.python.org/downloads/) (the
   "Windows installer (64-bit)").
2. Run it and check **"Add python.exe to PATH"** on the first screen,
   then choose **Install Now**.
3. Verify in a new terminal (PowerShell or cmd):

   ```
   python --version
   ```

### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip
python3.12 --version
```

If `python3.12` isn't available in your distro's repositories, use the
[deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) on
Ubuntu, or install via [pyenv](https://github.com/pyenv/pyenv).

## 3. Create and activate a virtual environment

Run these commands from inside this folder (the one containing
`check_cuda.py` and `test.onnx`).

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If activation is blocked by the execution policy, run PowerShell as
> the current user and allow local scripts once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Windows (cmd.exe)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### Linux (bash)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Your shell prompt should now show `(.venv)` at the start of the line.
Everything installed from here on stays local to this folder and won't
affect the system Python.

## 4. Install dependencies

With the virtual environment active:

```bash
python -m pip install --upgrade pip
pip install onnxruntime-gpu==1.18.1
```

`onnxruntime-gpu==1.18.1` is the version required by this script (it's
also the version already used in the project's `.venv`). Installing this
exact version matters because ONNX Runtime's GPU wheels are tied to a
specific CUDA/cuDNN ABI — a newer or older `onnxruntime-gpu` release may
need a different CUDA/cuDNN version than the one you installed in step 1.

This installs onto CUDA 12.x by default. If your machine only has CUDA
11.x installed instead, install from the CUDA-11 package feed instead of
the line above:

```bash
pip install onnxruntime-gpu==1.18.1 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```

## 5. Run the script

With the virtual environment still active, from this folder:

```bash
python check_cuda.py
```

Expected output ends with a `10. SUMMARY` section. Look for:

```
NVIDIA / nvidia-smi     : OK
CUDA Execution Provider : AVAILABLE
CUDA session            : CREATED
Session first provider  : CUDAExecutionProvider
Inference               : OK
```

## 6. Troubleshooting

- **`nvidia-smi` fails / not found** — the NVIDIA driver isn't installed
  or isn't on `PATH`. Install/update it before continuing.
- **`Cannot import onnxruntime`** — the virtual environment isn't active,
  or the `pip install` step above failed. Re-run step 3 then step 4.
- **`CUDAExecutionProvider` is NOT available** (but `onnxruntime` imports
  fine) — this is almost always a CUDA/cuDNN version mismatch:
  - Confirm CUDA Toolkit 12.x is installed: `nvcc --version` (Linux) or
    check `Program Files\NVIDIA GPU Computing Toolkit` (Windows).
  - Confirm cuDNN 9.x is installed and its DLLs/`.so` files are on
    `PATH` / `LD_LIBRARY_PATH`.
  - On Windows, make sure the CUDA `bin` folder (e.g.
    `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`) and
    the cuDNN `bin` folder are both in your `PATH` environment variable.
- **`[FAIL] Model file does not exist.`** — `test.onnx` isn't in the same
  folder as `check_cuda.py`. Move it there and re-run.

## 7. Deactivating

When you're done:

```bash
deactivate
```

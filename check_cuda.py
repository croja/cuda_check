import os
import sys
import platform
import subprocess
import traceback

MODEL_NAME = "test.onnx"


def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def run_command(command):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )

        print(result.stdout.strip())

        return result.returncode == 0

    except FileNotFoundError:
        print(f"Command not found: {' '.join(command)}")
        return False

    except Exception as e:
        print(f"Command failed: {e}")
        return False


def main():
    print("=" * 70)
    print("ONNX Runtime / CUDA diagnostic")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Python
    # ------------------------------------------------------------

    header("1. Python")

    print(f"Python version : {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"OS             : {platform.platform()}")
    print(f"Architecture   : {platform.machine()}")

    # ------------------------------------------------------------
    # 2. NVIDIA GPU
    # ------------------------------------------------------------

    header("2. NVIDIA GPU / driver")

    print("Running nvidia-smi...")

    nvidia_ok = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv",
        ]
    )

    if nvidia_ok:
        print("[OK] NVIDIA driver is accessible.")
    else:
        print("[FAIL] nvidia-smi failed or is not available.")

    # ------------------------------------------------------------
    # 3. ONNX Runtime
    # ------------------------------------------------------------

    header("3. ONNX Runtime")

    try:
        import onnxruntime as ort

    except Exception as e:
        print("[FAIL] Cannot import onnxruntime.")
        print()
        traceback.print_exc()
        return 1

    print(f"ONNX Runtime version : {ort.__version__}")
    print(f"ONNX Runtime module  : {ort.__file__}")

    providers = ort.get_available_providers()

    print()
    print("Available Execution Providers:")

    for provider in providers:
        print(f"  {provider}")

    cuda_available = "CUDAExecutionProvider" in providers

    print()

    if cuda_available:
        print("[OK] CUDAExecutionProvider is available.")
    else:
        print("[FAIL] CUDAExecutionProvider is NOT available.")

    # ------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------

    header("4. ONNX model")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_NAME)

    print(f"Model path: {model_path}")

    if not os.path.isfile(model_path):
        print("[FAIL] Model file does not exist.")
        print()
        print("Put test.onnx next to this script.")
        return 1

    model_size = os.path.getsize(model_path)

    print(f"Model size: {model_size / 1024 / 1024:.2f} MB")
    print("[OK] Model file exists.")

    # ------------------------------------------------------------
    # 5. Create CUDA session
    # ------------------------------------------------------------

    header("5. Create ONNX Runtime session")

    if not cuda_available:
        print("[SKIP] CUDAExecutionProvider is not available.")
        return 1

    print("Trying to create session with:")

    print("  CUDAExecutionProvider")
    print("  CPUExecutionProvider")

    try:
        session = ort.InferenceSession(
            model_path,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        print()
        print("[OK] InferenceSession created.")

    except Exception:
        print()
        print("[FAIL] Failed to create CUDA session.")
        print()
        traceback.print_exc()

        return 1

    # ------------------------------------------------------------
    # 6. Actual providers of the session
    # ------------------------------------------------------------

    header("6. Session providers")

    session_providers = session.get_providers()

    print("Providers used by the session:")

    for provider in session_providers:
        print(f"  {provider}")

    if session_providers:
        first_provider = session_providers[0]

        print()

        if first_provider == "CUDAExecutionProvider":
            print("[OK] CUDAExecutionProvider is the first provider.")
        elif first_provider == "CPUExecutionProvider":
            print("[WARNING] CPUExecutionProvider is the first provider.")
        else:
            print(f"[INFO] First provider: {first_provider}")

    # ------------------------------------------------------------
    # 7. Model information
    # ------------------------------------------------------------

    header("7. Model information")

    print("Inputs:")

    for inp in session.get_inputs():
        print(f"  name={inp.name}, " f"type={inp.type}, " f"shape={inp.shape}")

    print()
    print("Outputs:")

    for out in session.get_outputs():
        print(f"  name={out.name}, " f"type={out.type}, " f"shape={out.shape}")

    # ------------------------------------------------------------
    # 8. Provider options
    # ------------------------------------------------------------

    header("8. Provider options")

    try:
        provider_options = session.get_provider_options()

        for provider, options in provider_options.items():
            print(f"{provider}:")
            print(f"  {options}")

    except Exception as e:
        print(f"Could not get provider options: {e}")

    # ------------------------------------------------------------
    # 9. Run inference
    # ------------------------------------------------------------

    header("9. Run inference")

    print(
        "The script will try to create input data automatically "
        "from the first model input."
    )

    try:
        import numpy as np

        input_info = session.get_inputs()[0]

        input_name = input_info.name
        input_shape = input_info.shape

        print(f"Input name : {input_name}")
        print(f"Input shape: {input_shape}")

        # Resolve dynamic dimensions.
        #
        # For example:
        # [1, 3, 640, 640]
        #
        # is already concrete.
        #
        # If the model contains dimensions such as
        # "height" / "width" / None, use 640.

        concrete_shape = []

        for dimension in input_shape:
            if isinstance(dimension, int) and dimension > 0:
                concrete_shape.append(dimension)
            else:
                concrete_shape.append(640)

        print(f"Test input shape: {concrete_shape}")

        test_input = np.zeros(
            concrete_shape,
            dtype=np.float32,
        )

        print("Running inference...")

        outputs = session.run(
            None,
            {
                input_name: test_input,
            },
        )

        print()
        print("[OK] Inference completed.")

        print()
        print(f"Number of outputs: {len(outputs)}")

        for index, output in enumerate(outputs):
            print(
                f"Output {index}: "
                f"shape={getattr(output, 'shape', None)}, "
                f"dtype={getattr(output, 'dtype', None)}"
            )

    except Exception:
        print()
        print("[FAIL] Inference failed.")
        print()
        traceback.print_exc()

        return 1

    # ------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------

    header("10. SUMMARY")

    print(f"NVIDIA / nvidia-smi     : {'OK' if nvidia_ok else 'FAIL'}")
    print(
        f"CUDA Execution Provider : "
        f"{'AVAILABLE' if cuda_available else 'NOT AVAILABLE'}"
    )
    print("CUDA session            : CREATED")
    print(
        f"Session first provider  : "
        f"{session_providers[0] if session_providers else 'UNKNOWN'}"
    )
    print("Inference               : OK")

    print()
    print("Diagnostic finished.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

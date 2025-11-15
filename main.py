# main.py
import os
import io
import subprocess
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
import numpy as np
from gpu_kernel import gpu_matrix_add
from prometheus_client import CollectorRegistry, Gauge, generate_latest

app = FastAPI()

# Port config: Set via env STUDENT_PORT, default 8001
STUDENT_PORT = int(os.environ.get("STUDENT_PORT", "8009"))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/add")
async def add_matrices(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    # Validate extension
    if not file_a.filename.endswith(".npz") or not file_b.filename.endswith(".npz"):
        raise HTTPException(status_code=400, detail="Both files must be .npz")

    try:
        data_a = await file_a.read()
        data_b = await file_b.read()
        # load from bytes
        npz_a = np.load(io.BytesIO(data_a))
        npz_b = np.load(io.BytesIO(data_b))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded npz files: {e}")

    # Expect single array, commonly 'arr_0'
    arrs_a = [npz_a[k] for k in npz_a.files]
    arrs_b = [npz_b[k] for k in npz_b.files]

    if len(arrs_a) == 0 or len(arrs_b) == 0:
        raise HTTPException(status_code=400, detail="Each .npz must contain one array")
    A = arrs_a[0]
    B = arrs_b[0]

    # Validate 2D and shape
    if A.shape != B.shape:
        raise HTTPException(status_code=400, detail="Matrices have different shapes")
    if A.ndim != 2 or B.ndim != 2:
        raise HTTPException(status_code=400, detail="Only 2D matrices are supported")

    # Convert to float32 if needed (kernel expects float32)
    # call GPU add
    try:
        C, elapsed = gpu_matrix_add(A, B)
    except Exception as e:
        # If CUDA not available or kernel error
        raise HTTPException(status_code=500, detail=f"GPU computation failed: {e}")

    # (We do not return the matrix per spec)
    rows, cols = A.shape
    return JSONResponse({"matrix_shape": [int(rows), int(cols)], "elapsed_time": float(elapsed), "device": "GPU"})

@app.get("/gpu-info")
async def gpu_info():
    """
    Run nvidia-smi and parse memory used/total per GPU.
    Output:
    {
      "gpus": [
        {"gpu": "0", "memory_used_MB": 312, "memory_total_MB": 4096}
      ]
    }
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"nvidia-smi failed: {e.output}")
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            idx, used, total = parts[0], parts[1], parts[2]
            try:
                gpus.append({"gpu": idx, "memory_used_MB": int(used), "memory_total_MB": int(total)})
            except ValueError:
                # fallback: return raw strings
                gpus.append({"gpu": idx, "memory_used_MB": used, "memory_total_MB": total})
    return {"gpus": gpus}

@app.get("/metrics")
async def metrics():
    """
    Simple Prometheus metrics that run nvidia-smi and export memory used / total.
    Exposes metrics with names:
      gpu_memory_used_mb{gpu="0"} and gpu_memory_total_mb{gpu="0"}
    """
    reg = CollectorRegistry()
    g_used = Gauge("gpu_memory_used_mb", "GPU memory used (MB)", ["gpu"], registry=reg)
    g_total = Gauge("gpu_memory_total_mb", "GPU memory total (MB)", ["gpu"], registry=reg)

    # Query nvidia-smi
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        # return empty metrics + error code 500
        raise HTTPException(status_code=500, detail=f"nvidia-smi failed: {e.output}")

    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            idx, used, total = parts[0], parts[1], parts[2]
            try:
                g_used.labels(gpu=idx).set(float(used))
                g_total.labels(gpu=idx).set(float(total))
            except Exception:
                continue

    data = generate_latest(reg)
    return Response(content=data, media_type="text/plain; version=0.0.4; charset=utf-8")

# Run with: python3 main.py (or via uvicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=STUDENT_PORT, reload=False)

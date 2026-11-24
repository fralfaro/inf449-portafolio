"""Reproducible PCA experiments for Homework 5 (GPU linear algebra when available)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "tokio-dataset"
OUT = ROOT / "resultados"
FIG = ROOT / "Fig"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKIO_FILES = [
    "Butterfly_sRGB.jpg", "CD_sRGB.jpg", "Character_sRGB.jpg", "Cloth_sRGB.jpg",
    "Cloth2_sRGB.jpg", "Color_sRGB.jpg", "Colorchart_sRGB.jpg", "Fan_sRGB.jpg",
    "Party_sRGB.jpg", "Tshirts_sRGB.jpg",
]
BAND_FILES = [
    ROOT.parent.parent / "extra/material_curso/homework4/DIP3E_CH11_Original_Images"
    / f"Fig1138({chr(97+i)})(WashingtonDC_Band{i+1}_564).tif"
    for i in range(6)
]


def pca_fit(cube: np.ndarray):
    """Population PCA along pixels; cube shape is (rows, cols, channels)."""
    h, w, c = cube.shape
    x = torch.as_tensor(cube, dtype=torch.float32, device=DEVICE).reshape(-1, c)
    mu = x.mean(dim=0)
    xc = x - mu
    cov = xc.T @ xc / x.shape[0]
    # eigh returns ascending eigenvalues with eigenvectors in columns.
    vals, vecs = torch.linalg.eigh(cov)
    order = torch.argsort(vals, descending=True)
    vals, vecs = vals[order], vecs[:, order]
    scores = xc @ vecs
    return mu, cov, vals, vecs, scores


def reconstruct(mu, vecs, scores, k: int, shape):
    c = shape[-1]
    rec = scores[:, :k] @ vecs[:, :k].T + mu
    return rec.reshape(shape).clamp(0.0, 1.0).detach().cpu().numpy()


def metrics(reference: np.ndarray, estimate: np.ndarray):
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    mse = float(np.mean((ref - est) ** 2))
    psnr = float(peak_signal_noise_ratio(ref, est, data_range=1.0))
    ssim = float(structural_similarity(ref, est, data_range=1.0, channel_axis=-1))
    a = ref.reshape(-1, ref.shape[-1])
    b = est.reshape(-1, est.shape[-1])
    dot = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = den > 1e-12
    cos = dot[valid] / den[valid]
    sam = float(np.mean(np.arccos(np.clip(cos, -1.0, 1.0)))) if np.any(valid) else 0.0
    return {"mse": mse, "psnr_db": psnr, "ssim": ssim, "sam_rad": sam, "sam_deg": float(np.degrees(sam))}


def run_six_band():
    cube = np.stack([np.asarray(Image.open(p), dtype=np.float32) / 255.0 for p in BAND_FILES], axis=-1)
    mu, cov, vals, vecs, scores = pca_fit(cube)
    rec2 = reconstruct(mu, vecs, scores, 2, cube.shape)
    # Compare the RGB guide image visually; it is a rendered reference, not raw band data.
    reference = np.asarray(Image.open(ROOT / "referencia.png").convert("RGB").resize((564, 564)), dtype=np.float32) / 255.0
    rgb321 = cube[..., [2, 1, 0]]
    fig, ax = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for j in range(3):
        ax[0, j].imshow(cube[..., j], cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(f"Banda {j+1}")
        ax[1, j].imshow(rec2[..., j], cmap="gray", vmin=0, vmax=1)
        ax[1, j].set_title(f"Reconstrucción 2 PC · banda {j+1}")
        for a in (ax[0, j], ax[1, j]): a.axis("off")
    fig.savefig(FIG / "washington_bandas_pca2.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    ax[0].imshow(reference); ax[0].set_title("referencia.png (RGB renderizada)")
    ax[1].imshow(rgb321); ax[1].set_title("Bandas 3–2–1 (escala linealizada)")
    ax[2].imshow(np.abs(rgb321 - reference).mean(axis=2), cmap="magma"); ax[2].set_title("Diferencia absoluta media")
    for a in ax: a.axis("off")
    fig.savefig(FIG / "referencia_vs_bandas.png", dpi=170)
    plt.close(fig)
    return {
        "shape": list(cube.shape), "bands": [p.name for p in BAND_FILES],
        "mean": mu.detach().cpu().numpy().tolist(),
        "covariance": cov.detach().cpu().numpy().tolist(),
        "eigenvalues": vals.detach().cpu().numpy().tolist(),
        "explained_variance_ratio": (vals / vals.sum()).detach().cpu().numpy().tolist(),
        "eigenvectors_columns": vecs.detach().cpu().numpy().tolist(),
        "reconstruction_first_two": metrics(cube, rec2),
        "device": str(DEVICE),
    }


def run_tokio():
    start = time.perf_counter()
    originals = []
    for name in TOKIO_FILES:
        im = np.asarray(Image.open(DATA / name).convert("RGB"), dtype=np.float32) / 255.0
        originals.append(im)
    # Batch all ten PCA problems on the selected device.
    batch = torch.as_tensor(np.stack(originals), dtype=torch.float32, device=DEVICE)
    n, h, w, c = batch.shape
    x = batch.reshape(n, h * w, c)
    mu = x.mean(dim=1, keepdim=True)
    xc = x - mu
    cov = xc.transpose(1, 2) @ xc / (h * w)
    vals, vecs = torch.linalg.eigh(cov)
    vals, vecs = vals.flip(dims=[-1]), vecs.flip(dims=[-1])
    scores = xc @ vecs
    per_image = []
    for i, name in enumerate(TOKIO_FILES):
        row = {"image": name, "explained_variance_ratio": (vals[i] / vals[i].sum()).cpu().numpy().tolist(), "metrics": {}}
        for k in (1, 2, 3):
            rec = scores[i, :, :k] @ vecs[i, :, :k].T + mu[i]
            rec = rec.reshape(h, w, c).clamp(0, 1).cpu().numpy()
            row["metrics"][str(k)] = metrics(originals[i], rec)
            if name == "Colorchart_sRGB.jpg":
                Image.fromarray(np.uint8(np.round(rec * 255))).save(FIG / f"colorchart_pca{k}.png")
        per_image.append(row)
    mean_metrics = {}
    for k in (1, 2, 3):
        mean_metrics[str(k)] = {m: float(np.mean([r["metrics"][str(k)][m] for r in per_image])) for m in ("mse", "psnr_db", "ssim", "sam_rad", "sam_deg")}
    # Define 90% recovery as at least 90% of the improvement from one PC to the full RGB rank,
    # evaluated in distortion space for MSE (PSNR is logarithmic), and directly for SSIM/SAM.
    progress = {}
    for k in (1, 2, 3):
        vals_k = []
        for row in per_image:
            m1, mk, m3 = (row["metrics"][str(j)] for j in (1, k, 3))
            vals_k.append({
                "mse_reduction_fraction": float((m1["mse"] - mk["mse"]) / max(m1["mse"] - m3["mse"], 1e-15)),
                "ssim_gain_fraction": float((mk["ssim"] - m1["ssim"]) / max(m3["ssim"] - m1["ssim"], 1e-15)),
                "sam_reduction_fraction": float((m1["sam_rad"] - mk["sam_rad"]) / max(m1["sam_rad"] - m3["sam_rad"], 1e-15)),
            })
        progress[str(k)] = {key: float(np.mean([v[key] for v in vals_k])) for key in vals_k[0]}
    acceptable = [int(k) for k, vals_ in progress.items() if min(vals_.values()) >= .90]
    needed = min(acceptable) if acceptable else None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    keys = [("psnr_db", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("sam_deg", "SAM (grados) ↓")]
    for ax, (key, title) in zip(axes, keys):
        ax.plot([1, 2, 3], [mean_metrics[str(k)][key] for k in (1, 2, 3)], marker="o", linewidth=2)
        ax.set_xticks([1, 2, 3]); ax.set_xlabel("Componentes retenidas"); ax.set_title(title); ax.grid(True, alpha=.3)
    fig.savefig(FIG / "metricas_tokio_pca.png", dpi=170)
    plt.close(fig)
    return {
        "selected_files": TOKIO_FILES, "number_of_images": n, "image_shape": [h, w, c],
        "selection_rule": "Una imagen por tipo/escena: se incluye Butterfly_sRGB y se dejan fuera Butterfly2–6 para evitar sobrerrepresentar esa serie.",
        "per_image": per_image, "mean_metrics": mean_metrics, "recovery_progress": progress,
        "components_for_90_percent": needed, "criterion": "Promedio por imagen de la fracción de mejora respecto de 1 PC y rango completo; MSE usa reducción de error, SSIM usa ganancia, SAM usa reducción angular. Se exige >=0.90 en las tres.",
        "device": str(DEVICE), "elapsed_seconds": time.perf_counter() - start,
    }


def main():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    six = run_six_band()
    tokio = run_tokio()
    result = {"device": str(DEVICE), "gpu_name": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
              "six_band": six, "tokio": tokio, "total_seconds": time.perf_counter() - t0}
    if DEVICE.type == "cuda": result["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    (OUT / "resultados_pca.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"device": result["device"], "gpu": result["gpu_name"], "six_band_mean": six["mean"],
                      "six_band_eigenvalues": six["eigenvalues"], "six_band_metrics": six["reconstruction_first_two"],
                      "tokio_mean_metrics": tokio["mean_metrics"], "tokio_recovery_progress": tokio["recovery_progress"],
                      "tokio_90pct_components": tokio["components_for_90_percent"], "seconds": result["total_seconds"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__": main()

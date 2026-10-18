"""Tarea 4: sistema difractivo RGB y deconvolución.

El script implementa el flujo completo solicitado en la guía:

    RGB -> DOE/PSF por longitud de onda -> filtro Bayer -> demosaicking
        -> Wiener o Richardson-Lucy -> PSNR, SSIM y SAM.

Incluye implementaciones propias de Wiener y Richardson-Lucy y las compara con
skimage.restoration usando la misma imagen, PSF, regularización e iteraciones.
El dataset McMaster se descarga solo cuando no se encuentra en ``data/McMaster``.

Uso rápido desde la raíz del repositorio::

    python3 tareas/tarea-04/homework4.py

Ejecución de prueba::

    python3 tareas/tarea-04/homework4.py --limit 2 --iterations 10 --sin-barrido

Regenerar tablas y gráficos a partir de los CSV, sin repetir la simulación::

    python3 tareas/tarea-04/homework4.py --solo-tablas

Dependencias: numpy, scipy, scikit-image, matplotlib, Pillow y certifi.
"""

from __future__ import annotations

import argparse
import csv
import re
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from skimage import restoration
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


TASK_DIR = Path(__file__).resolve().parent
DATA_DIR = TASK_DIR / "data"
DATASET_DIR = DATA_DIR / "McMaster"
FIG_DIR = TASK_DIR / "Fig"
RESULTS_DIR = TASK_DIR / "resultados"
METRICS_CSV = RESULTS_DIR / "metricas_mcmaster.csv"
DIAGNOSTICS_CSV = RESULTS_DIR / "diagnosticos_por_imagen.csv"
SWEEP_CSV = RESULTS_DIR / "barrido_parametros.csv"
# API pública de Kaggle; corresponde al enlace de McMaster de la guía.
# La página original del dataset también se cita en el informe.
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/liikke/mcmaster"
DATASET_PASSWORD = None
WAVELENGTHS_NM = np.array([450.0, 550.0, 650.0])
REFERENCE_WAVELENGTH_NM = 550.0
BAYER_PATTERN = "RGGB"
CUBIC_STRENGTH = 18.0  # alfa [rad] de la fase cúbica a 550 nm
# Coeficientes [rad] de los modos de Zernike (indexación de Noll).
ZERNIKE_COEFFS = {"Z4": 0.30, "Z5": 0.18, "Z6": -0.12, "Z7": 0.08}
RL_EPSILON = 1e-12  # mismo valor que skimage.restoration.richardson_lucy

METHODS = ["demosaic", "wiener_scratch", "wiener_skimage", "wiener_skimage_lap",
           "rl_scratch", "rl_skimage"]
METHOD_LABELS = {
    "demosaic": "Demosaicking (sin deconvolución)",
    "wiener_scratch": "Wiener propio",
    "wiener_skimage": "Wiener skimage ($\\tau$ constante)",
    "wiener_skimage_lap": "Wiener skimage (Laplaciano, por defecto)",
    "rl_scratch": "Richardson--Lucy propio",
    "rl_skimage": "Richardson--Lucy skimage",
}
PLOT_LABELS = {
    "demosaic": "Demosaicking",
    "wiener_scratch": "Wiener propio",
    "wiener_skimage": "Wiener skimage (τ)",
    "wiener_skimage_lap": "Wiener skimage (Laplaciano)",
    "rl_scratch": "RL propio",
    "rl_skimage": "RL skimage",
}
PLOT_COLORS = {
    "demosaic": "#8c8b86",
    "wiener_scratch": "#2a78d6",
    "wiener_skimage": "#2a78d6",
    "wiener_skimage_lap": "#1baf7a",
    "rl_scratch": "#eb6834",
    "rl_skimage": "#eb6834",
}
METRIC_NAMES = ["PSNR [dB]", "SSIM", "SAM [deg]"]
TAU_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
ITERATION_GRID = [5, 10, 25, 50, 100]


def natural_key(path: Path):
    """Orden natural para que 2.png aparezca antes de 10.png."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def find_images(directory: Path) -> list[Path]:
    """Lista las imágenes de un directorio en orden natural."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in extensions],
        key=natural_key,
    )


def ensure_mcmaster(directory: Path = DATASET_DIR, download: bool = True) -> list[Path]:
    """Obtiene las 18 imágenes McMaster o informa cómo instalarlas."""
    paths = find_images(directory) if directory.exists() else []
    if len(paths) >= 18:
        return paths[:18]
    if not download:
        raise FileNotFoundError(
            f"No se encontraron 18 imágenes en {directory}. "
            "Ejecute sin --no-download para descargarlas."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / "McM.zip"
    if not archive.exists():
        print(f"Descargando McMaster desde {DATASET_URL}")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(DATASET_URL, context=ssl_context) as response:
            archive.write_bytes(response.read())
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Extrayendo {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(directory, pwd=DATASET_PASSWORD)
    paths = find_images(directory)
    if len(paths) < 18:
        raise RuntimeError(f"La descarga no contiene 18 imágenes RGB: {directory}")
    return paths[:18]


def load_rgb(path: Path, size: int = 256) -> np.ndarray:
    """Carga una imagen RGB, la normaliza a [0,1] y la redimensiona."""
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float64) / 255.0


def zernike_modes(rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Combinación de modos de Zernike Z4--Z7 (Noll) sin normalización RMS.

    Cada polinomio se usa con amplitud de pico unitaria en el borde de la
    pupila (se omiten los factores sqrt(3), sqrt(6) y sqrt(8) de Noll).
    """
    z4 = 2.0 * rho**2 - 1.0                          # desenfoque
    z5 = rho**2 * np.sin(2.0 * theta)                # astigmatismo oblicuo
    z6 = rho**2 * np.cos(2.0 * theta)                # astigmatismo vertical
    z7 = (3.0 * rho**3 - 2.0 * rho) * np.sin(theta)  # coma vertical
    c = ZERNIKE_COEFFS
    return c["Z4"] * z4 + c["Z5"] * z5 + c["Z6"] * z6 + c["Z7"] * z7


def create_doe_phase(size: int, cubic_strength: float = CUBIC_STRENGTH
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Crea la fase del DOE: fase cúbica más cuatro modos de Zernike."""
    coords = np.linspace(-1.0, 1.0, size, endpoint=False) + 1.0 / size
    x, y = np.meshgrid(coords, coords)
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    aperture = (rho <= 1.0).astype(np.float64)
    cubic = cubic_strength * (x**3 + y**3)
    phase = aperture * (cubic + zernike_modes(rho, theta))
    return phase, aperture


def psf_from_phase(phase: np.ndarray, aperture: np.ndarray, wavelength_nm: float) -> np.ndarray:
    """Calcula una PSF incoherente normalizada a partir de la pupila compleja."""
    scale = REFERENCE_WAVELENGTH_NM / wavelength_nm
    pupil = aperture * np.exp(1j * scale * phase)
    amplitude = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    psf = np.abs(amplitude) ** 2
    psf /= max(float(psf.sum()), np.finfo(float).eps)
    return psf


def build_psfs(size: int) -> np.ndarray:
    """Devuelve las PSF de los canales R, G y B apiladas en el último eje."""
    phase, aperture = create_doe_phase(size)
    return np.stack(
        [psf_from_phase(phase, aperture, wavelength) for wavelength in WAVELENGTHS_NM[::-1]],
        axis=-1,
    )


def bayer_masks(shape: tuple[int, int], pattern: str = BAYER_PATTERN) -> np.ndarray:
    """Devuelve tres máscaras binarias en orden RGB para un patrón Bayer."""
    if pattern != "RGGB":
        raise ValueError("Esta implementación usa el patrón Bayer RGGB.")
    h, w = shape
    masks = np.zeros((h, w, 3), dtype=np.float64)
    tile = np.array([[0, 1], [1, 2]])  # R G / G B
    for row in range(2):
        for col in range(2):
            masks[row::2, col::2, tile[row, col]] = 1.0
    return masks


def rgb_to_bayer(rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Conserva una muestra de color por píxel según las máscaras Bayer."""
    return np.sum(rgb * masks, axis=2)


def bilinear_demosaic(mosaic: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Demosaicking bilineal con preservación de las muestras observadas."""
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
    result = np.empty((*mosaic.shape, 3), dtype=np.float64)
    for channel in range(3):
        observed = mosaic * masks[..., channel]
        numerator = fftconvolve(observed, kernel, mode="same")
        denominator = fftconvolve(masks[..., channel], kernel, mode="same")
        result[..., channel] = numerator / np.maximum(denominator, 1e-12)
        result[..., channel][masks[..., channel] == 1] = mosaic[masks[..., channel] == 1]
    return np.clip(result, 0.0, 1.0)


def forward_model(rgb: np.ndarray, psfs: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Propaga cada canal por su PSF y aplica el filtro Bayer (sin ruido)."""
    blurred = np.empty_like(rgb)
    for channel in range(3):
        blurred[..., channel] = fftconvolve(rgb[..., channel], psfs[..., channel], mode="same")
    measurement = rgb_to_bayer(blurred, masks)
    return blurred, measurement


def psf_to_otf(psf: np.ndarray) -> np.ndarray:
    """Convierte una PSF centrada en una OTF compatible con fft2."""
    return np.fft.fft2(np.fft.ifftshift(psf))


def wiener_from_scratch(blurred: np.ndarray, psf: np.ndarray, balance: float = 0.01) -> np.ndarray:
    """Filtro de Wiener de la ecuación (4) con tau constante, en Fourier."""
    h = psf_to_otf(psf)
    estimate = np.fft.ifft2(np.conj(h) * np.fft.fft2(blurred) /
                            (np.abs(h) ** 2 + balance))
    return np.real(estimate)


def richardson_lucy_from_scratch(
    blurred: np.ndarray, psf: np.ndarray, iterations: int = 25, initial_value: float = 0.5
) -> np.ndarray:
    """Richardson--Lucy de la ecuación (6), con número fijo de iteraciones.

    La estimación inicial es la constante ``initial_value``; como la
    actualización es multiplicativa y la PSF suma 1, el resultado no depende
    de esa constante (siempre que sea positiva).
    """
    estimate = np.full_like(blurred, initial_value)
    psf_mirror = psf[::-1, ::-1]
    for _ in range(iterations):
        reblurred = fftconvolve(estimate, psf, mode="same") + RL_EPSILON
        estimate = estimate * fftconvolve(blurred / reblurred, psf_mirror, mode="same")
    return estimate


def deconvolve_channel(image: np.ndarray, psf: np.ndarray, method: str,
                       balance: float, iterations: int) -> np.ndarray:
    """Deconvoluciona un canal con el método indicado (sin recorte final)."""
    if method == "wiener_scratch":
        return wiener_from_scratch(image, psf, balance)
    if method == "wiener_skimage":
        # reg = impulso unitario => |reg|^2 = 1, es decir, el mismo tau constante
        # de la ecuación (4). Sin este argumento skimage usa un Laplaciano.
        return restoration.wiener(image, psf, balance=balance,
                                  reg=np.ones((1, 1)), clip=False)
    if method == "wiener_skimage_lap":
        return restoration.wiener(image, psf, balance=balance, clip=False)
    if method == "rl_scratch":
        return richardson_lucy_from_scratch(image, psf, iterations)
    if method == "rl_skimage":
        return restoration.richardson_lucy(image, psf, num_iter=iterations, clip=False)
    raise ValueError(f"Método desconocido: {method}")


def deconvolve_rgb(blurred_rgb: np.ndarray, psfs: np.ndarray, method: str,
                   balance: float, iterations: int) -> np.ndarray:
    """Deconvoluciona cada canal por separado y recorta el resultado a [0,1]."""
    result = np.empty_like(blurred_rgb)
    for channel in range(3):
        result[..., channel] = deconvolve_channel(
            blurred_rgb[..., channel], psfs[..., channel], method, balance, iterations)
    return np.clip(np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def spectral_angle_mapper(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SAM medio por píxel, en grados, para imágenes RGB."""
    ref = reference.reshape(-1, 3)
    est = estimate.reshape(-1, 3)
    numerator = np.sum(ref * est, axis=1)
    denominator = np.linalg.norm(ref, axis=1) * np.linalg.norm(est, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-12)
    # Dos vectores negros no tienen dirección; se define su error como cero.
    both_zero = (np.linalg.norm(ref, axis=1) < 1e-12) & (np.linalg.norm(est, axis=1) < 1e-12)
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    angles[both_zero] = 0.0
    return float(np.degrees(angles).mean())


def quality_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """PSNR (MAX_I = 1), SSIM (ventana uniforme 7x7, K1=0.01, K2=0.03) y SAM."""
    return {
        "PSNR [dB]": float(peak_signal_noise_ratio(reference, estimate, data_range=1.0)),
        "SSIM": float(structural_similarity(reference, estimate, data_range=1.0,
                                            channel_axis=2, win_size=7,
                                            gaussian_weights=False, K1=0.01, K2=0.03)),
        "SAM [deg]": spectral_angle_mapper(reference, estimate),
    }


def tau_from_equation5(reference: np.ndarray, blurred: np.ndarray,
                       demosaicked: np.ndarray) -> float:
    """Estimación escalar de tau = S_w / S_f (ecuación 5).

    El "ruido" es el error del demosaicking respecto de la imagen borrosa; por
    Parseval, el cociente de potencias espectrales medias es el cociente de
    energías en el dominio espacial.
    """
    noise = demosaicked - blurred
    return float(np.sum(noise**2) / np.sum(reference**2))


def high_frequency_fraction(image: np.ndarray, cutoff: float = 0.25) -> float:
    """Fracción de la energía espectral con frecuencia radial > cutoff [ciclos/píxel]."""
    fx = np.fft.fftfreq(image.shape[0])
    fy = np.fft.fftfreq(image.shape[1])
    radius = np.sqrt(fx[:, None] ** 2 + fy[None, :] ** 2)
    energy = np.abs(np.fft.fft2(image, axes=(0, 1))) ** 2
    if energy.ndim == 3:
        energy = energy.sum(axis=2)
    return float(energy[radius > cutoff].sum() / energy.sum())


def save_sensor_figure(rgb: np.ndarray, blurred: np.ndarray, mosaic: np.ndarray, masks: np.ndarray) -> None:
    """Figura del modelo de sensado para una imagen."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB original")
    axes[1].imshow(np.clip(blurred, 0, 1))
    axes[1].set_title("RGB después del DOE")
    axes[2].imshow(mosaic, cmap="gray")
    axes[2].set_title("Medición Bayer")
    axes[3].imshow(masks[:16, :16])
    axes[3].set_title("Máscara RGGB (16×16 píxeles)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "modelo_sensor.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_psf_figure(psfs: np.ndarray) -> None:
    """Figura de las PSF (escala logarítmica) de los tres canales."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = ["R", "G", "B"]
    for channel, ax in enumerate(axes):
        wavelength = WAVELENGTHS_NM[::-1][channel]
        ax.imshow(np.log10(psfs[..., channel] + 1e-12), cmap="magma")
        ax.set_title(f"PSF canal {names[channel]} ({int(wavelength)} nm)")
        ax.axis("off")
    fig.suptitle("PSF del elemento difractivo por canal (log10)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "psf_doe.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_figure(original: np.ndarray, demosaicked: np.ndarray,
                               reconstructions: dict[str, np.ndarray], name: str) -> None:
    """Figura con la imagen original, el demosaicking y las reconstrucciones."""
    items = [("Original", original), ("Demosaicking", demosaicked),
             *[(PLOT_LABELS[k], v) for k, v in reconstructions.items()]]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
    for ax, (title, image) in zip(axes.flat, items):
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(title)
        ax.axis("off")
    for ax in axes.flat[len(items):]:
        ax.axis("off")
    fig.suptitle(f"Reconstrucciones — imagen McMaster {name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reconstruccion_ejemplo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _values(rows: list[dict[str, object]], method: str, metric: str) -> np.ndarray:
    """Valores de una métrica para un método, en el orden de las imágenes."""
    return np.array([float(r[metric]) for r in rows if r["metodo"] == method])


def save_metric_figure(rows: list[dict[str, object]]) -> None:
    """Barras con la media y la desviación estándar de cada métrica."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, metric in zip(axes, METRIC_NAMES):
        means = [_values(rows, m, metric).mean() for m in METHODS]
        stds = [_values(rows, m, metric).std() for m in METHODS]
        positions = np.arange(len(METHODS))
        ax.bar(positions, means, yerr=stds, capsize=3, width=0.7,
               color=[PLOT_COLORS[m] for m in METHODS],
               hatch=None, edgecolor="white", linewidth=2, error_kw={"lw": 1, "ecolor": "#52514e"})
        for bar_index, method in enumerate(METHODS):
            if method.endswith("_skimage"):
                ax.patches[bar_index].set_alpha(0.55)
        ax.set_xticks(positions, [PLOT_LABELS[m] for m in METHODS], rotation=35, ha="right")
        ax.set_title(f"{metric}: media ± desv. est. (18 imágenes)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for x, value in zip(positions, means):
            ax.text(x, value / 2, f"{value:.2f}" if metric != "SSIM" else f"{value:.3f}",
                    ha="center", va="center", fontsize=8, color="white")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metricas_dataset.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_per_image_figure(rows: list[dict[str, object]]) -> None:
    """Métricas por imagen para todo McMaster (un panel por métrica).

    Las versiones skimage con tau constante coinciden con las propias, por lo
    que se grafican solo las propias y el Wiener de skimage con Laplaciano.
    """
    shown = ["demosaic", "wiener_scratch", "rl_scratch", "wiener_skimage_lap"]
    markers = {"demosaic": "s", "wiener_scratch": "o", "rl_scratch": "D",
               "wiener_skimage_lap": "^"}
    offsets = {"demosaic": -0.24, "wiener_scratch": -0.08, "rl_scratch": 0.08,
               "wiener_skimage_lap": 0.24}
    images = [r["imagen"] for r in rows if r["metodo"] == "demosaic"]
    x = np.arange(len(images))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, metric in zip(axes, METRIC_NAMES):
        for method in shown:
            ax.plot(x + offsets[method], _values(rows, method, metric), linestyle="none",
                    marker=markers[method], markersize=7, color=PLOT_COLORS[method],
                    markeredgecolor="white", markeredgewidth=1, label=PLOT_LABELS[method])
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for boundary in x[:-1] + 0.5:
            ax.axvline(boundary, color="#e4e3df", linewidth=0.8, zorder=0)
    axes[0].set_title("Métricas por imagen en McMaster (PSNR y SSIM: mayor es mejor; SAM: menor es mejor)")
    axes[-1].set_xticks(x, images)
    axes[-1].set_xlabel("Imagen McMaster")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.02 + 1.32), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metricas_por_imagen.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], path: Path, fields: list[str]) -> None:
    """Escribe una lista de diccionarios como CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Lee un CSV como lista de diccionarios."""
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_summary_tex(rows: list[dict[str, object]], path: Path) -> None:
    """Escribe la tabla de promedios (media ± desviación estándar) del informe."""
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Método & PSNR [dB] & SSIM & SAM [$^\\circ$] \\\\",
        "\\midrule",
    ]
    for method in METHODS:
        cells = []
        for metric, digits in zip(METRIC_NAMES, [2, 4, 2]):
            values = _values(rows, method, metric)
            cells.append(f"${values.mean():.{digits}f} \\pm {values.std():.{digits}f}$")
        lines.append(f"{METHOD_LABELS[method]} & {' & '.join(cells)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _latex_sci(value: float) -> str:
    """Formatea un número en notación científica de LaTeX."""
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.1e}".split("e")
    if -2 <= int(exponent) <= 1:
        return f"${value:.3f}$"
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def write_comparison_tex(rows: list[dict[str, object]], diagnostics: list[dict[str, object]],
                         path: Path) -> None:
    """Tabla con las diferencias entre implementación propia y skimage."""
    pairs = [("Wiener", "wiener_scratch", "wiener_skimage", "dif_max_wiener"),
             ("Wiener (vs. Laplaciano)", "wiener_scratch", "wiener_skimage_lap",
              "dif_max_wiener_lap"),
             ("Richardson--Lucy", "rl_scratch", "rl_skimage", "dif_max_rl")]
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Par comparado & $\\max|\\Delta \\widehat I|$ & $\\max|\\Delta\\mathrm{PSNR}|$ [dB] & "
        "$\\max|\\Delta\\mathrm{SSIM}|$ & $\\max|\\Delta\\mathrm{SAM}|$ [$^\\circ$] \\\\",
        "\\midrule",
    ]
    for label, own, ref, key in pairs:
        cells = [_latex_sci(max(float(d[key]) for d in diagnostics))]
        for metric in METRIC_NAMES:
            diff = np.abs(_values(rows, own, metric) - _values(rows, ref, metric)).max()
            cells.append(_latex_sci(diff))
        lines.append(f"{label} & {' & '.join(cells)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sweep_tex(sweep: list[dict[str, object]], diagnostics: list[dict[str, object]],
                    path: Path) -> None:
    """Tablas del barrido de tau (Wiener) y de iteraciones (RL)."""
    taus = np.array([float(d["tau_ec5"]) for d in diagnostics])
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Parámetro & PSNR [dB] & SSIM & SAM [$^\\circ$] \\\\",
        "\\midrule",
    ]
    previous = None
    for row in sweep:
        if previous is not None and row["parametro"] != previous:
            lines.append("\\midrule")
        previous = row["parametro"]
        value = float(row["valor"])
        name = f"$\\tau = {value:g}$" if row["parametro"] == "tau" else f"$K = {int(value)}$"
        lines.append(f"{name} & {float(row['PSNR [dB]']):.2f} & {float(row['SSIM']):.4f} & "
                     f"{float(row['SAM [deg]']):.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    # Macros con el resumen de la estimación de la ecuación (5).
    lines += ["\\gdef\\TauEcCincoMedia{" + _latex_sci(taus.mean()) + "}",
              "\\gdef\\TauEcCincoMin{" + _latex_sci(taus.min()) + "}",
              "\\gdef\\TauEcCincoMax{" + _latex_sci(taus.max()) + "}"]
    if "frac_alta_frec_error" in diagnostics[0]:
        err = np.mean([float(d["frac_alta_frec_error"]) for d in diagnostics])
        sig = np.mean([float(d["frac_alta_frec_senal"]) for d in diagnostics])
        lines += [f"\\gdef\\FracAltaError{{{100 * err:.0f}\\,\\%}}",
                  f"\\gdef\\FracAltaSenal{{{100 * sig:.1f}\\,\\%}}"]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tables_and_figures(rows, diagnostics, sweep) -> None:
    """Genera las tablas LaTeX y los gráficos de métricas a partir de los datos."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_summary_tex(rows, RESULTS_DIR / "promedios.tex")
    save_metric_figure(rows)
    save_per_image_figure(rows)
    if diagnostics:
        write_comparison_tex(rows, diagnostics, RESULTS_DIR / "comparacion_implementaciones.tex")
    if sweep and diagnostics:
        write_sweep_tex(sweep, diagnostics, RESULTS_DIR / "barrido_parametros.tex")


def run_sweeps(images: list[tuple[np.ndarray, np.ndarray]], psfs: np.ndarray,
               balance: float, iterations: int) -> list[dict[str, object]]:
    """Promedios sobre el dataset al variar tau (Wiener) y K (RL)."""
    sweep: list[dict[str, object]] = []
    configs = [("tau", tau, "wiener_scratch") for tau in TAU_GRID]
    configs += [("iteraciones", k, "rl_scratch") for k in ITERATION_GRID]
    for parameter, value, method in configs:
        metrics = []
        for rgb, demosaicked in images:
            estimate = deconvolve_rgb(
                demosaicked, psfs, method,
                value if parameter == "tau" else balance,
                int(value) if parameter == "iteraciones" else iterations)
            metrics.append(quality_metrics(rgb, estimate))
        row = {"parametro": parameter, "valor": value, "metodo": method}
        row.update({m: float(np.mean([d[m] for d in metrics])) for m in METRIC_NAMES})
        sweep.append(row)
        print(f"Barrido {parameter}={value}: PSNR {row['PSNR [dB]']:.2f} dB")
    return sweep


def run_experiment(size: int = 256, iterations: int = 25, balance: float = 0.01,
                   limit: int | None = None, download: bool = True,
                   sweeps: bool = True) -> list[dict[str, object]]:
    """Ejecuta todo el experimento y retorna una fila por imagen y método."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = ensure_mcmaster(download=download)
    if limit is not None:
        paths = paths[:limit]
    psfs = build_psfs(size)
    masks = bayer_masks((size, size))
    save_psf_figure(psfs)

    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    sweep_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    for index, path in enumerate(paths, start=1):
        rgb = load_rgb(path, size)
        blurred, mosaic = forward_model(rgb, psfs, masks)
        demosaicked = bilinear_demosaic(mosaic, masks)
        sweep_inputs.append((rgb, demosaicked))

        reconstructions: dict[str, np.ndarray] = {}
        for method in METHODS:
            if method == "demosaic":
                estimate = demosaicked
            else:
                estimate = deconvolve_rgb(demosaicked, psfs, method, balance, iterations)
            reconstructions[method] = estimate
            row = {"imagen": path.stem, "metodo": method}
            row.update(quality_metrics(rgb, estimate))
            rows.append(row)
        diagnostics.append({
            "imagen": path.stem,
            "dif_max_wiener": float(np.abs(reconstructions["wiener_scratch"]
                                           - reconstructions["wiener_skimage"]).max()),
            "dif_max_wiener_lap": float(np.abs(reconstructions["wiener_scratch"]
                                               - reconstructions["wiener_skimage_lap"]).max()),
            "dif_max_rl": float(np.abs(reconstructions["rl_scratch"]
                                       - reconstructions["rl_skimage"]).max()),
            "tau_ec5": tau_from_equation5(rgb, blurred, demosaicked),
            "frac_alta_frec_error": high_frequency_fraction(demosaicked - blurred),
            "frac_alta_frec_senal": high_frequency_fraction(rgb),
        })
        if index == 1:
            save_sensor_figure(rgb, blurred, mosaic, masks)
            save_reconstruction_figure(
                rgb, demosaicked,
                {k: v for k, v in reconstructions.items() if k != "demosaic"}, path.stem)
        print(f"[{index:02d}/{len(paths):02d}] {path.name}")

    write_csv(rows, METRICS_CSV, ["imagen", "metodo", *METRIC_NAMES])
    write_csv(diagnostics, DIAGNOSTICS_CSV, list(diagnostics[0].keys()))
    sweep = run_sweeps(sweep_inputs, psfs, balance, iterations) if sweeps else []
    if sweep:
        write_csv(sweep, SWEEP_CSV, ["parametro", "valor", "metodo", *METRIC_NAMES])
    write_tables_and_figures(rows, diagnostics, sweep)
    print(f"Resultados guardados en {METRICS_CSV}")
    return rows


def regenerate_from_csv() -> None:
    """Regenera tablas y gráficos de métricas desde los CSV existentes."""
    rows = read_csv(METRICS_CSV)
    diagnostics = read_csv(DIAGNOSTICS_CSV) if DIAGNOSTICS_CSV.exists() else []
    sweep = read_csv(SWEEP_CSV) if SWEEP_CSV.exists() else []
    write_tables_and_figures(rows, diagnostics, sweep)
    print(f"Tablas y gráficos regenerados desde {METRICS_CSV}")


def main() -> None:
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=256, help="Resolución cuadrada de trabajo")
    parser.add_argument("--iterations", type=int, default=25, help="Iteraciones Richardson-Lucy")
    parser.add_argument("--balance", type=float, default=0.01, help="Regularización tau de Wiener")
    parser.add_argument("--limit", type=int, default=None, help="Número de imágenes, útil para pruebas")
    parser.add_argument("--no-download", action="store_true", help="No descargar el dataset")
    parser.add_argument("--sin-barrido", action="store_true",
                        help="Omitir el barrido de tau y de iteraciones")
    parser.add_argument("--solo-tablas", action="store_true",
                        help="Regenerar tablas y gráficos desde los CSV sin simular")
    args = parser.parse_args()
    plt.switch_backend("Agg")  # en terminal solo se guardan las figuras
    if args.solo_tablas:
        regenerate_from_csv()
        return
    run_experiment(args.size, args.iterations, args.balance, args.limit,
                   not args.no_download, not args.sin_barrido)


if __name__ == "__main__":
    main()

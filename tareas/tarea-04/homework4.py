"""Tarea 4: sistema difractivo RGB y deconvolución.

El script implementa el flujo completo solicitado en la guía:

    RGB -> DOE/PSF por longitud de onda -> filtro Bayer + ruido -> demosaicking
        -> Wiener o Richardson-Lucy -> PSNR, SSIM y SAM.

Además resuelve directamente el problema regularizado de la ecuación (2) sobre la
medición Bayer (Tikhonov por gradiente conjugado), compara tres diseños de DOE
(fase cúbica, modos de Zernike y su combinación) frente al desenfoque, compara
demosaicking bilineal y Malvar, y repite la cadena sobre las bandas Washington DC
entregadas en el material del curso (lecture4a, diapositiva 6).

Los modos de Zernike son un port de ``homework4/Zernike`` (código MATLAB del
profesor): indexación y normalización de Noll, base ``zerni`` y combinación lineal.

Uso local desde la raíz del repositorio (entorno ``.venv``)::

    .venv/Scripts/python tareas/tarea-04/homework4.py

Ejecución de prueba::

    .venv/Scripts/python tareas/tarea-04/homework4.py --limit 2 --sin-barrido --sin-doe

Regenerar tablas y gráficos a partir de los CSV, sin repetir la simulación::

    .venv/Scripts/python tareas/tarea-04/homework4.py --solo-tablas

Dependencias: numpy, scipy, scikit-image, matplotlib, Pillow y certifi.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import ssl
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from math import factorial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
from skimage import restoration
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:  # PyTorch es opcional; el pipeline conserva un backend NumPy/SciPy.
    import torch
except ImportError:  # pragma: no cover - depende del entorno del usuario
    torch = None


TASK_DIR = Path(__file__).resolve().parent
DATASET_DIRS = [TASK_DIR / "McMaster", TASK_DIR / "data" / "McMaster"]
WASHINGTON_DIR = TASK_DIR / "homework4" / "DIP3E_CH11_Original_Images"
ZERNIKE_INDEX_M = TASK_DIR / "homework4" / "Zernike" / "zernike_index.m"
FIG_DIR = TASK_DIR / "Fig"
RESULTS_DIR = TASK_DIR / "resultados"
METRICS_CSV = RESULTS_DIR / "metricas_mcmaster.csv"
DIAGNOSTICS_CSV = RESULTS_DIR / "diagnosticos_por_imagen.csv"
SWEEP_CSV = RESULTS_DIR / "barrido_parametros.csv"
DOE_CSV = RESULTS_DIR / "doe_desenfoque.csv"
PSF_CSV = RESULTS_DIR / "psf_diagnosticos.csv"
WASHINGTON_CSV = RESULTS_DIR / "washington_dc.csv"
# Espejo de Kaggle del dataset McMaster; solo se usa si no está la carpeta local.
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/liikke/mcmaster"

# --- Óptica -----------------------------------------------------------------
WAVELENGTHS_NM = np.array([650.0, 550.0, 450.0])  # canales R, G, B
REFERENCE_WAVELENGTH_NM = 550.0                   # longitud de onda de diseño del DOE
PUPIL_GRID = 256       # muestras de la grilla de la pupila (FFT)
PSF_SIZE = 161         # ventana centrada en la que se recorta la PSF (impar)
CUBIC_ALPHA = 6.0      # fase cúbica mod(alpha*pi*(s^3+t^3), 2*pi), lecture4b diap. 6
# Coeficientes [rad] de los modos de Zernike normalizados de Noll (lecture4a diap. 8 y 11).
ZERNIKE_COEFFS = {4: 0.6, 5: 0.4, 6: -0.3, 7: 0.3, 8: -0.2, 11: 0.3}
DOE_DESIGNS = ["cubica", "zernike", "cubica_zernike"]
MAIN_DESIGN = "cubica_zernike"
DESIGN_LABELS = {"libre": "Apertura libre", "cubica": "Fase cúbica",
                 "zernike": "Zernike", "cubica_zernike": "Cúbica + Zernike"}
DEFOCUS_GRID = [0.0, 3.0, 6.0, 9.0, 12.0]  # Psi [rad] a 550 nm (lecture4b diap. 13)

# --- Sensor y reconstrucción ----------------------------------------------------
BAYER_PATTERN = "RGGB"
SEED = 2026
NOISE_SIGMA = 0.0      # la guía define el modelo ideal sin ruido aditivo
TAU = 1e-2             # tau elegido por barrido para la ecuación (4)
RL_ITERATIONS = 25     # K de Richardson-Lucy (ecuación 6)
TAU_BAYER = 1e-3       # tau de la ecuación (2) resuelta sobre la medición Bayer
CG_ITERATIONS = 60     # iteraciones de gradiente conjugado para la ecuación (2)
RL_EPSILON = 1e-12     # mismo valor que skimage.restoration.richardson_lucy

METHODS = ["demosaic", "wiener_scratch", "wiener_skimage", "wiener_skimage_lap",
           "rl_scratch", "rl_skimage", "tikhonov_bayer", "demosaic_malvar", "wiener_malvar"]
MAIN_METHODS = METHODS[:7]
METHOD_LABELS = {
    "demosaic": "Demosaicking bilineal (sin deconvolución)",
    "wiener_scratch": "Wiener propio",
    "wiener_skimage": "Wiener skimage ($\\tau$ constante)",
    "wiener_skimage_lap": "Wiener skimage (Laplaciano, por defecto)",
    "rl_scratch": "Richardson--Lucy propio",
    "rl_skimage": "Richardson--Lucy skimage",
    "tikhonov_bayer": "Ec.~(2) sobre Bayer (Tikhonov, GC)",
    "demosaic_malvar": "Demosaicking Malvar (sin deconvolución)",
    "wiener_malvar": "Malvar + Wiener propio",
}
PLOT_LABELS = {
    "demosaic": "Demosaicking",
    "wiener_scratch": "Wiener propio",
    "wiener_skimage": "Wiener skimage (τ)",
    "wiener_skimage_lap": "Wiener skimage (Laplaciano)",
    "rl_scratch": "RL propio",
    "rl_skimage": "RL skimage",
    "tikhonov_bayer": "Ec. (2) Bayer",
    "demosaic_malvar": "Malvar",
    "wiener_malvar": "Malvar + Wiener",
}
# Paleta categórica en orden fijo; el gris neutro es la línea base sin deconvolución.
PLOT_COLORS = {
    "demosaic": "#8c8b86",
    "wiener_scratch": "#2a78d6",
    "wiener_skimage": "#2a78d6",
    "rl_scratch": "#eb6834",
    "rl_skimage": "#eb6834",
    "tikhonov_bayer": "#1baf7a",
    "wiener_skimage_lap": "#eda100",
    "demosaic_malvar": "#8c8b86",
    "wiener_malvar": "#e87ba4",
}
DESIGN_COLORS = {"libre": "#8c8b86", "cubica": "#2a78d6", "zernike": "#eb6834",
                 "cubica_zernike": "#1baf7a"}
METRIC_NAMES = ["PSNR [dB]", "SSIM", "SAM [deg]"]
SIGMA_GRID = [0.0, 0.005, 0.02]
TAU_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
ITERATION_GRID = [5, 10, 25, 50, 100]
TAU_BAYER_GRID = [1e-4, 1e-3, 1e-2]

# ``auto`` ocupa CUDA si está disponible; ``cpu`` mantiene una ruta reproducible
# para equipos sin PyTorch/CUDA. Solo los cálculos FFT propios (PSF y Wiener)
# pasan por Torch; SciPy y scikit-image siguen siendo referencias independientes.
DEVICE = "auto"


def active_device() -> str:
    """Dispositivo efectivo del backend FFT propio."""
    if DEVICE != "auto":
        return DEVICE
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def torch_device():
    """Dispositivo Torch para FFT o ``None`` cuando se usa CPU NumPy."""
    device = active_device()
    return torch.device(device) if torch is not None and device == "cuda" else None


# =============================================================================
# Datos
# =============================================================================

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


def ensure_mcmaster(download: bool = False) -> list[Path]:
    """Devuelve las 18 imágenes McMaster de la carpeta local de la tarea.

    Busca en ``McMaster/`` y luego en ``data/McMaster``; solo descarga el espejo
    de Kaggle si ``download`` es verdadero y no hay copia local.
    """
    for directory in DATASET_DIRS:
        paths = find_images(directory) if directory.exists() else []
        if len(paths) >= 18:
            return paths[:18]
    if not download:
        raise FileNotFoundError(
            f"No se encontraron 18 imágenes en {DATASET_DIRS[0]}. "
            "Copie el dataset McMaster o ejecute con --descargar.")

    import certifi  # solo se necesita para la descarga
    target = DATASET_DIRS[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.parent / "McM.zip"
    if not archive.exists():
        print(f"Descargando McMaster desde {DATASET_URL}")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(DATASET_URL, context=ssl_context) as response:
            archive.write_bytes(response.read())
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    paths = find_images(target)
    if len(paths) < 18:
        raise RuntimeError(f"La descarga no contiene 18 imágenes RGB: {target}")
    return paths[:18]


def load_rgb(path: Path, size: int = 256) -> np.ndarray:
    """Carga una imagen RGB, la normaliza a [0,1] y la redimensiona (Lanczos)."""
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float64) / 255.0


def load_washington_rgb(size: int = 256) -> np.ndarray:
    """Compone un RGB con las bandas 3, 2 y 1 (rojo, verde, azul) de Washington DC."""
    bands = find_images(WASHINGTON_DIR)
    channels = []
    for band in (bands[2], bands[1], bands[0]):
        image = Image.open(band).convert("L").resize((size, size), Image.Resampling.LANCZOS)
        channels.append(np.asarray(image, dtype=np.float64) / 255.0)
    return np.stack(channels, axis=-1)


# =============================================================================
# Modos de Zernike (port de homework4/Zernike/*.m)
# =============================================================================

def noll_to_nm(j: int) -> tuple[int, int]:
    """Índices (n, m) del modo j en la indexación de Noll (``zernike_index.m``)."""
    if j < 1:
        raise ValueError("El índice de Noll parte en 1 (pistón).")
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    # Dentro del orden n, m crece; m = 0 aparece una vez y m > 0 dos veces (cos y sen).
    m_values = [m for m in range(n % 2, n + 1, 2) for _ in range(1 if m == 0 else 2)]
    return n, m_values[j - n * (n + 1) // 2 - 1]


def zernike_radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """Polinomio radial R_n^m(r) (función ``zrf`` de ``zernike2.m``)."""
    radial = np.zeros_like(r)
    for k in range((n - m) // 2 + 1):
        coefficient = ((-1) ** k * factorial(n - k)
                       / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)))
        radial = radial + coefficient * r ** (n - 2 * k)
    return radial


def zernike_noll(j: int, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Modo Z_j normalizado de Noll (``zernike2.m``; lecture4a, diapositiva 11).

    m = 0: sqrt(n+1) R_n^0. m > 0: sqrt(2(n+1)) R_n^m cos(m theta) si j es par y
    sen(m theta) si j es impar.
    """
    n, m = noll_to_nm(j)
    if m == 0:
        return np.sqrt(n + 1) * zernike_radial(n, 0, r)
    angular = np.cos(m * theta) if j % 2 == 0 else np.sin(m * theta)
    return np.sqrt(2 * (n + 1)) * zernike_radial(n, m, r) * angular


def zernike_basis(size: int, n_modes: int, delta: float | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Matriz Z (size^2 x n_modes) con los modos 1..n_modes (``zerni.m``).

    Como en el código MATLAB, la grilla es (-N/2 : N/2-1)*delta; por defecto
    delta = 2/N, de modo que la pupila unitaria ocupa la grilla. Devuelve también
    la máscara de la apertura circular.
    """
    delta = 2.0 / size if delta is None else delta
    coords = (np.arange(size) - size // 2) * delta
    x, y = np.meshgrid(coords, coords)
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    basis = np.stack([zernike_noll(j, r, theta).ravel() for j in range(1, n_modes + 1)], axis=1)
    return basis, (r <= 1.0).ravel()


def zernike_combination(basis: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Frente de onda W = Z a como imagen cuadrada (``zernikeLinearCombination.m``).

    El original además aplica ``mat2gray``; aquí se conserva W en radianes porque
    se usa directamente como fase.
    """
    wavefront = basis @ coefficients
    size = int(round(np.sqrt(wavefront.size)))
    return wavefront.reshape(size, size)


def zernike_fit(basis: np.ndarray, wavefront: np.ndarray, aperture: np.ndarray) -> np.ndarray:
    """Coeficientes por mínimos cuadrados dentro de la apertura (``A = Z\\W`` de ``test.m``)."""
    coefficients, *_ = np.linalg.lstsq(basis[aperture], wavefront.ravel()[aperture], rcond=None)
    return coefficients


def read_professor_noll_table(path: Path = ZERNIKE_INDEX_M) -> list[tuple[int, int]]:
    """Lee la tabla (n, m) de ``zernike_index.m`` para verificar ``noll_to_nm``."""
    text = path.read_text(encoding="latin-1")
    start = text.index("zer_index = [") + len("zer_index = [")
    body = text[start:text.index("]", start)]
    return [(int(a), int(b)) for a, b in re.findall(r"^\s*(\d+)\s+(\d+)", body, re.M)]


# =============================================================================
# Elemento difractivo y PSF
# =============================================================================

def pupil_grid(wavelength_nm: float, grid: int = PUPIL_GRID):
    """Coordenadas normalizadas de la pupila para una longitud de onda.

    El píxel del sensor es fijo, por lo que el radio de la pupila en muestras de
    la FFT escala como 1/lambda. Se elige R = grid/4 para 450 nm: así la OTF (de
    radio 2R) cabe en la grilla para todas las longitudes de onda, es decir, la
    PSF de intensidad queda muestreada sin aliasing (relleno de ceros x2).
    """
    radius = (grid / 4) * WAVELENGTHS_NM.min() / wavelength_nm
    coords = (np.arange(grid) - grid // 2) / radius
    x, y = np.meshgrid(coords, coords)
    rho, theta = np.hypot(x, y), np.arctan2(y, x)
    return x, y, rho, theta


def doe_phase(design: str, x: np.ndarray, y: np.ndarray, rho: np.ndarray,
              theta: np.ndarray) -> np.ndarray:
    """Fase del DOE a 550 nm, envuelta en [0, 2pi) como un perfil de alturas.

    - ``cubica``: alpha*pi*(s^3 + t^3) de Dowski y Cathey (lecture4b, diap. 6).
    - ``zernike``: W = sum_j a_j Z_j con modos de Noll (lecture4a, diap. 8).
    - ``cubica_zernike``: suma de ambas; ``libre``: sin DOE.
    """
    phase = np.zeros_like(x)
    if "cubica" in design:
        phase = phase + CUBIC_ALPHA * np.pi * (x**3 + y**3)
    if "zernike" in design:
        for j, coefficient in ZERNIKE_COEFFS.items():
            phase = phase + coefficient * zernike_noll(j, rho, theta)
    return np.mod(phase, 2.0 * np.pi)


def compute_psf(design: str, wavelength_nm: float, defocus: float = 0.0,
                size: int = PSF_SIZE) -> tuple[np.ndarray, float]:
    """PSF incoherente h = |F{P}|^2 recortada a ``size`` y normalizada a suma 1.

    La pupila es P = A exp(i (lambda0/lambda) (phi_DOE + Psi rho^2)): el retardo
    de un DOE de altura fija y el desenfoque Psi escalan como 1/lambda
    (lecture4a diap. 8; lecture4b diap. 13). Devuelve además la fracción de
    energía que queda dentro de la ventana.
    """
    x, y, rho, theta = pupil_grid(wavelength_nm)
    aperture = rho <= 1.0
    phase = (REFERENCE_WAVELENGTH_NM / wavelength_nm) * (
        doe_phase(design, x, y, rho, theta) + defocus * rho**2)
    device = torch_device()
    if device is None:
        pupil = aperture * np.exp(1j * phase)
        psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))) ** 2
    else:
        aperture_t = torch.as_tensor(aperture, dtype=torch.float64, device=device)
        phase_t = torch.as_tensor(phase, dtype=torch.float64, device=device)
        pupil_t = aperture_t * torch.exp(1j * phase_t)
        psf_t = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pupil_t))).abs().square()
        psf = psf_t.cpu().numpy()
    psf /= psf.sum()
    center, half = PUPIL_GRID // 2, size // 2
    cropped = psf[center - half:center + half + 1, center - half:center + half + 1]
    energy = float(cropped.sum())
    return cropped / energy, energy


def build_psfs(design: str = MAIN_DESIGN, defocus: float = 0.0) -> np.ndarray:
    """PSF de los canales R, G y B apiladas en el último eje."""
    return np.stack([compute_psf(design, wl, defocus)[0] for wl in WAVELENGTHS_NM], axis=-1)


def radial_mtf(psf: np.ndarray, n_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Perfil radial medio de |OTF| en función de la frecuencia [ciclos/píxel]."""
    otf = np.abs(np.fft.fft2(np.fft.ifftshift(psf)))
    fx = np.fft.fftfreq(psf.shape[0])
    radius = np.hypot(fx[:, None], fx[None, :])
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    index = np.digitize(radius.ravel(), edges) - 1
    valid = (index >= 0) & (index < n_bins)
    sums = np.bincount(index[valid], otf.ravel()[valid], minlength=n_bins)
    counts = np.bincount(index[valid], minlength=n_bins)
    return 0.5 * (edges[1:] + edges[:-1]), sums / np.maximum(counts, 1)


def psf_diagnostics() -> list[dict[str, object]]:
    """Energía en la ventana, ancho RMS y MTF a 0.25 ciclos/píxel por diseño y canal."""
    rows = []
    for design in ["libre", *DOE_DESIGNS]:
        for channel, wavelength in zip("RGB", WAVELENGTHS_NM):
            psf, energy = compute_psf(design, wavelength)
            yy, xx = np.indices(psf.shape) - psf.shape[0] // 2
            cy, cx = (psf * yy).sum(), (psf * xx).sum()
            rms = np.sqrt((psf * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum())
            freq, mtf = radial_mtf(psf)
            rows.append({"diseno": design, "canal": channel, "lambda_nm": wavelength,
                         "energia_ventana": energy, "ancho_rms_px": float(rms),
                         "mtf_025": float(np.interp(0.25, freq, mtf))})
    return rows


# =============================================================================
# Sensor Bayer y demosaicking
# =============================================================================

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


def malvar_demosaic(mosaic: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Demosaicking lineal de Malvar, He y Cutler (2004) para RGGB (de la tarea 1)."""
    g_at_rb = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
                        [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], dtype=np.float64) / 8
    rb_at_g_row = np.array([[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1],
                            [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]], dtype=np.float64) / 8
    rb_at_g_col = rb_at_g_row.T
    rb_at_br = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5],
                         [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], dtype=np.float64) / 8
    red_mask, green_mask, blue_mask = (masks[..., c] == 1 for c in range(3))
    rows, cols = np.indices(mosaic.shape)
    green_in_red_row = green_mask & (rows % 2 == 0)
    green_in_blue_row = green_mask & (rows % 2 == 1)

    green = np.where(green_mask, mosaic, convolve(mosaic, g_at_rb, mode="mirror"))
    by_row = convolve(mosaic, rb_at_g_row, mode="mirror")
    by_col = convolve(mosaic, rb_at_g_col, mode="mirror")
    diagonal = convolve(mosaic, rb_at_br, mode="mirror")
    red = np.where(red_mask, mosaic, 0.0)
    red = np.where(green_in_red_row, by_row, red)
    red = np.where(green_in_blue_row, by_col, red)
    red = np.where(blue_mask, diagonal, red)
    blue = np.where(blue_mask, mosaic, 0.0)
    blue = np.where(green_in_blue_row, by_row, blue)
    blue = np.where(green_in_red_row, by_col, blue)
    blue = np.where(red_mask, diagonal, blue)
    return np.clip(np.stack([red, green, blue], axis=-1), 0.0, 1.0)


def blur_channel(channel: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """h * f con extensión simétrica de la escena (la luz de fuera del campo también llega)."""
    half = psf.shape[0] // 2
    return fftconvolve(np.pad(channel, half, mode="symmetric"), psf, mode="valid")


def forward_model(rgb: np.ndarray, psfs: np.ndarray, masks: np.ndarray,
                  sigma: float = 0.0, rng: np.random.Generator | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Ecuación (1) con ruido: y = Sensor(h_l * f_l)_Bayer + omega, recortada a [0, 1]."""
    blurred = np.stack([blur_channel(rgb[..., c], psfs[..., c]) for c in range(3)], axis=-1)
    measurement = rgb_to_bayer(blurred, masks)
    if sigma > 0:
        rng = np.random.default_rng(SEED) if rng is None else rng
        measurement = measurement + sigma * rng.standard_normal(measurement.shape)
    return blurred, np.clip(measurement, 0.0, 1.0)


# =============================================================================
# Deconvolución
# =============================================================================

def psf_at_origin(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rellena la PSF con ceros hasta ``shape`` y mueve su centro al origen (``ir2tf``)."""
    padded = np.zeros(shape)
    padded[:psf.shape[0], :psf.shape[1]] = psf
    return np.roll(padded, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))


def psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """OTF de una PSF centrada, con el mismo convenio que ``skimage`` (``ir2tf``)."""
    return np.fft.fft2(psf_at_origin(psf, shape))


def wiener_from_scratch(blurred: np.ndarray, psf: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Filtro de Wiener de la ecuación (4) con tau constante, en Fourier."""
    device = torch_device()
    if device is not None:
        image_t = torch.as_tensor(blurred, dtype=torch.float64, device=device)
        psf_t = torch.as_tensor(psf_at_origin(psf, blurred.shape), dtype=torch.float64, device=device)
        h = torch.fft.fft2(psf_t)
        estimate = torch.fft.ifft2(h.conj() * torch.fft.fft2(image_t) /
                                   (h.abs().square() + tau)).real
        return estimate.cpu().numpy()
    h = psf_to_otf(psf, blurred.shape)
    estimate = np.fft.ifft2(np.conj(h) * np.fft.fft2(blurred) / (np.abs(h) ** 2 + tau))
    return np.real(estimate)


def richardson_lucy_from_scratch(blurred: np.ndarray, psf: np.ndarray,
                                 iterations: int = RL_ITERATIONS, initial_value: float = 0.5,
                                 checkpoints: list[int] | None = None):
    """Richardson--Lucy de la ecuación (6), con número fijo de iteraciones.

    La estimación inicial es la constante ``initial_value``; como la
    actualización es multiplicativa y la PSF suma 1, el resultado no depende
    de esa constante (siempre que sea positiva). Con ``checkpoints`` devuelve un
    diccionario con la estimación en esas iteraciones (para el barrido de K).
    """
    estimate = np.full_like(blurred, initial_value)
    psf_mirror = psf[::-1, ::-1]
    saved = {}
    for k in range(1, iterations + 1):
        reblurred = fftconvolve(estimate, psf, mode="same") + RL_EPSILON
        estimate = estimate * fftconvolve(blurred / reblurred, psf_mirror, mode="same")
        if checkpoints and k in checkpoints:
            saved[k] = estimate.copy()
    return saved if checkpoints else estimate


def deconvolve_channel(image: np.ndarray, psf: np.ndarray, method: str,
                       tau: float = TAU, iterations: int = RL_ITERATIONS) -> np.ndarray:
    """Deconvoluciona un canal con el método indicado (sin recorte de valores).

    La imagen se extiende simétricamente en medio ancho de la PSF antes de
    deconvolucionar y se recorta después, igual para todos los métodos; así la
    periodicidad de la FFT no genera artefactos en los bordes.
    """
    half = psf.shape[0] // 2
    padded = np.pad(image, half, mode="symmetric")
    if method == "wiener_scratch":
        estimate = wiener_from_scratch(padded, psf, tau)
    elif method == "wiener_skimage":
        # reg = impulso unitario => |reg|^2 = 1, es decir, el mismo tau constante
        # de la ecuación (4). Sin este argumento skimage usa un Laplaciano.
        estimate = restoration.wiener(padded, psf, balance=tau, reg=np.ones((1, 1)), clip=False)
    elif method == "wiener_skimage_lap":
        estimate = restoration.wiener(padded, psf, balance=tau, clip=False)
    elif method == "rl_scratch":
        estimate = richardson_lucy_from_scratch(padded, psf, iterations)
    elif method == "rl_skimage":
        estimate = restoration.richardson_lucy(padded, psf, num_iter=iterations, clip=False)
    else:
        raise ValueError(f"Método desconocido: {method}")
    return estimate[half:-half, half:-half]


def finish(image: np.ndarray) -> np.ndarray:
    """Recorta a [0, 1] (y reemplaza valores no finitos) al final de cada método."""
    return np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def deconvolve_rgb(blurred_rgb: np.ndarray, psfs: np.ndarray, method: str,
                   tau: float = TAU, iterations: int = RL_ITERATIONS) -> np.ndarray:
    """Deconvoluciona cada canal por separado y recorta el resultado a [0,1]."""
    return finish(np.stack([deconvolve_channel(blurred_rgb[..., c], psfs[..., c], method,
                                               tau, iterations) for c in range(3)], axis=-1))


def tikhonov_bayer(measurement: np.ndarray, masks: np.ndarray, psfs: np.ndarray,
                   tau: float = TAU_BAYER, iterations: int = CG_ITERATIONS,
                   initial: np.ndarray | None = None) -> np.ndarray:
    """Resuelve la ecuación (2) directamente sobre la medición Bayer.

    Como las máscaras B_c son disjuntas y suman 1,
    ||y - sum_c B_c (h_c * f_c)||^2 = sum_c ||B_c (y - h_c * f_c)||^2,
    de modo que cada canal resuelve por gradiente conjugado
    (A_c^T A_c + tau I) f_c = A_c^T y, con A_c f = B_c (h_c * f)|_campo.
    La incógnita f_c vive en un dominio extendido en medio ancho de la PSF, de
    modo que la convolución circular en ese dominio coincide con la lineal
    dentro del campo y no se impone ninguna condición de borde a la escena.
    """
    half = psfs.shape[0] // 2
    size = measurement.shape
    extended = (size[0] + 2 * half, size[1] + 2 * half)
    field = (slice(half, half + size[0]), slice(half, half + size[1]))
    result = np.empty((*size, 3))
    for c in range(3):
        otf = np.fft.rfft2(psf_at_origin(psfs[..., c], extended))
        mask = masks[..., c]

        def forward(f):
            return mask * np.fft.irfft2(otf * np.fft.rfft2(f), s=extended)[field]

        def adjoint(r):
            padded = np.zeros(extended)
            padded[field] = mask * r
            return np.fft.irfft2(np.conj(otf) * np.fft.rfft2(padded), s=extended)

        def normal(f):
            return adjoint(forward(f)) + tau * f

        start = initial[..., c] if initial is not None else np.full(size, 0.5)
        x = np.pad(start, half, mode="symmetric")
        residual = adjoint(mask * measurement) - normal(x)
        direction = residual.copy()
        rr = float(np.sum(residual**2))
        for _ in range(iterations):
            ad = normal(direction)
            step = rr / float(np.sum(direction * ad))
            x = x + step * direction
            residual = residual - step * ad
            rr_new = float(np.sum(residual**2))
            direction = residual + (rr_new / rr) * direction
            rr = rr_new
        result[..., c] = x[field]
    return finish(result)


# =============================================================================
# Métricas y diagnósticos
# =============================================================================

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

    El "ruido" omega es todo lo que separa la imagen demosaickeada de la imagen
    desenfocada: ruido del sensor interpolado más error del demosaicking. Con
    espectros planos, por Parseval el cociente de potencias espectrales medias es
    el cociente de energías en el dominio espacial.
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


# =============================================================================
# Experimentos (una función por imagen, para poder paralelizar)
# =============================================================================

def image_rng(index: int, sigma: float) -> np.random.Generator:
    """Generador reproducible por imagen y nivel de ruido (independiente del orden)."""
    return np.random.default_rng([SEED, index, int(round(sigma * 1e6))])


def reconstruct_all(rgb: np.ndarray, psfs: np.ndarray, masks: np.ndarray, sigma: float,
                    rng: np.random.Generator, tau: float, iterations: int,
                    tau_bayer: float, cg_iterations: int) -> dict[str, object]:
    """Simula una medición y aplica todos los métodos del experimento principal."""
    blurred, mosaic = forward_model(rgb, psfs, masks, sigma, rng)
    demosaicked = bilinear_demosaic(mosaic, masks)
    malvar = malvar_demosaic(mosaic, masks)
    estimates = {"demosaic": demosaicked}
    for method in ["wiener_scratch", "wiener_skimage", "wiener_skimage_lap",
                   "rl_scratch", "rl_skimage"]:
        estimates[method] = deconvolve_rgb(demosaicked, psfs, method, tau, iterations)
    estimates["tikhonov_bayer"] = tikhonov_bayer(mosaic, masks, psfs, tau_bayer,
                                                 cg_iterations, initial=demosaicked)
    estimates["demosaic_malvar"] = malvar
    estimates["wiener_malvar"] = deconvolve_rgb(malvar, psfs, "wiener_scratch", tau)
    return {"blurred": blurred, "mosaic": mosaic, "estimates": estimates}


def evaluate_image(task: tuple) -> dict[str, object]:
    """Experimento principal para una imagen McMaster."""
    index, path, psfs, cfg = task
    rgb = load_rgb(path, cfg["size"])
    masks = bayer_masks(rgb.shape[:2])
    out = reconstruct_all(rgb, psfs, masks, cfg["sigma"], image_rng(index, cfg["sigma"]),
                          cfg["tau"], cfg["iterations"], cfg["tau_bayer"], cfg["cg_iterations"])
    est = out["estimates"]
    rows = []
    for method in METHODS:
        row = {"imagen": path.stem, "metodo": method}
        row.update(quality_metrics(rgb, est[method]))
        rows.append(row)
    diagnostics = {
        "imagen": path.stem,
        "dif_max_wiener": float(np.abs(est["wiener_scratch"] - est["wiener_skimage"]).max()),
        "dif_max_wiener_lap": float(np.abs(est["wiener_scratch"] - est["wiener_skimage_lap"]).max()),
        "dif_max_rl": float(np.abs(est["rl_scratch"] - est["rl_skimage"]).max()),
        "tau_ec5": tau_from_equation5(rgb, out["blurred"], est["demosaic"]),
        "frac_alta_frec_error": high_frequency_fraction(est["demosaic"] - out["blurred"]),
        "frac_alta_frec_senal": high_frequency_fraction(rgb),
    }
    result = {"rows": rows, "diagnostics": diagnostics}
    if cfg.get("keep_images"):
        result.update(rgb=rgb, **out)
    return result


def sweep_image(task: tuple) -> list[dict[str, object]]:
    """Barrido de tau (Wiener y ec. 2) y de K (RL) para una imagen y un nivel de ruido."""
    index, path, psfs, cfg, sigma = task
    rgb = load_rgb(path, cfg["size"])
    masks = bayer_masks(rgb.shape[:2])
    blurred, mosaic = forward_model(rgb, psfs, masks, sigma, image_rng(index, sigma))
    demosaicked = bilinear_demosaic(mosaic, masks)
    base = {"imagen": path.stem, "sigma": sigma}
    rows = [{**base, "parametro": "tau_ec5", "valor": tau_from_equation5(rgb, blurred, demosaicked),
             "metodo": "diagnostico", **{m: np.nan for m in METRIC_NAMES}}]
    for tau in TAU_GRID:
        estimate = deconvolve_rgb(demosaicked, psfs, "wiener_scratch", tau)
        rows.append({**base, "parametro": "tau", "valor": tau, "metodo": "wiener_scratch",
                     **quality_metrics(rgb, estimate)})
    half = psfs.shape[0] // 2
    per_channel = []
    for c in range(3):
        padded = np.pad(demosaicked[..., c], half, mode="symmetric")
        saved = richardson_lucy_from_scratch(padded, psfs[..., c], max(ITERATION_GRID),
                                             checkpoints=ITERATION_GRID)
        per_channel.append({k: v[half:-half, half:-half] for k, v in saved.items()})
    for k in ITERATION_GRID:
        estimate = finish(np.stack([per_channel[c][k] for c in range(3)], axis=-1))
        rows.append({**base, "parametro": "iteraciones", "valor": k, "metodo": "rl_scratch",
                     **quality_metrics(rgb, estimate)})
    for tau in TAU_BAYER_GRID:
        estimate = tikhonov_bayer(mosaic, masks, psfs, tau, cfg["cg_iterations"], demosaicked)
        rows.append({**base, "parametro": "tau_bayer", "valor": tau, "metodo": "tikhonov_bayer",
                     **quality_metrics(rgb, estimate)})
    return rows


def doe_image(task: tuple) -> list[dict[str, object]]:
    """Diseños de DOE frente al desenfoque: la escena se mide con PSF(Psi) y se
    deconvoluciona con la PSF enfocada PSF(0), como en un sistema EDoF."""
    index, path, bank, cfg = task
    rgb = load_rgb(path, cfg["size"])
    masks = bayer_masks(rgb.shape[:2])
    rows = []
    for design in ["libre", *DOE_DESIGNS]:
        for defocus in DEFOCUS_GRID:
            _, mosaic = forward_model(rgb, bank[(design, defocus)], masks, cfg["sigma"],
                                      image_rng(index, cfg["sigma"]))
            demosaicked = bilinear_demosaic(mosaic, masks)
            estimate = deconvolve_rgb(demosaicked, bank[(design, 0.0)], "wiener_scratch", cfg["tau"])
            rows.append({"imagen": path.stem, "diseno": design, "desenfoque": defocus,
                         **quality_metrics(rgb, estimate)})
    return rows


def parallel_map(function, tasks: list, workers: int) -> list:
    """Aplica ``function`` a las tareas, en paralelo por procesos si workers > 1."""
    if workers <= 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, tasks))


def default_workers() -> int:
    """Número de procesos por defecto (deja núcleos libres)."""
    return max(1, min(9, (os.cpu_count() or 2) // 2))


def default_config(**overrides) -> dict[str, object]:
    """Configuración del experimento principal."""
    cfg = {"size": 256, "sigma": NOISE_SIGMA, "tau": TAU, "iterations": RL_ITERATIONS,
           "tau_bayer": TAU_BAYER, "cg_iterations": CG_ITERATIONS}
    cfg.update(overrides)
    return cfg


def run_main(paths: list[Path], cfg: dict, workers: int) -> tuple[list, list]:
    """Experimento principal sobre las imágenes McMaster; guarda CSV y figuras de ejemplo."""
    psfs = build_psfs(MAIN_DESIGN)
    tasks = [(i, p, psfs, {**cfg, "keep_images": i == 0}) for i, p in enumerate(paths)]
    results = parallel_map(evaluate_image, tasks, workers)
    rows = [r for res in results for r in res["rows"]]
    diagnostics = [res["diagnostics"] for res in results]
    first = results[0]
    save_sensor_figure(first["rgb"], first["blurred"], first["mosaic"], bayer_masks(first["rgb"].shape[:2]))
    save_reconstruction_figure(first["rgb"], first["estimates"], paths[0].stem)
    write_csv(rows, METRICS_CSV, ["imagen", "metodo", *METRIC_NAMES])
    write_csv(diagnostics, DIAGNOSTICS_CSV, list(diagnostics[0].keys()))
    return rows, diagnostics


def run_sweeps(paths: list[Path], cfg: dict, workers: int) -> list[dict[str, object]]:
    """Barridos de parámetros para cada nivel de ruido de ``SIGMA_GRID``."""
    psfs = build_psfs(MAIN_DESIGN)
    tasks = [(i, p, psfs, cfg, s) for s in SIGMA_GRID for i, p in enumerate(paths)]
    sweep = [r for res in parallel_map(sweep_image, tasks, workers) for r in res]
    write_csv(sweep, SWEEP_CSV, ["imagen", "sigma", "parametro", "valor", "metodo", *METRIC_NAMES])
    return sweep


def run_doe(paths: list[Path], cfg: dict, workers: int) -> list[dict[str, object]]:
    """Comparación de diseños de DOE frente al desenfoque."""
    bank = {(d, z): build_psfs(d, z) for d in ["libre", *DOE_DESIGNS] for z in DEFOCUS_GRID}
    tasks = [(i, p, bank, cfg) for i, p in enumerate(paths)]
    rows = [r for res in parallel_map(doe_image, tasks, workers) for r in res]
    write_csv(rows, DOE_CSV, ["imagen", "diseno", "desenfoque", *METRIC_NAMES])
    save_edof_psf_figure(bank)
    return rows


def run_washington(cfg: dict) -> list[dict[str, object]]:
    """Cadena completa sobre el RGB Washington DC (bandas 3, 2, 1) del material del curso."""
    rgb = load_washington_rgb(cfg["size"])
    psfs = build_psfs(MAIN_DESIGN)
    masks = bayer_masks(rgb.shape[:2])
    out = reconstruct_all(rgb, psfs, masks, cfg["sigma"], image_rng(99, cfg["sigma"]),
                          cfg["tau"], cfg["iterations"], cfg["tau_bayer"], cfg["cg_iterations"])
    rows = [{"imagen": "WashingtonDC", "metodo": m, **quality_metrics(rgb, out["estimates"][m])}
            for m in MAIN_METHODS]
    write_csv(rows, WASHINGTON_CSV, ["imagen", "metodo", *METRIC_NAMES])
    save_washington_figure(rgb, out["mosaic"], out["estimates"])
    return rows


def run_experiment(size: int = 256, sigma: float = NOISE_SIGMA, tau: float = TAU,
                   iterations: int = RL_ITERATIONS, tau_bayer: float = TAU_BAYER,
                   cg_iterations: int = CG_ITERATIONS, limit: int | None = None,
                   download: bool = False, sweeps: bool = True, doe: bool = True,
                   workers: int | None = None) -> list[dict[str, object]]:
    """Ejecuta todos los experimentos y genera CSV, tablas LaTeX y figuras."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    workers = default_workers() if workers is None else workers
    paths = ensure_mcmaster(download)[:limit]
    cfg = default_config(size=size, sigma=sigma, tau=tau, iterations=iterations,
                         tau_bayer=tau_bayer, cg_iterations=cg_iterations)
    write_csv(psf_diagnostics(), PSF_CSV,
              ["diseno", "canal", "lambda_nm", "energia_ventana", "ancho_rms_px", "mtf_025"])
    save_dataset_figure(paths)
    save_zernike_figure()
    save_psf_figure()
    save_mtf_figure()
    print(f"Backend FFT: {active_device()}" +
          (f" ({torch.cuda.get_device_name(0)})" if active_device() == "cuda" else ""))
    print(f"Experimento principal: {len(paths)} imágenes, {workers} procesos")
    rows, _ = run_main(paths, cfg, workers)
    if sweeps:
        print("Barrido de tau, K y ruido")
        run_sweeps(paths, cfg, workers)
    if doe:
        print("Diseños de DOE frente al desenfoque")
        run_doe(paths, cfg, workers)
    if WASHINGTON_DIR.exists():
        run_washington(cfg)
    regenerate_from_csv(cfg)
    print(f"Resultados guardados en {RESULTS_DIR}")
    return rows


# =============================================================================
# Figuras
# =============================================================================

def _save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _style(ax) -> None:
    """Ejes recesivos: grilla horizontal tenue y sin bordes superior/derecho."""
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save_dataset_figure(paths: list[Path]) -> None:
    """Miniaturas de las imágenes McMaster (como la Figura 2 de la guía)."""
    cols = 6
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.3 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, paths):
        ax.imshow(load_rgb(path, 128))
        ax.set_title(path.stem, fontsize=9)
    fig.tight_layout()
    _save(fig, "dataset_mcmaster.png")


def save_zernike_figure(max_order: int = 5, size: int = 96) -> None:
    """Pirámide de modos de Zernike de Noll (como lecture4a, diapositiva 12)."""
    coords = (np.arange(size) - size // 2) * (2.0 / size)
    x, y = np.meshgrid(coords, coords)
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    n_modes = (max_order + 1) * (max_order + 2) // 2
    fig, axes = plt.subplots(max_order + 1, 2 * max_order + 1,
                             figsize=(1.35 * (2 * max_order + 1), 1.4 * (max_order + 1)))
    for ax in axes.flat:
        ax.axis("off")
    for j in range(1, n_modes + 1):
        n, m = noll_to_nm(j)
        signed_m = m if (j % 2 == 0 or m == 0) else -m  # sen -> m negativo
        ax = axes[n, max_order + signed_m]
        mode = np.where(r <= 1, zernike_noll(j, r, theta), np.nan)
        ax.imshow(mode, cmap="RdBu_r", vmin=-np.nanmax(np.abs(mode)), vmax=np.nanmax(np.abs(mode)))
        ax.set_title(f"$Z_{{{j}}}$", fontsize=9)
    fig.suptitle("Modos de Zernike (Noll), filas n = 0..5, columnas m con signo")
    fig.tight_layout()
    _save(fig, "zernike_piramide.png")


def save_psf_figure() -> None:
    """Fase envuelta del DOE y PSF (log10) de los tres canales para cada diseño."""
    designs = DOE_DESIGNS
    fig, axes = plt.subplots(len(designs), 4, figsize=(13, 3.3 * len(designs)))
    for row, design in enumerate(designs):
        x, y, rho, theta = pupil_grid(REFERENCE_WAVELENGTH_NM)
        phase = np.where(rho <= 1, doe_phase(design, x, y, rho, theta), np.nan)
        c, half = PUPIL_GRID // 2, int(PUPIL_GRID / 4 * 450 / 550) + 2
        im = axes[row, 0].imshow(phase[c - half:c + half, c - half:c + half], cmap="twilight",
                                 vmin=0, vmax=2 * np.pi)
        axes[row, 0].set_title(f"{DESIGN_LABELS[design]}: fase [rad]")
        fig.colorbar(im, ax=axes[row, 0], fraction=0.046)
        for col, (channel, wl) in enumerate(zip("RGB", WAVELENGTHS_NM), start=1):
            psf = compute_psf(design, wl)[0]
            axes[row, col].imshow(np.log10(psf + 1e-9), cmap="magma", vmin=-7)
            axes[row, col].set_title(f"PSF {channel} ({int(wl)} nm), log10")
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    _save(fig, "psf_doe.png")


def save_mtf_figure() -> None:
    """Perfil radial de la MTF (canal G) para cada diseño y la apertura libre."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (channel, wl) in zip(axes, zip("RGB", WAVELENGTHS_NM)):
        for design in ["libre", *DOE_DESIGNS]:
            freq, mtf = radial_mtf(compute_psf(design, wl)[0])
            ax.semilogy(freq, np.maximum(mtf, 1e-6), color=DESIGN_COLORS[design], lw=2,
                        label=DESIGN_LABELS[design])
        ax.set_title(f"MTF radial, canal {channel} ({int(wl)} nm)")
        ax.set_xlabel("Frecuencia [ciclos/píxel]")
        _style(ax)
    axes[0].set_ylabel("|OTF| media (escala log)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "mtf_disenos.png")


def save_sensor_figure(rgb: np.ndarray, blurred: np.ndarray, mosaic: np.ndarray,
                       masks: np.ndarray) -> None:
    """Figura del modelo de sensado para una imagen."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB original $f_l$")
    axes[1].imshow(np.clip(blurred, 0, 1))
    axes[1].set_title("Después del DOE $h_l * f_l$")
    axes[2].imshow(mosaic, cmap="gray")
    axes[2].set_title("Medición Bayer $y$ (con ruido)")
    axes[3].imshow(masks[:16, :16])
    axes[3].set_title("Máscara RGGB (16×16 píxeles)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "modelo_sensor.png")


def save_reconstruction_figure(original: np.ndarray, estimates: dict[str, np.ndarray],
                               name: str) -> None:
    """Imagen completa y detalle ampliado para los métodos principales."""
    shown = ["demosaic", "wiener_scratch", "rl_scratch", "tikhonov_bayer", "wiener_skimage_lap"]
    items = [("Original", original), *[(PLOT_LABELS[k], estimates[k]) for k in shown]]
    size = original.shape[0]
    zoom = (slice(size // 4, size // 4 + size // 3), slice(size // 3, size // 3 + size // 3))
    fig, axes = plt.subplots(2, len(items), figsize=(3.1 * len(items), 6.6))
    for col, (title, image) in enumerate(items):
        axes[0, col].imshow(np.clip(image, 0, 1))
        axes[0, col].set_title(title)
        axes[1, col].imshow(np.clip(image[zoom], 0, 1), interpolation="nearest")
        axes[1, col].set_title("detalle")
        if col == 0:
            r, c = zoom
            axes[0, 0].add_patch(plt.Rectangle((c.start, r.start), c.stop - c.start,
                                               r.stop - r.start, fill=False, ec="white", lw=1.5))
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Reconstrucciones — imagen McMaster {name}")
    fig.tight_layout()
    _save(fig, "reconstruccion_ejemplo.png")


def save_washington_figure(rgb: np.ndarray, mosaic: np.ndarray,
                           estimates: dict[str, np.ndarray]) -> None:
    """Réplica de lecture4a, diapositiva 6, con las bandas Washington DC."""
    items = [("RGB (bandas 3, 2, 1)", rgb), ("Medición Bayer $y$", mosaic),
             ("Demosaicking", estimates["demosaic"]), ("Wiener propio", estimates["wiener_scratch"]),
             ("RL propio", estimates["rl_scratch"]), ("Ec. (2) Bayer", estimates["tikhonov_bayer"])]
    fig, axes = plt.subplots(1, len(items), figsize=(3.1 * len(items), 3.6))
    for ax, (title, image) in zip(axes, items):
        ax.imshow(np.clip(image, 0, 1), cmap="gray" if image.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "washington_dc.png")


def save_edof_psf_figure(bank: dict) -> None:
    """PSF del canal G a distintas distancias de desenfoque (lecture4b, diapositiva 6)."""
    designs = ["libre", "cubica"]
    fig, axes = plt.subplots(len(designs), len(DEFOCUS_GRID), figsize=(2.6 * len(DEFOCUS_GRID), 5.4))
    half = 40
    for row, design in enumerate(designs):
        for col, defocus in enumerate(DEFOCUS_GRID):
            psf = bank[(design, defocus)][..., 1]
            c = psf.shape[0] // 2
            axes[row, col].imshow(np.sqrt(psf[c - half:c + half + 1, c - half:c + half + 1]), cmap="magma")
            axes[row, col].set_title(f"{DESIGN_LABELS[design]}, Ψ={defocus:g} rad", fontsize=9)
            axes[row, col].axis("off")
    fig.suptitle("PSF del canal G frente al desenfoque (raíz cuadrada, ventana de 81×81 px)")
    fig.tight_layout()
    _save(fig, "edof_psf.png")


def _values(rows: list[dict[str, object]], method: str, metric: str) -> np.ndarray:
    """Valores de una métrica para un método, en el orden de las imágenes."""
    return np.array([float(r[metric]) for r in rows if r["metodo"] == method])


def save_metric_figure(rows: list[dict[str, object]]) -> None:
    """Barras con la media y la desviación estándar de cada métrica."""
    methods = [m for m in METHODS if m in {r["metodo"] for r in rows}]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for ax, metric in zip(axes, METRIC_NAMES):
        means = [_values(rows, m, metric).mean() for m in methods]
        stds = [_values(rows, m, metric).std() for m in methods]
        positions = np.arange(len(methods))
        bars = ax.bar(positions, means, yerr=stds, capsize=3, width=0.72,
                      color=[PLOT_COLORS[m] for m in methods], edgecolor="white", linewidth=2,
                      error_kw={"lw": 1, "ecolor": "#52514e"})
        for bar, method in zip(bars, methods):
            if method.endswith("_skimage") or method == "demosaic_malvar":
                bar.set_alpha(0.55)
        ax.set_xticks(positions, [PLOT_LABELS[m] for m in methods], rotation=40, ha="right")
        ax.set_title(f"{metric}: media ± desv. est.")
        _style(ax)
        for x, value in zip(positions, means):
            ax.text(x, value / 2, f"{value:.2f}" if metric != "SSIM" else f"{value:.3f}",
                    ha="center", va="center", fontsize=8, color="white")
    fig.tight_layout()
    _save(fig, "metricas_dataset.png")


def save_per_image_figure(rows: list[dict[str, object]]) -> None:
    """Métricas por imagen para todo McMaster (un panel por métrica).

    Las versiones skimage con tau constante coinciden con las propias, por lo
    que se grafican solo las propias, la ec. (2) y el Wiener de skimage con Laplaciano.
    """
    shown = ["demosaic", "wiener_scratch", "rl_scratch", "tikhonov_bayer", "wiener_skimage_lap"]
    markers = {"demosaic": "s", "wiener_scratch": "o", "rl_scratch": "D",
               "tikhonov_bayer": "v", "wiener_skimage_lap": "^"}
    offsets = dict(zip(shown, np.linspace(-0.28, 0.28, len(shown))))
    images = [r["imagen"] for r in rows if r["metodo"] == "demosaic"]
    x = np.arange(len(images))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, metric in zip(axes, METRIC_NAMES):
        for method in shown:
            ax.plot(x + offsets[method], _values(rows, method, metric), linestyle="none",
                    marker=markers[method], markersize=7, color=PLOT_COLORS[method],
                    markeredgecolor="white", markeredgewidth=1, label=PLOT_LABELS[method])
        ax.set_ylabel(metric)
        _style(ax)
        for boundary in x[:-1] + 0.5:
            ax.axvline(boundary, color="#e4e3df", linewidth=0.8, zorder=0)
    axes[0].set_title("Métricas por imagen en McMaster (PSNR y SSIM: mayor es mejor; SAM: menor es mejor)",
                      pad=34)
    axes[-1].set_xticks(x, images)
    axes[-1].set_xlabel("Imagen McMaster")
    axes[0].legend(ncol=len(shown), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    _save(fig, "metricas_por_imagen.png")


def sweep_means(sweep: list[dict[str, object]]) -> list[dict[str, object]]:
    """Promedia el barrido por (sigma, parámetro, valor)."""
    keys = sorted({(float(r["sigma"]), r["parametro"], float(r["valor"])) for r in sweep
                   if r["parametro"] != "tau_ec5"})
    out = []
    for sigma, parameter, value in keys:
        group = [r for r in sweep if float(r["sigma"]) == sigma and r["parametro"] == parameter
                 and float(r["valor"]) == value]
        out.append({"sigma": sigma, "parametro": parameter, "valor": value,
                    **{m: float(np.mean([float(r[m]) for r in group])) for m in METRIC_NAMES}})
    return out


def tau_eq5_by_sigma(sweep: list[dict[str, object]]) -> dict[float, np.ndarray]:
    """Valores por imagen del tau de la ecuación (5) para cada nivel de ruido."""
    sigmas = sorted({float(r["sigma"]) for r in sweep})
    return {s: np.array([float(r["valor"]) for r in sweep
                         if r["parametro"] == "tau_ec5" and float(r["sigma"]) == s]) for s in sigmas}


def save_sweep_figure(sweep: list[dict[str, object]]) -> None:
    """PSNR medio frente a tau (Wiener y ec. 2) y a K (RL) para cada nivel de ruido."""
    means = sweep_means(sweep)
    taus5 = tau_eq5_by_sigma(sweep)
    sigmas = sorted(taus5)
    colors = dict(zip(sigmas, ["#2a78d6", "#eb6834", "#1baf7a"]))
    panels = [("tau", "Wiener propio: PSNR vs τ", "τ"),
              ("iteraciones", "RL propio: PSNR vs K", "K (iteraciones)"),
              ("tau_bayer", "Ec. (2) sobre Bayer: PSNR vs τ", "τ")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, (parameter, title, xlabel) in zip(axes, panels):
        for sigma in sigmas:
            pts = [r for r in means if r["parametro"] == parameter and r["sigma"] == sigma]
            ax.semilogx([r["valor"] for r in pts], [r["PSNR [dB]"] for r in pts], marker="o",
                        lw=2, markersize=8, color=colors[sigma], label=f"σ = {sigma:g}")
            if parameter == "tau":
                ax.axvline(np.median(taus5[sigma]), color=colors[sigma], ls=":", lw=1.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        _style(ax)
    axes[0].set_ylabel("PSNR medio [dB]")
    axes[0].legend(frameon=False, title="Ruido del sensor")
    axes[0].text(0.02, 0.03, "líneas punteadas: τ de la ec. (5) (mediana)",
                 transform=axes[0].transAxes, fontsize=8, color="#52514e")
    fig.tight_layout()
    _save(fig, "barrido_ruido.png")


def save_doe_figure(doe_rows: list[dict[str, object]]) -> None:
    """PSNR y SSIM medios frente al desenfoque Psi para cada diseño de DOE."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, metric in zip(axes, ["PSNR [dB]", "SSIM"]):
        for design in ["libre", *DOE_DESIGNS]:
            values = [np.mean([float(r[metric]) for r in doe_rows
                               if r["diseno"] == design and float(r["desenfoque"]) == z])
                      for z in DEFOCUS_GRID]
            ax.plot(DEFOCUS_GRID, values, marker="o", lw=2, markersize=8,
                    color=DESIGN_COLORS[design], label=DESIGN_LABELS[design])
        ax.set_xlabel("Desenfoque Ψ a 550 nm [rad]")
        ax.set_ylabel(f"{metric} medio")
        ax.set_title(f"Wiener con la PSF enfocada: {metric}")
        _style(ax)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "edof_metricas.png")


# =============================================================================
# Tablas LaTeX
# =============================================================================

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
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Método & PSNR [dB] & SSIM & SAM [$^\\circ$] \\\\", "\\midrule"]
    for method in METHODS:
        if method == "demosaic_malvar":
            lines.append("\\midrule")
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


def write_sweep_tex(sweep: list[dict[str, object]], path: Path) -> None:
    """Tabla del mejor parámetro por método y nivel de ruido, y tau de la ec. (5)."""
    means = sweep_means(sweep)
    taus5 = tau_eq5_by_sigma(sweep)
    labels = {"tau": "Wiener propio ($\\tau$)", "iteraciones": "RL propio ($K$)",
              "tau_bayer": "Ec.~(2) Bayer ($\\tau$)"}
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "$\\sigma$ & Método & Mejor valor & PSNR [dB] & SSIM & SAM [$^\\circ$] \\\\", "\\midrule"]
    for i, sigma in enumerate(sorted(taus5)):
        if i:
            lines.append("\\midrule")
        for parameter in ["tau", "iteraciones", "tau_bayer"]:
            pts = [r for r in means if r["sigma"] == sigma and r["parametro"] == parameter]
            best = max(pts, key=lambda r: r["PSNR [dB]"])
            value = f"$K={int(best['valor'])}$" if parameter == "iteraciones" else _latex_sci(best["valor"])
            lines.append(f"{sigma:g} & {labels[parameter]} & {value} & {best['PSNR [dB]']:.2f} & "
                         f"{best['SSIM']:.4f} & {best['SAM [deg]']:.2f} \\\\")
        lines.append(f" & $\\tau$ ec.~(5), mediana [mín., máx.] & "
                     f"\\multicolumn{{4}}{{l}}{{{_latex_sci(np.median(taus5[sigma]))} "
                     f"[{_latex_sci(taus5[sigma].min())}, {_latex_sci(taus5[sigma].max())}]}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_doe_tex(doe_rows: list[dict[str, object]], path: Path) -> None:
    """PSNR medio por diseño y desenfoque, más la caída entre Psi = 0 y el máximo."""
    lines = ["\\begin{tabular}{l" + "c" * (len(DEFOCUS_GRID) + 1) + "}", "\\toprule",
             "Diseño & " + " & ".join(f"$\\Psi={z:g}$" for z in DEFOCUS_GRID)
             + " & Caída [dB] \\\\", "\\midrule"]
    for design in ["libre", *DOE_DESIGNS]:
        values = [np.mean([float(r["PSNR [dB]"]) for r in doe_rows
                           if r["diseno"] == design and float(r["desenfoque"]) == z])
                  for z in DEFOCUS_GRID]
        lines.append(f"{DESIGN_LABELS[design]} & " + " & ".join(f"{v:.2f}" for v in values)
                     + f" & {values[0] - values[-1]:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_psf_tex(psf_rows: list[dict[str, object]], path: Path) -> None:
    """Energía en la ventana, ancho RMS y MTF(0.25) por diseño y canal."""
    lines = ["\\begin{tabular}{lccccccccc}", "\\toprule",
             " & \\multicolumn{3}{c}{Energía en la ventana} & \\multicolumn{3}{c}{Ancho RMS [px]}"
             " & \\multicolumn{3}{c}{MTF a 0.25 ciclos/px} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
             "Diseño & R & G & B & R & G & B & R & G & B \\\\", "\\midrule"]
    for design in ["libre", *DOE_DESIGNS]:
        sub = [r for r in psf_rows if r["diseno"] == design]
        cells = ([f"{float(r['energia_ventana']):.3f}" for r in sub]
                 + [f"{float(r['ancho_rms_px']):.1f}" for r in sub]
                 + [f"{float(r['mtf_025']):.3f}" for r in sub])
        lines.append(f"{DESIGN_LABELS[design]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_macros_tex(rows, diagnostics, sweep, doe_rows, washington, cfg, path: Path) -> None:
    """Macros con las cifras que cita el texto del informe (todas provienen de los CSV)."""
    def mean(method, metric):
        return _values(rows, method, metric).mean()

    def fmt(value, digits=2):
        return f"{value:.{digits}f}"

    macros = {
        "CfgSigma": f"{cfg['sigma']:g}", "CfgTau": f"{cfg['tau']:g}",
        "CfgK": str(cfg["iterations"]), "CfgTauBayer": f"{cfg['tau_bayer']:g}",
        "CfgCG": str(cfg["cg_iterations"]), "CfgAlpha": f"{CUBIC_ALPHA:g}",
        "CfgPSF": str(PSF_SIZE), "NImagenes": str(len(diagnostics)),
    }
    keys = {"demosaic": "Dem", "wiener_scratch": "Wie", "wiener_skimage_lap": "Lap",
            "rl_scratch": "RL", "tikhonov_bayer": "Bay", "demosaic_malvar": "Mal",
            "wiener_malvar": "MalWie"}
    for method, key in keys.items():
        macros[f"Psnr{key}"] = fmt(mean(method, "PSNR [dB]"))
        macros[f"Ssim{key}"] = fmt(mean(method, "SSIM"), 4)
        macros[f"Sam{key}"] = fmt(mean(method, "SAM [deg]"))
    for method, key in [("wiener_scratch", "Wie"), ("rl_scratch", "RL"), ("tikhonov_bayer", "Bay")]:
        macros[f"Gan{key}"] = fmt(mean(method, "PSNR [dB]") - mean("demosaic", "PSNR [dB]"))
    psnr = {m: _values(rows, m, "PSNR [dB]") for m in METHODS}
    ssim = {m: _values(rows, m, "SSIM") for m in METHODS}
    sam = {m: _values(rows, m, "SAM [deg]") for m in METHODS}
    macros["WieGanaRLPsnr"] = str(int(np.sum(psnr["wiener_scratch"] > psnr["rl_scratch"])))
    macros["RLGanaWieSsim"] = str(int(np.sum(ssim["rl_scratch"] > ssim["wiener_scratch"])))
    macros["BayGanaWiePsnr"] = str(int(np.sum(psnr["tikhonov_bayer"] > psnr["wiener_scratch"])))
    macros["WieMejoraSam"] = str(int(np.sum(sam["wiener_scratch"] < sam["demosaic"])))
    macros["DifWiener"] = _latex_sci(max(float(d["dif_max_wiener"]) for d in diagnostics))
    macros["DifRL"] = _latex_sci(max(float(d["dif_max_rl"]) for d in diagnostics))
    if sweep:
        taus5 = tau_eq5_by_sigma(sweep)
        means = sweep_means(sweep)
        for sigma, key in zip(sorted(taus5), ["Cero", "Bajo", "Alto"]):
            macros[f"TauCincoMed{key}"] = _latex_sci(float(np.median(taus5[sigma])))
            for parameter, name in [("tau", "TauOpt"), ("iteraciones", "KOpt"), ("tau_bayer", "TauBayOpt")]:
                pts = [r for r in means if r["sigma"] == sigma and r["parametro"] == parameter]
                best = max(pts, key=lambda r: r["PSNR [dB]"])
                macros[f"{name}{key}"] = (str(int(best["valor"])) if parameter == "iteraciones"
                                          else f"{best['valor']:g}")
                macros[f"{name}Psnr{key}"] = fmt(best["PSNR [dB]"])
    if doe_rows:
        for design, key in [("libre", "Libre"), ("cubica", "Cub"), ("zernike", "Zer"),
                            ("cubica_zernike", "CubZer")]:
            values = [np.mean([float(r["PSNR [dB]"]) for r in doe_rows
                               if r["diseno"] == design and float(r["desenfoque"]) == z])
                      for z in DEFOCUS_GRID]
            macros[f"Doe{key}Foco"] = fmt(values[0])
            macros[f"Doe{key}Max"] = fmt(values[-1])
            macros[f"Doe{key}Caida"] = fmt(values[0] - values[-1])
        macros["DoePsiMax"] = f"{DEFOCUS_GRID[-1]:g}"
    if washington:
        for method, key in [("demosaic", "Dem"), ("wiener_scratch", "Wie"), ("rl_scratch", "RL"),
                            ("tikhonov_bayer", "Bay")]:
            value = [float(r["PSNR [dB]"]) for r in washington if r["metodo"] == method][0]
            macros[f"Wdc{key}"] = fmt(value)
    lines = [f"\\gdef\\{name}{{{value}}}" for name, value in macros.items()]
    lines += [f"\\gdef\\FracAltaError{{{100 * np.mean([float(d['frac_alta_frec_error']) for d in diagnostics]):.0f}\\,\\%}}",
              f"\\gdef\\FracAltaSenal{{{100 * np.mean([float(d['frac_alta_frec_senal']) for d in diagnostics]):.1f}\\,\\%}}",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def regenerate_from_csv(cfg: dict | None = None) -> None:
    """Regenera tablas LaTeX y gráficos de métricas desde los CSV existentes."""
    cfg = default_config() if cfg is None else cfg
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(METRICS_CSV)
    diagnostics = read_csv(DIAGNOSTICS_CSV)
    sweep = read_csv(SWEEP_CSV) if SWEEP_CSV.exists() else []
    if sweep and not {"sigma", "imagen"}.issubset(sweep[0]):
        # Ignore the legacy summary-only CSV; the current analysis requires
        # per-image rows and cannot safely reconstruct them from old averages.
        print(f"Se omite barrido legado sin columnas por imagen: {SWEEP_CSV}")
        sweep = []
    doe_rows = read_csv(DOE_CSV) if DOE_CSV.exists() else []
    washington = read_csv(WASHINGTON_CSV) if WASHINGTON_CSV.exists() else []
    write_summary_tex(rows, RESULTS_DIR / "promedios.tex")
    write_comparison_tex(rows, diagnostics, RESULTS_DIR / "comparacion_implementaciones.tex")
    save_metric_figure(rows)
    save_per_image_figure(rows)
    if PSF_CSV.exists():
        write_psf_tex(read_csv(PSF_CSV), RESULTS_DIR / "psf_diagnosticos.tex")
    if sweep:
        write_sweep_tex(sweep, RESULTS_DIR / "barrido_parametros.tex")
        save_sweep_figure(sweep)
    else:
        (RESULTS_DIR / "barrido_parametros.tex").write_text(
            "% Barrido omitido: falta un CSV con resultados por imagen.\n"
            "\\begin{tabular}{l}Barrido no disponible para la corrida actual.\\end{tabular}\n",
            encoding="utf-8",
        )
    if doe_rows:
        write_doe_tex(doe_rows, RESULTS_DIR / "doe_desenfoque.tex")
        save_doe_figure(doe_rows)
    write_macros_tex(rows, diagnostics, sweep, doe_rows, washington, cfg, RESULTS_DIR / "macros.tex")


def main() -> None:
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=256, help="Resolución cuadrada de trabajo")
    parser.add_argument("--sigma", type=float, default=NOISE_SIGMA, help="Ruido gaussiano del sensor")
    parser.add_argument("--tau", type=float, default=TAU, help="Regularización tau de Wiener")
    parser.add_argument("--iterations", type=int, default=RL_ITERATIONS, help="Iteraciones Richardson-Lucy")
    parser.add_argument("--tau-bayer", type=float, default=TAU_BAYER, help="tau de la ec. (2) sobre Bayer")
    parser.add_argument("--cg-iterations", type=int, default=CG_ITERATIONS, help="Iteraciones de GC")
    parser.add_argument("--limit", type=int, default=None, help="Número de imágenes, útil para pruebas")
    parser.add_argument("--workers", type=int, default=None, help="Procesos en paralelo")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Backend FFT propio (auto usa CUDA cuando está disponible)")
    parser.add_argument("--descargar", action="store_true", help="Descargar McMaster si no está local")
    parser.add_argument("--sin-barrido", action="store_true", help="Omitir los barridos de parámetros")
    parser.add_argument("--sin-doe", action="store_true", help="Omitir la comparación de diseños de DOE")
    parser.add_argument("--solo-tablas", action="store_true",
                        help="Regenerar tablas y gráficos desde los CSV sin simular")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    if DEVICE == "cuda" and (torch is None or not torch.cuda.is_available()):
        parser.error("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA disponible.")
    plt.switch_backend("Agg")  # en terminal solo se guardan las figuras
    cfg = default_config(size=args.size, sigma=args.sigma, tau=args.tau, iterations=args.iterations,
                         tau_bayer=args.tau_bayer, cg_iterations=args.cg_iterations)
    if args.solo_tablas:
        regenerate_from_csv(cfg)
        return
    run_experiment(args.size, args.sigma, args.tau, args.iterations, args.tau_bayer,
                   args.cg_iterations, args.limit, args.descargar, not args.sin_barrido,
                   not args.sin_doe, args.workers)


if __name__ == "__main__":
    main()

"""Revelado y demosaicking AHD para capturas RAW/DNG.

La funcion principal devuelve una imagen RGB float32 en [0, 1].
"""

from pathlib import Path

import numpy as np
import rawpy


def demosaick_raw_ahd(raw_path):
    """Revela y demosaifica un archivo RAW/DNG mediante el algoritmo AHD.

    Parameters
    ----------
    raw_path : str or pathlib.Path
        Ruta al archivo RAW o DNG.

    Returns
    -------
    numpy.ndarray
        Imagen RGB con tipo ``float32`` y valores en el intervalo [0, 1].
    """
    raw_path = Path(raw_path)

    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            use_auto_wb=False,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            no_auto_bright=True,
            gamma=(1, 1),
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )

    return np.clip(rgb.astype(np.float32) / 65535.0, 0.0, 1.0)

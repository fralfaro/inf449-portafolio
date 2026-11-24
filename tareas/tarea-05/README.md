# Tarea 5: PCA multiespectral

## Archivos

- `homework5.ipynb`: analisis ejecutado de principio a fin, con formulas, resultados y figuras.
- `homework5.py`: funciones reproducibles para cargar los cubos, calcular PCA, reconstruir y guardar las metricas.
- `informe.tex` / `informe.pdf`: informe final compilado desde esta carpeta.
- `Fig/`: imagenes de resultados.
- `resultados/resultados_pca.json`: estadisticos y metricas completos, incluyendo resultados por cada imagen Tokio.
- `tokio-dataset/`: conjunto original de imagenes RGB.
- `referencia.png`: imagen guia Washington D. C.

## Ejecucion local

Desde esta carpeta, activar el entorno `.venv` del repositorio y ejecutar:

```powershell
..\..\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute homework5.ipynb --output homework5-ejecutado.ipynb --ExecutePreprocessor.kernel_name=inf449-portafolio --ExecutePreprocessor.timeout=300
```

El notebook usa `torch`, `numpy`, `pandas`, `Pillow`, `matplotlib` y `scikit-image`. Los productos matriciales y la diagonalizacion se ejecutan en CUDA cuando esta disponible; si no, el codigo utiliza CPU. No hay dependencias de Google Colab.

Para regenerar los resultados y las figuras directamente:

```powershell
..\..\.venv\Scripts\python.exe homework5.py
```

El informe compila con `pdflatex -interaction=nonstopmode informe.tex` desde esta carpeta. `extra/material_curso/homework4/` y `extra/material_oficial/` se leen como fuentes; no se modifican.

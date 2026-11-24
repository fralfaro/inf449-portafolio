# Tarea 2: ejecución local con GPU

El notebook detecta CUDA automáticamente y usa la GPU para entrenar las CNN y SIREN cuando PyTorch puede acceder a ella. El entorno único del repositorio, `.venv`, instala PyTorch con CUDA 12.8 y las dependencias de las tareas.

## Windows (PowerShell)

Desde la raíz del repositorio, crea el entorno con Python 3.11 e instala todo con `uv`:

```powershell
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
python -m ipykernel install --user --name inf449-portafolio --display-name "Python (INF449 Portafolio CUDA)"
python -m notebook tareas\tarea-02\homework2.ipynb
```

Comprueba que PyTorch ve la GPU:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Debe mostrar la versión `2.11.0+cu128`, `True` y el nombre de la GPU NVIDIA. En Jupyter, selecciona **Python (INF449 Portafolio CUDA)** y ejecuta todas las celdas en orden.

Los datos de entrada están en `tareas/tarea-02/`: `demo.dng`, `demo.jpg` y `PhotoCD_PCD0992/` con 24 imágenes Kodak. No se clonan repositorios ni se descargan datos durante la ejecución.

El entrenamiento completo realiza tres entrenamientos CNN de cien épocas y dos ajustes SIREN de mil pasos; puede tomar varios minutos. La ejecución genera métricas CSV en `tareas/tarea-02/resultados/` y conserva figuras en `tareas/tarea-02/Fig/`.

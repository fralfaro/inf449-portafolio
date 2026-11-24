# INF449 – Portafolio de tareas (Imagen Computacional)

Repositorio personal con el material y las tareas del curso INF449. Este archivo indica cómo debe ayudarme Claude con las tareas. Responder siempre en español neutro y formal.

## Estructura del repositorio

```
files/
  lecture1.pdf ... lecture6.pdf, ...   # clases (lecture4a y lecture4b incluidas)
  computationalImaging.pdf        # libro oficial del curso
  inf449.pdf                      # programa / información general del curso
extra/
  material_curso/                 # material del curso: dataset, homework1, homework2, ..., README.md
  material_oficial/               # notebooks oficiales del libro, por capítulo
    Chapter2ForwardModels, Chapter3ImageEstimation, Chapter4GeometricImaging,
    Chapter5WaveImaging, Chapter6CoherentSystems, Chapter7CoherenceImaging,
    Chapter8FocalImaging, Chapter9DigitalImaging, Chapter10SamplingStrategy,
    Chapter11Design, images
tareas/
  dataset/                        # datos compartidos por las tareas
  tarea-01 ... tarea-05/          # mi trabajo
  README.md
exposicion/  img/
```

Cada `tarea-XX/` contiene normalmente:

- `homeworkXX.pdf`: enunciado oficial (fuente de verdad de lo que se pide).
- `informe.tex`: informe que debo completar con los resultados.
- `*.ipynb`: notebook con los análisis realizados.

## Fuentes de referencia (usar en este orden)

1. **Enunciado**: `tareas/tarea-XX/homeworkXX.pdf`. Extraer cada ítem pedido y numerarlo; todo lo demás se mide contra esa lista.
2. **Clases**: `files/lecture?.pdf`. Identificar qué lecture(s) cubren el tópico de la tarea y qué ideas, fórmulas y notación se esperan.
3. **Avance o ayuda del curso**: `extra/material_curso/homework?/`. Revisar si existe material para esa tarea (la carpeta puede no existir; hoy hay homework1, homework2 y homework4). Leer también `extra/material_curso/README.md`.
4. **Notebooks oficiales**: `extra/material_oficial/Chapter*/`. Buscar notebooks que ilustren o resuelvan partes de la tarea (funciones, simulaciones, visualizaciones reutilizables).
5. **Libro**: `files/computationalImaging.pdf`. Indicar qué capítulos y secciones respaldan cada ítem de la tarea.

Notas de lectura:
- Los PDF son extensos: leer solo las páginas relevantes (buscar el tópico primero), no el documento completo.
- Citar siempre la fuente concreta (lecture y diapositiva, capítulo y sección, ruta del notebook).
- Si un material no existe para una tarea, decirlo explícitamente en lugar de suponerlo.

## Flujo A: tarea nueva o sin avance ("ayúdame con la tarea-XX")

Hacer un **análisis previo** antes de escribir código:

1. Resumen del enunciado: objetivo, ítems pedidos, entregables (informe, notebook, figuras) y restricciones (formato, librerías, fecha si aparece).
2. Mapa de fuentes: lecture(s) relevantes, capítulos del libro, notebooks oficiales útiles y material del curso si existe.
3. Conceptos teóricos necesarios para cada ítem, de forma breve.
4. Plan de trabajo propuesto: pasos del notebook, qué figuras o tablas producir y qué resultados deben ir a cada sección de `informe.tex`.
5. Riesgos o ambigüedades del enunciado que convenga confirmar conmigo.

No implementar la solución completa hasta que lo pida; ofrecer el plan y avanzar por partes.

## Flujo B: tarea ya avanzada ("valida la tarea-XX")

Revisar y entregar un informe con estas secciones:

1. **Cobertura**: tabla ítem del enunciado → dónde se resuelve en el notebook → dónde aparece en `informe.tex` (completo, parcial, faltante).
2. **Corrección técnica**: fórmulas, supuestos, unidades, parámetros, normalización, semillas, manejo de bordes e interpretación de resultados. Contrastar con lectures y libro.
3. **Notebook**: reproducibilidad (se ejecuta de arriba abajo), orden, celdas con ruido o código muerto, figuras con títulos, ejes y leyendas, texto explicativo en markdown.
4. **Informe (`informe.tex`)**: que no queden campos sin rellenar (`TODO`, `XXX`, `\ldots`, placeholders), que las figuras y tablas coincidan con los resultados del notebook, que las referencias y etiquetas compilen, y que las conclusiones se sustenten en los resultados.
5. **Mejoras priorizadas**: lista ordenada por impacto (crítico, importante, opcional), con la corrección sugerida y la fuente que la respalda.

## Reglas de trabajo

- No modificar `files/`, `extra/material_oficial/` ni `extra/material_curso/`; son material de referencia (solo lectura).
- No sobrescribir mis notebooks ni mi `informe.tex` sin avisar. Para validaciones, proponer los cambios y aplicarlos solo si se me pide.
- No inventar resultados numéricos ni figuras para el informe: todo valor debe provenir de una ejecución del notebook.
- Mantener el estilo y la notación de las clases y del libro.
- Si el notebook es costoso de ejecutar, avisar antes de correrlo.
- Al compilar el informe, usar la ruta de la tarea correspondiente y reportar errores de LaTeX de forma resumida.

## Mapa lecture → capítulos del libro

Completar la primera vez que se revise el material y mantenerlo actualizado:

| Lecture | Tópico | Capítulos / secciones del libro | Notebooks relacionados |
|---|---|---|---|
| lecture1 | Fotografía computacional y arreglo de filtros Bayer, diapositiva 9 | Cap. 3, estimación de imágenes y demosaicking, pp. 116-121 | `Chapter3ImageEstimation/demosaickingSiren.ipynb`, `cnnDemosaicing.ipynb` |
| lecture2 | _por completar_ | | |
| lecture3 | _por completar_ | | |
| lecture4a / 4b | lecture4a: modelo Bayer y demosaicking, diapositivas 6 y 8; lecture4b: referencias a demosaicking direccional | Cap. 3, demosaicking, pp. 116-121 | `Chapter3ImageEstimation/demosaickingSiren.ipynb`, `cnnDemosaicing.ipynb` |
| lecture5 | _por completar_ | | |
| lecture6 | _por completar_ | | |

## Mapa tarea → fuentes

| Tarea | Lecture(s) | Material del curso | Notebooks oficiales | Capítulos del libro | Estado |
|---|---|---|---|---|---|
| tarea-01 | lecture1 (diap. 9), lecture4a (diap. 6, 8), lecture4b (referencias) | `extra/material_curso/homework1/` | `extra/material_oficial/Chapter3ImageEstimation/demosaickingSiren.ipynb`, `cnnDemosaicing.ipynb` | Cap. 3, demosaicking, pp. 116-121 | revisada: demosaicking RGGB bilineal y bonus Malvar2004; ejecución local documentada |
| tarea-02 | lecture1 (diap. 9), lecture4a (diap. 6, 8), lecture4b (CNN y demosaicking) | `extra/material_curso/homework2/` (`homework2a.ipynb`) | `extra/material_oficial/Chapter3ImageEstimation/cnnDemosaicing.ipynb`, `demosaickingSiren.ipynb` | Cap. 3, estimación de imágenes y demosaicking neuronal, pp. 116-121 | revisada: tarea-02 local con PyTorch CUDA, split por imagen 20/4, CNN y SIREN; corrida completa en RTX 4060 |
| tarea-03 | | (no existe) | | | _por revisar_ |
| tarea-04 | lecture4a (diap. 7–11; difracción y Zernike), lecture4b (diap. 6, 13; fase cúbica y desenfoque) | `extra/material_curso/homework4/` (bandas Washington DC y código MATLAB Zernike) | `extra/material_oficial/Chapter5WaveImaging/numericalDiffraction.ipynb`, `extra/material_oficial/Chapter3ImageEstimation/wienerFiltering.ipynb` | Cap. 5, §5.3 difracción (p. 196); ecuaciones de Wiener/RL según el enunciado | revisada: McMaster 18 imágenes, comparación propia/skimage, ecuación (2) Bayer; CUDA en FFT de PSF y Wiener; PDF recompilado |
| tarea-05 | lecture6 (diap. 2: imágenes multiespectrales, contexto) | `extra/material_curso/homework4/DIP3E_CH11_Original_Images/` (seis bandas Washington DC); `tareas/tarea-05/referencia.png` (referencia visual); no hay guía homework5 en material_curso | `extra/material_oficial/Chapter10SamplingStrategy/PrincipalComponentsOfImages.ipynb`, `multiSpectralImages.ipynb` | Brady, *Computational Imaging*, cap. 4, imágenes espectrales; el DIP citado en homework5, pp. 861–867, no está en `files/` | revisada: PCA poblacional, 6 bandas y 10 Tokio; métricas PSNR/SSIM/SAM; notebook local ejecutado con CUDA; informe PDF compilado |

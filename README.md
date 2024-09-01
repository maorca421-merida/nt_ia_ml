# Diplomado en Machine Learning - Actividades Prácticas

Este repositorio contiene los archivos de código para las actividades prácticas del módulo de Machine Learning en el Diplomado. Cada actividad está diseñada para fortalecer las habilidades de los participantes en la implementación y análisis de modelos de Machine Learning utilizando Python.

## Contenido del Repositorio

### 1. Actividad 1: Clasificadores Lineales
**Archivo**: `Actividad 1.1.py`

**Descripción**:  
En esta actividad, los participantes cargarán un conjunto de datos, lo dividirán en conjuntos de entrenamiento y prueba, y entrenarán dos modelos de clasificación lineal: una regresión logística y un SVM lineal. Luego, evaluarán el rendimiento de estos modelos utilizando varias métricas de clasificación, como precisión, recall, F1 Score y la matriz de confusión. Los participantes también modificarán parámetros clave para observar su impacto en los resultados.

### 2. Actividad 2: Clasificadores No Lineales
**Archivo**: `Actividad 2.1.py`

**Descripción**:  
Esta actividad explora los clasificadores no lineales aplicados a un conjunto de datos que no es linealmente separable. Los participantes entrenarán un modelo SVM con kernel RBF y un Árbol de Decisión, ajustando hiperparámetros como `gamma` y `max_depth`. El rendimiento se evaluará mediante métricas de clasificación, y los resultados se compararán con los obtenidos de modelos lineales.

### 3. Actividad 3: Aprendizaje No Supervisado - Clustering y Reducción de Dimensionalidad
**Archivo**: `Actividad 3.1.py`

**Descripción**:  
En esta actividad, los participantes aplicarán técnicas de aprendizaje no supervisado para realizar clustering en un conjunto de datos en 4 dimensiones. A través de la reducción de dimensionalidad con PCA, los datos se reducirán primero a 3D y luego a 2D, donde se aplicará K-Means. Se evaluará la calidad del clustering en cada espacio utilizando el Silhouette Score, y se comparará la varianza explicada en cada reducción dimensional.

## Instrucciones de Uso

### Opción 1: Clonar el Repositorio

1. **Clonar el Repositorio**:
   - Si prefieres clonar el repositorio completo en tu máquina local, puedes hacerlo utilizando Git:
     ```bash
     git clone https://github.com/tu-usuario/nombre-repositorio.git
     ```
   - Navega al directorio clonado:
     ```bash
     cd nombre-repositorio
     ```

2. **Ejecutar los Archivos**:
   - Asegúrate de tener instalado Python 3.x y las bibliotecas necesarias (`numpy`, `matplotlib`, `sklearn`).
   - Ejecuta cada archivo `.py` utilizando un IDE de tu preferencia (como PyCharm, VSCode, Spyder) o desde la consola:
     ```bash
     python Actividad 1.1.py
     ```

### Opción 2: Descargar los Archivos Directamente

1. **Descargar los Archivos**:
   - Navega a la página del repositorio en GitHub y descarga los archivos `.py` que necesitas. Puedes hacerlo seleccionando los archivos y haciendo clic en el botón "Download" o descargando el repositorio completo como un archivo ZIP.

2. **Extraer y Ejecutar los Archivos**:
   - Extrae los archivos `.py` en tu computadora.
   - Asegúrate de tener instalado Python 3.x y las bibliotecas necesarias (`numpy`, `matplotlib`, `sklearn`).
   - Ejecuta los archivos utilizando un IDE de tu preferencia (recomendado para ver las gráficas correctamente) o desde la consola:
     ```bash
     python Actividad 1.1.py
     ```

3. **Interpretar los Resultados**:
   - Sigue las instrucciones dentro de cada archivo para entender los resultados generados y cómo interpretar las métricas y gráficos.

## Requisitos

- **Python 3.x**
- **Bibliotecas**:
  - `numpy`
  - `matplotlib`
  - `sklearn`

## Contribuciones

Si encuentras algún problema o tienes sugerencias para mejorar las actividades, siéntete libre de abrir un issue o realizar un pull request.

## Contacto

Para cualquier consulta relacionada con el contenido del diplomado, por favor contacta a [tu nombre] a través de [tu correo electrónico].

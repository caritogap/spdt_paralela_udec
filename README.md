# SPDT Paralela UdeC

Implementación de un algoritmo de árboles de decisión paralelos basado en SPDT, desarrollada para el proyecto semestral de la asignatura **Introducción a la Computación Paralela**.

## Datasets

Para reproducir los resultados, se deben descargar los datasets desde el siguiente enlace:

https://drive.google.com/drive/folders/1tFtXRuk7t0gG2dYulkIuagihbRpAIlLu?usp=sharing

Los archivos descargados deben guardarse en una carpeta llamada `data`, ubicada en el mismo directorio que el código:

```text
spdt_paralela_udec/
├── data/
├── main.cpp
├── main_parallel_hists.cpp
└── ...
```

Los datasets disponibles son:

- `adult`
- `isolet`
- `letter`
- `epsilon`
- `covtype`

## Versión secuencial

Para compilar la versión secuencial, se utiliza el siguiente comando:

```bash
g++ main.cpp -fopenmp -O3 -o main
```

Para ejecutar el programa:

```bash
./main dataset
```

El argumento `dataset` corresponde al nombre del conjunto de datos que se desea procesar. Por ejemplo:

```bash
./main adult
```

## Versión paralela

Para compilar la versión paralela:

```bash
g++ main_parallel_hists.cpp -fopenmp -O3 -o main_p
```

Para ejecutar el programa:

```bash
./main_p dataset threads
```

Donde:

- `dataset` es el nombre del conjunto de datos que se desea procesar.
- `threads` es un número entero que indica la cantidad de hebras utilizadas durante la construcción de los histogramas.

Por ejemplo, para procesar el dataset `epsilon` utilizando 8 hebras:

```bash
./main_p epsilon 8
```

## Descripción de archivos

- `main.cpp`: contiene la función `main` de la implementación secuencial.
- `main_parallel_hists.cpp`: contiene la función `main` de la implementación paralela.
- `histogram.hpp` y `histogram.cpp`: contienen la definición y la implementación de los métodos de la clase `Histogram`.
- `constructhist.cpp`: contiene las funciones relacionadas con la construcción y combinación de los histogramas.
- `gainandimpurity.cpp`: contiene las funciones relacionadas con la construcción del árbol, el cálculo de la impureza y la selección del mejor punto de división.

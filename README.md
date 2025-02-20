
Hace poco tuve las ganas de experimentar con algoritmos de [dithering](http://alex-charlton.com/posts/Dithering_on_the_GPU/) para imagenes. Siempre me meto de cabeza al tema pero me termino desviando, y esta no fue la excepcion.

Cuando se hace dithering se limita la paleta de colores, a veces se la selecciona con un proposito experimental para ver combinaciones llamativas y extravagantes. Otras veces se busca los colores que van a hacer que la imagen con dithering se parezca lo mas posible a la original.

De ahi vino mi pregunta, cuales son dichos colores? o mas bien, cuales son los k colores mas representativos de una imagen? Necesitamos tener una buena nocion de cercania entre colores para poder agruparlos... chatgpt me recomendo usar el espacio de colores LAB por sobre RGB, voy a probar con los 2, pero primero con LAB. Voy a usar un algoritmo de agrupamiento como [K-means](https://www.youtube.com/watch?v=4b5d3muPQmA&t=182s&ab_channel=StatQuestwithJoshStarmer) para identificar los k clusters de colores mas representativos (el color que representa cada clúster es el centroide del clúster, que se calcula como el promedio aritmético de todos los puntos de ese cluster).

### Imagen de Prueba:
![Imagen de prueba](data/test.png "")  

## LAB

![Imagen de prueba](data/colors_lab.png "")
![Imagen de prueba](data/clusters_lab.png "")


## RGB

![Imagen de prueba](data/colors_rgb.png "")
![Imagen de prueba](data/clusters_rgb.png "")

## Naive Dithering

Consiste en reemplazar cada color de la imagen por el mas cercano de nuestra paleta de colores (en este caso la obtenida con k-means). El output va a depender de nuestra funcion de cercania. En este caso usamos distancia euclideana sobre el espacio de color especifico. 

### Paleta LAB
![Imagen de prueba](data/naive_dither_lab.png "")


### Paleta RGB
![Imagen de prueba](data/naive_dither_rgb.png "")

## Aumentando la cantidad de colores

Los ejemplos anteriores eran usando la paleta de 8 colores, pero que pasa si usamos 16 colores? (Notemos que es todavia un numero muy bajito de colores, la imagen original tiene 322.015 colores distintos)

### Paleta LAB 16 colores.
![Imagen de prueba](data/clusters_lab_16.png "")

| Original    | Dither |
| -------- | ------- |
| ![Imagen de prueba](data/test.png "")  | ![Imagen de prueba](data/naive_dither_lab_16.png "")    |

### Paleta RGB 16 colores.
![Imagen de prueba](data/clusters_rgb_16.png "")

| Original    | Dither |
| -------- | ------- |
| ![Imagen de prueba](data/test.png "")  | ![Imagen de prueba](data/naive_dither_rgb_16.png "")   

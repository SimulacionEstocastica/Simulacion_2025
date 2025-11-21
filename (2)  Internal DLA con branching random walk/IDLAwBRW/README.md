# Internal DLA with Branching Random Walk (IDLA-BRW)

Este proyecto implementa un modelo de **Internal Diffusion Limited Aggregation (IDLA)** combinado con **Branching Random Walk (BRW)**. El objetivo es simular la formación de clusters mediante caminatas aleatorias que pueden ramificarse, generando estructuras emergentes a partir de dinámicas locales.

## 🧠 Descripción General

El concepto de DLA es el siguiente, particulas hacen caminata aleatoria hasta que "tocan" una estructura y luego se vuelven parte de ella, en el caso de Internal DLA, estas partículas se introducen al origen, caminan por un "cluster" hasta salir de él, momento en que "mueren" y pasan a ser parte del cluster.

Esto se puede interpretar como una grilla de "hoyos", donde vamos introduciendo pelotas que solo pueden caminar sobre los hoyos tapados, cuando se salen de esta "superficie tapada", tapan un hoyo.

![Que es Idla](https://github.com/newencastillo/IDLAwBRW/blob/main/gifs/IDLA.gif)

Introducimos una partícula al origen hasta que esta muera, momento en el que añadimos la siguiente y así sucesivamente, la pregunta que manejamos primeramente es ¿cómo crece este clúster?, ¿tiene forma regular?

Esta pregunta fue respondida y demostrada en 1970~ por [REF] y la respuesta ( para cualquier dimensión) es LA bola! Resultados que se pueden observar:


![Crece como Bola](https://github.com/newencastillo/IDLAwBRW/blob/main/gifs/ComoBola.gif)

## BRanching random Walk 

Nuestro primer objetivo fue replicar una cierta variación a este concepto propuesta por Silvestri etal [REF2]
Alteramos la caminata aleatoria normal de esta simulación por BRW (Branching random walk). Esto es que nuestras partículas, ahora deciden con prob. $p$,  entre morir o duplicarse, con sus "hijas" cada una haciendo el siguiente paso del RW.

![Que es BRW](https://github.com/newencastillo/IDLAwBRW/blob/main/gifs/Branching1.gif)

El resultado que se expone y que pretendemos replicar, es que con esta dinámica, no se mantiene el comprotamiento asintótico sobre la forma de "la mancha", si no más bien que se forman "cototitos". lo que comprobamos rápidamente:

![COTOTOS](https://github.com/newencastillo/IDLAwBRW/blob/main/gifs/BigCrit.gif)

Lo interesante es que esto se cumple solo para el caso crítico, con $p = 1/2$, con valores mas bajos, o más altos, se vuelve a obtener el resultado del crecimiento asintótico, considerando que estas simulaciones son además muy costosas; cuando muerern mucho, la bola ya "no crece" desde cierto N (mentira), cuando se duplican mucho, la cantidad de particulas explota muy rápido, dificultando la simulación.

![](https://github.com/newencastillo/IDLAwBRW/blob/main/gifs/SubCrit.gif)
![](https://github.com/newencastillo/IDLAwBRW/blob/main/gifs/Crit.gif)
![](https://github.com/newencastillo/IDLAwBRW/blob/main/gifs/SuperCrit.gif)

## Lo percolación (Parte 2)

Cambiamos nuestro mundo, nuestro espacio donde esta ocurriendo esta recreación

Se implementa una *percolación* sobre la malla de $\mathbb{Z}^2$, esto es, quitar aristas para restringir el movimiento posible de las partículas.
La idea será eliminar aleatoriamente aristas en un campo finito (suficientemente grande para la simulación), con distintas probabilidades de eliminación para los "movimientos" verticales y los horizontales.

Es de esperar que de esta forma, el "cluster" que se formará asintóticamente será una elipse, y sus parámetros que lo definen estarán relacionados con la elección de las probabilidades para las aristas horizontales y verticales

![Obalo](https://github.com/newencastillo/IDLAwBRW/blob/main/gifs/PercOvalo.gif)



Queremos poner a prueba que sucede con el branching => Comportamiento esperado
![](https://github.com/newencastillo/IDLAwBRW/blob/main/gifs/3000BranchingPerc.gif)
![](https://github.com/newencastillo/IDLAwBRW/blob/main/newen.gif)

Trabajando en esto nos dimos cuenta de dos cosas:
  1. Acercar el valor de $p_v + p_h$ a 1 produce resultados ¿inesperados?

  **PERCOTOTOS:**
    
![Percototos](https://github.com/newencastillo/IDLAwBRW/blob/main/gifs/PercototosBolaLight.gif)

![Percototos](https://github.com/newencastillo/IDLAwBRW/blob/main/gifs/PercototosBola.gif)

![Percototos3](https://github.com/newencastillo/IDLAwBRW/blob/main/5000%20no%20branch.png)

  1. CUÁLES SON LOS PARÁMETROS A Y B
     
     A esto se le dedicó un buen tiempo computacional, se implementó un algoritmo tipo montecarlo para encontrar el factor de "escala" del óvalo como elipse, calculando los valores de a y b de varias simulaciones.

$$A = \pi a b = \pi (a*S)(b*S) \Rightarrow S \approx \sqrt{\frac{N}{\pi ab}}$$

  El encontrar una funcion que aproxime bien este factor de escala, dependiendo de $p_v$ y $p_h$ probó ser un verdadero desafío, tanto por la forma de la función, como un posible factor de escala que la acompañe.

  Nuestro mejor resultado es el utilizado para dibujar los elipses en las simulaciones y es $a \aprox \frac{1-p_v}{p_v+p_h}$, que no funciona tan bien para elipses muy "estiradas", en verdad sigue siendo muy bueno

  
![Ovalo](https://github.com/newencastillo/IDLAwBRW/blob/main/gifs/OvaloMalo.gif)



## ✅ Estado Actual

- Implementación base en **Python**.
- Simulación central de Random walk con posibilidad de branching.
- Clase de BRW 
  + restricción de IDLA
  + restricción de Percolationi
- Simulación con animación
  + Herramienta para fabricar gifs



### notas propias

*El problema fundamental*

Ya no hay problema fundamental

Este readme  NO fue creado (en parte) con inteligencia artificiel
usos de ia: la primera version del readme que ya casi ni existe y la funcion validar percolacion pq que lata y las maquetas para hacer las animaciones de matplot.


## TODO s

- organizar bien el proyecto () (onda hacer un __main__)

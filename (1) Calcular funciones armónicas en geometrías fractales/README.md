Este repositorio corresponde a el proyecto de "Calcular funciónes harmonicas sobre dominios fractales", hecho por Alonso Núñez C. y Antonia Valenzuela C.

# Contenidos

En la carpeta "Campos" se encuentra los campos originales que fueron entregados por el profesor Avelio Sepúlveda, mientras que en "Campos2" se encuentran los que fueron generados mediante el script de python "comprimir.py" y el ejecutable de C++ "testGenerator.cpp".

La mayoría de los resultados provienen del campo "Campo1real10.txt", estos resultados son campos que se guardaron como archivos .txt en la carpeta "outputs". Para graficar estos resultado se ocupa el scrip "graficarDatos.py", que al ejecutarse da las instrucciones de uso.

Tambien está el script "crearContornoSinusoidal.py", como su nombre indica crea las condiciones de borde para generar el ejemplo "tipico".

# Ejecución

El código principal del proyecto fue hecho en C++, esto por la ventaja de rapidez y dado que permite generar ejecutables que no necesitan recursos adicionales para correr en Windows. Para ejecutar directamente están los archivos "calcularCampo.exe" y "crearTests.exe", estos corresponden a el compilado de "harmonic.cpp" y "testGenerator.cpp" respectivamente. Si se desea ejecutar en otro ambiente estos deben ser compilados nuevamente. 

En la carpeta "inputs" se encuentran inputs de ejemplo para la aplicación "calcularCampo.exe". "tipico.in" toma alrededor de 6 o 7 horas, mientras que "fractal.in" toma alrededor de 3 horas.

Los scripts de python son autocontenidos salvo los inputs pedidos, estos se tienen que entregar mediante inputs que serán solicitados al ejecutar el script.
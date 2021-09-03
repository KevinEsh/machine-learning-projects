Para conseguir más datos del historial de valores de acciones del mercado Americano,
puede visitar la pagina Yahoo Finance: https://finance.yahoo.com/ y descargar los activos
de su preferencia. Asegurese de que el historial contenga la información concerniente a un
año de observaciones (251 observaciones). Deposite los .csv en la carpeta "Data" de este 
mismo folder.

Para calcular el retorno de todos sus datos y almacenarlo en un archivo .csv, ejecute el Notebook
"get_retuns.ipynb" con ubicado en la carpeta "Data" de la siguiente manera:

	-> jupyter notebook get_returns.ipynb

Su base de datos se guardará en la carpeta "Returns"

Como paso final, si quieres ejecutar el programa principal para la seleccion de portafolios, solamente
ejecute el script "main.py". No necesita de argumentos por consola. 

	-> python main.py

El programa imprimirá en pantalla el radio de Sharpie del mejor individuo de la población del algoritmo
genético de cada generación. Finalmente obtendrá la gráfica del retorno esperado vs volatilidad de cada
portafolio conseguido. Las cruces (x) indican un portafolio conseguido por un individuo del GA. Mientras
que los circulos (o) son portafolios generados de forma aleatoria; unicamente ploteados para que note como 
los individuos del GA superan el intento más naïve de asignación de capital a activos. 

Si desea modificar los resultados conseguidos, unicamente modifique los parámtros de GA en el mismo script
"main.py".
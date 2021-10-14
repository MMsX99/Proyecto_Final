# Proyecto_Final
PROYECTO GESTOS ARROJAR O DEJAR CAER BASURA.

Para vectores con referencia en 0(WRIST)
Se definen dos gestos: inicial y final para ambos casos:
1.	Sostener un objeto,
2.	Objeto soltado
Para el gesto 2 los vectores MCP son cercanos a ser paralelos (aproximadamente) a los vectores de la palma; los vectores PIP-MCP paralelos a MCP, y así sucesivamente hasta llegar a los puntos TIP.


1.	Sostener basura:
Al sostener basura, el Angulo entre los vectores falange MCP-PIP y el plano de la palma de la mano tiene un valor entre 0 y 90 grados.
Se crearon vectores con referencia en 0(WRIST) de los puntos 1,5,9,13,17( puntos MCP).
se compara el angulo entre estos vectores y los vectores PIP-MCP.
Se realiza este proceso sucesivamente en cada falange hasta llegar a los puntos TIP

La diferenciación va a ser en el gesto intermedio, en el caso de arrojar basura se toma el gesto “impulso”. Tengo pensado obtener este gesto visualizando la posición general de toda la mano con respecto por ejemplo al origen, y ver si hubo una cantidad relativa de variación en la posición hacia una dirección(tomar impulso), y después una cantidad relativa de variación en la posición en el sentido contrario(lanzar).
Estas cantidades de variación se trataran de relacionar en porcentajes del tamaño de la mano. Ya que pues en cada posible video de una mano variaran los tamaños.




referencias:  https://google.github.io/mediapipe/solutions/hands#hand-landmark-model

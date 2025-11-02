def reconstruir_camino(next_node: list, origen: int, destino: int) -> list:
    """Reconstruye el camino desde `origen` hasta `destino` usando la matriz `next_node`.
    Devuelve una lista de nodos (incluyendo origen y destino). Si no hay camino devuelve []"""
    if next_node[origen][destino] is None:
        return []
    path = [origen]
    u = origen
    # Avanzar hasta llegar a destino
    while u != destino:
        u = next_node[u][destino]
        # Proteccion por seguridad (evita bucles infinitos)
        if u is None:
            return []
        path.append(u)
    return path


def floydWarshallConCaminos(matrizDeAdyacencia: list) -> tuple:
    """Versión de Floyd-Warshall que además devuelve los caminos.

    Retorna una tupla (distancia, caminos) donde:
    - distancia es la matriz de distancias mínimas (float, inf si no hay camino)
    - caminos es una matriz donde caminos[i][j] es una lista de nodos que forman
      el camino mínimo desde i hasta j (vacía si no hay camino)

    La entrada `matrizDeAdyacencia` se interpreta como matriz de adyacencia
    con 0 indicando ausencia de arista (salvo la diagonal que se considera 0).
    """
    n = len(matrizDeAdyacencia)
    distancia = [[float('inf')] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                distancia[i][j] = 0
                next_node[i][j] = i
            elif matrizDeAdyacencia[i][j] != 0:
                distancia[i][j] = matrizDeAdyacencia[i][j]
                next_node[i][j] = j

    for k in range(n):
        for i in range(n):
            for j in range(n):
                # Evitar sumar infinitos
                if distancia[i][k] == float('inf') or distancia[k][j] == float('inf'):
                    continue
                if distancia[i][j] > distancia[i][k] + distancia[k][j]:
                    distancia[i][j] = distancia[i][k] + distancia[k][j]
                    next_node[i][j] = next_node[i][k]

    # Construir matriz de caminos (listas de nodos)
    caminos = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if distancia[i][j] == float('inf'):
                caminos[i][j] = []
            else:
                caminos[i][j] = reconstruir_camino(next_node, i, j)

    return distancia, caminos


def hubEnCamino(camino: list, hubs: list) -> bool:
    """Devuelve True si en el camino hay al menos un hub."""
    for nodo in camino:
        if nodo in hubs:
            return True
    return False

def recogerPaquete(capacidadActual:int, cantidadDeposito:int,capacidadMaxima:int)->int:
    """
        Devuelve la nueva capacidad del camion despues de recoger paquetes del deposito.
        Si no se pueden recoger todos los paquetes, devuelve la cantidad de paquetes que quedan en el deposito.
        Parametros:
        capacidadActual: Capacidad actual del camión.
        cantidadDeposito: Cantidad de paquetes en el depósito.
        capacidadMaxima: Capacidad máxima del camión.
        Salida: (nuevaCapacidad, paquetesRestantesEnDeposito)
    """
    espacioDisponible=capacidadMaxima-capacidadActual
    if cantidadDeposito>=espacioDisponible:
        return capacidadMaxima, cantidadDeposito-espacioDisponible
    else:
        return capacidadActual+cantidadDeposito, 0
# funciones.py
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# ============================================================
#  Floyd–Warshall con reconstrucción de caminos
# ============================================================

def reconstruir_camino(next_node: List[List[Optional[int]]], origen: int, destino: int) -> List[int]:
    """Reconstruye el camino desde `origen` hasta `destino` usando la matriz `next_node`.
    Devuelve una lista de nodos (incluyendo origen y destino). Si no hay camino devuelve []."""
    if next_node[origen][destino] is None:
        return []
    path = [origen]
    u = origen
    # Avanzar hasta llegar a destino
    while u != destino:
        u = next_node[u][destino]
        # Protección por seguridad (evita bucles infinitos)
        if u is None:
            return []
        path.append(u)
    return path


def floydWarshallConCaminos(matrizDeAdyacencia: List[List[float]]) -> Tuple[List[List[float]], List[List[List[int]]]]:
    """Versión de Floyd–Warshall que además devuelve los caminos.

    Retorna una tupla (distancia, caminos) donde:
      - distancia es la matriz de distancias mínimas (float, inf si no hay camino)
      - caminos[i][j] es una lista de nodos que forman el camino mínimo i->j (vacía si no hay camino)

    La entrada `matrizDeAdyacencia` se interpreta como matriz de adyacencia
    con 0 indicando ausencia de arista (salvo la diagonal que se considera 0).
    """
    n = len(matrizDeAdyacencia)
    distancia = [[float('inf')] * n for _ in range(n)]
    next_node: List[List[Optional[int]]] = [[None] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                distancia[i][j] = 0.0
                next_node[i][j] = i
            elif matrizDeAdyacencia[i][j] != 0:
                distancia[i][j] = matrizDeAdyacencia[i][j]
                next_node[i][j] = j

    for k in range(n):
        for i in range(n):
            dik = distancia[i][k]
            if dik == float('inf'):
                continue
            for j in range(n):
                kj = distancia[k][j]
                if kj == float('inf'):
                    continue
                nd = dik + kj
                if nd < distancia[i][j]:
                    distancia[i][j] = nd
                    next_node[i][j] = next_node[i][k]

    # Construir matriz de caminos (listas de nodos)
    caminos: List[List[List[int]]] = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if distancia[i][j] == float('inf'):
                caminos[i][j] = []
            else:
                caminos[i][j] = reconstruir_camino(next_node, i, j)

    return distancia, caminos


# ============================================================
#  Utilidades varias que ya tenías
# ============================================================

def hubEnCamino(camino: List[int], hubs: List[int]) -> bool:
    """Devuelve True si en el camino hay al menos un hub."""
    set_hubs = set(hubs)
    for nodo in camino:
        if nodo in set_hubs:
            return True
    return False


def recogerPaquete(capacidadActual: int, cantidadDeposito: int, capacidadMaxima: int) -> Tuple[int, int]:
    """
    Devuelve la nueva capacidad del camión y los paquetes restantes en depósito
    después de recoger paquetes.

    Parámetros:
      - capacidadActual: Capacidad actualmente ocupada del camión.
      - cantidadDeposito: Cantidad de paquetes disponibles en el depósito.
      - capacidadMaxima: Capacidad total del camión.

    Salida: (nuevaCapacidad, paquetesRestantesEnDeposito)
    """
    espacioDisponible = capacidadMaxima - capacidadActual
    if espacioDisponible <= 0:
        return capacidadActual, cantidadDeposito

    if cantidadDeposito >= espacioDisponible:
        return capacidadMaxima, cantidadDeposito - espacioDisponible
    else:
        return capacidadActual + cantidadDeposito, 0


# ============================================================
#  Nueva parte: Backtracking Opción A – Ruteo Básico
# ============================================================

@dataclass
class Solucion:
    """Estructura para guardar la mejor solución encontrada."""
    distancia: float = float('inf')
    ruta: List[int] = None

    def set(self, dist: float, ruta: List[int]) -> None:
        self.distancia = dist
        self.ruta = ruta.copy()


def resolver_opcion_a_backtracking(
    matriz_distancias: List[List[float]],
    deposito_id: int,
    hubs: List[int],
    demanda_por_nodo: Dict[int, int],
    capacidad_camion: int,
    forzar_regreso_al_deposito: bool = False,
) -> Solucion:
    """
    Explora por backtracking la mejor ruta (menor distancia total) para entregar
    todos los paquetes cumpliendo capacidad. Puede recargar en cualquier hub o
    en el depósito sin costo adicional (más allá del viaje hasta allí).

    Parámetros:
      - matriz_distancias: matriz de distancias mínimas (salida de floydWarshallConCaminos()[0])
      - deposito_id: id del nodo depósito (origen)
      - hubs: lista de ids de hubs (todos activos en Opción A)
      - demanda_por_nodo: dict {nodo_destino: cantidad_paquetes}
      - capacidad_camion: capacidad del camión
      - forzar_regreso_al_deposito: si True, obliga a volver al depósito al terminar

    Devuelve:
      - Solucion(distancia, ruta)
    """
    n = len(matriz_distancias)
    demanda = demanda_por_nodo.copy()
    total_restante = sum(demanda.values())
    nodos_recarga = set(hubs) | {deposito_id}

    mejor = Solucion()
    ruta_inicial: List[int] = [deposito_id]

    # -----------------------------------------------
    # Bound optimista muy barato (poda simple)
    # -----------------------------------------------
    def bound_optimista(u: int, restante: int) -> float:
        if restante == 0:
            return 0.0
        minimo = float('inf')
        for v, cnt in demanda.items():
            if cnt > 0:
                d = matriz_distancias[u][v]
                if d < minimo:
                    minimo = d
        return 0.0 if minimo == float('inf') else minimo

    # -----------------------------------------------
    # Cerrar ruta (opcionalmente forzar volver al depósito)
    # -----------------------------------------------
    def cerrar_ruta(dist_actual: float, u: int, ruta: List[int]) -> Tuple[float, List[int]]:
        if not forzar_regreso_al_deposito:
            return dist_actual, ruta
        d = matriz_distancias[u][deposito_id]
        if d == float('inf'):
            return float('inf'), ruta  # imposible volver
        nueva_ruta = ruta.copy()
        if u != deposito_id:
            nueva_ruta.append(deposito_id)
        return dist_actual + d, nueva_ruta

    # -----------------------------------------------
    # Backtracking
    # -----------------------------------------------
    # Contadores para depuración del backtracking
    contador_llamadas = 0
    intervalo_report = 1000

    def bt(u: int, carga: int, restante: int, dist: float, ruta: List[int]) -> None:
        nonlocal mejor, contador_llamadas
        contador_llamadas += 1
        if contador_llamadas % intervalo_report == 0:
            print(f"[DEBUG] llamadas={contador_llamadas:,} | mejor={mejor.distancia:.2f} | restante={restante}")

        # Poda por mejor conocido
        if dist >= mejor.distancia:
            return

        # Poda por bound optimista
        if dist + bound_optimista(u, restante) >= mejor.distancia:
            return

        # Caso final: todo entregado y camión vacío
        if restante == 0 and carga == 0:
            dist_final, ruta_final = cerrar_ruta(dist, u, ruta)
            if dist_final < mejor.distancia:
                mejor.set(dist_final, ruta_final)
            return

        # Si no tengo carga pero quedan paquetes -> ir a recargar (hub o depósito)
        if carga == 0:
            if restante == 0:
                dist_final, ruta_final = cerrar_ruta(dist, u, ruta)
                if dist_final < mejor.distancia:
                    mejor.set(dist_final, ruta_final)
                return

            # Cantidad a cargar (todo lo que permita el camión o lo que falte, lo que sea menor)
            para_cargar = min(capacidad_camion, restante)

            # Si ya estoy parado sobre un nodo de recarga, puedo cargar sin moverme
            if u in nodos_recarga:
                bt(u, para_cargar, restante, dist, ruta)

            # Alternativamente, moverme a otro nodo de recarga
            for r in nodos_recarga:
                if r == u:
                    continue
                d = matriz_distancias[u][r]
                if d == float('inf'):
                    continue  # sin camino
                ruta.append(r)
                bt(r, para_cargar, restante, dist + d, ruta)
                ruta.pop()
            return

        # Tengo carga > 0: elegir un destino con demanda > 0 y entregar lo máximo posible
        for destino, cnt in list(demanda.items()):
            if cnt <= 0:
                continue
            d = matriz_distancias[u][destino]
            if d == float('inf'):
                continue  # sin camino

            entrego = min(carga, cnt)  # en Opción A conviene entregar al máximo

            # Aplicar movimiento y entrega
            demanda[destino] -= entrego
            nueva_carga = carga - entrego
            nuevo_restante = restante - entrego

            ruta.append(destino)
            bt(destino, nueva_carga, nuevo_restante, dist + d, ruta)
            ruta.pop()

            # Deshacer
            demanda[destino] += entrego

    # Inicio en depósito, sin carga
    bt(deposito_id, 0, total_restante, 0.0, ruta_inicial)
    return mejor

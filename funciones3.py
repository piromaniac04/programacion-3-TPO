# funciones.py
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from math import ceil, sqrt

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
#  Utilidades varias
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
#  Backtracking Opción A – Ruteo Básico (con early-stop auto-escalable)
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
    # --- parámetros auto-escalables para early-stop y reporting ---
    max_llamadas_sin_mejora: Optional[int] = None,
    intervalo_report: Optional[int] = None,
) -> Solucion:
    """
    Explora por backtracking la mejor ruta (menor distancia total) para entregar
    todos los paquetes cumpliendo capacidad. Puede recargar en cualquier hub o
    en el depósito sin costo adicional (más allá del viaje hasta allí).

    Early-stop por 'meseta': si pasan `max_llamadas_sin_mejora` llamadas sin
    mejorar la mejor distancia, se corta y se devuelve la mejor hallada.
    Este umbral se puede pasar o se calcula automáticamente en función del tamaño.

    Retorna:
      - Solucion(distancia, ruta)
    """
    n = len(matriz_distancias)
    demanda = demanda_por_nodo.copy()
    total_restante = sum(demanda.values())
    nodos_recarga = set(hubs) | {deposito_id}

    mejor = Solucion()
    ruta_inicial: List[int] = [deposito_id]

    # -----------------------------------------------
    # Umbrales auto-escalables (meseta y reporte)
    # -----------------------------------------------
    m = sum(1 for _, cnt in demanda.items() if cnt > 0) or 1  # destinos con demanda
    T = ceil(total_restante / max(1, capacidad_camion)) or 1   # viajes mínimos

    def _auto_meseta(n_: int, m_: int, T_: int) -> int:
        """
        Heurística: el branching real lo marcan los destinos con demanda (m) y los viajes (T).
        n entra suave con sqrt(n). 'base' ajusta agresividad de corte.
        """
        base = 300  # subir para ser más conservador; bajar para cortar antes
        val = int(base * sqrt(max(1, n_)) * m_ * max(1, T_))
        # Encajonar para evitar extremos
        return max(5_000_000, val)

    if max_llamadas_sin_mejora is None:
        max_llamadas_sin_mejora = _auto_meseta(n, m, T)

    if intervalo_report is None:
        # reporte ~100 veces por búsqueda; mínimo 1000 para no inundar stdout
        intervalo_report = max(1_000, max_llamadas_sin_mejora // 100)

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
    # Backtracking + early-stop por meseta
    # -----------------------------------------------
    contador_llamadas = 0
    llamadas_desde_mejora = 0
    stop = False  # bandera global para cortar

    def bt(u: int, carga: int, restante: int, dist: float, ruta: List[int]) -> None:
        """
            Función recursiva de backtracking para encontrar la mejor ruta de entrega.

            Parámetros:
                u (int): Nodo actual en el que se encuentra el camión
                carga (int): Cantidad de paquetes que lleva el camión actualmente
                restante (int): Cantidad total de paquetes que faltan entregar
                dist (float): Distancia acumulada hasta el momento
                ruta (List[int]): Lista de nodos visitados en orden

            Efectos:
                - Actualiza la variable global 'mejor' cuando encuentra una solución mejor
                - Actualiza contadores y puede activar early-stop
                - Modifica el diccionario 'demanda' temporalmente (con backtracking)

            Returns:
                None: La función actualiza el estado global pero no retorna valores
        """
        nonlocal mejor, contador_llamadas, llamadas_desde_mejora, stop #Posible mejora, que estas variables esten en un data class
        if stop:
            return

        # Señales de vida + contadores
        contador_llamadas += 1
        llamadas_desde_mejora += 1
        if intervalo_report > 0 and (contador_llamadas % intervalo_report == 0):
            print(
                f"[DEBUG] llamadas={contador_llamadas:,} | sin_mejora={llamadas_desde_mejora:,} | "
                f"mejor={mejor.distancia:.2f} | restante={restante} | nodo={u} | "
                f"umbral={max_llamadas_sin_mejora:,}"
            )

        # Early-stop por meseta
        if llamadas_desde_mejora >= max_llamadas_sin_mejora:
            print(
                f"[EARLY-STOP] sin mejora en {llamadas_desde_mejora:,} llamadas "
                f"(umbral={max_llamadas_sin_mejora:,}). Devuelvo mejor={mejor.distancia:.2f}"
            )
            stop = True
            return

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
                llamadas_desde_mejora = 0  # ¡hubo mejora! reseteo la meseta
            return

        # Si no tengo carga pero quedan paquetes -> ir a recargar (hub o depósito)
        if carga == 0:
            para_cargar = min(capacidad_camion, restante)

            # Si ya estoy en nodo de recarga, cargar sin moverse
            if u in nodos_recarga:
                bt(u, para_cargar, restante, dist, ruta)
                if stop:
                    return

            # O moverse a otro nodo de recarga
            for r in nodos_recarga:
                if stop:
                    return
                if r == u:
                    continue
                d = matriz_distancias[u][r]
                if d == float('inf'):
                    continue  # sin camino
                ruta.append(r)
                bt(r, para_cargar, restante, dist + d, ruta)
                ruta.pop()
            return

        # Tengo carga > 0: elegir un destino con demanda > 0 y entregar al máximo
        for destino, cnt in list(demanda.items()):
            if stop:
                return
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

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from math import ceil, sqrt

# ============================================================
#  Modelos / Dataclasses
# ============================================================

@dataclass
class Solucion:
    """Guarda la mejor solución encontrada (ruta compacta entre nodos clave)."""
    distancia: float = float('inf')
    ruta: List[int] = None

    def set(self, dist: float, ruta: List[int]) -> None:
        self.distancia = dist
        self.ruta = ruta.copy()

@dataclass
class EstadoBT:
    """Estado mutable del backtracking (reemplaza nonlocals)."""
    mejor: Solucion
    contador_llamadas: int = 0
    llamadas_desde_mejora: int = 0
    stop: bool = False
    max_llamadas_sin_mejora: int = 0
    intervalo_report: int = 0


# ============================================================
#  Floyd–Warshall con reconstrucción de caminos
# ============================================================

def reconstruir_camino(next_node: List[List[Optional[int]]], origen: int, destino: int) -> List[int]:
    """Reconstruye el camino desde `origen` hasta `destino` usando la matriz `next_node`."""
    if next_node[origen][destino] is None:
        return []
    path = [origen]
    u = origen
    while u != destino:
        u = next_node[u][destino]
        if u is None:
            return []
        path.append(u)
    return path


def floydWarshallConCaminos(matrizDeAdyacencia: List[List[float]]) -> Tuple[List[List[float]], List[List[List[int]]]]:
    """Floyd–Warshall que devuelve matriz de distancias y caminos mínimos (listas de nodos)."""
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

    caminos: List[List[List[int]]] = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if distancia[i][j] != float('inf'):
                caminos[i][j] = reconstruir_camino(next_node, i, j)
    return distancia, caminos


# ============================================================
#  Utilidades varias
# ============================================================

def hubEnCamino(camino: List[int], hubs: List[int]) -> bool:
    """Devuelve True si en el camino hay al menos un hub."""
    set_h = set(hubs)
    return any(n in set_h for n in camino)


def recogerPaquete(capacidadActual: int, cantidadDeposito: int, capacidadMaxima: int) -> Tuple[int, int]:
    """Devuelve nueva capacidad del camión y paquetes restantes en depósito."""
    espacioDisponible = capacidadMaxima - capacidadActual
    if espacioDisponible <= 0:
        return capacidadActual, cantidadDeposito
    if cantidadDeposito >= espacioDisponible:
        return capacidadMaxima, cantidadDeposito - espacioDisponible
    return capacidadActual + cantidadDeposito, 0


# ============================================================
#  Heurísticas / Bounds / Chequeos (top-level)  # [REFACTOR]
# ============================================================

def auto_meseta(n: int, m: int, T: int) -> int:
    """Umbral auto-escalable para early-stop por meseta."""
    base = 1300
    val = int(base * sqrt(max(1, n)) * m * max(1, T))
    return max(50_000, min(5_000_000, val))


def cerrar_ruta(dist_actual: float,
                u: int,
                ruta: List[int],
                deposito_id: int,
                matriz_distancias: List[List[float]]) -> Tuple[float, List[int]]:
    """Cierra la ruta volviendo siempre al depósito (si es posible).

    Nota: se fuerza el regreso al depósito independientemente de flags externos.
    """
    d = matriz_distancias[u][deposito_id]
    if d == float('inf'):
        return float('inf'), ruta
    nueva_ruta = ruta.copy()
    if u != deposito_id:
        nueva_ruta.append(deposito_id)
    return dist_actual + d, nueva_ruta


def greedy_seed(matriz_distancias: List[List[float]],
                deposito_id: int,
                nodos_recarga: set,
                demanda: Dict[int, int],
                capacidad_camion: int) -> Solucion:
    """
    Construye una solución inicial (no óptima) para apretar la cota superior del BT.
    Ruta compacta: depósito/hub/destinos sin expandir intermedios.
    """
    dem = demanda.copy()
    total_restante = sum(dem.values())
    u = deposito_id
    restante = total_restante
    carga = 0
    dist = 0.0
    ruta = [deposito_id]

    while restante > 0:
        if carga == 0:
            carga = min(capacidad_camion, restante)
            mejor_r, mejor_d = None, float('inf')
            for r in nodos_recarga:
                d = matriz_distancias[u][r]
                if d < mejor_d:
                    mejor_r, mejor_d = r, d
            if mejor_r is None or mejor_d == float('inf'):
                break
            if mejor_r != u:
                dist += mejor_d
                ruta.append(mejor_r)
                u = mejor_r

        destinos = [d for d, cnt in dem.items() if cnt > 0 and matriz_distancias[u][d] != float('inf')]
        if not destinos:
            break
        destino = min(destinos, key=lambda d: matriz_distancias[u][d])
        d = matriz_distancias[u][destino]
        dist += d
        ruta.append(destino)
        entrego = min(carga, dem[destino])
        dem[destino] -= entrego
        carga -= entrego
        restante -= entrego
        u = destino

    # Forzamos siempre el regreso al depósito si es alcanzable
    if ruta:
        dv = matriz_distancias[u][deposito_id]
        if dv != float('inf') and ruta[-1] != deposito_id:
            dist += dv
            ruta.append(deposito_id)

    s = Solucion()
    s.set(dist, ruta)
    return s


# ============================================================
#  Núcleo del Backtracking (top-level)  # [REFACTOR]
# ============================================================

def bt(u: int,
                carga: int,
                restante: int,
                dist: float,
                ruta: List[int],
                matriz_distancias: List[List[float]],
                nodos_recarga: set,
                capacidad_camion: int,
                demanda: Dict[int, int],
                estado: EstadoBT,
                deposito_id: int) -> None:
    """
    Backtracking para Opción A (misma lógica original; ahora top-level).
    Parametros:
    - u: nodo actual
    - carga: carga actual del camión
    - restante: paquetes restantes por entregar
    - dist: distancia acumulada hasta ahora
    - ruta: ruta acumulada hasta ahora (lista de nodos)
    - matriz_distancias: matriz de distancias entre nodos
    - nodos_recarga: set de nodos donde se puede recargar (hubs + depósito)
    - capacidad_camion: capacidad máxima del camión
    - demanda: diccionario de demanda restante por nodo
    - estado: estado mutable del backtracking
    - deposito_id: id del nodo depósito
    """
    if estado.stop:
        return

    # Señales de vida
    estado.contador_llamadas += 1
    estado.llamadas_desde_mejora += 1
    if estado.intervalo_report > 0 and (estado.contador_llamadas % estado.intervalo_report == 0):
        print(f"[DEBUG] llamadas={estado.contador_llamadas:,} | sin_mejora={estado.llamadas_desde_mejora:,} | "
              f"mejor={estado.mejor.distancia:.2f} | restante={restante} | nodo={u} | "
              f"umbral={estado.max_llamadas_sin_mejora:,}")

    # Early-stop por meseta
    if estado.llamadas_desde_mejora >= estado.max_llamadas_sin_mejora:
        print(f"[EARLY-STOP] sin mejora en {estado.llamadas_desde_mejora:,} llamadas "
              f"(umbral={estado.max_llamadas_sin_mejora:,}). mejor={estado.mejor.distancia:.2f}")
        estado.stop = True
        return

    # Poda
    if dist >= estado.mejor.distancia:
        return

    # Caso final
    if restante == 0 and carga == 0:
        dist_final, ruta_final = cerrar_ruta(
            dist, u, ruta, deposito_id, matriz_distancias
        )
        if dist_final < estado.mejor.distancia:
            estado.mejor.set(dist_final, ruta_final)
            estado.llamadas_desde_mejora = 0
        return

    # Recarga
    if carga == 0:
        para_cargar = min(capacidad_camion, restante)

        # Cargar sin moverse si ya estoy en recarga
        if u in nodos_recarga:
            bt(u, para_cargar, restante, dist, ruta,
                        matriz_distancias, nodos_recarga, capacidad_camion,
                        demanda, estado, deposito_id)
            if estado.stop:
                return

        # Explorar moverse a cada recarga
        for r in nodos_recarga:
            if estado.stop:
                return
            if r == u:
                continue
            d = matriz_distancias[u][r]
            if d == float('inf'):
                continue
            ruta.append(r)
            bt(r, para_cargar, restante, dist + d, ruta,
                        matriz_distancias, nodos_recarga, capacidad_camion,
                        demanda, estado, deposito_id)
            ruta.pop()
        return

    # Entrega: ordenar destinos por cercanía
    destinos = [d for d, cnt in demanda.items() if cnt > 0 and matriz_distancias[u][d] != float('inf')]
    destinos.sort(key=lambda d: matriz_distancias[u][d])

    for destino in destinos:
        if estado.stop:
            return
        cnt = demanda[destino]
        d = matriz_distancias[u][destino]
        if d == float('inf'):
            continue

        entrego = min(carga, cnt)

        # Aplicar
        demanda[destino] -= entrego
        nueva_carga = carga - entrego
        nuevo_restante = restante - entrego

        ruta.append(destino)
        bt(destino, nueva_carga, nuevo_restante, dist + d, ruta,
            matriz_distancias, nodos_recarga, capacidad_camion,
            demanda, estado, deposito_id)
        ruta.pop()

        # Deshacer
        demanda[destino] += entrego


# ============================================================
#  Orquestador de Opción A (misma lógica, mejor estructurado)
# ============================================================

def resolver_opcion_a_backtracking(
    matriz_distancias: List[List[float]],
    deposito_id: int,
    hubs: List[int],
    demanda_por_nodo: Dict[int, int],
    capacidad_camion: int,
    max_llamadas_sin_mejora: Optional[int] = None,
    intervalo_report: Optional[int] = None,
) -> Solucion:
    """
    Explora por backtracking la mejor ruta para Opción A (hubs activos, recarga gratis).
    Misma lógica que antes; ahora sin funciones anidadas.
    """
    n = len(matriz_distancias)
    demanda = demanda_por_nodo.copy()
    total_restante = sum(demanda.values())
    nodos_recarga = set(hubs) | {deposito_id}
    mejor = Solucion()
    ruta_inicial: List[int] = [deposito_id]

    # Umbrales auto-escalables (meseta y report)
    m = sum(1 for _, cnt in demanda.items() if cnt > 0) or 1
    T = ceil(total_restante / max(1, capacidad_camion)) or 1
    if max_llamadas_sin_mejora is None:
        max_llamadas_sin_mejora = auto_meseta(n, m, T)  # [REFACTOR]
    if intervalo_report is None:
        intervalo_report = max(1_000, max_llamadas_sin_mejora // 100)

    # Semilla greedy
    semilla = greedy_seed(matriz_distancias, deposito_id, nodos_recarga,
                          demanda, capacidad_camion)  # [REFACTOR]
    if semilla.distancia < mejor.distancia:
        mejor.set(semilla.distancia, semilla.ruta)

    # Estado del BT
    estado = EstadoBT(
        mejor=mejor,
        max_llamadas_sin_mejora=max_llamadas_sin_mejora,
        intervalo_report=intervalo_report,
    )

    # Ejecutar BT
    bt(deposito_id, 0, total_restante, 0.0, ruta_inicial,
                matriz_distancias, nodos_recarga, capacidad_camion,
                demanda, estado, deposito_id)

    return estado.mejor

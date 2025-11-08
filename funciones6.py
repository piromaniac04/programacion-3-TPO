from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from math import ceil, sqrt

#  Modelos / Dataclasses
@dataclass
class Solucion:
    """Guarda la mejor solución encontrada"""
    distancia: float = float('inf')
    ruta: List[int] = None
    hubs_usados: set = field(default_factory=set)  # conjunto de hubs realmente usados

    def set(self, dist: float, ruta: List[int], hubs_usados: Optional[set] = None) -> None:
        """Actualiza la mejor solución con nueva distancia, ruta y hubs usados."""
        self.distancia = dist
        self.ruta = ruta.copy()
        self.hubs_usados = hubs_usados.copy() if hubs_usados else set()


@dataclass
class EstadoBT:
    """Estado mutable del backtracking."""
    mejor: Solucion
    contador_llamadas: int = 0
    llamadas_desde_mejora: int = 0
    stop: bool = False
    max_llamadas_sin_mejora: int = 0
    intervalo_report: int = 0


#  Floyd–Warshall con reconstrucción de caminos
def reconstruir_camino(next_node: List[List[Optional[int]]], origen: int, destino: int) -> List[int]:
    """Reconstruye el camino desde `origen` hasta `destino` usando la matriz `next_node`.
    Parametros:
    - next_node: matriz de nodos siguientes en el camino mínimo
    - origen: nodo de inicio
    - destino: nodo final
    Salida:
    - lista de nodos en el camino desde `origen` hasta `destino`
    """
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
    """Floyd-Warshall que devuelve matriz de distancias y caminos mínimos (listas de nodos).
    Parametros:
    - matrizDeAdyacencia: matriz de adyacencia con pesos (0 = sin arista).
    Salida:
    - distancia: matriz de distancias mínimas entre nodos
    - caminos: matriz de caminos mínimos entre nodos (listas de nodos)
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

    caminos: List[List[List[int]]] = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if distancia[i][j] != float('inf'):
                caminos[i][j] = reconstruir_camino(next_node, i, j)
    return distancia, caminos


def hubEnCamino(camino: List[int], hubs: List[int]) -> bool:
    """Devuelve True si en el camino hay al menos un hub.
    Parametros:
    - camino: lista de nodos del camino
    - hubs: lista de nodos que son hubs
    Salida:
    - True si hay al menos un hub en el camino, False en caso contrario
    """
    set_h = set(hubs)
    return any(n in set_h for n in camino)


def recogerPaquete(capacidadActual: int, cantidadDeposito: int, capacidadMaxima: int) -> Tuple[int, int]:
    """Devuelve nueva capacidad del camión y paquetes restantes en depósito.
    Parametros:
    - capacidadActual: capacidad actual del camión
    - cantidadDeposito: cantidad de paquetes en el depósito
    - capacidadMaxima: capacidad máxima del camión
    Salida:
    - tupla (nueva capacidad del camión, paquetes restantes en depósito)
    """
    espacioDisponible = capacidadMaxima - capacidadActual
    if espacioDisponible <= 0:
        return capacidadActual, cantidadDeposito
    if cantidadDeposito >= espacioDisponible:
        return capacidadMaxima, cantidadDeposito - espacioDisponible
    return capacidadActual + cantidadDeposito, 0


#  Heurísticas
def auto_meseta(n: int, m: int, T: int, base: int) -> int:
    """Umbral auto-escalable para early-stop por meseta.
    Parametros:
    - n: cantidad de nodos en el grafo
    - m: cantidad de nodos con demanda > 0
    - T: cantidad mínima de viajes necesarios (demanda total / capacidad camión)
    - base: valor base para el cálculo
    Salida:
    - umbral calculado (int)
    """
    val = int(base * sqrt(max(1, n)) * m * max(1, T))
    return max(50_000, min(5_000_000, val))


def cerrar_ruta(dist_actual: float,
                u: int,
                ruta: List[int],
                deposito_id: int,
                matriz_distancias: List[List[float]]) -> Tuple[float, List[int]]:
    """Cierra la ruta volviendo siempre al depósito (si es posible).
    Parametros:
    - dist_actual: distancia acumulada hasta el nodo actual `u`
    - u: nodo actual
    - ruta: ruta actual (lista de nodos)
    - deposito_id: id del nodo depósito
    - matriz_distancias: matriz de distancias entre nodos
    Salida:
    - tupla (distancia total cerrada, ruta cerrada)
    """
    d_vuelta = matriz_distancias[u][deposito_id]
    if d_vuelta == float('inf'):
        return float('inf'), ruta
    nueva_ruta = ruta.copy()
    if u != deposito_id:
        nueva_ruta.append(deposito_id)
    return dist_actual + d_vuelta, nueva_ruta


def primer_solucion_greedy(matriz_distancias: List[List[float]],
                           deposito_id: int,
                           nodos_recarga: set,
                           demanda: Dict[int, int],
                           capacidad_camion: int) -> Solucion:
    """
    Greedy:
      1) Cuando carga=0 elige la recarga r que minimiza: dist(u,r) + min_v{dist(r,v)} con demanda>0.
      2) Con carga>0 elige el destino más cercano; en empate, prioriza mayor demanda pendiente.
      3) Siempre fuerza el regreso al depósito si es alcanzable.
    Parametros:
    - matriz_distancias: matriz de distancias entre nodos
    - deposito_id: id del nodo depósito
    - nodos_recarga: conjunto de nodos donde se puede recargar (hubs + depósito)
    - demanda: diccionario {nodo: cantidad de paquetes a entregar}
    - capacidad_camion: capacidad máxima del camión
    Salida:
    - mejor solución encontrada (objeto Solucion)
    """
    dem = demanda.copy()
    total_restante = sum(dem.values())
    u = deposito_id
    restante = total_restante
    carga = 0
    dist = 0.0
    ruta = [deposito_id]
    hubs_usados = set()

    while restante > 0:
        if carga == 0:
            carga = min(capacidad_camion, restante)
            mejor_nodo_r, mejor_distancia_r = None, float('inf')
            for r in nodos_recarga:
                d_ur = matriz_distancias[u][r]
                if d_ur == float('inf'):
                    continue
                mejor_min_rv = min(
                    (matriz_distancias[r][v] for v, cnt in dem.items() if cnt > 0 and matriz_distancias[r][v] != float('inf')),
                    default=float('inf')
                )
                if mejor_min_rv == float('inf'):
                    continue
                distancia_u_r_v = d_ur + mejor_min_rv
                if distancia_u_r_v < mejor_distancia_r:
                    mejor_nodo_r, mejor_distancia_r = r, distancia_u_r_v
            if mejor_nodo_r is None:
                raise ValueError("No se encontró recarga válida; verificar conectividad del grafo.")
            if mejor_nodo_r != u:
                dist += matriz_distancias[u][mejor_nodo_r]
                ruta.append(mejor_nodo_r)
                u = mejor_nodo_r
                if u != deposito_id:
                    hubs_usados.add(u)

        candidatos = [v for v, cnt in dem.items() if cnt > 0 and matriz_distancias[u][v] != float('inf')]
        if not candidatos:
            break
        candidatos.sort(key=lambda v: (matriz_distancias[u][v], -dem[v]))
        destino = candidatos[0]
        d_ud = matriz_distancias[u][destino]
        dist += d_ud
        ruta.append(destino)
        entrego = min(carga, dem[destino])
        dem[destino] -= entrego
        carga -= entrego
        restante -= entrego
        u = destino

    if ruta:
        d_back = matriz_distancias[u][deposito_id]
        if d_back != float('inf') and ruta[-1] != deposito_id:
            dist += d_back
            ruta.append(deposito_id)

    s = Solucion()
    s.set(dist, ruta, hubs_usados)
    return s


#  Núcleo del Backtracking

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
       deposito_id: int,
       debug: bool,
       hubs_en_rama: Optional[set] = None) -> None:
    """
    Backtracking recursivo para encontrar la mejor solución posible. con poda y early-stop por meseta (finaliza si no hay mejora).
    Parametros:
    - u: nodo actual
    - carga: carga actual del camión
    - restante: paquetes restantes por entregar
    - dist: distancia acumulada hasta el nodo actual
    - ruta: ruta actual (lista de nodos)
    - matriz_distancias: matriz de distancias entre nodos
    - nodos_recarga: conjunto de nodos donde se puede recargar (hubs + depósito)
    - capacidad_camion: capacidad máxima del camión
    - demanda: diccionario {nodo: cantidad de paquetes a entregar}
    - estado: estado mutable del backtracking
    - deposito_id: id del nodo depósito
    - debug: si es True, imprime información de depuración
    - hubs_en_rama: conjunto de hubs usados en la rama actual
    Salida:
    - None (actualiza el estado.mejor si encuentra una mejor solución)
    """
    if hubs_en_rama is None:
        hubs_en_rama = set()

    if estado.stop:
        return

    estado.contador_llamadas += 1
    estado.llamadas_desde_mejora += 1
    if estado.intervalo_report > 0 and (estado.contador_llamadas % estado.intervalo_report == 0) and debug:
        print(f"[DEBUG] llamadas={estado.contador_llamadas:,} | sin_mejora={estado.llamadas_desde_mejora:,} | "
              f"mejor={estado.mejor.distancia:.2f} | restante={restante} | nodo={u}")

    if estado.llamadas_desde_mejora >= estado.max_llamadas_sin_mejora:
        estado.stop = True
        return

    # Poda por distancia
    if dist >= estado.mejor.distancia:
        return

    if restante == 0 and carga == 0:
        dist_final, ruta_final = cerrar_ruta(dist, u, ruta, deposito_id, matriz_distancias)
        if dist_final < estado.mejor.distancia:
            estado.mejor.set(dist_final, ruta_final, hubs_en_rama)
            estado.llamadas_desde_mejora = 0
        return

    if carga == 0:
        para_cargar = min(capacidad_camion, restante)

        if u in nodos_recarga and u != deposito_id:
            hubs_en_rama.add(u)
        if u in nodos_recarga:
            bt(u, para_cargar, restante, dist, ruta,
               matriz_distancias, nodos_recarga, capacidad_camion,
               demanda, estado, deposito_id, debug, hubs_en_rama.copy())
            if estado.stop:
                return

        for r in nodos_recarga:
            if estado.stop:
                return
            if r == u:
                continue
            d_ur = matriz_distancias[u][r]
            if d_ur == float('inf'):
                continue
            nuevos_hubs = hubs_en_rama.copy()
            if r != deposito_id:
                nuevos_hubs.add(r)
            ruta.append(r)
            bt(r, para_cargar, restante, dist + d_ur, ruta,
               matriz_distancias, nodos_recarga, capacidad_camion,
               demanda, estado, deposito_id, debug, nuevos_hubs)
            ruta.pop()
        return

    destinos = [v for v, cnt in demanda.items() if cnt > 0 and matriz_distancias[u][v] != float('inf')]
    destinos.sort(key=lambda v: matriz_distancias[u][v])

    for destino in destinos:
        if estado.stop:
            return
        cnt = demanda[destino]
        d_ud = matriz_distancias[u][destino]
        if d_ud == float('inf'):
            continue
        entrego = min(carga, cnt)
        demanda[destino] -= entrego
        nueva_carga = carga - entrego
        nuevo_restante = restante - entrego
        ruta.append(destino)
        bt(destino, nueva_carga, nuevo_restante, dist + d_ud, ruta,
           matriz_distancias, nodos_recarga, capacidad_camion,
           demanda, estado, deposito_id, debug, hubs_en_rama.copy())
        ruta.pop()
        demanda[destino] += entrego


def resolver_problema(
    matriz_distancias: List[List[float]],
    deposito_id: int,
    hubs: List[int],
    demanda_por_nodo: Dict[int, int],
    capacidad_camion: int,
    max_llamadas_sin_mejora: Optional[int] = None,
    intervalo_report: Optional[int] = None,
    debug: bool = False,
    base_meseta: int = 1300
) -> Solucion:
    """
    Resuelve el problema usando backtracking con poda y early-stop por meseta.
    Parametros:
    - matriz_distancias: matriz de distancias entre nodos
    - deposito_id: id del nodo depósito
    - hubs: lista de nodos que son hubs
    - demanda_por_nodo: diccionario {nodo: cantidad de paquetes a entregar}
    - capacidad_camion: capacidad máxima del camión
    - max_llamadas_sin_mejora: umbral de llamadas sin mejora para early-stop
    - intervalo_report: intervalo de llamadas para reporte de depuración
    - debug: si es True, imprime información de depuración
    - base_meseta: valor base para el cálculo del umbral de meseta
    Salida:
    - mejor solución encontrada (objeto Solucion)
    """
    n = len(matriz_distancias)
    demanda = demanda_por_nodo.copy()
    total_restante = sum(demanda.values())
    nodos_recarga = set(hubs) | {deposito_id}
    mejor = Solucion()
    ruta_inicial: List[int] = [deposito_id]

    if base_meseta <= 0:
        raise ValueError("base_meseta debe ser un entero positivo.")
    
    m = sum(1 for _, cnt in demanda.items() if cnt > 0) or 1
    T = ceil(total_restante / max(1, capacidad_camion)) or 1
    if max_llamadas_sin_mejora is None:
        max_llamadas_sin_mejora = auto_meseta(n, m, T, base_meseta)
    if intervalo_report is None:
        intervalo_report = max(1_000, max_llamadas_sin_mejora // 100)

    puntoDePartida = primer_solucion_greedy(
        matriz_distancias, deposito_id, nodos_recarga, demanda, capacidad_camion)
    if puntoDePartida.distancia < mejor.distancia:
        mejor.set(puntoDePartida.distancia, puntoDePartida.ruta, puntoDePartida.hubs_usados)

    estado = EstadoBT(
        mejor=mejor,
        max_llamadas_sin_mejora=max_llamadas_sin_mejora,
        intervalo_report=intervalo_report,
    )

    bt(deposito_id, 0, total_restante, 0.0, ruta_inicial,
       matriz_distancias, nodos_recarga, capacidad_camion,
       demanda, estado, deposito_id, debug)

    return estado.mejor

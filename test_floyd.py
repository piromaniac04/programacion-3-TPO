import funciones as f

# Grafo de ejemplo con 4 nodos
# 0 -> 1 (3), 1 -> 2 (1), 2 -> 3 (7), 3 -> 0 (2)
G = [
    [0, 3, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 7],
    [2, 0, 0, 0],
]

dist, caminos = f.floydWarshallConCaminos(G)

print("Matriz de distancias:")
for row in dist:
    print(['{:.0f}'.format(x) if x!=float('inf') else 'inf' for x in row])

print('\nMatriz de caminos (listas de nodos):')
for i, row in enumerate(caminos):
    for j, path in enumerate(row):
        print(f"{i} -> {j}: {path}")

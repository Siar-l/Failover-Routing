import os
import networkx as nx

folder = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\Data"
print("Dateien im Ordner:")
for f in os.listdir(folder):
    if f.endswith(".graphml"):
        print("  ->", f)


path = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\Data\Geant2012.graphml"
G = nx.read_graphml(path)
print(" Datei erfolgreich geladen!")
print("Knoten:", G.number_of_nodes())
print("Kanten:", G.number_of_edges())

import networkx as nx
path = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\Data\Abilene.graphml"
G = nx.read_graphml(path).to_undirected()

print("Edge connectivity:", nx.edge_connectivity(G))
print("Node connectivity:", nx.node_connectivity(G))
print("Anzahl der Komponenten:", nx.number_connected_components(G))

# Zeigt alle Brücken im Graph:
print("Brücken (kritische Kanten):", list(nx.bridges(G)))

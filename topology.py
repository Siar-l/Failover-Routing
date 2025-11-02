import os 
from networkx.algorithms.connectivity import edge_connectivity
import networkx as nx


folder = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\Data"
print(f"\n=== Edge Connectivity Analyse für alle Topologien in: {folder} ===\n")
list = []
print("Dateien im Ordner:")
for f in os.listdir(folder):
    if not f.endswith(".graphml"):
       continue  
    path= os.path.join(folder,f) 

    try:
        G0 = nx.read_graphml(path)
        G = G0.to_undirected()
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_component(G), key=len)).copy()
        lam = edge_connectivity(G)
        node_count= G.number_of_nodes()
        edge_count = G.number_of_edges()


        list.append((f,lam,node_count,edge_count))
        print(f"{f:25s} | connectivity = {lam:<2d} | Nodes = {node_count:<3d} | Edges = {edge_count:<3d}") 

    except Exception as e:
        print (f"{f:25s} Fehler: {e}")
list.sort(key = lambda x: x[1], reverse=True)
print ("\n=== Zusammefassung (nach connectivity sortiert)===")
for f, lam, n, m in list:
    print(f"{f:25s} connectivity = {lam:<2d} (Nodes = {n}, Edges = {m})")

if list:
    avg_conn = sum(lam for _ , lam, _, _ in list)/ len(list)
    print(f"\nDurchschnittliche Edge Connectivity über alle Topologien: {avg_conn:.2f}")




#print("  ->", f)
        



path = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\Data\Geant2012.graphml"
G = nx.read_graphml(path)
print(" Datei erfolgreich geladen!")
print("Knoten:", G.number_of_nodes())
print("Kanten:", G.number_of_edges())


path = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\Data\Abilene.graphml"
G = nx.read_graphml(path).to_undirected()

print("Edge connectivity:", nx.edge_connectivity(G))
print("Node connectivity:", nx.node_connectivity(G))
print("Anzahl der Komponenten:", nx.number_connected_components(G))

# Zeigt alle Brücken im Graph:
print("Brücken (kritische Kanten):", list(nx.bridges(G)))

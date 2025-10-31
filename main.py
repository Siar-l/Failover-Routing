import networkx as nx
from networkx.algorithms.connectivity import edge_connectivity
import random



#def create_graph():

   # G = nx.complete_graph(6)

    #return G
path = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\Data\Geant2012.graphml"
G0 = nx.read_graphml(path)

G = G0.to_undirected()
if not nx.is_connected(G):
     G = G.subgraph(max(nx.connected_components(G),key=len)).copy()
for u,v, data in G .edges(data=True):
     data.setdefault("weight",1.0)

deg = dict(G.degree())
target = max(deg, key=deg.get)
source = min(deg, key=deg.get)
lam = edge_connectivity(G)
k_desired = 4 
k= min(k_desired,lam)
print(f"k={k} (edge_connectivity = {lam})")
print(f"source = {source},target = {target}")
def build_spanning_trees(G, target, k):
    trees = []
    for _ in range(k):

        for(u, v)in G.edges():
             G[u][v] ['weight'] = random.random()

        T = nx.minimum_spanning_tree(G)
        trees.append(T)
    return trees
    
    
def simulate_failover(G, trees, source, target,  faild_edges, k):
    current_tree = 0
    current_node = source
    visited = set ()
    bounces = 0

    while current_node != target:
        visited.add((current_node, current_tree))
        T = trees[current_tree]

        try: 
            path = nx.shortest_path(T, source=current_node,target=target)
            next_node = path[1]

        except:
                return False,bounces
        if (current_node, next_node) in faild_edges or (next_node, current_node) in faild_edges:
            current_tree = (current_tree + 1) % k
            bounces += 1
            if (current_node,current_tree) in visited:
                    return False, bounces
            continue
        current_node = next_node
    return True, bounces

def update_failures(G, failed_edges,failure_rate=0.2, recovery_rate=0.1):
    edges = list(G.edges())

    for e in edges:
         if e not in failed_edges and random.random() < failure_rate:
              failed_edges.add(e)
              failed_edges.add((e[1], e[0]))
    for e in list(failed_edges):
         if random.random() < recovery_rate:
            failed_edges.discard(e)
            failed_edges.discard((e[1],e[0]))
    return failed_edges       
def dynamic_simulation(G, trees, source, target, k, steps=10,
                        failure_rate = 0.2, recovery_rate=0.1):
     failed_edges = set()
     success_log = []
     bounces_log = []
     failed_count_log = []

     for t in range(steps):
          print(f"\n Zeit = {t+1}")
          failed_edges = update_failures(G, failed_edges,failure_rate=failure_rate, recovery_rate=recovery_rate)

          success, bounces = simulate_failover(G, trees,source, target, failed_edges, k)

          success_log.append(1 if success else 0)
          bounces_log.append(bounces)
          failed_count_log.append(len(failed_edges)//2)

          print("-> Routing erfolgreich:", success)
          print("Bounce:", bounces)
          print("-> Aktuel defekte Kanten:", len(failed_edges)//2)

     return {
         "success_log" : success_log,
         "bounce_log": bounces_log,
         "failed_count_log": failed_count_log
     
     }


def main():
     #k = 3
    # G = create_graph()
    # target = 0
    # source = 5 

     trees = build_spanning_trees(G, target, k)
     result = dynamic_simulation(G, trees, source, target, k, steps=10,
                                 failure_rate=0.25, recovery_rate = 0.15)
     
     success_rate = sum(result["success_log"]) / len(result["success_log"])
     avg_bounces = sum(result["bounce_log"]) / len(result["bounce_log"])
     #failed_edges = random.sample(list(G.edges()),2)

     #success, bounces = simulate_failover(G, trees, source, target, k, steps=10)
     #print("Erfolg:", success)
     #print("Bounce:", bounces)
     #print("Ausgefallene Kanten", failed_edges)

     print("\n==== Zusammenfassung=====")
     print(f"Erfolgsrate {success_rate:.2f}")
     print(f"Durchschnitliche Bounce: {avg_bounces: .2f}")
     print("Defekte Links pro Step:", result["failed_count_log"])
     print("Erfolg pro Step       :", result["success_log"])
     print("Bounces pro Step      :", result ["bounce_log"])

if    __name__=="__main__":
     main()




    


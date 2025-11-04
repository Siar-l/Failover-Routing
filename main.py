import networkx as nx
from networkx.algorithms.connectivity import edge_connectivity
import random
import topology
import os
import matplotlib.pyplot as plt
import pandas as pd


#def create_graph():

   # G = nx.complete_graph(6)

    #return G
path = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\Data\Abilene.graphml"
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
print(f"Edge connectivity= {lam}")
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
    hops = 0

    while current_node != target:
        if (current_node, current_tree) in visited:
             return False, bounces, hops
        visited.add((current_node, current_tree))
        T = trees[current_tree]

        try: 
            path = nx.shortest_path(T, source=current_node,target=target)
            if len(path) < 2:
                 raise nx.NetworkXNoPath
            
            next_node = path[1]

        except Exception:
                current_tree = (current_tree+1)%k
                bounces += 1
                continue
               
        if (current_node, next_node) in faild_edges or (next_node, current_node) in faild_edges:
            current_tree = (current_tree + 1) % k
            bounces += 1
            continue
        current_node = next_node
        hops += 1
    return True, bounces, hops

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
     pathlen_log = [] 

     for t in range(steps):
         
          failed_edges = update_failures(G, failed_edges,failure_rate=failure_rate, recovery_rate=recovery_rate)

          success, bounces, hops = simulate_failover(G, trees,source, target, failed_edges, k)

          success_log.append(1 if success else 0)
          bounces_log.append(bounces)
          failed_count_log.append(len(failed_edges)//2)
          if success:
               pathlen_log.append(hops)
          print(f"\n Zeit = {t+1}")
          print("-> Routing erfolgreich:", success)
          print("Bounce:", bounces)
          print("-> Aktuel defekte Kanten:", len(failed_edges)//2)

     return {
         "success_log" : success_log,
         "bounce_log": bounces_log,
         "failed_count_log": failed_count_log,
         "pathlen_log": pathlen_log,
     
     }


def main():
     def main():
    # Lade und analysiere nur den Abilene Graph
      path = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\Data\Abilene.graphml"
      G0 = nx.read_graphml(path)
      G = G0.to_undirected()
    
      if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
      for u, v, data in G.edges(data=True):
         data.setdefault("weight", 1.0)

    # Berechne und zeige Grapheigenschaften
     deg = dict(G.degree())
     target = max(deg, key=deg.get)
     source = min(deg, key=deg.get)
     lam = edge_connectivity(G)
    
     print("\n===== Graph Information =====")
     print(f"Nodes: {G.number_of_nodes()}")
     print(f"Edges: {G.number_of_edges()}")
     print(f"Edge connectivity (lambda): {lam}")
     print(f"Source node: {source}")
     print(f"Target node: {target}")
     print("============================\n")

     #k = 3
    # G = create_graph()
    # target = 0
    # source = 5 
     random.seed(42)
     if lam <=4:
        k_values =[1,2,3,4]
     if lam <=3:
          k_values = [1,2,3]
     if lam <= 2 :
          k_values = [1,2]   
     if lam <= 1 :
          k_values = [1]      
     results={}


     for k_desired in k_values:
          k_effective=min(k_desired,edge_connectivity(G))
          print("\n==========================")
          print(f"Experiment mit k_desired = {k_desired}(effektive: k = {k_effective})")
          print("==========================")

          trees = build_spanning_trees(G, target, k_effective)
          result = dynamic_simulation(G, trees, source, target, k_effective, steps=10,
                                 failure_rate=0.12, recovery_rate = 0.25)
     
          success_rate = sum(result["success_log"]) / len(result["success_log"]) if result["success_log"] else 0.0
          avg_bounces = sum(result["bounce_log"]) / len(result["bounce_log"]) if result["bounce_log"] else 0.0
          avg_failed = sum(result["failed_count_log"])/ len(result["failed_count_log"])if result["failed_count_log"] else 0.0
          avg_hops = (sum(result["pathlen_log"]) / len(result["pathlen_log"])) if result["pathlen_log"] else 0.0
          #failed_edges = random.sample(list(G.edges()),2)

     #success, bounces = simulate_failover(G, trees, source, target, k, steps=10)
     #print("Erfolg:", success)
     #print("Bounce:", bounces)
     #print("Ausgefallene Kanten", failed_edges)
          results[k_desired] = {
             "success_rate": success_rate,
             "avg_bounces": avg_bounces,
             "k_effective": k_effective,
             "avg_failed": avg_failed,
             "avg_hops": avg_hops,
             "raw": result
          
       }
    # print("\n==== Zusammenfassung=====")
     #print(f"Erfolgsrate {success_rate:.2f}")
     #print(f"Durchschnitliche Bounce: {avg_bounces: .2f}")
     #print("Defekte Links pro Step:", result["failed_count_log"])
     #print("Erfolg pro Step       :", result["success_log"])
     #print("Bounces pro Step      :", result ["bounce_log"])
     print("\n=====Geamtübersicht ====")
     for k in sorted(results.keys()):          
         metrics = results[k]                    
         print(f"k_desired={k} | k={metrics['k_effective']} | "
               f"Erfolg={metrics['success_rate']:.2f} | "
               f"Bounces={metrics['avg_bounces']:.2f} | "
               f"def. Kanten/Step={metrics['avg_failed']:.2f} | "
               f"Hops = {metrics['avg_hops']:.2f}")
         
     out_dir = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\results"
     os.makedirs(out_dir, exist_ok=True)

    
     rows = []
     for kd, m in results.items():
         rows.append({
             "k_desired": kd,
             "k_effective": m["k_effective"],
             "success_rate": m["success_rate"],
             "avg_bounces": m["avg_bounces"],
             "avg_failed": m["avg_failed"],
             "avg_hops": m["avg_hops"]
         })
     df = pd.DataFrame(rows).sort_values("k_desired")
     csv_path = os.path.join(out_dir, "summary_results.csv")
     df.to_csv(csv_path, index=False, sep=';')
     print(f"CSV gespeichert: {csv_path}")

    
     plt.figure(figsize=(8,4))
     plt.bar(df["k_desired"].astype(str), df["success_rate"])
     plt.title(f"Success rate per k (Edge Connectivity = {lam})")
     
     plt.xlabel("k_desired")
     plt.ylabel("Success rate")
     plt.tight_layout()
     plt.savefig(os.path.join(out_dir, "success_rate_per_k.png"))
     plt.close()

     plt.figure(figsize=(8,4))
     plt.bar(df["k_desired"].astype(str), df["avg_bounces"])
     plt.title(f"Success rate per k (Edge Connectivity = {lam})")
     plt.xlabel("k_desired")
     plt.ylabel("Avg bounces")
     plt.tight_layout()
     plt.savefig(os.path.join(out_dir, "avg_bounces_per_k.png"))
     plt.close()


     plt.figure(figsize=(8,4))
     plt.bar(df["k_desired"].astype(str), df["avg_failed"])
     plt.title(f"Success rate per k (Edge Connectivity = {lam})")
     plt.xlabel("k_desired")
     plt.ylabel("Avg failed links")
     plt.tight_layout()
     plt.savefig(os.path.join(out_dir, "avg_failed_per_k.png"))
     plt.close()

     
     example_k = max(results.keys())
     raw = results[example_k]["raw"]
     steps = len(raw["success_log"])
     plt.figure(figsize=(10,4))
     plt.plot(range(1, steps+1), raw["success_log"], marker='o', label='success (1/0)')
     plt.plot(range(1, steps+1), raw["bounce_log"], marker='x', label='bounces')
     plt.xlabel("Step")
     plt.legend()
     plt.title(f"Time series (k={example_k}), Edge Connectivity = {lam})")
     plt.tight_layout()
     plt.savefig(os.path.join(out_dir, f"time_series_k{example_k}.png"))
     plt.close()

     print(f"Plots gespeichert in: {out_dir}")
          
     
          
if    __name__=="__main__":
     main()




    


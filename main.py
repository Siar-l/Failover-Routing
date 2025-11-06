import networkx as nx
from networkx.algorithms.connectivity import edge_connectivity
import random

import os
import matplotlib.pyplot as plt
import pandas as pd


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
def analyze_resilience(G, trees, source, target, k, num_trials=50, max_failures=50):
     
    if max_failures is None:
          max_failures = min(G.number_of_edges()//2, 15)

    resilience_data = []

    for num_fails in range(0, max_failures + 1):
        successes = 0; 
        total_bounces = 0
        total_hops = 0
        successful_hops= []
        for trial in range(num_trials):
             all_edges = list(G.edges())
             failed = set()
             if num_fails > 0:
                  fail_list = random.sample(all_edges, min(num_fails, len(all_edges)))
                  for e in fail_list:
                       failed.add(e)
                       failed.add((e[1], e[0]))

             success, bounces, hops = simulate_failover(G, trees, source, target, failed, k)
        
             if success:
                successes += 1
                successful_hops.append(hops)
                total_bounces += bounces
                total_hops += hops

        success_rate = successes / num_trials 
        avg_bounces = total_bounces / num_trials 
        avg_hops = sum(successful_hops) / len(successful_hops) if successful_hops else 0

        resilience_data.append({
           "num_failures": num_fails,
           "success_rate": success_rate,
           "avg_bounces": avg_bounces,
           "avg_hops": avg_hops,
           "failure_percentage": (num_fails / G.number_of_edges()) * 100
       })
        print(f"Aufälle: {num_fails} / {G.number_of_edges()} | Erfolg: {success_rate * 100:.2%} | Bounces: {avg_bounces:.2f} | Hops: {avg_hops:.2f}")

    return resilience_data
 

def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(base_dir, "Data")
    abs_data_dir = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\Data"
    data_dir = default_data_dir if os.path.exists(default_data_dir) else abs_data_dir

    print("Quelle wählen:")
    print("   1) Topology Zoo (GraphML aus Data)")
    print("   2) Graph generieren")
    sel = (input("Auswahl [1/2](Default) 1): ").strip() or "1")

    
    if sel == "2":
        print("\nGraph-Generator:")
        print("  a) Erdos-Renyi G(n,p)")
        print("  b) Barabasi-Albert BA(n,m)")
        print("  c) Complete K_n")
        choice = (input("Auswahl [a/b/c] (Default a): ").strip() or "a").lower()
        if choice == "b":
            n = int(input("n (Default 20): ").strip() or "20")
            m = int(input("m (Default 2): ").strip() or "2")
            G = nx.barabasi_albert_graph(n, m).to_undirected()
            graph_name = f"BA_n{n}_m{m}"
        elif choice == "c":
            n = int(input("n (Default 20): ").strip() or "20")
            G = nx.complete_graph(n).to_undirected()
            graph_name = f"Complete_n{n}"
        else:
            n = int(input("n (Default 30): ").strip() or "30")
            p = float(input("p (Default 0.08): ").strip() or "0.08")
            G = nx.erdos_renyi_graph(n, p).to_undirected()
            graph_name = f"ER_n{n}_p{p}"
    else:
        # Topology Zoo
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(".graphml")]
        if not files:
            raise FileNotFoundError(f"Keine .graphml Dateien in: {data_dir}")
        files.sort()
        print("\nTopology Zoo Dateien:")
        for i, fname in enumerate(files, 1):
            print(f"  {i}) {fname}")
        idx = int((input(f"Auswahl [1-{len(files)}] (Default 1): ").strip() or "1"))
        idx = max(1, min(len(files), idx))
        path = os.path.join(data_dir, files[idx - 1])
        G0 = nx.read_graphml(path)
        G = G0.to_undirected()
        graph_name = os.path.splitext(os.path.basename(path))[0]

    custom = input(f"Optional: Graphname überschreiben (Enter für '{graph_name}'): ").strip()
    if custom:
        graph_name = custom    

    # Lade und analysiere nur den Abilene Graph
    #path = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\Data\Abilene.graphml"
    #G0 = nx.read_graphml(path)
    #G = G0.to_undirected()
    
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
    k_values = list(range(1,lam+1))     
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
    print("\n========Resilienz-Analyse=========")

    resilience_results = {}

    
    for k_desired in k_values:
         k_effective = min(k_desired, edge_connectivity(G))
         print(f"Resilienz-Analyse für k = {k_effective}")
         trees = build_spanning_trees(G, target, k_effective)
         res_data = analyze_resilience(G, trees, source, target, k_effective, num_trials=50, max_failures=None)
         resilience_results[k_effective]=res_data

    out_dir = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\results"
    os.makedirs(out_dir, exist_ok=True)

    print("\n========Erstelle Resilienz_Plots=========")

    plt.figure(figsize=(10, 6))
    for k_effective in sorted(resilience_results.keys()):
        data = resilience_results[k_effective]
        failures = [d["num_failures"] for d in data]
        success = [d["success_rate"] for d in data]
        plt.plot(failures, success, marker='o', label=f'k={k_effective}', linewidth=2)
        

    plt.xlabel("Anzahl ausgefallener Kanten")
    plt.ylabel("Erfolgsrate")
    plt.title(f"Resilienz bei steigenden Ausfällen(Edge Connectivity = {lam})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "resilience_analysis.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for k_effective in sorted(resilience_results.keys()):
        data = resilience_results[k_effective]
        failures = [d["num_failures"] for d in data]
        success = [d["avg_bounces"] for d in data]
        plt.plot(failures, success, marker='o', label=f'k={k_effective}', linewidth=2)

    
    plt.xlabel("Anzahl ausgefallener Kanten")
    plt.ylabel("Durchschnittliche Bounces")
    plt.title(f"Bounces bei steigenden Ausfällen (Edge Connectivity = {lam})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "resilience_bounces_per_k.png"))

    resiliennce_rows = []
    for k_effective, data in resilience_results.items():
         for entry in data:
              resiliennce_rows.append({"k_effective": k_effective, **entry})

    df_res = pd.DataFrame(resiliennce_rows)
    csv_res_path = os.path.join(out_dir, "resilience_analysis.csv")
    df_res.to_csv(csv_res_path, index = False,sep = ";")
    print(f"CSV gespeichert: {csv_res_path}")

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




    
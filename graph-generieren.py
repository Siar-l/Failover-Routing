import networkx as nx
import matplotlib.pyplot as plt
import random

def gen_graph_with_min_edge_connectivity(n = 20, p = 0.2, K = 2, max_tries = 200, seed =42, draw=True,
                                         save_plot = False):
    
    rng = random.Random(seed)
    best = None
    best_lambda = -1
    if K < n:
        try:
            G = nx.random_regular_graph(K, n, seed=rng.randint(0,10**9))
            if nx.is_connected(G):
                lam = nx.edge_connectivity(G)
                print(f"Random-regular d={K} produced connected graph, edge_connectivity={lam}")
                if lam >= K:
                    return G, lam
                best, best_lambda = (G, lam) if lam > best_lambda else (best, best_lambda)
        except Exception as e:
            
            print("random_regular_graph failed:", e)

    for attempt in range(1,max_tries + 1):
        G = nx.erdos_renyi_graph(n=n, p=p, seed=rng.randint(0, 10**9))

        if not nx.is_connected(G):
          lam = 0
        else:
           degs = dict(G.degree())
           min_deg = min(degs.values()) if degs else 0
           if min_deg < K:
             lam = min_deg
           else:
             lam = nx.edge_connectivity(G)
        
        if  lam > best_lambda:
           best, best_lambda = G, lam

        print(f"Try {attempt:03d}: connected={nx.is_connected(G)} edge_connectivity = {lam}")
        if lam>=K:
           
            print(f"\nZiel erreicht: connectivity={lam} ≥ {K} in Versuch #{attempt}")
            if draw:
                pos = nx.spring_layout(G, seed=seed + attempt)
                plt.figure(figsize=(6,6))
                nx.draw(G, pos, with_labels=True, node_color="lightgreen", node_size=700)
                plt.title(f"Erdos–Rényi n={n}, p={p}, edge_connectivity={lam}")
                if save_plot:
                    fname = f"er_n{n}_p{p}_lam{lam}_try{attempt}.png"
                    plt.savefig(fname, bbox_inches="tight")
                    print("Plot gespeichert:", fname)
                plt.close()
            return G, lam

    print(f"\nZiel (connectivity >= {K}) nicht erreicht. Bester Wert: connectivity={best_lambda}")
    if draw and best is not None:
        pos = nx.spring_layout(best, seed=seed)
        plt.figure(figsize=(6,6))
        nx.draw(best, pos, with_labels=True, node_color="salmon", node_size=700)
        plt.title(f"Best found: Erdos–Rényi n={n}, p={p}, edge_connectivity={best_lambda}")
        if save_plot:
            fname = f"er_best_n{n}_p{p}_lam{best_lambda}.png"
            plt.savefig(fname, bbox_inches="tight")
            print("Plot gespeichert:", fname)
        plt.close()
    return best, best_lambda 





if __name__ == "__main__":
   
    n = 20
    seed = 42
    rng = random.Random(seed)
    K_random = rng.randint(1, max(1, n-1))
    print(f"Gewähltes zufälliges K (gewünschte edge-connectivity): {K_random}")

    G, lambda_val = gen_graph_with_min_edge_connectivity(
        n=n,           
        p=0.2,         
        K=K_random,   
        max_tries=200, 
        seed=seed,     
        draw=True,     
        save_plot=True 
    )
    print(f"\nFinaler Graph:")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Edge Connectivity: {lambda_val}")
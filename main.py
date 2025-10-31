import networkx as nx
import random



def create_graph():

    G = nx.complete_graph(6)
    return G

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
def dynamic_simulation(G, trees, source, target, k, steps=10):
     failed_edges = set()
     for t in range(steps):
          print(f"\n Zeit = {t+1}")
          failed_edges = update_failures(G, failed_edges)
          success, bounces = simulate_failover(G, trees,source, target, failed_edges, k)

          print("-> Routing erfolgreich:", success)
          print("Bounce:", bounces)
          print("-> Aktuel defekte Kanten:", failed_edges)


def main():
     k = 10
     G = create_graph()
     target = 0
     source = 5 

     trees = build_spanning_trees(G, target, k)
     dynamic_simulation(G, trees, source, target, k, steps=10)
     #failed_edges = random.sample(list(G.edges()),2)

     #success, bounces = simulate_failover(G, trees, source, target, k, steps=10)
     #print("Erfolg:", success)
     #print("Bounce:", bounces)
     #print("Ausgefallene Kanten", failed_edges)

if    __name__=="__main__":
     main()




    


import networkx as nx
import random



def create_graph():

    G = nx.complete_graph(6)
    return G

def build_spanning_trees(G, target, k):
    trees = []
    for i in range(k):

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


def main():
     k = 3 
     G = create_graph()
     target = 0
     source = 5 

     trees = build_spanning_trees(G, target, k)
     failed_edges = random.sample(list(G.edges()),2)

     success, bounces = simulate_failover(G, trees, source, target, failed_edges,k)
     print("Erfolg:", success)
     print("Bounce:", bounces)
     print("Ausgefallene Kanten", failed_edges)

if    __name__=="__main__":
     main()




    


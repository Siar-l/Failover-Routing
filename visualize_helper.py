# Helper file - wird in main.py importiert
import matplotlib.pyplot as plt
import networkx as nx

def visualize_path_on_graph(G, path, source, target, failed_edges=None, title="Routing Path", save_path=None, show_all_failures=True):
    """
    Visualisiert einen Pfad auf dem Graphen mit farbigen Knoten und Kanten.
    
    Args:
        G: NetworkX Graph
        path: Liste von Knoten im Pfad
        source: Source-Knoten
        target: Target-Knoten
        failed_edges: Set von ausgefallenen Kanten (optional)
        title: Titel für die Visualisierung
        save_path: Speicherpfad (optional)
        show_all_failures: Ob alle Failures angezeigt werden sollen
    """
    if failed_edges is None:
        failed_edges = set()
    
    # Erstelle Figure
    plt.figure(figsize=(14, 10))
    
    # Layout mit festem Seed für Konsistenz
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
    # Kanten-Sets erstellen
    # Normalisiere Kanten für ungerichtete Graphen (min, max)
    path_edges = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        path_edges.add((min(u, v), max(u, v)))
    
    # Normalisiere failed_edges ebenfalls
    normalized_failed = set()
    for u, v in failed_edges:
        normalized_failed.add((min(u, v), max(u, v)))
    
    # Zeichne alle normalen Kanten (grau, dünn)
    all_edges = []
    for u, v in G.edges():
        edge_norm = (min(u, v), max(u, v))
        if edge_norm not in path_edges and edge_norm not in normalized_failed:
            all_edges.append((u, v))
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, 
                          edge_color='lightgray', width=0.5, alpha=0.3)
    
    # Zeichne ausgefallene Kanten (rot, gestrichelt)
    if show_all_failures and failed_edges:
        failed_edge_list = [(u, v) for u, v in G.edges() 
                           if (min(u, v), max(u, v)) in normalized_failed]
        if failed_edge_list:
            nx.draw_networkx_edges(G, pos, edgelist=failed_edge_list,
                                  edge_color='red', width=2, alpha=0.8, 
                                  style='dashed', label='Failed Edge')
    
    # Zeichne Pfad-Kanten (blau, dick)
    path_edge_list = [(path[i], path[i+1]) for i in range(len(path)-1)]
    if path_edge_list:
        nx.draw_networkx_edges(G, pos, edgelist=path_edge_list,
                              edge_color='blue', width=3, alpha=0.9,
                              arrows=True, arrowsize=20, arrowstyle='->')
    
    # Zeichne Hop-Nummern auf Pfad-Kanten
    for i, (u, v) in enumerate(path_edge_list, 1):
        # Berechne Mittelpunkt der Kante
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        
        # Zeichne gelben Kreis mit Hop-Nummer
        plt.scatter([x], [y], c='yellow', s=300, zorder=5, edgecolors='black', linewidths=1.5)
        plt.text(x, y, str(i), fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=6)
    
    # Zeichne normale Knoten (hellgrau)
    other_nodes = [n for n in G.nodes() if n not in path]
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,
                          node_color='lightgray', node_size=200, alpha=0.4)
    
    # Zeichne Pfad-Knoten (hellblau)
    path_nodes = [n for n in path if n != source and n != target]
    if path_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=path_nodes,
                              node_color='dodgerblue', node_size=400)
    
    # Zeichne Source (dunkelgrün)
    nx.draw_networkx_nodes(G, pos, nodelist=[source],
                          node_color='darkgreen', node_size=500, label='Source')
    
    # Zeichne Target (dunkelrot)
    nx.draw_networkx_nodes(G, pos, nodelist=[target],
                          node_color='darkred', node_size=500, label='Target')
    
    # Beschriftungen
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pfad-Visualisierung gespeichert: {save_path}")
        plt.close()
    else:
        plt.show()

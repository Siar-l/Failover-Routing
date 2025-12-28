import networkx as nx
from networkx.algorithms.connectivity import edge_connectivity
import random
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class Packet:
    ttl: int = 64
    idx: int | None = None

def build_psn(G, target, k):
    """Algorithm 2: PSN Building - Erstellt Path Selection Network"""
    psn = {n: {'up_links': [], 'down_links': [], 'distances': {}} for n in G.nodes()}
    
    # Berechne k verschiedene Gewichtungen für Diversität
    for layer in range(k):
        # Randomisiere Gewichte für verschiedene Pfade
        weights = {}
        for u, v in G.edges():
            base_weight = G[u][v].get('weight', 1.0)
            # Füge Variation hinzu basierend auf Layer
            w = base_weight * (1 + 0.1 * layer + random.random() * 0.2)
            weights[(u, v)] = w
            weights[(v, u)] = w
        
        # Berechne shortest paths 1× pro Layer (effizient!)
        try:
            dist = nx.single_source_dijkstra_path_length(
                G, target,
                weight=lambda u, v, d: weights.get((u, v), 1.0)
            )
        except:
            dist = {}
        
        # Distanzen für ALLE Nodes setzen
        for node in G.nodes():
            psn[node]['distances'][layer] = dist.get(node, float('inf'))
        
        # Klassifiziere Links als Up-links oder Down-links
        for node in G.nodes():
            if node == target:
                continue
                
            node_dist = psn[node]['distances'][layer]
            
            for neighbor in G.neighbors(node):
                neighbor_dist = psn[neighbor]['distances'][layer]
                
                # Tuple (neighbor, layer) statt dict - stabiler für Vergleiche
                link = (neighbor, layer)
                
                # Up-link: führt näher zum Ziel
                if neighbor_dist < node_dist:
                    if link not in psn[node]['up_links']:
                        psn[node]['up_links'].append(link)
                # Down-link: führt weg vom Ziel
                elif neighbor_dist > node_dist:
                    if link not in psn[node]['down_links']:
                        psn[node]['down_links'].append(link)
    
    return psn

def link_is_up(u, v, failed_edges: set[tuple]) -> bool:
    """Check if edge is available (undirected failure storage)"""
    edge_norm = (min(u, v), max(u, v))
    return edge_norm not in failed_edges

def smart_selection(psn, current_node, target, failed_edges, current_layer, k):
    """Algorithm 3: Smart Selection - Wählt nächsten Link intelligent"""
    
    if current_node == target:
        return None, current_layer, False
    
    node_psn = psn[current_node]
    
    # 1) Up-link im selben Layer (kein Bounce) - Paper-konform
    for neighbor, layer in node_psn['up_links']:
        if layer == current_layer and link_is_up(current_node, neighbor, failed_edges):
            return neighbor, layer, False  # No bounce
    
    # 2) Up-link in anderem Layer (Bounce)
    for neighbor, layer in node_psn['up_links']:
        if layer != current_layer and link_is_up(current_node, neighbor, failed_edges):
            return neighbor, layer, True  # Bounce occurred
    
    # 3) Down-links als Fallback
    for neighbor, layer in node_psn['down_links']:
        if link_is_up(current_node, neighbor, failed_edges):
            return neighbor, layer, (layer != current_layer)
    
    return None, current_layer, False


def kf_route_packet(G, psn, source, target, failed_edges, k):
    """
    KF-Routing mit PSN und Smart Selection (echtes Keep-Forwarding)
    """
    pkt = Packet(ttl=64, idx=0)
    current_layer = 0
    current_node = source
    visited = set()
    bounces = 0
    hops = 0
    path = [source]
    
    max_iterations = 200  # Verhindere infinite loops
    iterations = 0
    
    while current_node != target and pkt.ttl > 0 and iterations < max_iterations:
        iterations += 1
        
        # Loop detection
        state = (current_node, current_layer)
        if state in visited:
            # Versuche Layer zu wechseln
            current_layer = (current_layer + 1) % k
            bounces += 1
            
            if (current_node, current_layer) in visited:
                return False, bounces, hops, path
        
        visited.add((current_node, current_layer))
        
        # Algorithm 3: Smart Selection
        next_node, new_layer, did_bounce = smart_selection(psn, current_node, target, failed_edges, current_layer, k)
        
        if next_node is None:
            # Kein Pfad gefunden, wechsle Layer
            current_layer = (current_layer + 1) % k
            bounces += 1
            
            # Wenn alle Layer probiert wurden
            if len([v for v in visited if v[0] == current_node]) >= k:
                return False, bounces, hops, path
            continue
        
        if did_bounce:
            bounces += 1
        
        pkt.ttl -= 1
        current_layer = new_layer
        current_node = next_node
        path.append(current_node)
        hops += 1
    
    return (current_node == target), bounces, hops, path


def update_failures(G, failed_edges, failure_rate=0.2, recovery_rate=0.1):
    """Undirected failure handling - each edge fails/recovers exactly once"""
    edges = list(G.edges())
    
    # Neue Ausfälle (undirected)
    for u, v in edges:
        edge_norm = (min(u, v), max(u, v))
        if edge_norm not in failed_edges and random.random() < failure_rate:
            failed_edges.add(edge_norm)
    
    # Recovery (undirected)
    for edge_norm in list(failed_edges):
        if random.random() < recovery_rate:
            failed_edges.discard(edge_norm)
    
    return failed_edges


def dynamic_simulation(G, psn, source, target, k, steps=10, failure_rate=0.4, recovery_rate=0.1):
    failed_edges = set()
    success_log = []
    bounces_log = []
    failed_count_log = []
    pathlen_log = []
    all_paths = []
    all_failed_edges = []
    
    for t in range(steps):
        failed_edges = update_failures(G, failed_edges, failure_rate=failure_rate, recovery_rate=recovery_rate)
        
        success, bounces, hops, path = kf_route_packet(G, psn, source, target, failed_edges, k)
        
        success_log.append(1 if success else 0)
        bounces_log.append(bounces)
        failed_count_log.append(len(failed_edges))  # Already undirected
        
        if success:
            pathlen_log.append(hops)
            all_paths.append(path)
            all_failed_edges.append(failed_edges.copy())
        
        # Ausgefallene Kanten formatieren (bereits undirected)
        failed_edges_list = sorted(list(failed_edges))
        failed_str = ", ".join([f"({u}-{v})" for u, v in failed_edges_list])
        
        print(f"Zeit {t+1}: Success={success}, Bounces={bounces}, Dynamic Pfad: {' → '.join(str(n) for n in path)}")
        print(f"  Ausgefallene Kanten ({len(failed_edges)}): {failed_str if failed_str else 'Keine'}")
    
    return {
        "success_log": success_log,
        "bounce_log": bounces_log,
        "failed_count_log": failed_count_log,
        "pathlen_log": pathlen_log,
        "all_paths": all_paths,
        "all_failed_edges": all_failed_edges
    }


def analyze_resilience(G, psn, source, target, k, num_trials=50, max_failures=50, static_failures=True):
    if max_failures is None:
        max_failures = min(G.number_of_edges() // 2, 15)
    
    resilience_data = []
    all_paths = []  # Global für Gesamtstatistik
    zero_failure_paths = []
    
    # Für statische Analyse: Erstelle feste Failure-Sets (undirected)
    fixed_failure_sets = {}
    if static_failures:
        all_edges = [(min(u, v), max(u, v)) for u, v in G.edges()]
        edge_betweenness = nx.edge_betweenness_centrality(G)
        # Normalisiere Keys
        edge_betweenness_norm = {(min(u, v), max(u, v)): bc for (u, v), bc in edge_betweenness.items()}
        sorted_edges = sorted(edge_betweenness_norm.items(), key=lambda x: x[1], reverse=True)
        
        for num_fails in range(1, max_failures + 1):
            num_important = int(num_fails * 0.7)
            num_random = num_fails - num_important
            
            fail_list = [edge for edge, _ in sorted_edges[:num_important]]
            remaining_edges = [e for e in all_edges if e not in fail_list]
            if num_random > 0 and remaining_edges:
                fail_list.extend(random.sample(remaining_edges, min(num_random, len(remaining_edges))))
            
            fixed_failure_sets[num_fails] = fail_list[:num_fails]
    
    for num_fails in range(0, max_failures + 1):
        successes = 0
        total_bounces = 0
        total_hops = 0
        successful_hops = []
        paths_this_level = []  # Per-level tracking!
        
        fixed_failed = set()
        if static_failures and num_fails > 0:
            fixed_failed = set(fixed_failure_sets[num_fails])
        
        for trial in range(num_trials):
            if static_failures:
                failed = fixed_failed
            else:
                all_edges = [(min(u, v), max(u, v)) for u, v in G.edges()]
                failed = set()
                if num_fails > 0:
                    failed = set(random.sample(all_edges, min(num_fails, len(all_edges))))
            
            success, bounces, hops, path = kf_route_packet(G, psn, source, target, failed, k)
            
            if success:
                successes += 1
                successful_hops.append(hops)
                total_bounces += bounces
                total_hops += hops
                paths_this_level.append(path)
                all_paths.append(path)
                
                if num_fails == 0:
                    zero_failure_paths.append(path)
        
        success_rate = successes / num_trials
        avg_bounces = total_bounces / num_trials
        avg_hops = sum(successful_hops) / len(successful_hops) if successful_hops else 0
        
        # Per-level statistics (FIX!)
        unique_paths = len(set(tuple(p) for p in paths_this_level)) if paths_this_level else 0
        avg_path_length = sum(len(p) for p in paths_this_level) / len(paths_this_level) if paths_this_level else 0
        
        resilience_data.append({
            "num_failures": num_fails,
            "success_rate": success_rate,
            "avg_bounces": avg_bounces,
            "avg_hops": avg_hops,
            "unique_paths": unique_paths,
            "avg_path_length": avg_path_length,
            "failure_percentage": (num_fails / G.number_of_edges()) * 100
        })
        
        mode_text = "(feste Failures)" if static_failures else "(zufällige Failures)"
        print(f"Ausfälle: {num_fails} / {G.number_of_edges()} {mode_text} | Erfolg: {success_rate:.1%} | Bounces: {avg_bounces:.2f} | Hops: {avg_hops:.2f} | Unique Paths: {unique_paths}")
    
    # Gesamtstatistiken
    unique_paths_total = len(set(tuple(p) for p in all_paths))
    avg_path_length_total = sum(len(p) - 1 for p in all_paths) / len(all_paths) if all_paths else 0
    
    unique_zero_failure = len(set(tuple(p) for p in zero_failure_paths))
    avg_zero_failure_length = sum(len(p) - 1 for p in zero_failure_paths) / len(zero_failure_paths) if zero_failure_paths else 0
    
    print(f"\n=== Pfad-Statistiken (KF-Routing) ===")
    print(f"Gesamte erfolgreiche Pfade: {len(all_paths)}")
    print(f"Einzigartige Pfade (gesamt): {unique_paths_total}")
    print(f"Durchschnittliche Pfadlänge (alle): {avg_path_length_total:.2f}")
    print(f"Bei 0 Failures: {unique_zero_failure} einzigartige Pfade, Ø Länge: {avg_zero_failure_length:.2f}")
    
    return resilience_data, all_paths, zero_failure_paths


def visualize_path_on_graph(G, path, source, target, failed_edges=None, title="KF-Routing Path", save_path=None, show_all_failures=True):
    """Visualisiert KF-Routing Pfad"""
    if failed_edges is None:
        failed_edges = set()
    
    plt.figure(figsize=(16, 11))
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
    # Normalisiere Kanten
    path_edges = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        path_edges.add((min(u, v), max(u, v)))
    
    # failed_edges bereits normalisiert (undirected)
    normalized_failed = failed_edges
    
    # Zeichne normale Kanten
    all_edges = []
    for u, v in G.edges():
        edge_norm = (min(u, v), max(u, v))
        if edge_norm not in path_edges and edge_norm not in normalized_failed:
            all_edges.append((u, v))
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color='lightgray', width=1, alpha=0.3)
    
    # Zeichne Failed Edges (ZUERST, damit sie sichtbar sind)
    if show_all_failures and failed_edges:
        failed_edge_list = [(u, v) for u, v in G.edges() if (min(u, v), max(u, v)) in normalized_failed]
        if failed_edge_list:
            nx.draw_networkx_edges(G, pos, edgelist=failed_edge_list,
                                  edge_color='red', width=4, alpha=0.9, style='dashed')
            print(f"  Visualisiere {len(failed_edge_list)} ausgefallene Kanten: {failed_edge_list}")
    
    # Zeichne Pfad-Kanten (darüber)
    path_edge_list = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    if path_edge_list:
        nx.draw_networkx_edges(G, pos, edgelist=path_edge_list,
                              edge_color='blue', width=5, alpha=0.9)
    
    # Hop-Nummern
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        
        plt.annotate('', xy=(mid_x + dx, mid_y + dy), xytext=(mid_x - dx, mid_y - dy),
                    arrowprops=dict(arrowstyle='->', color='darkblue', lw=3.5, alpha=0.9))
        
        plt.text(mid_x, mid_y, f'{i+1}', fontsize=11, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='yellow', edgecolor='black', alpha=0.9))
    
    # Knoten
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == source:
            node_colors.append('darkgreen')
            node_sizes.append(800)
        elif node == target:
            node_colors.append('darkred')
            node_sizes.append(800)
        elif node in path:
            node_colors.append('dodgerblue')
            node_sizes.append(600)
        else:
            node_colors.append('lightgray')
            node_sizes.append(400)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='white')
    
    # Legende
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='darkgreen', edgecolor='black', label='Source'),
        Patch(facecolor='darkred', edgecolor='black', label='Target'),
        Patch(facecolor='dodgerblue', edgecolor='black', label='Path Node'),
        Patch(facecolor='lightgray', edgecolor='black', label='Other Node'),
        Line2D([0], [0], color='blue', linewidth=5, label='Path Edge'),
        Line2D([0], [0], color='red', linewidth=3, linestyle='dashed', label='Failed Edge'),
        Line2D([0], [0], color='lightgray', linewidth=1, label='Available Edge')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)
    
    num_failed = len(failed_edges) if failed_edges else 0  # Already undirected
    title_text = f"{title}\nPath: {' → '.join(str(n) for n in path)} ({len(path)-1} hops)"
    if num_failed > 0:
        title_text += f"\nTotal Failed Edges: {num_failed}"
    
    plt.title(title_text, fontsize=13, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pfad-Visualisierung gespeichert: {save_path}")
    
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(base_dir, "Data")
    abs_data_dir = r"C:\Users\Faraz\OneDrive\Desktop\Bachelorarbeit\Code-Failover-Routing\Data"
    data_dir = default_data_dir if os.path.exists(default_data_dir) else abs_data_dir
    
    print("=== KF-ROUTING MIT IMPORT-AWARENESS ===\n")
    print("Quelle wählen:")
    print("   1) Topology Zoo (GraphML aus Data)")
    print("   2) Graph generieren")
    sel = (input("Auswahl [1/2] (Default 1): ").strip() or "1")
    
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
    
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    for u, v, data in G.edges(data=True):
        data.setdefault("weight", 1.0)
    
    # Seed BEFORE source/target selection for reproducibility
    random.seed(42)
    
    nodes = list(G.nodes())
    source, target = random.sample(nodes, 2)
    deg = dict(G.degree())
    lam = edge_connectivity(G)
    
    print("\n===== Graph Information =====")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Edge connectivity (lambda): {lam}")
    print(f"Source node: {source} (degree: {deg[source]})")
    print(f"Target node: {target} (degree: {deg[target]})")
    print("============================\n")
    
    print("Experiment-Modus wählen:")
    print("  1) Nur Dynamische Simulation")
    print("  2) Nur Statische Resilienz-Analyse")
    print("  3) Beide mit kombinierten Plots")
    mode_choice = (input("Auswahl [1/2/3] (Default 3): ").strip() or "3")
    
    k_values = list(range(1, lam + 1))
    
    dynamic_results = {}
    dynamic_detailed_results = {}
    resilience_results = {}
    
    # Dynamische Simulation
    if mode_choice in ["1", "3"]:
        print("\n" + "="*50)
        print("DYNAMISCHE SIMULATION (KF-ROUTING)")
        print("="*50)
        
        failure_rates = [0.05, 0.10, 0.15, 0.20, 0.25]
        
        for k_desired in k_values:
            k_effective = min(k_desired, edge_connectivity(G))
            print(f"\n[Dynamisch] k = {k_effective}")
            
            psn = build_psn(G, target, k_effective)
            
            k_results = []
            for fr in failure_rates:
                print(f"\n--- Failure Rate: {fr:.2f} ---")
                result = dynamic_simulation(G, psn, source, target, k_effective,
                                          steps=20, failure_rate=fr, recovery_rate=0.1)
                
                success_rate = sum(result["success_log"]) / len(result["success_log"]) if result["success_log"] else 0.0
                avg_bounces = sum(result["bounce_log"]) / len(result["bounce_log"]) if result["bounce_log"] else 0.0
                avg_failed = sum(result["failed_count_log"]) / len(result["failed_count_log"]) if result["failed_count_log"] else 0.0
                
                # Pfadanalyse für diese failure_rate
                unique_paths = len(set(tuple(p) for p in result["all_paths"])) if result["all_paths"] else 0
                avg_path_length = sum(len(p) - 1 for p in result["all_paths"]) / len(result["all_paths"]) if result["all_paths"] else 0
                
                print(f"\n{'='*60}")
                print(f"DYNAMISCHE PFAD-ANALYSE")
                print(f"k={k_effective}, failure_rate={fr:.2f}")
                print(f"{'='*60}")
                print(f"✓ Erfolgreiche Pfade: {len(result['all_paths'])} von 20 Versuchen")
                print(f"✓ Einzigartige Pfade: {unique_paths}")
                print(f"✓ Durchschnittliche Pfadlänge: {avg_path_length:.2f} Hops")
                print(f"✓ Success Rate: {success_rate:.1%}")
                print(f"✓ Avg Bounces: {avg_bounces:.2f}")
                print(f"✓ Avg Failed Edges: {avg_failed:.2f}")
                print(f"{'='*60}")
                
                k_results.append({
                    "failure_rate": fr,
                    "success_rate": success_rate,
                    "avg_bounces": avg_bounces,
                    "avg_failed": avg_failed
                })
            
            k_results_sorted = sorted(k_results, key=lambda x: x["avg_failed"])
            dynamic_detailed_results[k_effective] = k_results_sorted
            
            avg_success = sum(r["success_rate"] for r in k_results) / len(k_results)
            avg_bounces_all = sum(r["avg_bounces"] for r in k_results) / len(k_results)
            avg_failed_all = sum(r["avg_failed"] for r in k_results) / len(k_results)
            
            dynamic_results[k_effective] = {
                "success_rate": avg_success,
                "avg_bounces": avg_bounces_all,
                "avg_failed": avg_failed_all
            }
        
        # Gesamtübersicht Dynamische Pfad-Analyse
        print("\n" + "="*70)
        print("GESAMTÜBERSICHT: DYNAMISCHE PFAD-ANALYSE")
        print("="*70)
        
        # Detaillierte Tabelle pro k
        for k_effective in sorted(k_values):
            if k_effective <= edge_connectivity(G):
                print(f"\n{'─'*70}")
                print(f"k={k_effective}")
                print(f"{'─'*70}")
                
                if k_effective in dynamic_detailed_results:
                    for entry in dynamic_detailed_results[k_effective]:
                        fr = entry["failure_rate"]
                        sr = entry["success_rate"]
                        ab = entry["avg_bounces"]
                        af = entry["avg_failed"]
                        print(f"  FR={fr:.2f}: Success={sr:>6.1%}, Bounces={ab:>5.2f}, AvgFail={af:>5.2f}")
                
                print(f"\n  Durchschnitt über alle failure_rates:")
                print(f"    Erfolgsrate: {dynamic_results[k_effective]['success_rate']:.1%}")
                print(f"    Bounces: {dynamic_results[k_effective]['avg_bounces']:.2f}")
                print(f"    Failures: {dynamic_results[k_effective]['avg_failed']:.2f}")
        
        print(f"\n{'='*70}")
    
    # Statische Resilienz-Analyse
    static_paths = {}
    zero_failure_paths = {}
    if mode_choice in ["2", "3"]:
        print("\n" + "="*50)
        print("STATISCHE RESILIENZ-ANALYSE (KF-ROUTING)")
        print("="*50)
        print("Hinweis: Verwendet deterministische Ausfälle basierend auf Kanten-Wichtigkeit\n")
        
        for k_desired in k_values:
            k_effective = min(k_desired, edge_connectivity(G))
            print(f"\n[Statisch] k = {k_effective}")
            psn = build_psn(G, target, k_effective)
            res_data, all_paths, zero_paths = analyze_resilience(G, psn, source, target, k_effective,
                                                                 num_trials=50, max_failures=None, static_failures=True)
            resilience_results[k_effective] = res_data
            static_paths[k_effective] = all_paths
            zero_failure_paths[k_effective] = zero_paths
    
    # Output-Verzeichnis
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_kf", graph_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Kombinierte Plots
    if mode_choice == "3" and dynamic_detailed_results and resilience_results:
        print("\n" + "="*50)
        print("ERSTELLE KOMBINIERTE PLOTS")
        print("="*50)
        
        # Success Rate
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for k_effective in sorted(resilience_results.keys()):
            data = resilience_results[k_effective]
            failures = [d["num_failures"] for d in data]
            success = [d["success_rate"] for d in data]
            ax1.plot(failures, success, marker='o', label=f'k={k_effective}', linewidth=2)
        ax1.set_xlabel("Anzahl ausgefallener Kanten", fontsize=11)
        ax1.set_ylabel("Erfolgsrate", fontsize=11)
        ax1.set_title("Statisch: Erfolgsrate bei festen Failures", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        for k_effective in sorted(dynamic_detailed_results.keys()):
            data = dynamic_detailed_results[k_effective]
            failures = [d["avg_failed"] for d in data]
            success = [d["success_rate"] for d in data]
            ax2.plot(failures, success, marker='o', label=f'k={k_effective}', linewidth=2)
        ax2.set_xlabel("Durchschnittliche ausgefallene Kanten", fontsize=11)
        ax2.set_ylabel("Erfolgsrate", fontsize=11)
        ax2.set_title("Dynamisch: Success Rate vs Avg Failures", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"KF-Routing: Statisch vs Dynamisch (λ={lam})", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "combined_success_rate.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bounces
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for k_effective in sorted(resilience_results.keys()):
            data = resilience_results[k_effective]
            failures = [d["num_failures"] for d in data]
            bounces = [d["avg_bounces"] for d in data]
            ax1.plot(failures, bounces, marker='s', label=f'k={k_effective}', linewidth=2)
        ax1.set_xlabel("Anzahl ausgefallener Kanten", fontsize=11)
        ax1.set_ylabel("Durchschnittliche Bounces", fontsize=11)
        ax1.set_title("Statisch: Bounces bei festen Failures", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        for k_effective in sorted(dynamic_detailed_results.keys()):
            data = dynamic_detailed_results[k_effective]
            failures = [d["avg_failed"] for d in data]
            bounces = [d["avg_bounces"] for d in data]
            ax2.plot(failures, bounces, marker='s', label=f'k={k_effective}', linewidth=2)
        ax2.set_xlabel("Durchschnittliche ausgefallene Kanten", fontsize=11)
        ax2.set_ylabel("Durchschnittliche Bounces", fontsize=11)
        ax2.set_title("Dynamisch: Bounces vs Avg Failures", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"KF-Routing: Bounces Statisch vs Dynamisch (λ={lam})", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "combined_bounces.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Kombinierte Plots gespeichert")
    
    # Statische Plots
    if mode_choice in ["2", "3"] and resilience_results:
        print("\n[Erstelle statische Einzelplots...]")
        
        plt.figure(figsize=(10, 6))
        for k_effective in sorted(resilience_results.keys()):
            data = resilience_results[k_effective]
            failures = [d["num_failures"] for d in data]
            success = [d["success_rate"] for d in data]
            plt.plot(failures, success, marker='o', label=f'k={k_effective}', linewidth=2)
        plt.xlabel("Anzahl ausgefallener Kanten")
        plt.ylabel("Erfolgsrate")
        plt.title(f"KF-Routing: Statische Erfolgsrate (λ={lam})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "static_success_rate.png"))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        for k_effective in sorted(resilience_results.keys()):
            data = resilience_results[k_effective]
            failures = [d["num_failures"] for d in data]
            bounces = [d["avg_bounces"] for d in data]
            plt.plot(failures, bounces, marker='o', label=f'k={k_effective}', linewidth=2)
        plt.xlabel("Anzahl ausgefallener Kanten")
        plt.ylabel("Durchschnittliche Bounces")
        plt.title(f"KF-Routing: Statische Bounces (λ={lam})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "static_bounces.png"))
        plt.close()
        
        # CSV Export
        resilience_rows = []
        for k_effective, data in resilience_results.items():
            for entry in data:
                resilience_rows.append({"k_effective": k_effective, **entry})
        df_res = pd.DataFrame(resilience_rows)
        csv_res_path = os.path.join(out_dir, "static_analysis.csv")
        df_res.to_csv(csv_res_path, index=False, sep=";")
        print(f"✓ Statische CSV gespeichert: {csv_res_path}")
    
    # Dynamische Plots
    if mode_choice in ["1", "3"] and dynamic_detailed_results:
        print("\n[Erstelle dynamische Einzelplots...]")
        
        plt.figure(figsize=(10, 6))
        for k_effective in sorted(dynamic_detailed_results.keys()):
            data = dynamic_detailed_results[k_effective]
            failures = [d["avg_failed"] for d in data]
            success = [d["success_rate"] for d in data]
            plt.plot(failures, success, marker='o', label=f'k={k_effective}', linewidth=2)
        plt.xlabel("Durchschnittliche ausgefallene Kanten")
        plt.ylabel("Erfolgsrate")
        plt.title(f"KF-Routing: Dynamische Erfolgsrate (λ={lam})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "dynamic_success_rate.png"))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        for k_effective in sorted(dynamic_detailed_results.keys()):
            data = dynamic_detailed_results[k_effective]
            failures = [d["avg_failed"] for d in data]
            bounces = [d["avg_bounces"] for d in data]
            plt.plot(failures, bounces, marker='o', label=f'k={k_effective}', linewidth=2)
        plt.xlabel("Durchschnittliche ausgefallene Kanten")
        plt.ylabel("Durchschnittliche Bounces")
        plt.title(f"KF-Routing: Dynamische Bounces (λ={lam})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "dynamic_bounces.png"))
        plt.close()
        
        # CSV Export
        dynamic_rows = []
        for k_effective, data in dynamic_detailed_results.items():
            for entry in data:
                dynamic_rows.append({"k_effective": k_effective, **entry})
        df_dyn = pd.DataFrame(dynamic_rows)
        csv_dyn_path = os.path.join(out_dir, "dynamic_analysis.csv")
        df_dyn.to_csv(csv_dyn_path, index=False, sep=";")
        print(f"✓ Dynamische CSV gespeichert: {csv_dyn_path}")
        
        # Dynamische Pfad-Visualisierungen (reduziert für Geschwindigkeit)
        print("\n[Erstelle 3 dynamische Pfad-Beispiele...]")
        
        middle_k = sorted(dynamic_detailed_results.keys())[len(dynamic_detailed_results) // 2] if dynamic_detailed_results else None
        if middle_k:
            psn_middle = build_psn(G, target, middle_k)
            middle_fr = failure_rates[len(failure_rates) // 2]
            
            failed_edges = set()
            step_samples = [0, 10, 19]  # Nur 3 Zeitschritte
            
            for step in range(20):
                failed_edges = update_failures(G, failed_edges, failure_rate=middle_fr, recovery_rate=0.1)
                
                if step in step_samples:
                    success, bounces, hops, path = kf_route_packet(G, psn_middle, source, target, failed_edges, middle_k)
                    
                    # Zeige ausgefallene Kanten im Terminal
                    failed_list = sorted(list({(min(u, v), max(u, v)) for u, v in failed_edges}))
                    failed_str = ", ".join([f"{u}-{v}" for u, v in failed_list])
                    
                    if success:
                        print(f"  ✓ Zeitschritt {step+1}: Pfad {' → '.join(str(n) for n in path)}, {len(failed_edges)//2} Failures")
                        print(f"    Ausgefallene Kanten: {failed_str if failed_str else 'Keine'}")
                        
                        title = f"Dynamisch: Zeitschritt {step+1} (k={middle_k}, FR={middle_fr}, {bounces} Bounces)"
                        save_path = os.path.join(out_dir, f"dynamic_path_step{step+1}_k{middle_k}.png")
                        visualize_path_on_graph(G, path, source, target, failed_edges=failed_edges, title=title, save_path=save_path)
                    else:
                        print(f"  ✗ Zeitschritt {step+1}: KEIN PFAD GEFUNDEN, {len(failed_edges)//2} Failures")
                        print(f"    Ausgefallene Kanten: {failed_str}")
                        print(f"    Versuchter Pfad: {' → '.join(str(n) for n in path)}")
                        
                        title = f"Dynamisch: Zeitschritt {step+1} FEHLGESCHLAGEN (k={middle_k}, FR={middle_fr})"
                        save_path = os.path.join(out_dir, f"dynamic_path_step{step+1}_k{middle_k}_FAILED.png")
                        visualize_path_on_graph(G, path, source, target, failed_edges=failed_edges, title=title, save_path=save_path)
            
            print(f"✓ Dynamische Pfad-Beispiele erstellt")
    
    # Pfad-Analyse (Statisch)
    if mode_choice in ["2", "3"] and static_paths:
        print("\n" + "="*50)
        print("STATISCHE PFAD-VISUALISIERUNGEN")
        print("="*50)
        
        middle_k = sorted(zero_failure_paths.keys())[len(zero_failure_paths) // 2] if zero_failure_paths else None
        if middle_k and zero_failure_paths[middle_k]:
            print(f"\nErstelle Beispiel-Pfadvisualisierungen für k={middle_k}...")
            
            psn_middle = build_psn(G, target, middle_k)
            failure_levels = [0, 2, 5, 8, 12, 15]
            
            for fail_count in failure_levels[:6]:
                all_edges = list(G.edges())
                failed = set()
                
                if fail_count > 0:
                    edge_betweenness = nx.edge_betweenness_centrality(G)
                    # Normalisiere für undirected
                    edge_betweenness_norm = {(min(u, v), max(u, v)): bc for (u, v), bc in edge_betweenness.items()}
                    sorted_edges = sorted(edge_betweenness_norm.items(), key=lambda x: x[1], reverse=True)
                    fail_list = [edge for edge, _ in sorted_edges[:fail_count]]
                    
                    failed = set(fail_list)
                
                success, bounces, hops, path = kf_route_packet(G, psn_middle, source, target, failed, middle_k)
                
                # Zeige ausgefallene Kanten im Terminal
                failed_list = sorted(list({(min(u, v), max(u, v)) for u, v in failed}))
                failed_str = ", ".join([f"{u}-{v}" for u, v in failed_list])
                
                if success:
                    print(f"  ✓ k={middle_k}, {fail_count} Failures: Pfad {' → '.join(str(n) for n in path)}")
                    print(f"    Ausgefallene Kanten: {failed_str if failed_str else 'Keine'}")
                    
                    title = f"KF-Routing: Pfad mit {fail_count} Failures (k={middle_k}, {bounces} Bounces)"
                    save_path = os.path.join(out_dir, f"static_path_{fail_count}failures_k{middle_k}.png")
                    visualize_path_on_graph(G, path, source, target, failed_edges=failed, title=title, save_path=save_path)
                else:
                    print(f"  ✗ k={middle_k}, {fail_count} Failures: KEIN PFAD GEFUNDEN")
                    print(f"    Ausgefallene Kanten: {failed_str}")
                    print(f"    Versuchter Pfad: {' → '.join(str(n) for n in path)}")
                    
                    # Erstelle auch für fehlgeschlagene Fälle eine Visualisierung
                    title = f"KF-Routing: FEHLGESCHLAGEN bei {fail_count} Failures (k={middle_k})"
                    save_path = os.path.join(out_dir, f"static_path_{fail_count}failures_k{middle_k}_FAILED.png")
                    visualize_path_on_graph(G, path, source, target, failed_edges=failed, title=title, save_path=save_path)
        
        print(f"✓ Pfad-Analyse-Plots erstellt")
    
    print(f"\n{'='*50}")
    print(f"ALLE ERGEBNISSE GESPEICHERT IN:")
    print(f"{out_dir}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()


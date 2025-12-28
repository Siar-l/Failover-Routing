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

def build_spanning_trees(G, target, k):
    """Erstellt k verschiedene Spanning Trees"""
    trees = []
    for _ in range(k):
        for (u, v) in G.edges():
            G[u][v]['weight'] = random.random()
        T = nx.minimum_spanning_tree(G)
        trees.append(T)
    return trees

def extend_edges_to_maximal(G, tree, target):
    """
    Erweitert einen Spanning Tree zu einem maximalen DAG durch Hinzufügen von Kanten,
    die keine Zyklen erzeugen und Richtung zum Target haben.
    """
    dag = tree.copy()
    
    # Berechne Distanzen zum Target im Tree
    try:
        distances = nx.single_source_shortest_path_length(tree, target)
    except:
        return dag
    
    # Versuche alle Kanten aus G hinzuzufügen
    for u, v in G.edges():
        if dag.has_edge(u, v) or dag.has_edge(v, u):
            continue
        
        # Prüfe ob beide Knoten im Tree sind
        if u not in distances or v not in distances:
            continue
        
        # Füge Kante hinzu wenn sie zum Target zeigt (kleinere Distanz)
        if distances[u] > distances[v]:
            # Kante u -> v zeigt zum Target
            dag.add_edge(u, v)
        elif distances[v] > distances[u]:
            # Kante v -> u zeigt zum Target
            dag.add_edge(v, u)
    
    return dag

def link_is_up(u, v, failed_edges: set[tuple]) -> bool:
    return (u, v) not in failed_edges and (v, u) not in failed_edges

def simulate_failover(G, trees, source, target, failed_edges, k):
    """
    Simuliert Failover-Routing mit DAG-Extension
    """
    use_header = k >= 6
    num_trees = len(trees)
    if num_trees == 0:
        return False, 0, 0, []
    
    pkt = Packet(ttl=64, idx=(0 if use_header else None))
    current_tree = 0
    current_node = source
    visited = set()
    bounces = 0
    hops = 0
    actual_path = [source]
    
    while current_node != target and pkt.ttl > 0:
        if not use_header:
            state = (current_node, current_tree % num_trees)
            if state in visited:
                return False, bounces, hops, actual_path
            visited.add(state)
        
        active_tree = (pkt.idx if use_header else current_tree) % num_trees
        T = trees[active_tree]
        
        try:
            tree_path = nx.shortest_path(T, source=current_node, target=target)
            if len(tree_path) < 2:
                raise nx.NetworkXNoPath
            
            next_node = tree_path[1]
        
        except Exception:
            if use_header:
                pkt.idx = (0 if pkt.idx is None else pkt.idx + 1)
            else:
                current_tree += 1
            bounces += 1
            if (use_header and (pkt.idx is not None and pkt.idx >= num_trees)) or \
               (not use_header and current_tree >= num_trees):
                return False, bounces, hops, actual_path
            continue
        
        if not link_is_up(current_node, next_node, failed_edges):
            if use_header:
                pkt.idx = (0 if pkt.idx is None else pkt.idx + 1)
            else:
                current_tree += 1
            bounces += 1
            if (use_header and (pkt.idx is not None and pkt.idx >= num_trees)) or \
               (not use_header and current_tree >= num_trees):
                return False, bounces, hops, actual_path
            continue
        
        pkt.ttl -= 1
        current_node = next_node
        actual_path.append(current_node)
        hops += 1
    
    return (current_node == target), bounces, hops, actual_path

def update_failures(G, failed_edges, failure_rate=0.2, recovery_rate=0.1):
    """
    Aktualisiert failed_edges basierend auf dem ursprünglichen Graphen G (ungerichtet).
    Speichert beide Richtungen für bidirektionale Failures.
    """
    # Verwende den ursprünglichen Graph für Failures, nicht die DAGs/Trees
    edges = list(G.edges())
    
    # Normalisiere Kanten für konsistente Zählung
    normalized_edges = [(min(u, v), max(u, v)) for u, v in edges]
    unique_edges = list(set(normalized_edges))
    
    for e in unique_edges:
        u, v = e
        # Prüfe ob Kante bereits ausgefallen (in normalisierter Form)
        if e not in {(min(a, b), max(a, b)) for a, b in failed_edges} and random.random() < failure_rate:
            # Füge beide Richtungen hinzu (bidirektionaler Ausfall)
            failed_edges.add((u, v))
            failed_edges.add((v, u))
    
    # Recovery: Entferne beide Richtungen
    for e in list(failed_edges):
        if random.random() < recovery_rate:
            failed_edges.discard(e)
            failed_edges.discard((e[1], e[0]))
    
    return failed_edges

def dynamic_simulation(G, trees, source, target, k, steps=10, failure_rate=0.4, recovery_rate=0.1):
    """
    Dynamische Simulation mit zeitlich variierenden Kanten-Ausfällen.
    G: Ursprünglicher Graph (ungerichtet) - wird für Failures verwendet
    trees: Liste von DAGs/Trees für Routing
    """
    failed_edges = set()
    success_log = []
    bounces_log = []
    failed_count_log = []
    pathlen_log = []
    all_paths = []
    all_failed_edges = []
    
    for t in range(steps):
        # Aktualisiere Failures basierend auf dem URSPRÜNGLICHEN Graph G
        failed_edges = update_failures(G, failed_edges, failure_rate=failure_rate, recovery_rate=recovery_rate)
        
        success, bounces, hops, path = simulate_failover(G, trees, source, target, failed_edges, k)
        
        success_log.append(1 if success else 0)
        bounces_log.append(bounces)
        # Zähle unique ausgefallene Kanten (normalisiert)
        unique_failed = len({(min(u, v), max(u, v)) for u, v in failed_edges})
        failed_count_log.append(unique_failed)
        
        if success:
            pathlen_log.append(hops)
            all_paths.append(path)
            all_failed_edges.append(failed_edges.copy())
        
        print(f"\n Zeit = {t+1}")
        print("-> Routing erfolgreich:", success)
        print("Bounce:", bounces)
        print("-> Aktuel defekte Kanten:", len(failed_edges) // 2)
    
    # Pfad-Statistiken
    unique_paths = len(set(tuple(p) for p in all_paths))
    avg_path_length = sum(len(p) - 1 for p in all_paths) / len(all_paths) if all_paths else 0
    
    print(f"\n=== Pfad-Statistiken (DAG-Extension) ===")
    print(f"Gesamte erfolgreiche Pfade: {len(all_paths)}")
    print(f"Einzigartige Pfade: {unique_paths}")
    print(f"Durchschnittliche Pfadlänge: {avg_path_length:.2f}")
    
    return {
        "success_log": success_log,
        "bounce_log": bounces_log,
        "failed_count_log": failed_count_log,
        "pathlen_log": pathlen_log,
        "all_paths": all_paths,
        "all_failed_edges": all_failed_edges
    }

def analyze_resilience(G, trees, source, target, k, num_trials=50, max_failures=50, static_failures=True):
    if max_failures is None:
        max_failures = min(G.number_of_edges() // 2, 15)
    
    resilience_data = []
    all_paths = []
    zero_failure_paths = []
    
    # Für statische Analyse: Erstelle feste Failure-Sets
    fixed_failure_sets = {}
    if static_failures:
        all_edges = list(G.edges())
        edge_betweenness = nx.edge_betweenness_centrality(G)
        sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
        
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
        
        fixed_failed = set()
        if static_failures and num_fails > 0:
            fail_list = fixed_failure_sets[num_fails]
            for e in fail_list:
                fixed_failed.add(e)
                fixed_failed.add((e[1], e[0]))
        
        for trial in range(num_trials):
            if static_failures:
                failed = fixed_failed
            else:
                all_edges = list(G.edges())
                failed = set()
                if num_fails > 0:
                    fail_list = random.sample(all_edges, min(num_fails, len(all_edges)))
                    for e in fail_list:
                        failed.add(e)
                        failed.add((e[1], e[0]))
            
            success, bounces, hops, path = simulate_failover(G, trees, source, target, failed, k)
            
            if success:
                successes += 1
                successful_hops.append(hops)
                total_bounces += bounces
                total_hops += hops
                all_paths.append(path)
                
                if num_fails == 0:
                    zero_failure_paths.append(path)
        
        success_rate = successes / num_trials
        avg_bounces = total_bounces / num_trials
        avg_hops = sum(successful_hops) / len(successful_hops) if successful_hops else 0
        
        unique_paths = len(set(tuple(p) for p in all_paths)) if all_paths else 0
        avg_path_length = sum(len(p) for p in all_paths) / len(all_paths) if all_paths else 0
        
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
    
    # Statistiken
    unique_paths_total = len(set(tuple(p) for p in all_paths))
    avg_path_length_total = sum(len(p) - 1 for p in all_paths) / len(all_paths) if all_paths else 0
    
    unique_zero_failure = len(set(tuple(p) for p in zero_failure_paths))
    avg_zero_failure_length = sum(len(p) - 1 for p in zero_failure_paths) / len(zero_failure_paths) if zero_failure_paths else 0
    
    print(f"\n=== Pfad-Statistiken (DAG-Extension) ===")
    print(f"Gesamte erfolgreiche Pfade: {len(all_paths)}")
    print(f"Einzigartige Pfade: {unique_paths_total}")
    print(f"Durchschnittliche Pfadlänge (alle): {avg_path_length_total:.2f}")
    print(f"Bei 0 Failures: {unique_zero_failure} einzigartige Pfade, Ø Länge: {avg_zero_failure_length:.2f}")
    
    return resilience_data, all_paths, zero_failure_paths

def visualize_path_on_graph(G, path, source, target, failed_edges=None, title="DAG-Extension Path", save_path=None, show_all_failures=True):
    """Visualisiert DAG-Extension Pfad"""
    if failed_edges is None:
        failed_edges = set()
    
    plt.figure(figsize=(16, 11))
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
    # Normalisiere Kanten
    path_edges = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        path_edges.add((min(u, v), max(u, v)))
    
    normalized_failed = set()
    for u, v in failed_edges:
        normalized_failed.add((min(u, v), max(u, v)))
    
    # Zeichne normale Kanten
    all_edges = []
    for u, v in G.edges():
        edge_norm = (min(u, v), max(u, v))
        if edge_norm not in path_edges and edge_norm not in normalized_failed:
            all_edges.append((u, v))
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color='lightgray', width=0.5, alpha=0.3)
    
    # Zeichne Failed Edges
    if show_all_failures and failed_edges:
        failed_edge_list = [(u, v) for u, v in G.edges() if (min(u, v), max(u, v)) in normalized_failed]
        if failed_edge_list:
            nx.draw_networkx_edges(G, pos, edgelist=failed_edge_list,
                                  edge_color='red', width=2, alpha=0.8, style='dashed')
    
    # Zeichne Pfad-Kanten
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
    
    num_failed = len(failed_edges) // 2 if failed_edges else 0
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
    
    print("=== DAG-EXTENSION FAILOVER ROUTING ===\n")
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
            seed = int(input("Seed (Default 42 für Reproduzierbarkeit): ").strip() or "42")
            G = nx.erdos_renyi_graph(n, p, seed=seed).to_undirected()
            graph_name = f"ER_n{n}_p{p}_seed{seed}"
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
    
    random.seed(42)
    k_values = list(range(1, lam + 1))
    
    dynamic_results = {}
    dynamic_detailed_results = {}
    resilience_results = {}
    dynamic_paths_for_viz = {}
    
    # Dynamische Simulation
    if mode_choice in ["1", "3"]:
        print("\n" + "="*50)
        print("DYNAMISCHE SIMULATION (DAG-EXTENSION)")
        print("="*50)
        
        failure_rates = [0.05, 0.10, 0.15, 0.20, 0.25]
        
        for k_desired in k_values:
            k_effective = min(k_desired, edge_connectivity(G))
            print(f"\n[Dynamisch] k = {k_effective}")
            
            trees = build_spanning_trees(G, target, k_effective)
            
            # Erweitere Trees zu DAGs
            dags = [extend_edges_to_maximal(G, tree, target) for tree in trees]
            
            k_results = []
            middle_fr = failure_rates[len(failure_rates)//2]
            
            for fr in failure_rates:
                result = dynamic_simulation(G, dags, source, target, k_effective,
                                          steps=20, failure_rate=fr, recovery_rate=0.1)
                
                success_rate = sum(result["success_log"]) / len(result["success_log"]) if result["success_log"] else 0.0
                avg_bounces = sum(result["bounce_log"]) / len(result["bounce_log"]) if result["bounce_log"] else 0.0
                avg_failed = sum(result["failed_count_log"]) / len(result["failed_count_log"]) if result["failed_count_log"] else 0.0
                
                k_results.append({
                    "failure_rate": fr,
                    "success_rate": success_rate,
                    "avg_bounces": avg_bounces,
                    "avg_failed": avg_failed
                })
                print(f"  failure_rate={fr:.2f} → avg_failed={avg_failed:.2f}, success={success_rate:.1%}, bounces={avg_bounces:.2f}")
                
                # Speichere Pfade für mittlere failure_rate
                if fr == middle_fr and result["all_paths"]:
                    dynamic_paths_for_viz[k_effective] = {
                        "paths": result["all_paths"],
                        "failed_edges": result["all_failed_edges"],
                        "bounces": result["bounce_log"],
                        "failed_counts": result["failed_count_log"]
                    }
            
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
    
    # Statische Resilienz-Analyse
    static_paths = {}
    zero_failure_paths = {}
    if mode_choice in ["2", "3"]:
        print("\n" + "="*50)
        print("STATISCHE RESILIENZ-ANALYSE (DAG-EXTENSION)")
        print("="*50)
        print("Hinweis: Verwendet deterministische Ausfälle basierend auf Kanten-Wichtigkeit\n")
        
        for k_desired in k_values:
            k_effective = min(k_desired, edge_connectivity(G))
            print(f"\n[Statisch] k = {k_effective}")
            trees = build_spanning_trees(G, target, k_effective)
            
            # Erweitere Trees zu DAGs
            dags = [extend_edges_to_maximal(G, tree, target) for tree in trees]
            
            res_data, all_paths, zero_paths = analyze_resilience(G, dags, source, target, k_effective,
                                                                 num_trials=50, max_failures=None, static_failures=True)
            resilience_results[k_effective] = res_data
            static_paths[k_effective] = all_paths
            zero_failure_paths[k_effective] = zero_paths
    
    # Output-Verzeichnis
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_DAG", graph_name)
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
        
        plt.suptitle(f"DAG-Extension: Statisch vs Dynamisch (λ={lam})", fontsize=14, y=1.02)
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
        
        plt.suptitle(f"DAG-Extension: Bounces Statisch vs Dynamisch (λ={lam})", fontsize=14, y=1.02)
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
        plt.title(f"DAG-Extension: Statische Erfolgsrate (λ={lam})")
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
        plt.title(f"DAG-Extension: Statische Bounces (λ={lam})")
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
        plt.title(f"DAG-Extension: Dynamische Erfolgsrate (λ={lam})")
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
        plt.title(f"DAG-Extension: Dynamische Bounces (λ={lam})")
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
    
    # Dynamische Pfad-Visualisierungen
    if mode_choice in ["1", "3"] and dynamic_paths_for_viz:
        print("\n========== DYNAMISCHE PFAD-VISUALISIERUNGEN ==========")
        
        for k_effective, data in dynamic_paths_for_viz.items():
            print(f"\nErstelle dynamische Visualisierungen für k={k_effective}...")
            paths = data["paths"]
            failed_edges_list = data["failed_edges"]
            bounces = data["bounces"]
            failed_counts = data["failed_counts"]
            
            # Visualisiere erste 6 Zeitschritte
            num_viz = min(6, len(paths))
            for t in range(num_viz):
                path = paths[t]
                failed = failed_edges_list[t]
                bounce = bounces[t] if t < len(bounces) else 0
                failed_count = failed_counts[t] if t < len(failed_counts) else 0
                
                title = f"DAG-Extension Dynamisch Zeit={t+1} (k={k_effective}, {failed_count} Failures, {bounce} Bounces)"
                save_path = os.path.join(out_dir, f"dynamic_time{t+1}_k{k_effective}.png")
                visualize_path_on_graph(G, path, source, target, failed_edges=failed, title=title, save_path=save_path)
            
            print(f"✓ {num_viz} dynamische Visualisierungen erstellt")
    
    # Pfad-Analyse
    if mode_choice in ["2", "3"] and static_paths:
        print("\n" + "="*50)
        print("PFAD-ANALYSE UND VISUALISIERUNGEN")
        print("="*50)
        
        middle_k = sorted(zero_failure_paths.keys())[len(zero_failure_paths) // 2] if zero_failure_paths else None
        if middle_k and zero_failure_paths[middle_k]:
            print(f"\nErstelle Beispiel-Pfadvisualisierungen für k={middle_k}...")
            
            trees_middle = build_spanning_trees(G, target, middle_k)
            dags_middle = [extend_edges_to_maximal(G, tree, target) for tree in trees_middle]
            
            failure_levels = [0, 2, 5, 8, 12, 15]
            
            for fail_count in failure_levels[:6]:
                all_edges = list(G.edges())
                failed = set()
                
                if fail_count > 0:
                    edge_betweenness = nx.edge_betweenness_centrality(G)
                    sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
                    fail_list = [edge for edge, _ in sorted_edges[:fail_count]]
                    
                    for e in fail_list:
                        failed.add(e)
                        failed.add((e[1], e[0]))
                
                success, bounces, hops, path = simulate_failover(G, dags_middle, source, target, failed, middle_k)
                
                if success:
                    title = f"DAG-Extension: Pfad mit {fail_count} Failures (k={middle_k}, {bounces} Bounces)"
                    save_path = os.path.join(out_dir, f"static_path_{fail_count}failures_k{middle_k}.png")
                    visualize_path_on_graph(G, path, source, target, failed_edges=failed, title=title, save_path=save_path)
                else:
                    print(f"  Warnung: Kein erfolgreicher Pfad mit {fail_count} Failures gefunden")
        
        print(f"✓ Pfad-Analyse-Plots erstellt")
    
    print(f"\n{'='*50}")
    print(f"ALLE ERGEBNISSE GESPEICHERT IN:")
    print(f"{out_dir}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()





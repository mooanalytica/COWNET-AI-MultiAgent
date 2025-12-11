import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Dict, Any, List
from scipy import stats
import math

def create_social_network_graph(interaction_counts: pd.DataFrame) -> dict:
    """
    Create NetworkX graph from interaction counts with inverse distance weights.
    
    Args:
        interaction_counts: DataFrame with 'cow_i', 'cow_j', 'count' columns
        
    Returns:
        (G, edge_lengths): Graph with 'distance' attributes, and edge_lengths dict
    """
    G = nx.Graph()
    
    # Add edges with interaction counts
    for _, row in interaction_counts.iterrows():
        cow1, cow2, count = row['cow_i'], row['cow_j'], row['count']
        G.add_edge(cow1, cow2, count=count)
    
    # Calculate edge lengths inversely proportional to interaction count
    max_count = interaction_counts['count'].max()
    min_length = 0.2
    max_length = 2.0
    edge_lengths = {}
    
    for _, row in interaction_counts.iterrows():
        count = row['count']
        # Inverse scaling: more interactions = shorter edge
        length = max_length - (count - 1) / (max_count - 1 + 1e-9) * (max_length - min_length)
        edge_lengths[(row['cow_i'], row['cow_j'])] = length
    
    # Set distance attributes on edges
    for (u, v), length in edge_lengths.items():
        G[u][v]['distance'] = length
    
    return nx.to_dict_of_dicts(G)

def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-np.clip(x, -10, 10)))


def robust_z_score(values: Dict[str, float]) -> Dict[str, float]:
    """Robust z-score normalization using median/IQR (herd-level)."""
    if not values:
        return {}
    
    scores_array = np.array(list(values.values()))
    median = np.median(scores_array)
    q75 = np.percentile(scores_array, 75)
    q25 = np.percentile(scores_array, 25)
    iqr = q75 - q25
    
    if iqr == 0:
        iqr = np.std(scores_array) * 1.35
        if iqr == 0:
            iqr = 1.0
    
    normalized_scores = {}
    for cow_id, score in values.items():
        z_robust = (score - median) / (iqr / 1.35)
        normalized_scores[cow_id] = z_robust
    return normalized_scores


def compute_centralities(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    """Compute all standard centralities for the graph."""
    if len(G.nodes()) == 0:
        return {'betweenness': {}, 'degree': {}, 'closeness': {}}
    
    return {
        'betweenness': nx.betweenness_centrality(G, normalized=True),
        'degree': nx.degree_centrality(G),
        'closeness': nx.closeness_centrality(G)
    }


def compute_community_disruption(G: nx.Graph, cow_id: str) -> float:
    """Compute community disruption score for a specific cow."""
    if cow_id not in G.nodes() or len(G.nodes()) < 3:
        return 0.0
    
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
    except:
        return 0.0
    
    # Find cow's community
    cow_community = None
    for i, comm in enumerate(communities):
        if cow_id in comm:
            cow_community = i
            break
    
    if cow_community is None:
        return 0.0
    
    # Count cross-community neighbors
    neighbors = list(G.neighbors(cow_id))
    if not neighbors:
        return 0.0
    
    cross_connections = sum(
        1 for n in neighbors 
        if any(n in comm for i, comm in enumerate(communities) if i != cow_community)
    )
    return cross_connections / len(neighbors)


def compute_per_cow_sna_metrics(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive SNA metrics for each cow (node-level).
    
    Returns: {cow_id: {'betweenness': float, 'degree': float, 'closeness': float, 
                       'community_disruption': float, ...}}
    """
    centralities = compute_centralities(G)
    
    # Compute additional metrics for each cow
    metrics = {}
    for cow_id in G.nodes():
        metrics[cow_id] = {
            **{k: v.get(cow_id, 0.0) for k, v in centralities.items()},
            'community_disruption': compute_community_disruption(G, cow_id),
            'degree': G.degree(cow_id),  # Raw degree (not normalized)
        }
    
    return metrics


def compute_herd_level_metrics(G: nx.Graph, per_cow_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Compute herd-level (aggregate) metrics.
    
    Returns: {'avg_degree': float, 'density': float, 'max_betweenness': float, ...}
    """
    if len(G.nodes()) == 0:
        return {}
    
    # Basic network stats
    herd_metrics = {
        'num_cows': len(G.nodes()),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': sum(G.degree(n) for n in G.nodes()) / len(G.nodes()),
        'diameter': nx.diameter(G) if nx.is_connected(G) else float('inf'),
    }
    
    # Aggregate centralities across herd
    all_betweenness = {cow: m['betweenness'] for cow, m in per_cow_metrics.items()}
    all_degree = {cow: m['degree'] for cow, m in per_cow_metrics.items()}
    
    herd_metrics.update({
        'avg_betweenness': np.mean(list(all_betweenness.values())),
        'max_betweenness': max(all_betweenness.values()) if all_betweenness else 0,
        'avg_degree_centrality': np.mean(list(all_degree.values())),
        'max_degree': max(herd_metrics['avg_degree'], max(G.degree(n) for n in G.nodes())),
    })
    
    return herd_metrics


def compute_risk_scores(G: nx.Graph, per_cow_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute conflict and isolation risk scores for each cow using the original weighted formula.
    
    Returns: {cow_id: {'conflict_risk': float, 'isolation_risk': float}}
    """
    # Normalize all metrics using robust z-score
    norm_betweenness = robust_z_score({c: m['betweenness'] for c, m in per_cow_metrics.items()})
    norm_degree = robust_z_score({c: m['degree'] for c, m in per_cow_metrics.items()})
    
    risk_scores = {}
    for cow_id in G.nodes():
        # Extract normalized components
        btw = norm_betweenness.get(cow_id, 0)
        degree = per_cow_metrics[cow_id]['degree']
        comm_disrupt = per_cow_metrics[cow_id]['community_disruption']
        deg_centrality = per_cow_metrics[cow_id]['degree']
        closeness = per_cow_metrics[cow_id]['closeness']
        
        # Original weighted formulas (simplified - no temporal components)
        conflict = (
            0.50 * sigmoid(btw) +
            0.15 * sigmoid(comm_disrupt) +
            0.35 * (1 - deg_centrality)  # High centrality reduces conflict risk
        )
        isolation = (
            0.50 * (1 - deg_centrality) +      # Low degree = isolation
            0.30 * (1 - closeness) +           # Low closeness = isolation
            0.20 * abs(degree - np.mean(list(norm_degree.values())))  # Degree deviation
        )
        
        risk_scores[cow_id] = {
            'conflict_risk': min(1.0, conflict),
            'isolation_risk': min(1.0, isolation)
        }
    
    return risk_scores


# ---------- Demo Usage ----------
def get_demo_sna_results(G: nx.Graph) -> Dict[str, Any]:
    """
    Main function for LangGraph agent tools - returns all metrics in LangGraph-friendly format.
    
    Returns:
    {
        'herd_metrics': {...},
        'per_cow_metrics': {cow_id: {...}},
        'risk_scores': {cow_id: {'conflict_risk': float, 'isolation_risk': float}},
        'top_risk_cows': [{'cow_id': str, 'conflict_risk': float, 'isolation_risk': float}, ...]
    }
    """
    per_cow_metrics = compute_per_cow_sna_metrics(G)
    herd_metrics = compute_herd_level_metrics(G, per_cow_metrics)
    risk_scores = compute_risk_scores(G, per_cow_metrics)
    
    # Top risk cows for quick agent access
    top_risks = sorted(
        risk_scores.items(),
        key=lambda x: max(x[1]['conflict_risk'], x[1]['isolation_risk']),
        reverse=True
    )[:5]
    
    return {
        'herd_metrics': herd_metrics,
        'per_cow_metrics': per_cow_metrics,
        'risk_scores': risk_scores,
        'top_risk_cows': [{'cow_id': cow_id, **scores} for cow_id, scores in top_risks]
    }


"""
Congress Network Visualization - Exercise 3 Part (a)
Visualizes the weighted directed Congress Twitter interaction network
"""

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Create Figures directory if it doesn't exist
figures_dir = Path(__file__).parent.parent / "Figures"
figures_dir.mkdir(exist_ok=True)

# Load the directed graph from edgelist
# The edgelist format is: source target {'weight': value}
G = nx.read_edgelist(
    "congress.edgelist",
    create_using=nx.DiGraph,
    nodetype=int,
    delimiter=' ',
    data=False
)

# Parse weights from the file manually
with open("congress.edgelist", 'r') as f:
    for line in f:
        parts = line.strip().split(' ')
        if len(parts) >= 3:
            u, v = int(parts[0]), int(parts[1])
            # Extract weight from the dictionary string
            weight_str = ' '.join(parts[2:]).strip("'{}'")
            weight_val = float(weight_str.split(': ')[1].rstrip('}'))
            G[u][v]['weight'] = weight_val

print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Compute layout - Spring layout (Fruchterman-Reingold algorithm)
# This is ideal for this network because:
# 1. It minimizes edge crossings, making the structure more visible
# 2. Connected components naturally cluster together
# 3. It handles weighted edges - we use edge weights inversely as distances
print("Computing spring layout (this may take a minute)...")

# Use weights inversely as distances: high weight (strong connection) = close nodes
pos = nx.spring_layout(
    G,
    k=0.5,  # Optimal distance between nodes
    iterations=50,
    weight='weight',
    seed=42
)

# Create visualization
fig, ax = plt.subplots(figsize=(16, 12), dpi=150)

# Draw nodes
node_sizes = 300
nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_color='lightblue',
    node_size=node_sizes,
    alpha=0.8,
    ax=ax,
    edgecolors='navy',
    linewidths=0.5
)

# Draw edges with transparency based on weight
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
max_weight = max(weights)

# Normalize weights for visualization
normalized_weights = [w / max_weight for w in weights]

nx.draw_networkx_edges(
    G,
    pos,
    width=[0.1 + 1.5 * w for w in normalized_weights],  # Edge width proportional to weight
    alpha=0.15,
    ax=ax,
    edge_color='gray',
    arrowsize=8,
    arrowstyle='->'
)

# Add labels for highly connected nodes (top 10 by degree)
degrees = dict(G.degree())
top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
labels = {node: str(node) for node, _ in top_nodes}

nx.draw_networkx_labels(
    G,
    pos,
    labels,
    font_size=8,
    font_color='darkblue',
    ax=ax
)

ax.set_title(
    "Congress Network: Twitter Interactions Between Members\n"
    "Spring Layout (Fruchterman-Reingold) | Node size proportional to degree",
    fontsize=14,
    fontweight='bold',
    pad=20
)
ax.axis('off')
plt.tight_layout()

# Save figure
output_path = figures_dir / "congress_network_layout.pdf"
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Figure saved to {output_path}")

# Also save as PNG for display
output_path_png = figures_dir / "congress_network_layout.png"
plt.savefig(output_path_png, format='png', bbox_inches='tight', dpi=150)
print(f"PNG also saved to {output_path_png}")

plt.close()

# Print network statistics
print("\nNetwork Statistics:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
print(f"Network density: {nx.density(G):.4f}")

# Check connectivity
if nx.is_weakly_connected(G):
    print("Network is weakly connected (connected when treated as undirected)")
else:
    print("Network is NOT weakly connected")
    num_components = nx.number_weakly_connected_components(G)
    print(f"Number of weakly connected components: {num_components}")

print("\nVisualization complete!")

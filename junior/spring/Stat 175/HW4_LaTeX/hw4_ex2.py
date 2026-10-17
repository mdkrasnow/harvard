"""
Stat 175 HW4 Exercise 2: Community Detection on Yeast PPI Network
"""
import numpy as np
import igraph as ig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import networkx as nx
import os

# ── Setup ──────────────────────────────────────────────────────────────
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures")
os.makedirs(FIGDIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
G = ig.Graph.Read_GraphML("yeast.graphml")
# Work with the largest connected component for clean spectral analysis
components = G.connected_components()
largest_cc = components.giant()
G = largest_cc
G = G.simplify()  # remove multi-edges and self-loops

print(f"Yeast PPI Network:")
print(f"  Nodes: {G.vcount()}")
print(f"  Edges: {G.ecount()}")
print(f"  Graph attributes: {G.attributes()}")
print(f"  Vertex attributes: {G.vs.attributes()}")
print(f"  Edge attributes: {G.es.attributes()}")

# ── (b) Louvain community detection ───────────────────────────────────
louvain = G.community_multilevel()
modularity_louvain = louvain.modularity
n_communities = len(louvain)
membership_louvain = louvain.membership

print(f"\n(b) Louvain Algorithm Results:")
print(f"  Modularity: {modularity_louvain:.4f}")
print(f"  Number of communities: {n_communities}")
print(f"  Community sizes: {sorted(louvain.sizes(), reverse=True)}")

# ── (c) Visualize the aggregated network ──────────────────────────────
# Build the aggregated (contracted) graph: one node per community
agg = louvain.cluster_graph(combine_edges=dict(weight="sum"))
# agg is an igraph Graph where each vertex = one community
n_agg = agg.vcount()
sizes = louvain.sizes()

# Convert to networkx for visualization
G_agg_nx = nx.Graph()
for i in range(n_agg):
    G_agg_nx.add_node(i, size=sizes[i])
for e in agg.es:
    src, tgt = e.source, e.target
    if src != tgt:
        w = e["weight"] if "weight" in agg.es.attributes() else 1
        G_agg_nx.add_edge(src, tgt, weight=w)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
pos = nx.spring_layout(G_agg_nx, seed=42, k=2.0/np.sqrt(n_agg))

# Node sizes proportional to community size
node_sizes = [sizes[i] * 15 for i in range(n_agg)]

# Edge widths proportional to number of inter-community edges
edge_weights = [G_agg_nx[u][v].get('weight', 1) for u, v in G_agg_nx.edges()]
max_w = max(edge_weights) if edge_weights else 1
edge_widths = [1 + 4 * w / max_w for w in edge_weights]

# Color by community index
cmap = plt.cm.Set3
colors = [cmap(i / n_agg) for i in range(n_agg)]

nx.draw_networkx_nodes(G_agg_nx, pos, node_size=node_sizes, node_color=colors,
                       edgecolors='black', linewidths=0.5, ax=ax)
nx.draw_networkx_edges(G_agg_nx, pos, width=edge_widths, alpha=0.4, ax=ax)
nx.draw_networkx_labels(G_agg_nx, pos, font_size=7, ax=ax,
                        labels={i: f"C{i}\n({sizes[i]})" for i in range(n_agg)})
ax.set_title("Aggregated Network from Louvain (one node per community)", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "ex2c_aggregated.pdf"), bbox_inches='tight')
plt.close()
print("\n(c) Aggregated network saved to Figures/ex2c_aggregated.pdf")

# ── (d) Laplacian Spectral Clustering Embeddings ─────────────────────
n = G.vcount()
# Build adjacency and Laplacian
A = np.array(G.get_adjacency().data)
degrees = np.array(A.sum(axis=1)).flatten()
D = np.diag(degrees)
L = D - A

# Convert to sparse for eigsh
L_sparse = csr_matrix(L.astype(float))

# We need eigenvalues 2 through d+1 (skip the trivial eigenvalue 0)
# eigsh with which='SM' gets smallest eigenvalues
max_d = 40
# Get max_d+1 smallest eigenvalues (index 0 is the trivial zero eigenvalue)
print("\n(d) Computing spectral embeddings...")
eigenvalues, eigenvectors = eigsh(L_sparse, k=max_d + 1, which='SM')

# Sort by eigenvalue
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"  Smallest eigenvalues: {eigenvalues[:5]}")

# Embeddings: use eigenvectors 1 through d (skip index 0, the constant vector)
dims = [3, 10, 40]
embeddings = {}
for d in dims:
    # Columns 1 through d (0-indexed), i.e., eigenvectors for lambda_2, ..., lambda_{d+1}
    X_embed = eigenvectors[:, 1:d+1]
    embeddings[d] = X_embed
    print(f"  d={d}: embedding shape = {X_embed.shape}")

# Visualize: plot first 2 components of each embedding colored by Louvain membership
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, d in enumerate(dims):
    X = embeddings[d]
    ax = axes[i]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=membership_louvain,
                        cmap='tab20', s=5, alpha=0.6)
    ax.set_title(f"Spectral Embedding (d={d})\nFirst 2 components", fontsize=12)
    ax.set_xlabel("$e_2$")
    ax.set_ylabel("$e_3$")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "ex2d_spectral_embeddings.pdf"), bbox_inches='tight')
plt.close()
print("  Spectral embeddings visualization saved to Figures/ex2d_spectral_embeddings.pdf")

# ── (e) K-means on spectral embeddings ───────────────────────────────
k = n_communities
print(f"\n(e) Running k-means with k={k} on each embedding...")

kmeans_results = {}
for d in dims:
    X = embeddings[d]
    km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=500)
    labels = km.fit_predict(X)
    kmeans_results[d] = labels

    # Compute modularity of these labels
    mod = G.modularity(labels)
    print(f"  d={d}: modularity = {mod:.4f}")

# Visualize k-means clusters for each d
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, d in enumerate(dims):
    X = embeddings[d]
    labels = kmeans_results[d]
    ax = axes[i]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels,
                        cmap='tab20', s=5, alpha=0.6)
    ax.set_title(f"k-means on Spectral Embedding (d={d})\nk={k}", fontsize=12)
    ax.set_xlabel("$e_2$")
    ax.set_ylabel("$e_3$")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "ex2e_kmeans_clusters.pdf"), bbox_inches='tight')
plt.close()
print("  K-means clustering visualization saved to Figures/ex2e_kmeans_clusters.pdf")

# ── (f) Compare Louvain vs Spectral Clustering ───────────────────────
print(f"\n(f) Comparison of Louvain vs Spectral Clustering:")
print(f"{'Metric':<30} {'d=3':>10} {'d=10':>10} {'d=40':>10}")
print("-" * 62)

# NMI
nmis = []
for d in dims:
    nmi = normalized_mutual_info_score(membership_louvain, kmeans_results[d])
    nmis.append(nmi)
print(f"{'NMI':<30} {nmis[0]:>10.4f} {nmis[1]:>10.4f} {nmis[2]:>10.4f}")

# ARI
aris = []
for d in dims:
    ari = adjusted_rand_score(membership_louvain, kmeans_results[d])
    aris.append(ari)
print(f"{'ARI':<30} {aris[0]:>10.4f} {aris[1]:>10.4f} {aris[2]:>10.4f}")

# Modularity
mods = []
for d in dims:
    mod = G.modularity(kmeans_results[d])
    mods.append(mod)
print(f"{'Modularity (spectral)':<30} {mods[0]:>10.4f} {mods[1]:>10.4f} {mods[2]:>10.4f}")
print(f"{'Modularity (Louvain)':<30} {modularity_louvain:>10.4f}")

# Number of non-empty communities
for d in dims:
    n_nonempty = len(set(kmeans_results[d]))
    print(f"  d={d}: {n_nonempty} non-empty clusters")

# Bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# NMI bar chart
ax = axes[0]
x = np.arange(len(dims))
ax.bar(x, nmis, color=['#4C72B0', '#55A868', '#C44E52'], width=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'd={d}' for d in dims])
ax.set_ylabel('NMI')
ax.set_title('Normalized Mutual Information\n(Louvain vs Spectral k-means)')
ax.set_ylim(0, 1)
for i, v in enumerate(nmis):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

# ARI bar chart
ax = axes[1]
ax.bar(x, aris, color=['#4C72B0', '#55A868', '#C44E52'], width=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'd={d}' for d in dims])
ax.set_ylabel('ARI')
ax.set_title('Adjusted Rand Index\n(Louvain vs Spectral k-means)')
ax.set_ylim(0, 1)
for i, v in enumerate(aris):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

# Modularity comparison
ax = axes[2]
all_mods = mods + [modularity_louvain]
labels_bar = [f'Spectral\nd={d}' for d in dims] + ['Louvain']
colors_bar = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
ax.bar(range(len(all_mods)), all_mods, color=colors_bar, width=0.5)
ax.set_xticks(range(len(all_mods)))
ax.set_xticklabels(labels_bar)
ax.set_ylabel('Modularity')
ax.set_title('Modularity Comparison')
for i, v in enumerate(all_mods):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "ex2f_comparison.pdf"), bbox_inches='tight')
plt.close()
print("\n  Comparison figure saved to Figures/ex2f_comparison.pdf")

print("\nDone! All figures saved to Figures/ directory.")

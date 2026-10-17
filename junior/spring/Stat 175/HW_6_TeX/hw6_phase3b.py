"""Exercise 4b: LSTM aggregator, implemented with neighbor sub-sampling (memory-safe).

We wrap SAGEConv(aggr='lstm') but cap each node's neighborhood at K neighbors
(uniform sub-sampling each epoch). Full-neighborhood LSTM OOMs on CPU for Citeseer.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from hw6_common import load_data, load_results, save_results, make_node_split, set_seed, device

ds, data = load_data()
y = data.y.cpu().numpy()
idx_train, idx_test = make_node_split(y)
num_features = ds.num_features
num_classes = ds.num_classes


def subsample_edges(edge_index, num_nodes, K=10, seed=0):
    """Cap neighborhood size at K by uniform subsampling (per destination node)."""
    rng = np.random.default_rng(seed)
    src, dst = edge_index.numpy()
    order = np.argsort(dst, kind="stable")
    src, dst = src[order], dst[order]
    keep = np.ones(len(dst), dtype=bool)
    start = 0
    for node in range(num_nodes):
        end = start
        while end < len(dst) and dst[end] == node:
            end += 1
        if end - start > K:
            idx = np.arange(start, end)
            chosen = rng.choice(idx, size=K, replace=False)
            mask = np.zeros(end - start, dtype=bool)
            mask[chosen - start] = True
            keep[start:end] = mask
        start = end
    new_src = src[keep]
    new_dst = dst[keep]
    return torch.tensor(np.stack([new_src, new_dst]), dtype=torch.long)


class SAGELSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden_dim, aggr="lstm")
        self.c2 = SAGEConv(hidden_dim, out_dim, aggr="lstm")

    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.c2(x, ei)

    def embed(self, x, ei):
        return F.relu(self.c1(x, ei))


set_seed()
K = 3
HIDDEN = 16
EPOCHS = 20
LR = 0.005

d = data.to(device)
edge_sub = subsample_edges(d.edge_index.cpu(), d.num_nodes, K=K, seed=0).to(device)
print(f"Original edges: {d.edge_index.size(1)}  Subsampled edges (K={K}): {edge_sub.size(1)}", flush=True)

model = SAGELSTM(num_features, HIDDEN, num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

t = time.time()
best_val, best_state = 0.0, None
for epoch in range(EPOCHS):
    model.train(); opt.zero_grad()
    out = model(d.x, edge_sub)
    loss = F.cross_entropy(out[d.train_mask], d.y[d.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(d.x, edge_sub).argmax(dim=1)
        v = (pred[d.val_mask] == d.y[d.val_mask]).float().mean().item()
        if v > best_val:
            best_val = v
            best_state = {k: v2.clone() for k, v2 in model.state_dict().items()}
    print(f"  epoch {epoch+1}/{EPOCHS} loss={loss.item():.4f} val_acc={v:.4f} t={time.time()-t:.1f}s", flush=True)

if best_state is not None:
    model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    emb = model.embed(d.x, edge_sub).cpu().numpy()
    pred = model(d.x, edge_sub).argmax(dim=1).cpu().numpy()

sc = StandardScaler()
Zt = sc.fit_transform(emb[idx_train]); Ze = sc.transform(emb[idx_test])
clf = LogisticRegression(max_iter=5000).fit(Zt, y[idx_train])
f_lr = f1_score(y[idx_test], clf.predict(Ze), average="macro")
f_direct = f1_score(y[idx_test], pred[idx_test], average="macro")
print(f"LSTM: F1(LR)={f_lr:.4f}  F1(direct)={f_direct:.4f}  time={time.time()-t:.1f}s")

R = load_results()
R.setdefault("ex4", {})
R["ex4"]["lstm_f1_lr"] = float(f_lr)
R["ex4"]["lstm_f1_direct"] = float(f_direct)
R["ex4"]["lstm_params"] = {"hidden_dim": HIDDEN, "epochs": EPOCHS, "neighbor_sample_K": K, "lr": LR, "grad_clip": 1.0}
save_results(R)

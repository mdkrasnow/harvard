"""HW6: GCN experiments on Citeseer. Runs Exercises 1-4 end-to-end."""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling, to_undirected
import torch_geometric.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cpu")

DATA_ROOT = "/tmp/Citeseer"
dataset = Planetoid(root=DATA_ROOT, name="Citeseer")
data = dataset[0]
num_nodes = data.num_nodes
num_features = dataset.num_features
num_classes = dataset.num_classes
print(f"Citeseer: {num_nodes} nodes, {data.num_edges//2} undirected edges, "
      f"{num_features} features, {num_classes} classes")


# ============================================================
# Exercise 1: GCN embeddings vs. original features
# ============================================================
class GCN(nn.Module):
    """Standard 2-layer GCN. forward returns logits; embed returns hidden representation."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def embed(self, x, edge_index):
        """Embedding = output of penultimate layer (before final classifier)."""
        if self.num_layers == 1:
            return x  # no hidden rep; return features
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


def train_gcn_node(num_layers=2, hidden_dim=64, epochs=200, lr=0.01, weight_decay=5e-4, verbose=False):
    torch.manual_seed(SEED)
    model = GCN(num_features, hidden_dim, num_classes, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    d = data.to(device)
    best_val = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(d.x, d.edge_index)
        loss = F.cross_entropy(out[d.train_mask], d.y[d.train_mask])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(d.x, d.edge_index).argmax(dim=1)
            val_acc = (pred[d.val_mask] == d.y[d.val_mask]).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb = model.embed(d.x, d.edge_index).cpu().numpy()
        pred = model(d.x, d.edge_index).argmax(dim=1).cpu().numpy()
    return model, emb, pred


print("\n===== Exercise 1: GCN vs. Original Features =====")
model1, emb1, _ = train_gcn_node(num_layers=2, hidden_dim=64)
y = data.y.cpu().numpy()
X_orig = data.x.cpu().numpy()

idx = np.arange(num_nodes)
idx_train, idx_test = train_test_split(idx, test_size=0.3, random_state=0, stratify=y)

# Logistic regression on GCN embeddings
sc_z = StandardScaler()
Z_train = sc_z.fit_transform(emb1[idx_train])
Z_test = sc_z.transform(emb1[idx_test])
clf_Z = LogisticRegression(max_iter=5000).fit(Z_train, y[idx_train])
f1_Z = f1_score(y[idx_test], clf_Z.predict(Z_test), average="macro")

# Logistic regression on original features
sc_x = StandardScaler(with_mean=False)
X_train = sc_x.fit_transform(X_orig[idx_train])
X_test = sc_x.transform(X_orig[idx_test])
clf_X = LogisticRegression(max_iter=5000).fit(X_train, y[idx_train])
f1_X = f1_score(y[idx_test], clf_X.predict(X_test), average="macro")

print(f"F1 macro (GCN embeddings):     {f1_Z:.4f}")
print(f"F1 macro (original features):  {f1_X:.4f}")


# ============================================================
# Exercise 2: Edge prediction with GCN
# ============================================================
print("\n===== Exercise 2: Edge Prediction =====")

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def decode(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


def split_edges_for_lp(data, val_frac=0.05, test_frac=0.10, seed=0):
    """Random link split. Uses RandomLinkSplit from PyG."""
    transform = T.RandomLinkSplit(
        num_val=val_frac, num_test=test_frac,
        is_undirected=True, add_negative_train_samples=False, split_labels=True,
    )
    torch.manual_seed(seed)
    train_data, val_data, test_data = transform(data.clone())
    return train_data, val_data, test_data


def train_edge_gcn(num_layers=2, hidden_dim=64, out_dim=64, epochs=200, lr=0.01, seed=0):
    torch.manual_seed(seed)
    train_data, val_data, test_data = split_edges_for_lp(data, seed=seed)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    model = GCNEncoder(num_features, hidden_dim, out_dim, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        z = model(train_data.x, train_data.edge_index)
        pos_edge = train_data.pos_edge_label_index
        neg_edge = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=num_nodes,
            num_neg_samples=pos_edge.size(1), method="sparse",
        )
        pos_logits = decode(z, pos_edge)
        neg_logits = decode(z, neg_edge)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones(pos_logits.size(0)), torch.zeros(neg_logits.size(0))]).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            z = model(train_data.x, train_data.edge_index)
            pos = torch.sigmoid(decode(z, val_data.pos_edge_label_index)).cpu().numpy()
            neg = torch.sigmoid(decode(z, val_data.neg_edge_label_index)).cpu().numpy()
            y_val = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
            s_val = np.concatenate([pos, neg])
            val_auc = roc_auc_score(y_val, s_val)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model(train_data.x, train_data.edge_index)
        pos = torch.sigmoid(decode(z, test_data.pos_edge_label_index)).cpu().numpy()
        neg = torch.sigmoid(decode(z, test_data.neg_edge_label_index)).cpu().numpy()
    y_test = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    s_test = np.concatenate([pos, neg])
    auc = roc_auc_score(y_test, s_test)
    preds = (s_test >= 0.5).astype(int)
    zero_one_loss = float(np.mean(preds != y_test))
    accuracy = 1.0 - zero_one_loss
    return auc, zero_one_loss, accuracy


auc2, loss2, acc2 = train_edge_gcn(num_layers=2, hidden_dim=64, out_dim=64, epochs=200)
print(f"Edge prediction (2-layer GCN):  ROC-AUC={auc2:.4f}  0/1 loss={loss2:.4f}  accuracy={acc2:.4f}")


# ============================================================
# Exercise 3: Number of layers (1..5)
# ============================================================
print("\n===== Exercise 3: Number of layers =====")

layer_results_node = {}
layer_results_edge = {}
for L in [1, 2, 3, 4, 5]:
    _, emb, _ = train_gcn_node(num_layers=L, hidden_dim=64, epochs=200)
    sc = StandardScaler()
    Zt = sc.fit_transform(emb[idx_train])
    Ze = sc.transform(emb[idx_test])
    clf = LogisticRegression(max_iter=5000).fit(Zt, y[idx_train])
    f1L = f1_score(y[idx_test], clf.predict(Ze), average="macro")
    layer_results_node[L] = f1L
    print(f"  Node classification: L={L}  F1-macro={f1L:.4f}")

for L in [1, 2, 3, 4, 5]:
    auc, lz, _ = train_edge_gcn(num_layers=L, hidden_dim=64, out_dim=64, epochs=200)
    layer_results_edge[L] = (auc, lz)
    print(f"  Edge prediction:     L={L}  AUC={auc:.4f}  0/1 loss={lz:.4f}")


# ============================================================
# Exercise 4: Aggregation methods (pooling vs LSTM)
# ============================================================
print("\n===== Exercise 4: Aggregation (pool vs LSTM) =====")

class SAGEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, aggr="max", num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_dim, out_dim, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggr))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            self.convs.append(SAGEConv(hidden_dim, out_dim, aggr=aggr))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def embed(self, x, edge_index):
        if self.num_layers == 1:
            return x
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


def train_sage(aggr, hidden_dim=64, epochs=200, lr=0.01, weight_decay=5e-4, seed=0):
    torch.manual_seed(seed)
    model = SAGEModel(num_features, hidden_dim, num_classes, aggr=aggr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    d = data.to(device)
    best_val = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(d.x, d.edge_index)
        loss = F.cross_entropy(out[d.train_mask], d.y[d.train_mask])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(d.x, d.edge_index).argmax(dim=1)
            val_acc = (pred[d.val_mask] == d.y[d.val_mask]).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb = model.embed(d.x, d.edge_index).cpu().numpy()
        pred = model(d.x, d.edge_index).argmax(dim=1).cpu().numpy()
    return emb, pred


def lr_f1(emb):
    sc = StandardScaler()
    Zt = sc.fit_transform(emb[idx_train])
    Ze = sc.transform(emb[idx_test])
    clf = LogisticRegression(max_iter=5000).fit(Zt, y[idx_train])
    return f1_score(y[idx_test], clf.predict(Ze), average="macro")


emb_pool, pred_pool = train_sage(aggr="max")
f1_pool_lr = lr_f1(emb_pool)
f1_pool_direct = f1_score(y[idx_test], pred_pool[idx_test], average="macro")

emb_lstm, pred_lstm = train_sage(aggr="lstm")
f1_lstm_lr = lr_f1(emb_lstm)
f1_lstm_direct = f1_score(y[idx_test], pred_lstm[idx_test], average="macro")

print(f"Pooling aggr (max):    F1 (LR on emb)={f1_pool_lr:.4f}  F1 (direct)={f1_pool_direct:.4f}")
print(f"LSTM aggr:             F1 (LR on emb)={f1_lstm_lr:.4f}  F1 (direct)={f1_lstm_direct:.4f}")


# Save all results to JSON for reference
out = {
    "ex1": {"f1_gcn_embed": f1_Z, "f1_original": f1_X},
    "ex2": {"auc": auc2, "zero_one_loss": loss2, "accuracy": acc2},
    "ex3_node_f1_by_layer": layer_results_node,
    "ex3_edge_by_layer": {str(k): {"auc": v[0], "zero_one_loss": v[1]} for k, v in layer_results_edge.items()},
    "ex4": {
        "pool_f1_lr": f1_pool_lr, "pool_f1_direct": f1_pool_direct,
        "lstm_f1_lr": f1_lstm_lr, "lstm_f1_direct": f1_lstm_direct,
    },
}
with open("/Users/mkrasnow/Desktop/HARVARD/junior/spring/Stat 175/HW_6_TeX/hw6_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nResults saved.")

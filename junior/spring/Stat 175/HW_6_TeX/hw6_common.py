"""Shared utilities for HW6 experiments."""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 0
DATA_ROOT = "/tmp/Citeseer"
RESULTS_PATH = "/Users/mkrasnow/Desktop/HARVARD/junior/spring/Stat 175/HW_6_TeX/hw6_results.json"

device = torch.device("cpu")


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_data():
    ds = Planetoid(root=DATA_ROOT, name="Citeseer")
    return ds, ds[0]


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(r):
    with open(RESULTS_PATH, "w") as f:
        json.dump(r, f, indent=2)


class GCN(nn.Module):
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
        if self.num_layers == 1:
            return x
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


def train_gcn_node(data, num_features, num_classes,
                   num_layers=2, hidden_dim=64, epochs=200, lr=0.01, weight_decay=5e-4):
    set_seed()
    model = GCN(num_features, hidden_dim, num_classes, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    d = data.to(device)
    best_val, best_state = 0.0, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(d.x, d.edge_index)
        F.cross_entropy(out[d.train_mask], d.y[d.train_mask]).backward()
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


def make_node_split(y, seed=0):
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)


def lr_macro_f1(emb, y, idx_train, idx_test):
    sc = StandardScaler()
    Zt = sc.fit_transform(emb[idx_train])
    Ze = sc.transform(emb[idx_test])
    clf = LogisticRegression(max_iter=5000).fit(Zt, y[idx_train])
    return f1_score(y[idx_test], clf.predict(Ze), average="macro")


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


def train_edge_gcn(data, num_features, num_nodes,
                   num_layers=2, hidden_dim=64, out_dim=64, epochs=200, lr=0.01, seed=0):
    set_seed(seed)
    tr = T.RandomLinkSplit(
        num_val=0.05, num_test=0.10,
        is_undirected=True, add_negative_train_samples=False, split_labels=True,
    )
    train_data, val_data, test_data = tr(data.clone())
    train_data = train_data.to(device); val_data = val_data.to(device); test_data = test_data.to(device)
    model = GCNEncoder(num_features, hidden_dim, out_dim, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_auc, best_state = 0.0, None
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        z = model(train_data.x, train_data.edge_index)
        pos = train_data.pos_edge_label_index
        neg = negative_sampling(edge_index=train_data.edge_index, num_nodes=num_nodes,
                                num_neg_samples=pos.size(1), method="sparse")
        logits = torch.cat([decode(z, pos), decode(z, neg)])
        labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]).to(device)
        F.binary_cross_entropy_with_logits(logits, labels).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            z = model(train_data.x, train_data.edge_index)
            p = torch.sigmoid(decode(z, val_data.pos_edge_label_index)).cpu().numpy()
            n = torch.sigmoid(decode(z, val_data.neg_edge_label_index)).cpu().numpy()
            y = np.concatenate([np.ones_like(p), np.zeros_like(n)])
            s = np.concatenate([p, n])
            v = roc_auc_score(y, s)
            if v > best_val_auc:
                best_val_auc = v
                best_state = {k: v2.clone() for k, v2 in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model(train_data.x, train_data.edge_index)
        p = torch.sigmoid(decode(z, test_data.pos_edge_label_index)).cpu().numpy()
        n = torch.sigmoid(decode(z, test_data.neg_edge_label_index)).cpu().numpy()
    y = np.concatenate([np.ones_like(p), np.zeros_like(n)])
    s = np.concatenate([p, n])
    auc = roc_auc_score(y, s)
    zero_one = float(np.mean((s >= 0.5).astype(int) != y))
    return auc, zero_one, 1.0 - zero_one


class SAGEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, aggr="max", num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers; self.dropout = dropout
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
                x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def embed(self, x, edge_index):
        if self.num_layers == 1:
            return x
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index); x = F.relu(x)
        return x


def train_sage_node(data, num_features, num_classes, aggr,
                    hidden_dim=64, epochs=100, lr=0.01, weight_decay=5e-4):
    set_seed()
    model = SAGEModel(num_features, hidden_dim, num_classes, aggr=aggr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    d = data.to(device)
    best_val, best_state = 0.0, None
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(d.x, d.edge_index)
        F.cross_entropy(out[d.train_mask], d.y[d.train_mask]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(d.x, d.edge_index).argmax(dim=1)
            v = (pred[d.val_mask] == d.y[d.val_mask]).float().mean().item()
            if v > best_val:
                best_val = v
                best_state = {k: v2.clone() for k, v2 in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb = model.embed(d.x, d.edge_index).cpu().numpy()
        pred = model(d.x, d.edge_index).argmax(dim=1).cpu().numpy()
    return emb, pred

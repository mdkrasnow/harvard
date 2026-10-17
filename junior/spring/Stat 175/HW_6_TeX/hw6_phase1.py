"""Exercise 1 & 2: GCN vs original features, and edge prediction."""
from hw6_common import (load_data, load_results, save_results, train_gcn_node,
                        make_node_split, lr_macro_f1, train_edge_gcn)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

ds, data = load_data()
num_nodes = data.num_nodes
num_features = ds.num_features
num_classes = ds.num_classes
print(f"Citeseer: {num_nodes} nodes, {data.num_edges//2} edges, {num_features} feats, {num_classes} classes")

y = data.y.cpu().numpy()
Xo = data.x.cpu().numpy()
idx_train, idx_test = make_node_split(y)

R = load_results()

print("== Ex 1 ==")
emb1, _ = train_gcn_node(data, num_features, num_classes, num_layers=2, hidden_dim=64, epochs=200)
f1_Z = lr_macro_f1(emb1, y, idx_train, idx_test)
sc = StandardScaler(with_mean=False)
Xt = sc.fit_transform(Xo[idx_train]); Xe = sc.transform(Xo[idx_test])
clf = LogisticRegression(max_iter=5000).fit(Xt, y[idx_train])
f1_X = f1_score(y[idx_test], clf.predict(Xe), average="macro")
R["ex1"] = {"f1_gcn_embed": float(f1_Z), "f1_original": float(f1_X)}
save_results(R)
print(f"  GCN embed F1={f1_Z:.4f}  Original F1={f1_X:.4f}")

print("== Ex 2 ==")
auc, zol, acc = train_edge_gcn(data, num_features, num_nodes, num_layers=2, hidden_dim=64, out_dim=64, epochs=200)
R["ex2"] = {"auc": float(auc), "zero_one_loss": float(zol), "accuracy": float(acc)}
save_results(R)
print(f"  AUC={auc:.4f}  0/1 loss={zol:.4f}  acc={acc:.4f}")

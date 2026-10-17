"""Exercise 3: Number of layers (1..5) for node classification and edge prediction."""
from hw6_common import (load_data, load_results, save_results,
                        train_gcn_node, make_node_split, lr_macro_f1, train_edge_gcn)

ds, data = load_data()
num_nodes = data.num_nodes
num_features = ds.num_features
num_classes = ds.num_classes
y = data.y.cpu().numpy()
idx_train, idx_test = make_node_split(y)
R = load_results()

print("== Ex 3a: node classification, varying layers ==")
R.setdefault("ex3_node_f1_by_layer", {})
for L in [1, 2, 3, 4, 5]:
    emb, _ = train_gcn_node(data, num_features, num_classes, num_layers=L, hidden_dim=64, epochs=200)
    f = lr_macro_f1(emb, y, idx_train, idx_test)
    R["ex3_node_f1_by_layer"][str(L)] = float(f)
    save_results(R)
    print(f"  L={L}  F1-macro={f:.4f}")

print("== Ex 3b: edge prediction, varying layers ==")
R.setdefault("ex3_edge_by_layer", {})
for L in [1, 2, 3, 4, 5]:
    auc, zol, _ = train_edge_gcn(data, num_features, num_nodes,
                                 num_layers=L, hidden_dim=64, out_dim=64, epochs=200)
    R["ex3_edge_by_layer"][str(L)] = {"auc": float(auc), "zero_one_loss": float(zol)}
    save_results(R)
    print(f"  L={L}  AUC={auc:.4f}  0/1 loss={zol:.4f}")

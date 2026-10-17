"""Exercise 4a: max-pooling aggregator (fast)."""
import time
from hw6_common import (load_data, load_results, save_results,
                        train_sage_node, make_node_split, lr_macro_f1)
from sklearn.metrics import f1_score

ds, data = load_data()
y = data.y.cpu().numpy()
idx_train, idx_test = make_node_split(y)

t = time.time()
emb_p, pred_p = train_sage_node(data, ds.num_features, ds.num_classes, aggr="max",
                                hidden_dim=64, epochs=200)
f_lr = lr_macro_f1(emb_p, y, idx_train, idx_test)
f_direct = f1_score(y[idx_test], pred_p[idx_test], average="macro")
print(f"Pool: F1(LR)={f_lr:.4f}  F1(direct)={f_direct:.4f}  time={time.time()-t:.1f}s")

R = load_results()
R.setdefault("ex4", {})
R["ex4"]["pool_f1_lr"] = float(f_lr)
R["ex4"]["pool_f1_direct"] = float(f_direct)
save_results(R)

"""Exercise 4: Aggregation methods (max pooling vs LSTM) for node classification.

NB: LSTM aggregation is expensive on CPU, so we use fewer epochs and a smaller hidden dim.
"""
import time
from hw6_common import (load_data, load_results, save_results,
                        train_sage_node, make_node_split, lr_macro_f1)
from sklearn.metrics import f1_score

ds, data = load_data()
num_features = ds.num_features
num_classes = ds.num_classes
y = data.y.cpu().numpy()
idx_train, idx_test = make_node_split(y)
R = load_results()

print("== Ex 4: pooling vs LSTM aggregation ==")

t = time.time()
emb_p, pred_p = train_sage_node(data, num_features, num_classes, aggr="max",
                                hidden_dim=64, epochs=200)
f_pool_lr = lr_macro_f1(emb_p, y, idx_train, idx_test)
f_pool_direct = f1_score(y[idx_test], pred_p[idx_test], average="macro")
print(f"  Pool: F1(LR)={f_pool_lr:.4f}  F1(direct)={f_pool_direct:.4f}  time={time.time()-t:.1f}s")

t = time.time()
emb_l, pred_l = train_sage_node(data, num_features, num_classes, aggr="lstm",
                                hidden_dim=32, epochs=50)
f_lstm_lr = lr_macro_f1(emb_l, y, idx_train, idx_test)
f_lstm_direct = f1_score(y[idx_test], pred_l[idx_test], average="macro")
print(f"  LSTM: F1(LR)={f_lstm_lr:.4f}  F1(direct)={f_lstm_direct:.4f}  time={time.time()-t:.1f}s")

R["ex4"] = {
    "pool_f1_lr": float(f_pool_lr), "pool_f1_direct": float(f_pool_direct),
    "lstm_f1_lr": float(f_lstm_lr), "lstm_f1_direct": float(f_lstm_direct),
}
save_results(R)
print("Saved.")

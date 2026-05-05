"""Graph convolutional network on the rainfall->storage->market graph.

Eleven nodes:
    0..4   five catchments (rain, flow)
    5..9   five storage sub-systems (energy in storage)
    10     the market (price, demand, basslink)

Edges follow physical water flow direction: each catchment flows into
its corresponding storage node; storage nodes feed the market node;
storage nodes are joined to each other where the Hydro Tas system has
inter-basin transfers (Gordon<->Pedder, Mersey<->Forth, etc.). Self
loops are added for stability.

We use a dense adjacency since N=11 — there's no point pulling in
PyTorch Geometric for a graph this small."""

import numpy as np
import torch
import torch.nn as nn

CATCHMENT_NAMES = ["pieman", "gordon_pedder", "mersey_forth", "derwent",
                   "great_lake"]
STORAGE_NAMES = ["west_coast", "gordon", "mersey_forth", "derwent",
                 "great_lake"]
N_CATCH = 5
N_STOR = 5
MARKET = 10
N_NODES = 11

# physical edges
EDGES = [
    # catchment -> storage (one-to-one in the same order)
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    # storage <-> storage (known inter-basin transfers)
    (6, 5),  # gordon <-> west coast (geographic adjacency)
    (8, 9),  # derwent <-> great lake (the canals)
    # storage -> market (all storages feed dispatchable hydro)
    (5, MARKET), (6, MARKET), (7, MARKET), (8, MARKET), (9, MARKET),
]


def build_adjacency():
    A = np.zeros((N_NODES, N_NODES), dtype=np.float32)
    for i, j in EDGES:
        A[i, j] = A[j, i] = 1.0
    A = A + np.eye(N_NODES, dtype=np.float32)
    d = A.sum(axis=1)
    D_inv = np.diag(1.0 / np.sqrt(d + 1e-9))
    return D_inv @ A @ D_inv


def build_node_features(df, lags=(1, 7, 28)):
    """Per-node feature blocks per timestep.

    For each catchment i: [rain_today, *lags of rain]
    For each storage j: [storage_today, *lags of storage]
    For the market: [price_today, demand_today, basslink_today,
                     *lags of price]

    Flow is intentionally excluded: when WIST gauges are unavailable the
    `flow_*_cms` columns are constructed in `data.load_flows` as a
    deterministic exponential-kernel convolution of the same catchment
    rainfall already present as a node feature, so the column carries no
    information beyond rain-today + rain-lags. Including it as an extra
    feature inflates apparent capacity without adding signal. To keep
    tensor shape consistent we pad each node block to the same feature
    width F."""
    n = len(df)
    F = 1 + len(lags) + 2   # rain_today + 3 rain lags + 2 spare slots
    X = np.zeros((n, N_NODES, F), dtype=np.float32)

    for i, nm in enumerate(CATCHMENT_NAMES):
        rain = df[f"rain_{nm}_mm"].values
        X[:, i, 0] = rain
        for k, L in enumerate(lags, start=1):
            X[L:, i, k] = rain[:-L]

    for j, nm in enumerate(STORAGE_NAMES):
        col = f"{nm}_gwh"
        s = df[col].values if col in df.columns else df["storage_gwh"].values / 5
        X[:, N_CATCH + j, 0] = s
        for k, L in enumerate(lags, start=1):
            X[L:, N_CATCH + j, k] = s[:-L]

    price = df["price_aud_mwh"].values
    X[:, MARKET, 0] = price
    X[:, MARKET, 1] = df["demand_mw"].values
    X[:, MARKET, 2] = df["basslink_mw"].values
    for k, L in enumerate(lags, start=3):
        X[L:, MARKET, k] = price[:-L]

    valid = max(lags)
    y = np.zeros(n, dtype=np.float32)
    y[:-1] = price[1:]
    return X[valid:-1], y[valid:-1]


class CatchmentGCN(nn.Module):
    def __init__(self, in_features, hidden=32, target_node=MARKET):
        super().__init__()
        self.target = target_node
        self.W1 = nn.Linear(in_features, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, X, A):
        h = torch.relu(self.W1(X))
        h = torch.einsum("ij,bjf->bif", A, h)
        h = torch.relu(self.W2(h))
        h = torch.einsum("ij,bjf->bif", A, h)
        return self.head(h[:, self.target, :])


def train(X, y, A, tr, va, epochs=30, batch=256, lr=1e-3, hidden=32,
          seed=0):
    torch.manual_seed(seed)
    model = CatchmentGCN(in_features=X.shape[2], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A_t = torch.from_numpy(A.astype(np.float32))
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.reshape(-1, 1))
    n_tr = tr.stop - tr.start
    rng = np.random.default_rng(seed)
    loss_fn = nn.MSELoss()
    history = {"train": [], "val": []}
    for epoch in range(epochs):
        model.train()
        idx = rng.permutation(n_tr) + tr.start
        running = 0.0
        for s in range(0, n_tr, batch):
            b = idx[s:s + batch]
            yh = model(Xt[b], A_t)
            loss = loss_fn(yh, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.item()) * len(b)
        model.eval()
        with torch.no_grad():
            vh = model(Xt[va], A_t)
            vloss = float(loss_fn(vh, yt[va]).item())
        history["train"].append(running / n_tr)
        history["val"].append(vloss)
    model.eval()
    with torch.no_grad():
        all_pred = model(Xt, A_t).numpy().ravel()
    return model, history, all_pred

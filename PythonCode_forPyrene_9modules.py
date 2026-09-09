import os
import time
import dill
import numpy as np
import pandas as pd
import lingam
from lingam.utils import make_prior_knowledge, make_dot
from pathlib import Path
import graphviz
import networkx as nx





# --------------------------
# Settings
# --------------------------
base_dir = Path(__file__).resolve().parent
OUT_DOT_PREFIX = base_dir / "LiNGAM_dag_eigengene"
OUT_DOT_PREFIX_CONS = base_dir / "LiNGAM_dag_eigengene_consensus"

file = "logCPM_eigengenes_pyrene_9modules.csv"
BOOT_PKL = "lingam_bootstrap_9modules.pkl"

M = 1000  # number of bootstrap resamples

# Edge threshold used for network metrics (structure extraction from adjacency matrices)
# NOTE: In lingam, adjacency_matrix_[i, j] is the causal effect of x_j -> x_i.
MIN_CAUSAL_EFFECT_FOR_METRICS = 0.0001

# Thresholds for lingam summary outputs
MIN_CAUSAL_EFFECT_FOR_PRINT = 0.0001
N_DIRECTIONS_PRINT = 10000
N_DAGS_PRINT = 3

# Downstream module
DOWNSTREAM = "turquoise"
DOWNSTREAM_LABEL = f"ME{DOWNSTREAM}"


# Seed
np.random.seed(9999)



# --------------------------
# Load data
# --------------------------
data = pd.read_csv(file)

# Ensure Concentration exists and is the first column
if "Concentration" not in data.columns:
    raise ValueError("Column 'Concentration' not found in the input CSV.")
if data.columns[0] != "Concentration":
    cols = ["Concentration"] + [c for c in data.columns if c != "Concentration"]
    data = data[cols]

# Transform concentration
data["Concentration"] = np.log10(data["Concentration"].to_numpy(dtype=np.float64) + 1.0)
print("Data shape:", data.shape)

labels = list(data.columns)
data_np = data.to_numpy(dtype=np.float64)

# --------------------------
# Prior knowledge (Concentration is exogenous)
# --------------------------
print("\nColumns and their indices:")
for idx, col in enumerate(labels):
    print(idx, col)

# Concentration is at index 0 by construction above
prior_k = make_prior_knowledge(
    n_variables=data_np.shape[1],
    exogenous_variables=[0],
    # sink_variables=[...],
    # paths=[(cause_idx, effect_idx), ...],
)

print("\nPrior knowledge matrix:\n", prior_k)


# --------------------------
# Fit model
# --------------------------
model = lingam.DirectLiNGAM(prior_knowledge=prior_k)
model.fit(data_np)

adj = model.adjacency_matrix_
print("\nLearned adjacency matrix:\n", adj)

dot = make_dot(adj, labels=labels)
# dot.render(OUT_DOT_PREFIX)     # DAG


# --------------------------
# Bootstrap
# --------------------------
start = time.perf_counter()
result = model.bootstrap(data_np, n_sampling=M)
end = time.perf_counter()
print("\nElapsed time (bootstrap):", end - start)

with open(BOOT_PKL, "wb") as f:
    dill.dump(result, f)

# Reload (optional / reproducibility)
#with open(BOOT_PKL, "rb") as f:
#    result = dill.load(f)



# --------------------------
# Standard lingam summaries
# --------------------------
cdc = result.get_causal_direction_counts(
    n_directions=N_DIRECTIONS_PRINT,
    min_causal_effect=MIN_CAUSAL_EFFECT_FOR_PRINT,
    split_by_causal_effect_sign=True,
)

from lingam.utils import print_causal_directions, print_dagc

print("\n===== Causal directions (counts) =====")
print_causal_directions(cdc, M, labels=labels)

dagc = result.get_directed_acyclic_graph_counts(
    n_dags=N_DAGS_PRINT,
    min_causal_effect=MIN_CAUSAL_EFFECT_FOR_PRINT,
    split_by_causal_effect_sign=True,
)
print("\n===== DAG counts (top) =====")
print_dagc(dagc, M, labels=labels)

prob = result.get_probabilities(min_causal_effect=MIN_CAUSAL_EFFECT_FOR_METRICS)
print("\n===== Edge probabilities (min_causal_effect=%.4f) =====" % MIN_CAUSAL_EFFECT_FOR_METRICS)
print(prob)


# path-based probability
from_index = labels.index("Concentration")
to_index   = labels.index(DOWNSTREAM_LABEL)

paths = result.get_paths(
    from_index=from_index,
    to_index=to_index,
    min_causal_effect=MIN_CAUSAL_EFFECT_FOR_METRICS
)
path_df = pd.DataFrame(paths)
path_df.head(10)








# --------------------------
# Drow concensus DAG
# --------------------------
# Note: Sign-specific DAG, distinguishing positive and negative edges
edge_prob_threshold = 0.60
effect_threshold = MIN_CAUSAL_EFFECT_FOR_METRICS

# Extract bootstrap result
A_list = None
for attr in ["adjacency_matrices_", "adjacency_matrices", "_adjacency_matrices"]:
    if hasattr(result, attr):
        A_list = getattr(result, attr)
        break

if A_list is None:
    raise RuntimeError(
        "Could not find adjacency matrices inside the bootstrap result. "
        "Tried attributes: adjacency_matrices_, adjacency_matrices, _adjacency_matrices"
    )

A_list = list(A_list)
if len(A_list) != M:
    print(f"\n[Warning] Number of adjacency matrices in result: {len(A_list)} (expected {M}). Using {len(A_list)}.")
    M_eff = len(A_list)
else:
    M_eff = M

n_vars = len(labels)


A_array = np.asarray(A_list, dtype=float)
consensus_adj = np.zeros((n_vars, n_vars), dtype=float)

for eff in range(n_vars):
    for cause in range(n_vars):
        if eff == cause:
            continue

        coefs = A_array[:, eff, cause]

        positive = np.isfinite(coefs) & (coefs >= effect_threshold)
        negative = np.isfinite(coefs) & (coefs <= -effect_threshold)

        positive_probability = np.mean(positive)
        negative_probability = np.mean(negative)

        if positive_probability >= edge_prob_threshold:
            consensus_adj[eff, cause] = np.median(coefs[positive])

        elif negative_probability >= edge_prob_threshold:
            consensus_adj[eff, cause] = np.median(coefs[negative])

dot_consensus = make_dot(consensus_adj, labels=labels)

dot_consensus.render(OUT_DOT_PREFIX_CONS, cleanup=True)



# --------------------------
# Sign-specific dominant edge probability
# --------------------------

A_array = np.asarray(A_list, dtype=float)
effect_threshold = MIN_CAUSAL_EFFECT_FOR_METRICS

positive_prob = np.mean(
    np.isfinite(A_array) & (A_array >= effect_threshold),
    axis=0
)

negative_prob = np.mean(
    np.isfinite(A_array) & (A_array <= -effect_threshold),
    axis=0
)


# Sign-specific
signed_dominant_prob = np.where(
    positive_prob >= negative_prob,
    positive_prob,
    -negative_prob
)
print("\n===== Signed edge probabilities ===== minus values represent negative effects")
print(signed_dominant_prob.round(3))





# ==========================================================
# Sign-specific bootstrap path probabilities
# ==========================================================

from collections import defaultdict
from lingam.utils import find_all_paths

from_index = labels.index("Concentration")
to_index   = labels.index(DOWNSTREAM_LABEL)
threshold  = MIN_CAUSAL_EFFECT_FOR_METRICS

# Store positive and negative effects separately
path_records = defaultdict(lambda: {
    "positive": [],
    "negative": []
})

for A in A_list:

    A = np.asarray(A, dtype=float)

    paths, effects = find_all_paths(
        A,
        from_index,
        to_index,
        threshold
    )

    for path, effect in zip(paths, effects):

        key = tuple(path)

        if effect > 0:
            path_records[key]["positive"].append(float(effect))

        elif effect < 0:
            path_records[key]["negative"].append(float(effect))

# Summarize probabilities

rows = []
n_bootstrap = len(A_list)

for path, rec in path_records.items():

    n_pos = len(rec["positive"])
    n_neg = len(rec["negative"])

    p_pos = n_pos / n_bootstrap
    p_neg = n_neg / n_bootstrap

    # This should correspond to official get_paths() probability
    p_unsigned = p_pos + p_neg

    signed_dominant_prob = (
        p_pos if p_pos >= p_neg
        else -p_neg
    )

    rows.append({
        "path_name":
            " -> ".join(labels[i] for i in path),

        "positive_probability": p_pos,
        "negative_probability": p_neg,
        "unsigned_probability": p_unsigned,
        "signed_dominant_probability": signed_dominant_prob,

        "median_positive_effect":
            np.median(rec["positive"])
            if n_pos > 0 else np.nan,

        "median_negative_effect":
            np.median(rec["negative"])
            if n_neg > 0 else np.nan
    })


path_sign_df = pd.DataFrame(rows)
path_sign_df = path_sign_df.sort_values(
    "unsigned_probability",
    ascending=False
).reset_index(drop=True)

print("\n===== Path probabilities considering sign =====")
print(path_sign_df.head(10))

import argparse
import csv
import glob
import itertools
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import normalized_mutual_info_score


def read_scalar(data, key, default=None):
    if key not in data:
        return default
    value = np.asarray(data[key]).reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_artifact(path):
    data = np.load(path, allow_pickle=False)
    dataset = str(read_scalar(data, "dataset", ""))
    tau = float(read_scalar(data, "tau"))
    concept_tau = float(read_scalar(data, "concept_tau"))
    lp_tau_g = float(read_scalar(data, "lp_tau_g"))
    seed = int(read_scalar(data, "seed"))
    key = (round(tau, 8), round(concept_tau, 8), round(lp_tau_g, 8))
    return {
        "path": path,
        "data": data,
        "dataset": dataset,
        "seed": seed,
        "key": key,
    }


def get_aligned_pseudo(item):
    data = item["data"]
    gid = np.asarray(data["gid"], dtype=np.int64)
    pseudo = np.asarray(data["pseudo"], dtype=np.int64)
    order = np.argsort(gid)
    return gid[order], pseudo[order]


def pairwise_pseudo_nmi(items):
    scores = []
    for item_a, item_b in itertools.combinations(items, 2):
        gid_a, pseudo_a = get_aligned_pseudo(item_a)
        gid_b, pseudo_b = get_aligned_pseudo(item_b)
        common_gid, idx_a, idx_b = np.intersect1d(
            gid_a, gid_b, return_indices=True
        )
        if len(common_gid) == 0:
            continue
        score = normalized_mutual_info_score(pseudo_a[idx_a], pseudo_b[idx_b])
        scores.append(score)
    if not scores:
        return np.nan, np.nan, 0
    return float(np.mean(scores)), float(np.std(scores)), len(scores)


def normalized_cluster_entropy(labels, num_labels=None):
    labels = np.asarray(labels, dtype=np.int64)
    if num_labels is None:
        num_labels = max(int(labels.max()) + 1, len(np.unique(labels)))
    counts = np.bincount(labels, minlength=num_labels)
    probs = counts[counts > 0].astype(np.float64) / max(1, counts.sum())
    effective_clusters = len(probs)
    if effective_clusters <= 1 or num_labels <= 1:
        return 0.0, effective_clusters
    entropy = -float(np.sum(probs * np.log(probs))) / float(np.log(num_labels))
    return entropy, effective_clusters


def clustering_acc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    true_map = {v: i for i, v in enumerate(true_values)}
    pred_map = {v: i for i, v in enumerate(pred_values)}
    mat = np.zeros((len(pred_values), len(true_values)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        mat[pred_map[yp], true_map[yt]] += 1
    row_ind, col_ind = linear_sum_assignment(mat.max() - mat)
    return float(mat[row_ind, col_ind].sum() / len(y_true))


def evaluate_item(item):
    data = item["data"]
    if "test_y_true" in data and "test_y_pred" in data:
        y_true = np.asarray(data["test_y_true"], dtype=np.int64)
        y_pred = np.asarray(data["test_y_pred"], dtype=np.int64)
    else:
        y_true = np.asarray(data["y_true"], dtype=np.int64)
        y_pred = np.asarray(data["kmeans_pred"], dtype=np.int64)

    return np.asarray(
        [
            clustering_acc(y_true, y_pred),
            adjusted_rand_score(y_true, y_pred),
            normalized_mutual_info_score(y_true, y_pred),
            fowlkes_mallows_score(y_true, y_pred),
        ],
        dtype=np.float64,
    )


def summarize_group(key, items, min_clusters, min_entropy):
    stability_mean, stability_std, pair_count = pairwise_pseudo_nmi(items)

    entropies = []
    effective_clusters = []
    for item in items:
        data = item["data"]
        if "cluster_entropy" in data and "effective_clusters" in data:
            entropies.append(float(read_scalar(data, "cluster_entropy")))
            effective_clusters.append(float(read_scalar(data, "effective_clusters")))
        else:
            entropy, n_clusters = normalized_cluster_entropy(data["pseudo"])
            entropies.append(entropy)
            effective_clusters.append(float(n_clusters))

    metric_values = np.stack([evaluate_item(item) for item in items], axis=0)
    metric_mean = metric_values.mean(axis=0)
    metric_std = metric_values.std(axis=0)

    cluster_entropy = float(np.mean(entropies))
    effective_cluster_count = float(np.mean(effective_clusters))
    valid = (
        len(items) >= 2
        and not np.isnan(stability_mean)
        and cluster_entropy >= min_entropy
        and effective_cluster_count >= min_clusters
    )

    tau, concept_tau, lp_tau_g = key
    return {
        "tau": tau,
        "concept_tau": concept_tau,
        "lp_tau_g": lp_tau_g,
        "runs": len(items),
        "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda x: x["seed"])),
        "pair_count": pair_count,
        "stability_nmi_mean": stability_mean,
        "stability_nmi_std": stability_std,
        "cluster_entropy": cluster_entropy,
        "effective_clusters": effective_cluster_count,
        "valid": int(valid),
        "acc_mean": metric_mean[0],
        "ari_mean": metric_mean[1],
        "nmi_mean": metric_mean[2],
        "fmi_mean": metric_mean[3],
        "acc_std": metric_std[0],
        "ari_std": metric_std[1],
        "nmi_std": metric_std[2],
        "fmi_std": metric_std[3],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=".",
        help="Directory containing temperature-selection subdirectories.",
    )
    parser.add_argument("--dataset", default="MIntRec")
    parser.add_argument("--min_clusters", type=int, default=16)
    parser.add_argument("--min_entropy", type=float, default=0.75)
    parser.add_argument(
        "--out",
        default="temperature_stability_summary.csv",
        help="Output CSV summary path.",
    )
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.root, "**", "temperature_selection_*.npz"), recursive=True)
    items = []
    for path in files:
        item = load_artifact(path)
        if item["dataset"] == args.dataset:
            items.append(item)

    groups = {}
    for item in items:
        groups.setdefault(item["key"], []).append(item)

    rows = [
        summarize_group(key, group_items, args.min_clusters, args.min_entropy)
        for key, group_items in sorted(groups.items())
    ]

    valid_rows = [row for row in rows if row["valid"]]
    selection_pool = valid_rows if valid_rows else rows
    selected = None
    if selection_pool:
        selected = max(selection_pool, key=lambda row: row["stability_nmi_mean"])

    fieldnames = [
        "selected",
        "tau",
        "concept_tau",
        "lp_tau_g",
        "runs",
        "seeds",
        "pair_count",
        "stability_nmi_mean",
        "stability_nmi_std",
        "cluster_entropy",
        "effective_clusters",
        "valid",
        "acc_mean",
        "ari_mean",
        "nmi_mean",
        "fmi_mean",
        "acc_std",
        "ari_std",
        "nmi_std",
        "fmi_std",
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["selected"] = int(
                selected is not None
                and row["tau"] == selected["tau"]
                and row["concept_tau"] == selected["concept_tau"]
                and row["lp_tau_g"] == selected["lp_tau_g"]
            )
            writer.writerow(row)

    print(f"Scanned root: {args.root}")
    print(f"Found artifacts: {len(files)}")
    print(f"Matched dataset artifacts: {len(items)}")
    print(f"Temperature groups: {len(rows)}")
    if selected is not None:
        print(
            "Selected by label-free stability: "
            f"tau={selected['tau']}, concept_tau={selected['concept_tau']}, "
            f"lp_tau_g={selected['lp_tau_g']}, "
            f"stability={selected['stability_nmi_mean']:.4f}"
        )
    print(f"Wrote summary to: {args.out}")


if __name__ == "__main__":
    main()

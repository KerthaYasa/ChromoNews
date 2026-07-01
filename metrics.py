"""
metrics.py
Metrik evaluasi Information Retrieval standar.
ranked_ids : list of doc_id (urut dari paling relevan menurut sistem)
relevant_ids : set of doc_id yang relevan (ground truth / qrels)
"""
import math


def precision_at_k(ranked_ids, relevant_ids, k):
    topk = ranked_ids[:k]
    if not topk:
        return 0.0
    hits = sum(1 for d in topk if d in relevant_ids)
    return hits / len(topk)


def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    topk = ranked_ids[:k]
    hits = sum(1 for d in topk if d in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids):
    for i, d in enumerate(ranked_ids, start=1):
        if d in relevant_ids:
            return 1.0 / i
    return 0.0


def average_precision(ranked_ids, relevant_ids, k=None):
    if not relevant_ids:
        return 0.0
    if k:
        ranked_ids = ranked_ids[:k]
    hits = 0
    sum_prec = 0.0
    for i, d in enumerate(ranked_ids, start=1):
        if d in relevant_ids:
            hits += 1
            sum_prec += hits / i
    denom = min(len(relevant_ids), len(ranked_ids)) if k else len(relevant_ids)
    return sum_prec / len(relevant_ids) if len(relevant_ids) > 0 else 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k):
    def dcg(ids):
        return sum(
            (1.0 / math.log2(i + 1)) for i, d in enumerate(ids[:k], start=1) if d in relevant_ids
        )
    ideal = dcg(list(relevant_ids)[:k] + ranked_ids[:k])  # cukup: ideal = semua rel di posisi awal
    ideal_ids = list(relevant_ids)
    idcg = sum((1.0 / math.log2(i + 1)) for i in range(1, min(k, len(ideal_ids)) + 1))
    if idcg == 0:
        return 0.0
    return dcg(ranked_ids) / idcg


def evaluate_run(ranked_ids, relevant_ids, k_list=(5, 10)):
    out = {}
    for k in k_list:
        out[f"P@{k}"] = precision_at_k(ranked_ids, relevant_ids, k)
        out[f"R@{k}"] = recall_at_k(ranked_ids, relevant_ids, k)
        out[f"NDCG@{k}"] = ndcg_at_k(ranked_ids, relevant_ids, k)
    out["MRR"] = reciprocal_rank(ranked_ids, relevant_ids)
    out["AP"] = average_precision(ranked_ids, relevant_ids)
    return out

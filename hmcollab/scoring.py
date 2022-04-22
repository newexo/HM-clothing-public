import numpy as np


def precision_at_k(ranked_results):
    return sum(ranked_results) / len(ranked_results)


def ap_at_k(ranked_results):
    count = len(ranked_results)
    if not count:
        return 0
    s = 0
    for i in range(len(ranked_results)):
        if ranked_results[i]:
            s += precision_at_k(ranked_results[: i + 1])
    return s / count


def map_at_k(ranked_results):
    return np.mean([ap_at_k(r) for r in ranked_results])

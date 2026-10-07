"""Reference metrics and manual claim auditing; no model-generated ground truth."""
from .retrieval import tokens


def retrieval_metrics(ranked_ids, relevant_ids, k=5):
    if k <= 0 or not relevant_ids:
        raise ValueError("Positive k and nonempty relevance labels are required")
    ranked = list(dict.fromkeys(ranked_ids))[:k]
    relevant = set(relevant_ids)
    reciprocal = next((1 / i for i, x in enumerate(ranked, 1) if x in relevant), 0.0)
    return {f"recall@{k}": len(set(ranked) & relevant) / len(relevant),
            f"reciprocal_rank@{k}": reciprocal}


def rouge_l(reference, candidate):
    ref, cand = tokens(reference), tokens(candidate)
    if not ref or not cand:
        return 0.0
    row = [0] * (len(cand) + 1)
    for r in ref:
        previous = 0
        for j, c in enumerate(cand, 1):
            old = row[j]
            row[j] = previous + 1 if r == c else max(row[j], row[j-1])
            previous = old
    return 2 * row[-1] / (len(ref) + len(cand))


def bleu(reference, candidate):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    if not tokens(reference) or not tokens(candidate):
        return 0.0
    return float(sentence_bleu([tokens(reference)], tokens(candidate),
                              smoothing_function=SmoothingFunction().method4))


def claim_metrics(labels):
    """Manual labels distinguish unsupported from unverifiable claims."""
    allowed = {"supported", "unsupported", "unverifiable"}
    if any(x not in allowed for x in labels):
        raise ValueError("Invalid claim label")
    total = len(labels)
    counts = {x: labels.count(x) for x in sorted(allowed)}
    return {**counts, "total": total,
            "unsupported_fraction": counts["unsupported"] / total if total else None,
            "unverifiable_fraction": counts["unverifiable"] / total if total else None}

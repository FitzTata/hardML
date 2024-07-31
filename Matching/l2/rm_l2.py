from math import log2

from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    '''
    допустим, что swapped_pair - это пара элементов, находящихся в неправильном порядке
    в предсказанном тензоре ys_pred относительно истинного тензора ys_true
    '''
    if not len(ys_true) or not len(ys_pred):
        return 0

    if len(ys_true) != len(ys_pred):
        raise ValueError

    swapped_count = 0
    for i in range(len(ys_true)):
        for j in range(i + 1, len(ys_true)):
            condition1 = (ys_true[i] > ys_true[j]) and (ys_pred[i] < ys_pred[j])
            condition2 = (ys_true[i] < ys_true[j]) and (ys_pred[i] > ys_pred[j])
            if condition1 or condition2:
                swapped_count += 1

    return swapped_count


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return y_value
    if gain_scheme == 'exp2':
        return 2**y_value - 1


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    def sort_true_by_pred(ys_true: Tensor, ys_pred: Tensor) -> list:
        _, indices = sort(ys_pred, descending=True)
        ys_true_sorted = ys_true[indices]
        return ys_true_sorted.tolist()

    sorted_ys_true = sort_true_by_pred(ys_true.double(), ys_pred)
    return sum([compute_gain(val, gain_scheme) / log2(idx + 2) for idx, val in enumerate(sorted_ys_true)])


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    dcg_value = dcg(ys_true, ys_pred, gain_scheme)
    ideal_ys_true = sort(ys_true, descending=True).values
    idcg_value = dcg(ideal_ys_true, ideal_ys_true, gain_scheme)

    ndcg_value = dcg_value / idcg_value if idcg_value else 0.0

    return ndcg_value


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    def sort_true_by_pred(ys_true: Tensor, ys_pred: Tensor) -> list:
        _, indices = sort(ys_pred, descending=True)
        ys_true_sorted = ys_true[indices]
        return ys_true_sorted.tolist()

    sorted_ys_true = sort_true_by_pred(ys_true, ys_pred)
    precission_k = sum(sorted_ys_true[:k]) / min(sum(sorted_ys_true), k)

    return precission_k


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    try:
        correct_position = ys_true.tolist().index(1)
    except ValueError:
        return 0
    target = ys_pred[correct_position]
    ys_pred_sorted = sorted(ys_pred.tolist(), reverse=True)
    min_idx = ys_pred_sorted.index(target) + 1

    return 1 / min_idx


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15) -> float:
    def sort_true_by_pred(ys_true: Tensor, ys_pred: Tensor) -> list:
        _, indices = sort(ys_pred, descending=True)
        ys_true_sorted = ys_true[indices]
        return ys_true_sorted.tolist()

    p_continue = 1.0
    p_found_val = 0.0
    ys_true_sorted = sort_true_by_pred(ys_true, ys_pred)

    for rel in ys_true_sorted:
        p_found_val += p_continue * rel
        p_continue *= (1 - rel) * (1 - p_break)

    return p_found_val


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    def sort_true_by_pred(ys_true: Tensor, ys_pred: Tensor) -> list:
        _, indices = sort(ys_pred, descending=True)
        ys_true_sorted = ys_true[indices]
        return ys_true_sorted.tolist()

    n_relevant = 0
    cum_precision = 0.0
    ys_true_sorted = sort_true_by_pred(ys_true, ys_pred)

    for i, y_true in enumerate(ys_true_sorted):
        if y_true == 1.0:
            n_relevant += 1
            cum_precision += n_relevant / (i + 1)

    if n_relevant == 0:
        return 0.0

    return cum_precision / n_relevant

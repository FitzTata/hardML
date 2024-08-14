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

def test_num_swapped_pairs():
    tests = ((Tensor([5, 4, 3, 2, 1]), Tensor([0.9, 0.8, 0.7, 0.6, 0.5]), 0),
             (Tensor([4, 5, 3, 2, 1]), Tensor([0.9, 0.8, 0.7, 0.6, 0.5]), 1),
             (Tensor([1, 2, 3, 4, 5]), Tensor([0.9, 0.8, 0.7, 0.6, 0.5]), 10),
             (Tensor([0, 2, 2]), Tensor([0.1, 0.4, 0.5]), 0)
            )
    for test in tests:
        assert num_swapped_pairs(test[0], test[1]) == test[2], f'{num_swapped_pairs(test[0], test[1])} != {test[2]}'
    print('tests ok')
    

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
    
def test_dcg():
    tests = ((Tensor([3, 2, 3, 0, 1, 2]), Tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.55]), 'exp2', 13.909554869526021),
             (Tensor([3, 2, 3, 0, 1, 2]), Tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.55]), 'const', 6.891772308720021)
            )
    for test in tests:
        assert dcg(test[0], test[1], test[2]) == test[3], f'{dcg(test[0], test[1], test[2])} != {test[3]}'
    print('tests ok')
    

def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    dcg_value = dcg(ys_true, ys_pred, gain_scheme)
    ideal_ys_true = sort(ys_true, descending=True).values
    idcg_value = dcg(ideal_ys_true, ideal_ys_true, gain_scheme)

    ndcg_value = dcg_value / idcg_value if idcg_value else 0.0

    return ndcg_value
    
def test_ndcg():
    tests = ((Tensor([3, 2, 3, 0, 1, 2]), Tensor([0.9, 0.8, 0.9, 0.4, 0.5, 0.8]), 'exp2', 1.0),
             (Tensor([3, 2, 3, 0, 1, 2]), Tensor([0.9, 0.8, 0.9, 0.4, 0.5, 0.8]), 'const', 1.0)
            )
    for test in tests:
        assert ndcg(test[0], test[1], test[2]) == test[3], f'{ndcg(test[0], test[1], test[2])} != {test[3]}'
    print('tests ok')
    

def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    def sort_true_by_pred(ys_true: Tensor, ys_pred: Tensor) -> list:
        _, indices = sort(ys_pred, descending=True)
        ys_true_sorted = ys_true[indices]
        return ys_true_sorted.tolist()

    sorted_ys_true = sort_true_by_pred(ys_true, ys_pred)
    precission_k = sum(sorted_ys_true[:k]) / min(sum(sorted_ys_true), k)

    return precission_k
    
def test_precission_at_k():
    tests = ((Tensor([1, 0, 1, 0, 1, 1]), Tensor([.9, .8, .7, .6, .5, .4]), 3, 0.6666666666666666),
             (Tensor([1, 0, 1, 0]), Tensor([.9, .8, .7, .6]), 3, 1.0)
            )
    for test in tests:
        assert precission_at_k(test[0], test[1], test[2]) == test[3], f'{precission_at_k(test[0], test[1], test[2])} != {test[3]}'
    print('tests ok')
    

def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    try:
        correct_position = ys_true.tolist().index(1)
    except ValueError:
        return 0
    target = ys_pred[correct_position]
    ys_pred_sorted = sorted(ys_pred.tolist(), reverse=True)
    min_idx = ys_pred_sorted.index(target) + 1

    return 1 / min_idx
    
def test_reciprocal_rank():
    tests = ((Tensor([1, 0, 0, 0, 0, 0]), Tensor([.9, .8, .7, .6, .5, .4]), 1.0),
             (Tensor([0, 0, 0, 1]), Tensor([.9, .8, .7, .6]), 0.25),
             (Tensor([0, 0, 0, 0]), Tensor([.9, .8, .7, .6]), 0.0),
             (Tensor([0, 0, 0, 1]), Tensor([.6, .8, .9, .7]), 0.3333333333333333)
            )
              
    for test in tests:
        assert reciprocal_rank(test[0], test[1]) == test[2], f'{reciprocal_rank(test[0], test[1])} != {test[2]}'
    print('tests ok')
    

def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15) -> float:
    def sort_true_by_pred(ys_true: Tensor, ys_pred: Tensor) -> list:
        '''такая структура, чтобы проходили юниттесты на платформе'''
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

def test_p_found():
    tests = ((Tensor([0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 1., 0.,
        0., 1., 1., 1., 1., 1.]), Tensor([0.0577, 0.0117, 0.7932, 0.2888, 0.0691, 0.9759, 0.4000, 0.3519, 0.3288,
        0.4549, 0.0026, 0.2685, 0.6604, 0.9075, 0.7337, 0.4975, 0.5701, 0.0463,
        0.5535, 0.0775, 0.4004, 0.2248, 0.3393, 0.7370]), 0.6141249999999999),
             (Tensor([1.0]), Tensor([0.5]), 1.0),
            )
              
    for test in tests:
        assert p_found(test[0], test[1]) == test[2], f'{p_found(test[0], test[1])} != {test[2]}'
    print('tests ok')


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    def sort_true_by_pred(ys_true: Tensor, ys_pred: Tensor) -> list:
         '''такая структура, чтобы проходили юниттесты на платформе'''
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

def test_average_precision():
    tests = ((Tensor([1, 0, 1, 0, 0, 0]), Tensor([.9, .8, .7, .6, .5, .4]), 0.8333333333333333),
             (Tensor([0, 0, 0, 1]), Tensor([.9, .8, .7, .6]), 0.25),
             (Tensor([0, 0, 0, 0]), Tensor([.9, .8, .7, .6]), 0.0),
             (Tensor([0, 0, 0, 1]), Tensor([.6, .8, .9, .7]), 0.3333333333333333)
            )
              
    for test in tests:
        assert average_precision(test[0], test[1]) == test[2], f'{average_precision(test[0], test[1])} != {test[2]}'
    print('tests ok')
    return cum_precision / n_relevant

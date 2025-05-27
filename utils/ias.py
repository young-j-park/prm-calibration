
from typing import List
import math


def compute_N_IAS(p, denom=1.0, q=0.99, eps=1e-4):
    p = max(eps, min(p, 1.0 - eps))
    return math.ceil(math.log(1 - q) / math.log(1 - p) / denom)


def determine_M(
    scores: List[List[float]],
    max_samples: int,
    max_leaves: int,
    interleave: bool=False
) -> List[List[int]]:
    if not interleave:
        return [[max_samples] * len(row) for row in scores]
    else:
        m_is_list = []
        for row in scores:
            m_is = min(compute_N_IAS(min(row), max_leaves), max_samples)
            m_is_list.append([m_is] * len(row))
        return m_is_list


def determine_K(
    scores: List[List[float]],
    max_samples: int,
    max_leaves: int,
    interleave: bool=False
) -> List[int]:
    if not interleave:
        return [max_leaves for _ in scores]
    else:
        k_is_list = []
        for row in scores:
            kp1 = 1
            for k, s in enumerate(sorted(row, reverse=True)):
                kp1 = k + 1
                n_is = compute_N_IAS(s)
                if kp1 ==  max_leaves or kp1 * max_samples >= n_is:
                    break
            k_is_list.append(kp1)
        return k_is_list

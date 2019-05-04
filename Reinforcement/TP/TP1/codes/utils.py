import numpy as np


def policy_index_proba(A, s, d, det_pol=None, islist=None, value_base=False):
    """
    Args:
        A (int, list): if int it represents the number of actions per state,
                       if list it contains the available actions per state
        s (int): the code of the state
        d (list, list of list): the policy: S x 1 (det) or S x As (stoch)
        det_pol (none, bool): true if the policy is deterministic, false otherwise
        islist (none, bool): true if the action set is a list, false otherwise
        value_based (boolean): used to identify if the policy (det)
                               stores values or indeces
    Returns:
        A_idxs: the indexes of the actions selected by the policy
        proba: the probabilities associated to the selected actions
    """
    if det_pol is None:
        det_pol = False if isinstance(d[0], list) else True
    if islist is None:
        islist = isinstance(A, list)

    if det_pol:
        proba = 1
        if islist and value_base:
            A_idxs = A[s].index(d[s])
        else:
            A_idxs = d[s]

    else:
        proba = d[s]
        if islist:
            A_idxs = range(len(A[s]))
        else:
            A_idxs = range(A)
    return A_idxs, proba


def v_from_q(S, A, d, q):
    islist = isinstance(A, list)
    det_pol = False if isinstance(d[0], list) else True

    v = np.empty((S,))
    for s in range(S):
        A_idxs, proba = policy_index_proba(A=A, s=s, d=d, det_pol=det_pol, islist=islist)
        v[s] = np.dot(proba, q[s, A_idxs])

    return v

from collections import defaultdict

import numpy as np
import tqdm as tqdm
#from dataclasses import dataclass
from typing import Dict


class TrieNode:
    def __init__(self, depth=0):
        self.depth = depth
        self.counts = defaultdict(int)
        self.children = defaultdict(self.create_child)

    def create_child(self):
        return TrieNode(depth=self.depth + 1)

    def is_leaf(self):
        return len(self.children) == 0

    def __iter__(self):
        yield self

        for child in self.children.values():
            for grandchild in child:
                yield grandchild


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def add(self, id, s):
        node = self.root
        for c in s:
            node = node.children[c]

        node.counts[id] = 1

    @property
    def nodes(self):
        for node in self.root:
            yield node

    @property
    def num_nodes(self):
        return sum(1 for _ in self.nodes)

    @property
    def leaves(self):
        for node in self.root:
            if node.is_leaf():
                yield node

    @property
    def num_leaves(self):
        return sum(1 for _ in self.leaves)


class SpectrumKernel:

    def __init__(self, k, weight=None):
        self.k = k
        self.weight = weight
        self.trie = Trie()

        self.next_id = 0
        self.fitted_sequences = {}

        self.fitted_ = False
        self.fitted_on_ = None

    def fit(self, S):
        for s in S:
            self._fit_string(s)

        self.fitted_ = True
        self.fitted_on_ = S

        return self._build_kernel(S, self.fitted_on_)

    def predict(self, T):
        for t in T:
            self._fit_string(t)

        return self._build_kernel(T, self.fitted_on_)

    def _fit_string(self, s):
        if s in self.fitted_sequences:
            return

        id = self.next_id
        self.next_id += 1
        self.fitted_sequences[s] = id

        if self.is_weighted:
            substrings = sum(
                (self._substrings(s, k) for k in range(min(4, self.k), self.k + 1)),
                [],
            )
        else:
            substrings = self._substrings(s, self.k)

        for sub in substrings:
            self.trie.add(id, sub)

    def _build_kernel(self, T, S):
        T_ids = [
            self.fitted_sequences[t]
            for t in T
        ]

        S_ids = [
            self.fitted_sequences[s]
            for s in S
        ]

        dot_products = defaultdict(float)
        squared_norms = defaultdict(float)

        weight = self.weight or 1.0
        set_T_ids = set(T_ids)
        set_S_ids = set(S_ids)
        all_ids = set_T_ids | set_S_ids

        if self.weight is None:
            it = tqdm.tqdm(self.trie.leaves, total=self.trie.num_leaves)
        else:
            it = tqdm.tqdm(self.trie.nodes, total=self.trie.num_nodes)

        for node in it:
            node_ids = set(node.counts.keys())

            for idx in node_ids & all_ids:
                squared_norms[idx] += node.counts[idx] ** 2 * weight ** (self.k - node.depth)

            for t_idx in node_ids & set_T_ids:
                for s_idx in node_ids & set_S_ids:
                    dot_product = node.counts[t_idx] * node.counts[s_idx]
                    if self.is_weighted:
                        dot_product *= weight ** (self.k - node.depth)
                    dot_products[t_idx, s_idx] += dot_product

        K = np.zeros((len(T), len(S)))

        for i, t_idx in enumerate(T_ids):
            for j, s_idx in enumerate(S_ids):
                K[i, j] = dot_products[t_idx, s_idx] / np.sqrt(squared_norms[t_idx] * squared_norms[s_idx])

        return K

    @staticmethod
    def _substrings(s, k):
        return [
            s[i:i + k]
            for i in range(len(s) - k + 1)
        ]

    @property
    def is_weighted(self):
        return self.weight is not None

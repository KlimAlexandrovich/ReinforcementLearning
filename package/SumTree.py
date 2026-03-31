import numpy as np


class SumTree:
    """
    A binary tree-based structure for efficient handling of priorities.
    SumTree is used in priority sampling, often for tasks such as reinforcement learning
    or other probabilistic tasks where entries are sampled based on their priority.
    It efficiently manages priorities by leveraging a binary tree structure and provides
    methods to update priorities, access cumulative priorities, and perform proportional
    sampling.
    """

    def __init__(self, capacity: int):
        self.user_capacity = capacity
        self.capacity = 2 ** int(np.ceil(np.log2(capacity)))
        self.tree_levels = int(np.log2(self.capacity)) + 1
        self.tree_size = 2 * self.capacity - 1
        self.tree = np.zeros(self.tree_size)

    def __repr__(self) -> str:
        return (
            f"SumTree("
            f"user_capacity={self.user_capacity}, "
            f"capacity={self.capacity}, "
            f"tree_levels={self.tree_levels}, "
            f"tree_size={self.tree_size}"
            f")"
        )

    def _propagate(self, tree_index: int, change: float) -> None:
        """ Propagate priority change up the tree. """
        parent = (tree_index - 1) // 2
        self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)

    def _leaf_index(self, data_index: int) -> int:
        """ Convert data index to leaf node index in a tree. """
        assert 0 <= data_index < self.user_capacity, \
            f"Invalid data_index: {data_index} not in [0, {self.user_capacity})"
        return data_index + self.tree_size // 2

    def update(self, data_index: int, priority: float) -> None:
        """ Update the priority of a single data point. """
        assert 0 <= data_index < self.user_capacity, \
            f"Invalid data_index: {data_index} not in [0, {self.user_capacity})"
        assert priority >= 0, f"Invalid priority: {priority} must be ≥ 0."
        tree_index = self._leaf_index(data_index)
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    @property
    def total_priority(self) -> float:
        """ Returns the sum of all priorities (root node value). """
        return self.tree[0]

    def fill_tree(self, value: float) -> None:
        """ Fill leaf values with a single value and rebuild a tree. """
        assert value >= 0, f"Invalid value: {value} must be ≥ 0."
        # Fill only leaf nodes.
        leaf_start = self.tree_size // 2
        self.tree[leaf_start:] = value
        # Rebuild internal nodes from bottom to top.
        for i in range(leaf_start - 1, -1, -1):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            self.tree[i] = self.tree[left_child] + self.tree[right_child]

    def get(self, value: float) -> int:
        """
        Retrieve data index by cumulative priority value.
        Used for proportional sampling: sample ~ priority.
        Args: value: Random value in [0, total_priority]
        Returns: Data index corresponding to sampled priority
        """
        assert 0 <= value <= self.total_priority, \
            f"Invalid value: {value} not in [0, {self.total_priority}]"
        parent_idx = 0
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1

            if left_child >= self.tree_size:
                # Leaf node reached.
                leaf_index = parent_idx
                data_index = leaf_index - self.tree_size // 2
                return data_index

            # Traverse down the tree.
            if value <= self.tree[left_child]:
                parent_idx = left_child
            else:
                value -= self.tree[left_child]
                parent_idx = right_child

    def get_priority(self, data_index: int) -> float:
        """ Get priority of a specific data point. """
        tree_index = self._leaf_index(data_index)
        return self.tree[tree_index]

    def get_max_priority(self, default: float = 1.0) -> float:
        """ Return maximum priority value among stored entries. """
        leaf_start = self.tree_size // 2
        max_val = self.tree[leaf_start: leaf_start + self.user_capacity].max()
        return max_val if max_val > 0 else default

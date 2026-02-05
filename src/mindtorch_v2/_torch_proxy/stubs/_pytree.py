"""Stub for torch.utils._pytree - Tree utilities.

This is a Tier 3 stub that provides minimal tree_flatten/tree_unflatten
functionality that PyTorch-based libraries may use.
"""

from collections import namedtuple


# Tree spec for reconstruction
TreeSpec = namedtuple('TreeSpec', ['type', 'context', 'children_specs'])


def tree_flatten(obj):
    """Flatten a nested structure into a list of leaves and a TreeSpec.

    Args:
        obj: A nested structure (list, tuple, dict, or leaf)

    Returns:
        Tuple of (list of leaves, TreeSpec for reconstruction)
    """
    if obj is None:
        return [], TreeSpec(type(None), None, [])

    if isinstance(obj, (list, tuple)):
        leaves = []
        children_specs = []
        for item in obj:
            item_leaves, item_spec = tree_flatten(item)
            leaves.extend(item_leaves)
            children_specs.append(item_spec)
        return leaves, TreeSpec(type(obj), None, children_specs)

    if isinstance(obj, dict):
        leaves = []
        children_specs = []
        keys = list(obj.keys())
        for key in keys:
            item_leaves, item_spec = tree_flatten(obj[key])
            leaves.extend(item_leaves)
            children_specs.append(item_spec)
        return leaves, TreeSpec(dict, keys, children_specs)

    # Leaf node
    return [obj], TreeSpec(type(obj), None, [])


def tree_unflatten(leaves, treespec):
    """Reconstruct a nested structure from leaves and a TreeSpec.

    Args:
        leaves: List of leaf values
        treespec: TreeSpec describing the structure

    Returns:
        Reconstructed nested structure
    """
    if treespec.type is type(None):
        return None

    if treespec.type in (list, tuple):
        result = []
        leaf_idx = 0
        for child_spec in treespec.children_specs:
            num_leaves = _count_leaves(child_spec)
            child_leaves = leaves[leaf_idx:leaf_idx + num_leaves]
            result.append(tree_unflatten(child_leaves, child_spec))
            leaf_idx += num_leaves
        return treespec.type(result)

    if treespec.type is dict:
        result = {}
        keys = treespec.context
        leaf_idx = 0
        for i, child_spec in enumerate(treespec.children_specs):
            num_leaves = _count_leaves(child_spec)
            child_leaves = leaves[leaf_idx:leaf_idx + num_leaves]
            result[keys[i]] = tree_unflatten(child_leaves, child_spec)
            leaf_idx += num_leaves
        return result

    # Leaf node
    return leaves[0] if leaves else None


def _count_leaves(treespec):
    """Count the number of leaves in a TreeSpec."""
    if not treespec.children_specs:
        return 1
    return sum(_count_leaves(child) for child in treespec.children_specs)


def tree_map(fn, tree):
    """Apply a function to all leaves in a tree.

    Args:
        fn: Function to apply to each leaf
        tree: Nested structure

    Returns:
        New tree with fn applied to all leaves
    """
    leaves, spec = tree_flatten(tree)
    new_leaves = [fn(leaf) for leaf in leaves]
    return tree_unflatten(new_leaves, spec)


def tree_map_only(ty, fn, tree):
    """Apply function only to leaves of a specific type."""
    def conditional_fn(x):
        if isinstance(x, ty):
            return fn(x)
        return x
    return tree_map(conditional_fn, tree)


# PyTree registration (no-op for our purposes)
def register_pytree_node(cls, flatten_fn, unflatten_fn, *, serialized_type_name=None, **kwargs):
    """Register a custom type for pytree operations."""
    pass


# Context for pytree operations
class _TreeContext:
    pass


# Additional utilities for pytree operations
def tree_leaves(tree):
    """Get all leaves from a tree."""
    leaves, _ = tree_flatten(tree)
    return leaves


def tree_structure(tree):
    """Get the TreeSpec of a tree."""
    _, spec = tree_flatten(tree)
    return spec


def tree_all(fn, tree):
    """Check if fn returns True for all leaves."""
    return all(fn(leaf) for leaf in tree_leaves(tree))


def tree_any(fn, tree):
    """Check if fn returns True for any leaf."""
    return any(fn(leaf) for leaf in tree_leaves(tree))

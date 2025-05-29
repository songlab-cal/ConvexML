import pickle

from cassiopeia.data import CassiopeiaTree


def read_tree(
    tree_path: str,
) -> CassiopeiaTree:
    with open(tree_path, "rb") as f:
        return pickle.load(f)


def write_tree(
    tree: CassiopeiaTree,
    tree_path: str,
):
    with open(tree_path, "wb") as f:
        pickle.dump(tree, f)
        f.flush()

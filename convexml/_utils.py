import networkx as nx


def to_newick(
    tree: nx.DiGraph,
    record_branch_lengths: bool = False,
    record_node_names: bool = False,
) -> str:
    """Converts a networkx graph to a newick string.

    Args:
        tree: A networkx tree
        record_branch_lengths: Whether to record branch lengths on the tree in
            the newick string
        record_node_names: Whether to record internal node names on the tree in
            the newick string

    Returns:
        A newick string representing the topology of the tree
    """

    def _to_newick_str(g, node):
        is_leaf = g.out_degree(node) == 0
        weight_string = ""

        if record_branch_lengths and g.in_degree(node) > 0:
            parent = list(g.predecessors(node))[0]
            weight_string = ":" + str(g[parent][node]["length"])

        _name = str(node)

        name_string = ""
        if record_node_names:
            name_string = f"{_name}"

        return (
            "%s" % (_name,) + weight_string
            if is_leaf
            else (
                "("
                + ",".join(
                    _to_newick_str(g, child) for child in g.successors(node)
                )
                + ")"
                + name_string
                + weight_string
            )
        )

    root = [node for node in tree if tree.in_degree(node) == 0][0]
    return _to_newick_str(tree, root) + ";"

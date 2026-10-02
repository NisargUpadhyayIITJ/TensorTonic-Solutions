def build_expression_graph(leaves: list, operations: list) -> tuple:
    """
    Returns a tuple of the node list and final node ID.
    """
    result_nodes = []
    for node in leaves:
        result_nodes.append({
            "id": node['id'],
            "data": float(node['data']),
            "grad": 0.0,
            "op": "",
            "parents": []
        })
    for operation in operations:
        left_id = operation['left']
        right_id = operation['right']
        left = [node for node in result_nodes if node["id"] == left_id][0]
        right = [node for node in result_nodes if node["id"] == right_id][0]
        out_data = 0.0
        if(operation['op'] == "+"):
            out_data = left['data'] + right['data']
        if(operation['op'] == "*"):
            out_data = left['data'] * right['data']

        result_nodes.append({
            "id": operation['id'],
            "data": float(out_data),
            "grad": 0.0,
            "op": operation["op"],
            "parents": [left["id"], right["id"]]
        })
    return result_nodes, result_nodes[-1]["id"]

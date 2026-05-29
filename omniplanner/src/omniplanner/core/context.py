from collections import UserDict

from dsg_pddl.core.parsing import pddl_char_to_dsg_char
from spark_dsg import NodeSymbol


def string_as_nodesymbol(string):
    try:
        c = pddl_char_to_dsg_char(string[0])
        return NodeSymbol(c, int(string[1:]))
    except Exception:
        return None


class DsgNodeContext(UserDict):
    def __init__(self, dsg_dict, external_dict):
        super().__init__()
        self.dsg_dict = dsg_dict
        self.external_dict = external_dict
        self.data = dsg_dict | external_dict

    def __setitem__(self, key, value):
        if key in self.dsg_dict:
            raise Exception("Cannot override DSG context")
        self.external_dict[key] = value
        self.data[key] = value


class DsgContextProvider(dict):
    def __init__(self, dsg):
        self.dsg = dsg

    def __getitem__(self, key):
        ns = string_as_nodesymbol(key)
        symbol_info = {}
        if ns is not None:
            node = self.dsg.find_node(ns)
            if node is not None:
                symbol_info["position"] = node.attributes.position
                node_layer = node.layer.layer
                node_partition = node.layer.partition
                category = self.dsg.get_labelspace(
                    node_layer, node_partition
                ).get_node_category(node)
                symbol_info["semantic_label"] = category

        try:
            explicit_symbols = dict.__getitem__(self, key)
        except KeyError:
            dict.__setitem__(self, key, {})
            explicit_symbols = dict.__getitem__(self, key)
        return DsgNodeContext(symbol_info, explicit_symbols)

    def __setitem__(self, key, val):
        if not isinstance(val, dict):
            raise TypeError(
                "Context value must be a dictionary, e.g., {context_key: context_value}"
            )
        dict.__setitem__(self, key, val)

    def __contains__(self, key):
        if dict.__contains__(self, key) and len(dict.__getitem__(self, key)) > 0:
            return True

        ns = string_as_nodesymbol(key)
        if ns is None:
            return False
        return self.dsg.find_node(ns) is not None

# TODO: need to reexamine this whole parsing framework as some point.
# It's brittle and requires the :type section which should be optional
# similar for :functions
def ensure_pddl_domain(ast):
    if ast[0] != "define":
        raise Exception("Malformed PDDL Domain ast, missing define")

    if ast[2][0] != ":requirements":
        raise Exception("Missing :requirements, must go after name")

    if ast[3][0] != ":types":
        raise Exception("Missing :types, must go after requirements")

    if ast[4][0] != ":predicates":
        raise Exception("Missing :predicates, must go after types")

    if ast[5][0] != ":functions":
        raise Exception("Missing :functions, must go after predicates")

    if ast[5][0] != ":functions":
        raise Exception("Missing :functions, must go after predicates")

    for clause in ast[6:]:
        if clause[0] not in [":derived", ":action"]:
            raise Exception(f"Expected a :deried or :action, not {clause[0]}")


def get_domain_name(ast):
    return ast[1][1]


def get_domain_requirements(ast):
    return ast[2][1:]


def get_domain_types(ast):
    # TODO: this is arguably incomplete, because we don't
    # parse the type/subtype relationship
    return ast[3][1:]


def get_domain_predicates(ast):
    return ast[4][1:]


def get_functions(ast):
    return ast[5][1:]


def get_derived(ast):
    derived = ()
    for clause in ast:
        if type(clause) is not tuple:
            continue
        if clause[0] == ":derived":
            derived += clause[1:]
    return derived


def get_actions(ast):
    actions = ()
    for clause in ast:
        if type(clause) is not tuple:
            continue
        if clause[0] == ":action":
            actions += clause[1:]
    return actions

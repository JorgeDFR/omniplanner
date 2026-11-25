from functools import reduce


def extract_facts(goal, predicate):
    match goal:
        case tuple() | list():
            if len(goal) == 0:
                return ()
            elif goal[0] == predicate:
                return (goal,)
            else:
                children = (extract_facts(g, predicate) for g in goal)
                return reduce(lambda x, y: x + y, children)
        case _:
            return ()


def extract_negated_facts(goal, predicate, negated=False):
    return _extract_facts_with_polarity(goal, predicate, negated, True)


def extract_positive_facts(goal, predicate):
    return _extract_facts_with_polarity(goal, predicate, False, False)


def _extract_facts_with_polarity(goal, predicate, negated, required_polarity):
    if not isinstance(goal, (tuple, list)) or not goal:
        return ()
    head = goal[0]
    if head == "not" and len(goal) > 1:
        return _extract_facts_with_polarity(goal[1], predicate, not negated, required_polarity)
    if head == predicate:
        return (goal,) if negated == required_polarity else ()
    return tuple(
        fact
        for child in goal
        for fact in _extract_facts_with_polarity(child, predicate, negated, required_polarity)
    )


def tokenize_lisp(string):
    return string.replace("\n", "").replace("(", "( ").replace(")", " )").split()


def get_lisp_ast(toks):
    t = toks.pop(0)
    if t == "(":
        exp = ()
        while toks[0] != ")":
            exp += (get_lisp_ast(toks),)
        toks.pop(0)
        return exp
    elif t != ")":
        return t
    else:
        raise SyntaxError("Unexpected )")


def lisp_string_to_ast(string):
    return get_lisp_ast(tokenize_lisp(string))


def ast_to_string(ast):
    match ast:
        case list() | tuple():
            elements = [ast_to_string(e) for e in ast]
            elements_str = " ".join(elements)
            return f"({elements_str})"
        case _:
            return str(ast)


# PDDL is case-insensitive, but spark_dsg is case sensitive.
# Here to convert back and forth, but this is *extremely* brittle.
def pddl_char_to_dsg_char(c):
    match c:
        case "r":
            return "R"
        case "o":
            return "O"
        case "p":
            # NOTE: this means we can only ground to 2d places, not 3d places
            return "P"
        case _:
            return c

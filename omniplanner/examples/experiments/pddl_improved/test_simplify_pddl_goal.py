from itertools import product


def simplify(expr):
    """
    Convert a PDDL logical expression into DNF.

    Output:
        - single conjunction:
              ('and', ...)
        - multiple conjunctions:
              ('or', ('and', ...), ('and', ...))
    """

    def is_atom(x):
        return not isinstance(x, tuple) or (
            len(x) > 0 and x[0] not in ("and", "or", "not")
        )

    def negate(lit):
        if isinstance(lit, tuple) and lit[0] == "not":
            return lit[1]
        return ("not", lit)

    def to_dnf(e):
        # Atomic predicate
        if is_atom(e):
            return [frozenset([e])]

        op = e[0]

        # NOT
        if op == "not":
            arg = e[1]

            # Double negation
            if isinstance(arg, tuple) and arg[0] == "not":
                return to_dnf(arg[1])

            # De Morgan
            if isinstance(arg, tuple):
                if arg[0] == "and":
                    return to_dnf(
                        ("or", *[("not", x) for x in arg[1:]])
                    )

                if arg[0] == "or":
                    return to_dnf(
                        ("and", *[("not", x) for x in arg[1:]])
                    )

            return [frozenset([e])]

        # OR
        if op == "or":
            result = []
            for sub in e[1:]:
                result.extend(to_dnf(sub))
            return result

        # AND
        if op == "and":
            sub_dnfs = [to_dnf(sub) for sub in e[1:]]
            result = []

            for combo in product(*sub_dnfs):
                merged = set()
                contradiction = False

                for clause in combo:
                    for lit in clause:
                        if negate(lit) in merged:
                            contradiction = True
                            break
                        merged.add(lit)

                    if contradiction:
                        break

                if not contradiction:
                    result.append(frozenset(merged))

            return result

        raise ValueError(f"Unknown operator: {op}")

    def build_expr(dnf):
        # Remove duplicates
        unique = []
        seen = set()

        for clause in dnf:
            if clause not in seen:
                seen.add(clause)
                unique.append(clause)

        if not unique:
            return False

        and_terms = []

        for clause in unique:
            lits = sorted(clause, key=str)

            if len(lits) == 1:
                and_terms.append(("and", lits[0]))
            else:
                and_terms.append(("and", *lits))

        # Only one conjunction -> return AND directly
        if len(and_terms) == 1:
            return and_terms[0]

        return ("or", *and_terms)

    return build_expr(to_dnf(expr))


# -------------------------------------------------------------------
# Example
# -------------------------------------------------------------------

pddl_goal = (
    "and",

    # Must have visited a region
    ("or",
        ("visited-region", "r1"),
        ("visited-region", "r2"),
    ),

    # Either be at a specific POI or holding an object
    ("or",
        ("at-poi", "p12"),
        ("holding", "o3"),
    ),

    # Nested conjunction/disjunction
    ("and",

        # Either object is safe OR visited
        ("or",
            ("safe", "o7"),
            ("visited-object", "o7"),
        ),

        # Object location constraints
        ("or",
            ("and",
                ("object-in-place", "o7", "p44"),
                ("visited-place", "p44"),
            ),
            ("and",
                ("object-in-place", "o7", "p91"),
                ("visited-place", "p91"),
            ),
        ),

        # Negated branch
        ("or",
            ("not", ("at-place", "p3")),
            ("visited-place", "p3"),
        ),
    ),

    # Region membership
    ("or",
        ("and",
            ("in-region", "r1"),
            ("visited-place", "p18"),
        ),
        ("and",
            ("in-region", "r2"),
            ("visited-place", "p77"),
        ),
    ),

    # Contradiction branch (should disappear after simplify)
    ("or",
        ("and",
            ("holding", "o9"),
            ("not", ("holding", "o9")),
        ),
        ("visited-object", "o9"),
    ),
)

print(simplify(pddl_goal))
"""Exact optimistic bounds for rectangular injective assignment gains."""


def injective_assignment_upper_bound(rows):
    """Bound integer gains for r rows assigned to distinct columns (r <= c).

    Callers supply rectangular integer rows and have checked capacity. Independent
    row maxima give one bound. Every feasible assignment also selects r distinct
    columns, and each selected contribution is at most that column's maximum.
    Thus the r largest column maxima give another bound; their minimum remains
    optimistic, including when gains are negative. Neither bound is a solver.
    """
    if not rows:
        return 0
    row_bound = sum(max(row) for row in rows)
    if len(rows) == 1:
        return row_bound
    column_maxima = [max(column) for column in zip(*rows)]
    column_bound = sum(sorted(column_maxima, reverse=True)[:len(rows)])
    return min(row_bound, column_bound)

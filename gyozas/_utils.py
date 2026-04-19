"""Internal utility helpers shared across gyozas submodules."""


def is_fixed_domain(var) -> bool:
    """Return True if a SCIP variable has a fixed domain (lb >= ub)."""
    return var.getLbLocal() >= var.getUbLocal()

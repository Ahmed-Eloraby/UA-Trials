"""Shared Gurobi model for the LP relaxation and the exact ILP.

Both programs minimize the maximum node load. The only variables are the
orientation of each unlocked edge: ``x_u + x_v = 1``. Locked edges are
constants in those loads, so a later step can freeze part of the orientation
and solve the rest with the same builder. A candidate makespan ``T`` is used
only by the flow search and by the balance search, not by this minimum-load
model. :class:`CapacitatedRelaxationStep` is the related feasibility LP:
free edges must fit in given residual capacities, and the objective is zero.
"""

from __future__ import annotations

from makespan.context import SolveContext
from makespan.factory import Step


class SolverUnavailable(RuntimeError):
    """Raised when ``gurobipy`` is not installed."""


def require_gurobi():
    """Import and return ``gurobipy``.

    Importing this package stays safe when Gurobi is absent. The error is
    raised only when an LP or ILP step actually runs.

    Returns:
        module: The imported ``gurobipy`` module.
    """
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise SolverUnavailable(
            "gurobipy is not installed; install it (with a license) to use the "
            "LP/ILP models"
        ) from exc
    return gp


def solver_available() -> bool:
    """Whether the LP and ILP steps can run.

    Returns:
        bool: True when ``gurobipy`` can be imported.
    """
    try:
        require_gurobi()
    except SolverUnavailable:
        return False
    return True


class RelaxationStep(Step):
    """Solve the orientation LP, or the ILP when ``integral`` is true.

    The LP writes a fraction toward ``u`` for every unlocked edge and raises
    the lower bound to the optimal value. The ILP also proposes the optimal
    orientation of the unlocked edges.

    Args:
        integral: When true, variables are binary and the step proposes heads.
        time_limit: Gurobi time limit in seconds, or ``None`` for the solver default.
    """

    def __init__(self, integral: bool, time_limit: float | None = None):
        """Select the ILP when ``integral`` is true, otherwise the LP.

        ``time_limit`` is passed to Gurobi in seconds. ``None`` leaves the
        solver's own limit in place.

        Args:
            integral: When true, variables are binary and the step proposes heads.
            time_limit: Gurobi time limit in seconds, or ``None`` for the solver default.
        """
        self.integral = integral
        self.time_limit = time_limit

    def run(self, ctx: SolveContext) -> None:
        """Minimize the maximum load over the unlocked edges.

        The LP records a fraction toward ``u`` for each free edge. The ILP
        also proposes the optimal heads. In both cases the optimal value
        raises the context lower bound.

        Gurobi only accepts a linear or quadratic objective, so the maximum
        load is represented by one linear variable that is at least every
        node load. That variable is the objective. It is not a threshold
        supplied to the model.

        Args:
            ctx: Solve state whose unlocked edges become LP or ILP variables.
        """
        gp = require_gurobi()
        from gurobipy import GRB

        model = gp.Model("orientation")
        model.Params.OutputFlag = 0
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        variables: dict[int, tuple] = {}
        free = ctx.free_edges()
        kind = GRB.BINARY if self.integral else GRB.CONTINUOUS
        for edge in free:
            toward_u = model.addVar(lb=0.0, ub=1.0, vtype=kind, name=f"x_{edge.id}_u")
            toward_v = model.addVar(lb=0.0, ub=1.0, vtype=kind, name=f"x_{edge.id}_v")
            variables[edge.id] = (toward_u, toward_v)
            model.addConstr(toward_u + toward_v == 1, name=f"orient_{edge.id}")
        base = ctx.base_load()
        loads = []
        for node in ctx.instance.nodes:
            load = gp.LinExpr(base[node])
            for edge in ctx.instance.incident(node):
                if edge.id in ctx.locked:
                    continue
                toward_u, toward_v = variables[edge.id]
                chosen = toward_u if node == edge.u else toward_v
                load += edge.weight_toward(node) * chosen
            loads.append(load)
        if not loads:
            model.setObjective(0.0, GRB.MINIMIZE)
        else:
            # A max expression is not a legal Gurobi objective. This variable
            # is forced to sit above every node load, then minimized.
            maximum_load = model.addVar(lb=0.0, name="maximum_load")
            for index, load in enumerate(loads):
                model.addConstr(load <= maximum_load, name=f"maximum_{index}")
            model.setObjective(maximum_load, GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL or model.SolCount == 0:
            kind_name = "ILP" if self.integral else "LP"
            raise RuntimeError(
                f"{kind_name} ended with Gurobi status {model.Status} "
                f"and {model.SolCount} solutions"
            )
        objective = float(model.ObjVal)
        if self.integral:
            ctx.metadata["ilp_objective"] = objective
        else:
            ctx.metadata["lp_objective"] = objective
        free_heads: dict[int, int] = {}
        for edge in free:
            toward_u, toward_v = variables[edge.id]
            fraction_u = float(toward_u.X)
            ctx.fractional[edge.id] = fraction_u
            if self.integral:
                head = edge.u if fraction_u >= float(toward_v.X) else edge.v
                free_heads[edge.id] = head
                ctx.orientation[edge.id] = head
        if self.integral:
            ctx.consider(ctx.assignment_from(free_heads))
        ctx.raise_lower_bound(objective)


class CapacitatedRelaxationStep(Step):
    """Feasibility LP for the free edges under residual capacities.

    Locked edges stay fixed. Each open node can accept at most
    ``residual_cap[node]`` more load from free edges. Closed nodes are not
    constrained. The objective is zero: any feasible fractional orientation
    is enough for a later rounding step. An infeasible model sets
    ``ctx.metadata["residual_feasible"]`` to false and does not write fractions.

    Args:
        residual_cap: Remaining load each node may accept from free edges.
        open_nodes: Nodes that still receive a capacity constraint.
    """

    def __init__(self, residual_cap: dict[int, float], open_nodes: set[int]):
        """Store the residual capacities and the nodes the LP is allowed to load.

        Args:
            residual_cap: Remaining load each node may accept from free edges.
            open_nodes: Nodes that still receive a capacity constraint.
        """
        self.residual_cap = dict(residual_cap)
        self.open_nodes = set(open_nodes)

    def run(self, ctx: SolveContext) -> None:
        """Write a feasible fraction toward ``u`` for every free edge, or mark failure.

        Args:
            ctx: Solve state whose free edges are the LP variables.
        """
        gp = require_gurobi()
        from gurobipy import GRB

        free = ctx.free_edges()
        model = gp.Model("residual_orientation")
        model.Params.OutputFlag = 0
        variables: dict[int, tuple] = {}
        try:
            for edge in free:
                toward_u = model.addVar(lb=0.0, ub=1.0, name=f"x_{edge.id}_u")
                toward_v = model.addVar(lb=0.0, ub=1.0, name=f"x_{edge.id}_v")
                variables[edge.id] = (toward_u, toward_v)
                model.addConstr(toward_u + toward_v == 1, name=f"orient_{edge.id}")
            for node in self.open_nodes:
                rhs = self.residual_cap.get(node, 0.0)
                if -1e-8 <= rhs < 0.0:
                    rhs = 0.0
                terms = []
                for edge in free:
                    if node not in (edge.u, edge.v):
                        continue
                    toward_u, toward_v = variables[edge.id]
                    chosen = toward_u if node == edge.u else toward_v
                    terms.append(edge.weight_toward(node) * chosen)
                if terms:
                    model.addConstr(gp.quicksum(terms) <= rhs, name=f"cap_{node}")
                elif rhs < -1e-8:
                    ctx.metadata["residual_feasible"] = False
                    return
            model.setObjective(0.0, GRB.MINIMIZE)
            model.optimize()
            if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                ctx.metadata["residual_feasible"] = False
                return
            if model.Status != GRB.OPTIMAL or model.SolCount == 0:
                raise RuntimeError(
                    f"residual LP ended with Gurobi status {model.Status}"
                )
            for edge in free:
                toward_u, _toward_v = variables[edge.id]
                ctx.fractional[edge.id] = float(toward_u.X)
            ctx.metadata["residual_feasible"] = True
        finally:
            model.dispose()

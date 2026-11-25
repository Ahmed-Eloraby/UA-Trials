# Minimum-makespan orientation

Each edge of an undirected multigraph has two weights, one for each endpoint. An orientation sends the edge to exactly one endpoint and adds that weight to the endpoint's load. The solvers below choose an orientation whose maximum load, the makespan, is as small as possible.

Input is an export from `Graph Instance Builder.html`: text, CSV, or a JSON backup. Node ids are kept as they were exported.

## Run

From this directory, with Python able to import the `src` folder (the launcher adds it):

```text
python solve.py instances.txt --algo greedy-light
python solve.py instances.txt --algo flow-search
python solve.py instances.txt --algo all --format json
python solve.py instances.txt --algo greedy-light --format csv --out orientations.csv
python solve.py generated/n10_m15.txt -C score-ratio-peel-through score-ratio-peel-through-flow-asc score-ratio-peel-through-flow-room-asc score-ratio-peel-through-flow-room-desc score-ratio-peel-through-quota score-ratio-peel-through-topo score-ratio-raw-through score-ratio-raw-through-flow-asc score-ratio-raw-through-flow-room-asc score-ratio-raw-through-flow-room-desc score-ratio-raw-through-quota score-ratio-raw-through-topo score-diff-peel-through score-diff-peel-through-flow-asc score-diff-peel-through-flow-room-asc score-diff-peel-through-flow-room-desc score-diff-peel-through-quota score-diff-peel-through-topo score-diff-raw-through score-diff-raw-through-flow-asc score-diff-raw-through-flow-room-asc score-diff-raw-through-flow-room-desc score-diff-raw-through-quota score-diff-raw-through-topo -R brute
```

`python` must be the interpreter that has the packages you need. `lp-round` and `ilp` also need `gurobipy` and a Gurobi license. The other algorithms do not.

Each run also creates `Outputs/<YYYY-MM-DD_HH-MM-SS>/` in the working directory (`--output-dir` changes the parent). The terminal report is unchanged. The folder holds the files below.

| File | What it is |
| --- | --- |
| `assignments/<algorithm>.txt` | One file per algorithm that produced an orientation. Every instance keeps its name and tags. After `n m`, each edge is `u v head weight`: `head` is the node the edge points to, and `weight` is the load that choice adds. A `# loads` block then lists `node load` for every node, including isolates. |
| `solutions.csv` | One row per oriented edge: `instance,tags,algorithm,makespan,lower_bound,upper_bound,edge_id,u,v,head,weight,skipped`. |
| `loads.csv` | One row per node: `instance,tags,algorithm,node,load,makespan`. |
| `summary.csv` | One row per algorithm, across every instance in the run. Counts of `solved` and `skipped`, then `min`, `avg`, and `max` of `makespan`, `lower_bound`, and `upper_bound`. A comparison also fills `gap` and `ratio` the same way. A metric with no values is left blank. An infinite ratio is excluded from the average and shown as `inf` in the maximum. |
| `comparison.csv`, `comparison.txt` | Written for `-C` / `-R`. The text file is the same table the terminal prints. |
| `solutions.json` or `comparison.json` | Written when `--format json`. |
| `report.txt` | The terminal text report, when `--format text` and this is not a comparison. |

Open `Assignment Viewer.html` and choose a run folder (Open folder) or the files inside `assignments`. The arrow points at the node that received the edge. The number under a node is its load. Amber marks every node whose load equals the makespan. The algorithm menu switches drawings when more than one algorithm was saved.

## Algorithms

| Name | What it does |
| --- | --- |
| `greedy-light` | Send every edge to its lighter endpoint. |
| `greedy-load` | Send each edge to the endpoint that leaves the smaller load. |
| `ratio-greedy` | Peel pendant bundles by weight ratio, orient the core toward the lighter endpoint, then propagate overload for as many passes as the component has nodes. The threshold is bisected. A rejected threshold is not used as a lower bound. |
| `brute` | Exact branch and bound. Refuses instances with more than `--max-edges` free edges (default 20). |
| `flow-search` | Greedy upper bound, then binary search on a flow network. The searched value is a lower bound. On unit weights it is the optimum. |
| `balance-flow` | Fenced threshold search. Residual edges are rounded by max flow. |
| `balance-graph` | LP-free threshold search: exact graph reductions, a certified price-guided push, then flip-chain polishing. Within 2(1 + eps) of the optimum. See below. |
| `lp-round` | Gurobi LP, then round each edge toward an endpoint whose fraction is at least 1/2. |
| `balance-lp` | The same fenced search as `balance-flow`, with the residual edges rounded from a capacitated LP. Needs Gurobi. |
| `balance-cycle` | The same fenced search as `balance-lp`, with residual cycles broken by comparing both directions. No LP. |
| `balance-bridge` | The same search as `balance-cycle`, after splitting each bridge bundle by the spare capacity of the two sides. |
| `ilp` | The same model with binary orientation variables. This is exact. |

`score-{ratio|diff}-{peel|raw}-[early-]{sink|source|both|flow|topo|inflow}` is one threshold search with forty-eight names. `ratio` compares loads by quotient and `diff` by subtraction. `peel` prunes leaves before scoring; `raw` does not. Each trial absorbs any node that can take all of its free edges, scores the remaining free edges by a hypothetical in-load and out-load, then offloads under the threshold. A plain name orients edges inside one level after that pass. The `early` name orients them first, using the same metric, so the cuts see the residual load. `sink` pushes toward the lowest score, `source` pushes away from the highest, and `both` uses distance to each. `flow` and `topo` direct a simple arc from a higher score to a lower score and concede in max-flow order or topological order. `inflow` also opens an equal-score bundle in both directions, then claims edges from the largest in-flow. Each channel may place at most its share of that in-flow times the channel capacity on the node, and the edges of a channel are kept or handed to the neighbor as in a peel.

`score-{ratio|diff}-{peel|raw}-through[-flow-asc|-flow-room-asc|-flow-room-desc|-quota|-topo]` is a separate threshold search, implemented in `src/makespan/score_through.py`. Nodes and edges are ordered from the weighted graph, not from their ids: a node's place comes from its incident weights and its neighbours' places. With `peel`, leaves are pruned to a fixed point before absorb, and a parent that drops to one neighbour is pruned on the next scan. Absorb runs only after no leaf remains, and pruning repeats if absorb creates one. `raw` absorbs once. Both then use the same score. It directs each bundle from the higher score to the lower score. An equal score points from the smaller progress fraction toward the larger one: hop distance from that free component's highest score, divided by the sum of the distances from its highest and lowest scores. An equal fraction points from the earlier canonical place to the later place. Each node is split so the arc between the two copies is the node's room under the threshold. The plain `through` name takes the largest flow through a node first. `through-flow-asc` takes the smallest flow first. `through-flow-room-asc` and `through-flow-room-desc` order nodes by the flow divided by that room, smallest first or largest first. A node with no room and some flow sorts as an infinite ratio. A tie uses the larger score, then the smaller node id. On those turns the node keeps an outgoing edge while it fits and otherwise gives it to the neighbour. `through-quota` spends the flow on each bundle as weight pointed at the lower-score end. The largest flow-through node claims first. A bundle is filled by the metric at that end, and an edge that does not fit in the remaining flow stays with the higher-score neighbour. `through-topo` does not push a flow. Nodes take turns in topological order, a node before the neighbours it points to, and then use the same keep-or-give. That pass ends the trial: there is no level pass and no leftover pass, and `through` has no `early` name.

`--algo all` runs every name in that table, including the forty-eight score-family names and the twenty-four through names. Brute force over the edge limit, and LP or ILP when Gurobi is missing, are skipped instead of aborting the rest.

Text output is one block per instance: makespan, bounds, then one line `edge_id u v head weight` per edge. `--format json` writes the same records as a list, with `tags` and a `loads` array added. `--format csv` writes one row per edge, with `instance,tags,algorithm,makespan,lower_bound,upper_bound,edge_id,u,v,head,weight,skipped`. A skipped algorithm is a single row with the reason in `skipped`. The assignment files in `Outputs` are the text format the viewer reads; they are separate from this terminal report.

## `balance-graph`

Implemented in `src/makespan/graph_balance.py`, separately from `balance.py`. It never solves an LP.

The search bisects a threshold `T` between `lower`, which moves only when a trial proves that no orientation has makespan at most `T`, and `accepted`, the smallest `T` whose trial returned an orientation with makespan at most `2T`. The measured incumbent is kept apart, so the result is at most `2 * accepted <= 2 (1 + eps) * lower`. With integral weights the thresholds are integers, and `eps=0` searches to the exact boundary.

One trial at `T`:

1. Exact reductions, repeated until none applies. None of them can lose an orientation of makespan at most `T`:
   - zero-weight edges go to their free end;
   - a node whose free edges all fit takes them all;
   - an edge that fits on one end only goes there, and an edge that fits nowhere rejects;
   - a leaf keeps the part of its bundle that relieves the parent most, by an exact 0/1 knapsack, and a parent that still overflows rejects;
   - a node holds at most one edge heavier than half its room, so edges heavy at both ends must form trees and single cycles. A denser component rejects, and a component with one cycle forces its pendant edges and pushes other heavy edges away from its nodes.
2. A free component that is a plain cycle takes the better of its two rotations.
3. `certified_push` orients the rest with node prices and helpful paths (after Gairing, Monien and Woclaw, ICALP 2005). It ends with every load at most `2T`, or with a set `S` and prices that prove `T` infeasible. That proof is re-checked in exact rational arithmetic before the trial rejects. If a step limit is reached the trial is `unknown`, and the search stops.

`PolishStep` then applies chains of up to three edge flips that lower the makespan. Trees of bundles, and plain cycles with pendant bundles, are solved exactly with `eps=0`. The solution metadata records the fences, the trial count, and which rule produced each reject (`graph_reject_reasons`).

## Compare algorithms

`-C` / `--compare` takes one or more algorithm names and prints a table for each instance. `all` is allowed and expands to the full list.

```text
python solve.py instances.txt -C greedy-light greedy-load flow-search
python solve.py instances.txt --compare greedy-light flow-search -R brute --format csv
```

Without a reference, `gap` is `makespan - best` and `ratio` is `makespan / best`, where `best` is the smallest makespan in that table. `--format csv` writes one row per algorithm, including `gap`, `ratio`, and `skipped`.

`-R` / `--reference` sets the exact solver used as the optimum. The only accepted values are `brute` and `ilp`.

```text
python solve.py instances.txt -C greedy-light flow-search lp-round -R brute
python solve.py instances.txt --compare greedy-light flow-search --reference ilp
```

Every row is then measured against that optimum: `gap = makespan - optimum` and `ratio = makespan / optimum`. The reference is solved even when it is not in the `-C` list, and it appears in the table with role `reference`. If you pass only `-R`, the algorithm from `--algo` (default `lp-round`) is compared with the reference.

If the reference cannot run, the table is still printed, the optimum is marked unavailable, and the process exits with status 1. Brute force cannot run past `--max-edges`. The ILP cannot run without Gurobi.

## Add a step

A step is a class with `run(self, ctx)`. `ctx` is a `SolveContext`: the instance, the current orientation, the locked edges, any fractional solution, and the best complete orientation seen so far. A step may

- read `ctx.free_edges()` and leave `ctx.locked` edges alone,
- call `ctx.lock(edge_id, head)` so later steps treat that edge as base load,
- store a fraction toward `u` in `ctx.fractional[edge_id]`,
- call `ctx.consider(heads)` with a complete map of edge id to head. The context keeps the map only when its evaluated makespan is strictly better.

Put the class in a module under `src/makespan/` and give the class and `run` a docstring. Register a pipeline in `src/makespan/algorithms.py`:

```python
@AlgorithmFactory.register("my-algo")
def _my_algo() -> Pipeline:
    """Short description shown to the next reader of this catalog."""
    return Pipeline("my-algo", [GreedyLightestStep(), MyStep()])
```

Also append `"my-algo"` to `ALGORITHM_ORDER` in that file if it should appear in `--algo all` and `--compare all` at a chosen position. Importing `makespan` loads `algorithms.py`, which is what registers the name.

`AlgorithmFactory.create("my-algo").solve(instance)` runs the pipeline. `create("custom", steps=[...], name="trial")` does the same without a permanent name.

A later algorithm can freeze some edges and re-solve the rest, because the LP, the ILP, brute force, the greeds, and the flow search all operate on unlocked edges only:

```python
from makespan.factory import AlgorithmFactory, Pipeline
from makespan.relaxation import RelaxationStep
from makespan.rounding import ThresholdRoundStep

AlgorithmFactory.create(
    "custom",
    name="lp-then-round",
    steps=[
        RelaxationStep(integral=False),
        ThresholdRoundStep(0.5),
    ],
)
```

`SolveContext.lock(edge_id, head)` fixes an edge. Later steps see its weight as base load and do not change it, which is how a multi-step algorithm can re-solve only the edges it has not accepted.

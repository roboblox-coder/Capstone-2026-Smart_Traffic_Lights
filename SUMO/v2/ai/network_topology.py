"""Offline corridor-adjacency builder.

Walks the road graph of a SUMO ``.net.xml`` and, for every traffic light,
finds the single upstream and single downstream *traffic-light* neighbour
(the next signalised junction reached by following edges out of / into the
intersection, skipping plain non-signalised junctions in between).

Output is a committed artifact ``ai/adjacency.json``:

    { "<tls_id>": { "upstream": "<tls_id>"|null,
                     "downstream": "<tls_id>"|null }, ... }

It is read at training / eval / live time to build each agent's fixed-width
neighbour observation block, so the mapping must be deterministic and stable.

Run from ``SUMO/v2``::

    python ai/network_topology.py --net NE_8th_St_Corridor.net.xml \
                                  --out ai/adjacency.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import sumolib  # noqa: E402

# How many junctions deep to look before giving up on finding a signalised
# neighbour. The corridor's longest TLS-free gap is well under this.
_MAX_DEPTH = 8


def _tls_junction_nodes(tls) -> set:
    """Junction node ids this TLS physically controls.

    ``net.getNode(tls_id)`` KeyErrors for joined/cluster TLS, so derive the
    nodes from the connections instead: every controlled connection turns at
    the junction = the to-node of its incoming lane's edge.
    """
    nodes = set()
    for conn in tls.getConnections():
        in_lane = conn[0]
        try:
            nodes.add(in_lane.getEdge().getToNode().getID())
        except Exception:
            pass
    return nodes


def _build_node_tls_map(net) -> dict:
    """node id -> tls id (a junction belongs to at most one TLS)."""
    node_to_tls = {}
    for tls in net.getTrafficLights():
        tid = tls.getID()
        for nid in _tls_junction_nodes(tls):
            node_to_tls[nid] = tid
    return node_to_tls


def _nearest_tls(start_edges, node_to_tls, self_tls, direction):
    """BFS over the road graph from ``start_edges`` until the first edge
    incident to a *different* TLS junction. Returns (neighbour_id|None,
    weight) where weight = how many distinct start edges reach it (used to
    pick the dominant neighbour when several exist).
    """
    hits = {}
    for seed in start_edges:
        seen_edges = set()
        q = deque([(seed, 0)])
        while q:
            edge, depth = q.popleft()
            eid = edge.getID()
            if eid in seen_edges or depth > _MAX_DEPTH:
                continue
            seen_edges.add(eid)

            if direction == "down":
                node = edge.getToNode()
            else:
                node = edge.getFromNode()
            nid = node.getID()

            owner = node_to_tls.get(nid)
            if owner is not None and owner != self_tls:
                hits[owner] = hits.get(owner, 0) + 1
                break  # stop this seed at the first signalised junction

            nxt = node.getOutgoing() if direction == "down" else node.getIncoming()
            for ne in nxt:
                if ne.getID() not in seen_edges:
                    q.append((ne, depth + 1))

    if not hits:
        return None, 0
    best = max(hits.items(), key=lambda kv: (kv[1], kv[0]))
    return best[0], best[1]


def build_adjacency(net_file: str) -> dict:
    net = sumolib.net.readNet(net_file)
    node_to_tls = _build_node_tls_map(net)

    adjacency = {}
    for tls in net.getTrafficLights():
        tid = tls.getID()
        conns = tls.getConnections()
        out_edges, in_edges = {}, {}
        for conn in conns:
            try:
                oe = conn[1].getEdge()
                out_edges[oe.getID()] = oe
                ie = conn[0].getEdge()
                in_edges[ie.getID()] = ie
            except Exception:
                pass

        downstream, _ = _nearest_tls(
            out_edges.values(), node_to_tls, tid, "down"
        )
        upstream, _ = _nearest_tls(
            in_edges.values(), node_to_tls, tid, "up"
        )
        adjacency[tid] = {"upstream": upstream, "downstream": downstream}

    return adjacency


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", default="NE_8th_St_Corridor.net.xml")
    ap.add_argument("--out", default="ai/adjacency.json")
    args = ap.parse_args()

    adjacency = build_adjacency(args.net)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(adjacency, fh, indent=2, sort_keys=True)

    n = len(adjacency)
    n_up = sum(1 for v in adjacency.values() if v["upstream"])
    n_down = sum(1 for v in adjacency.values() if v["downstream"])
    self_refs = [
        t for t, v in adjacency.items()
        if v["upstream"] == t or v["downstream"] == t
    ]
    print(f"wrote {args.out}: {n} TLS  "
          f"({n_up} with upstream, {n_down} with downstream)")
    print(f"self-references: {self_refs if self_refs else 'none'}")
    for t in sorted(adjacency):
        v = adjacency[t]
        print(f"  {t:<55s} up={v['upstream']}  down={v['downstream']}")


if __name__ == "__main__":
    main()

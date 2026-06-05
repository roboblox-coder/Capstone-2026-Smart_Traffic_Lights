"""Build a calibrated counts XML from real-world turn ratios + tag provenance.

Refactors the monolithic ``real_life_simulation.py`` into a library with
explicit provenance tracking. The legacy script silently filled the
corridor from ``randomTrips`` fringe defaults for the 6 intersections
WITHOUT real turn-ratio data; this module writes those entries
explicitly with ``estimated`` markers in a sidecar JSON and (optionally)
in XML comments alongside each ``<edge>`` row.

Why a sidecar instead of inline XML attributes: ``routeSampler.py`` (the
SUMO tool that consumes the counts XML) ignores unknown attributes
silently -- a provenance attribute would be invisible downstream. A
separate JSON survives the routeSampler pass and is queryable from
eval reports.

Differences from ``real_life_simulation.py``:

  1. Non-interactive. The legacy fallback path ``input(...)`` is
     replaced with a hard error so batch / CI runs can never silently
     ask for a junction ID.
  2. Provenance JSON sidecar with ``source: real|estimated`` per edge.
  3. Estimated defaults are explicit (``DEFAULT_VOLUME_PER_DIR``,
     traceable to OSM lane mix in ``report.md``) rather than
     hand-buried in ``randomTrips --fringe-factor 100``.
  4. The legacy script's hardcoded ``TARGET_INTS`` list is now a
     function argument so the call site can extend coverage as the
     city engagement track lands more turn ratios.

Usage (from ``SUMO/v2``):

    python -m ai.sumo_calibration.build_calibrated_routes
    # Then drive routeSampler over the produced counts file:
    python $SUMO_HOME/tools/randomTrips.py \\
        -n NE_8th_St_Corridor.net.xml \\
        -r route_pool_calibrated.rou.xml --fringe-factor 100
    python $SUMO_HOME/tools/routeSampler.py \\
        -r route_pool_calibrated.rou.xml \\
        --edgedata-files real_world_counts_calibrated.xml \\
        -o harvard_simulation_calibrated.rou.xml
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Default per-direction volume for intersections without real data. Source:
# OSM lane-mix prior for two-lane minor arterials at midday (low end of
# Bellevue corridor counts). Documented in report.md so the choice is
# auditable; bump here only when new field data warrants it.
DEFAULT_VOLUME_PER_DIR = 300

# Six intersections in the corridor for which Harvard / Bellevue gave us
# turn-ratio counts. The remaining corridor intersections fall back to
# DEFAULT_VOLUME_PER_DIR and get tagged ``source=estimated`` below.
DEFAULT_TARGET_INTS = [7, 16, 21, 27, 26, 30]


@dataclass
class IntersectionRecord:
    int_id: int
    junction_id: Optional[str]
    source: str  # "real" or "estimated"
    ns_address: str = ""
    ew_address: str = ""
    volumes_by_dir: dict = field(default_factory=dict)  # "NB" -> count
    edge_ids_by_dir: dict = field(default_factory=dict)  # "NB" -> edge_id
    notes: list = field(default_factory=list)


# ---------- pure helpers (no SUMO dependency) ----------

def normalize_name(name) -> str:
    """Strip directional / suffix tokens to match xlsx 'Way Northeast'-style
    names against OSM 'NE 8th Street' etc."""
    if not name or (isinstance(name, float) and math.isnan(name)):
        return ""
    n = str(name).lower()
    for word in ["†", "street", "st", "avenue", "ave", "way", "wy",
                 "ne", "northeast"]:
        n = n.replace(word, "")
    return n.strip()


def compass_dir(angle_deg: float) -> Optional[str]:
    a = angle_deg % 360
    if 315 <= a or a < 45:
        return "NB"
    if 45 <= a < 135:
        return "EB"
    if 135 <= a < 225:
        return "SB"
    if 225 <= a < 315:
        return "WB"
    return None


def load_excel_rows(path: str) -> list:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    out = []
    for row in df.to_dict(orient="records"):
        cleaned = {
            str(k).strip(): (str(v).strip() if pd.notna(v) else "")
            for k, v in row.items()
        }
        out.append(cleaned)
    return out


# ---------- SUMO-dependent helpers ----------

def _read_net(net_file: str):
    """Lazy sumolib import so this module loads on machines without SUMO
    (e.g. CI nodes that only need the provenance schema)."""
    import sumolib
    return sumolib.net.readNet(net_file)


def match_junction(net, ns_name: str, ew_name: str) -> Optional[str]:
    """Find a junction whose incident edges share both normalized street
    names. Deterministic: returns the first match in
    ``net.getNodes()`` order. Returns None on no match -- caller must
    decide whether to error or fall back."""
    norm_ns = normalize_name(ns_name)
    norm_ew = normalize_name(ew_name)
    if not norm_ns or not norm_ew:
        return None
    for node in net.getNodes():
        edge_names = [normalize_name(e.getName())
                      for e in (node.getIncoming() + node.getOutgoing())]
        if (any(norm_ns in en for en in edge_names) and
                any(norm_ew in en for en in edge_names)):
            return node.getID()
    return None


def incoming_edges_by_compass(net, junction_id: str) -> dict:
    """For each incoming edge of ``junction_id``, infer NB/SB/EB/WB by
    the heading of the last shape segment, and return ``{dir: edge_id}``.

    Direction collisions (two NB incomers, etc.) keep the first one in
    iteration order to stay deterministic.
    """
    node = net.getNode(junction_id)
    out: dict = {}
    for edge in node.getIncoming():
        shape = edge.getShape()
        if len(shape) < 2:
            continue
        p1, p2 = shape[-2], shape[-1]
        # match legacy script's atan2(dx, dy) convention so this stays
        # byte-compatible with the existing pipeline.
        angle = (math.degrees(math.atan2(p2[0] - p1[0],
                                         p2[1] - p1[1])) + 360) % 360
        d = compass_dir(angle)
        if d and d not in out:
            out[d] = edge.getID()
    return out


# ---------- main pipeline ----------

def _row_int_id(row: dict) -> Optional[int]:
    raw = row.get("Int#", "")
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return None


def _row_volume(row: dict, direction: str) -> int:
    """Sum the L+T+R volumes for one direction in a turn-ratio row."""
    vol = 0
    for turn in ("L", "T", "R"):
        v = row.get(f"{direction}_{turn}", "")
        if v == "":
            continue
        try:
            vol += int(float(v))
        except (ValueError, TypeError):
            pass
    return vol


def collect_records(
    turn_ratio_xlsx: str,
    address_xlsx: str,
    net_file: str,
    target_int_ids: list,
    estimate_remaining_for_signalized: bool = True,
) -> tuple:
    """Build one ``IntersectionRecord`` per processed intersection, plus
    the global list of (edge_id, count, source) rows that go into the
    counts XML.

    The "estimated" intersections covered are the corridor's signalized
    nodes that are NOT in ``target_int_ids``. Those get
    ``DEFAULT_VOLUME_PER_DIR`` per direction so the corridor isn't
    artificially starved when routeSampler runs. Non-signalized
    junctions are left alone.
    """
    addresses_by_id = {}
    for row in load_excel_rows(address_xlsx):
        iid = _row_int_id(row)
        if iid is not None:
            addresses_by_id[iid] = row

    turn_rows_by_id = {}
    for row in load_excel_rows(turn_ratio_xlsx):
        iid = _row_int_id(row)
        if iid is not None:
            turn_rows_by_id[iid] = row

    net = _read_net(net_file)
    records: list = []
    count_entries: list = []  # tuples of (edge_id, count, source)

    # --- real entries (turn ratios from xlsx) ---
    for iid in target_int_ids:
        if iid not in turn_rows_by_id:
            print(f"WARN: target intersection {iid} has no turn-ratio row "
                  f"in {turn_ratio_xlsx}; skipping.")
            continue
        addr = addresses_by_id.get(iid, {})
        ns = addr.get("NS Address", "")
        ew = addr.get("EW Address", "")
        jid = match_junction(net, ns, ew)
        rec = IntersectionRecord(int_id=iid, junction_id=jid, source="real",
                                 ns_address=ns, ew_address=ew)
        if jid is None:
            rec.notes.append("junction_match_failed")
            print(f"WARN: int#{iid} ({ns} & {ew}) -- no junction match. "
                  f"Skipping (legacy script's interactive fallback is "
                  f"disabled in batch mode).")
            records.append(rec)
            continue
        edges_by_dir = incoming_edges_by_compass(net, jid)
        rec.edge_ids_by_dir = edges_by_dir
        turn_row = turn_rows_by_id[iid]
        for d in ("NB", "SB", "EB", "WB"):
            vol = _row_volume(turn_row, d)
            rec.volumes_by_dir[d] = vol
            if vol > 0 and d in edges_by_dir:
                count_entries.append((edges_by_dir[d], vol, "real"))
        records.append(rec)

    if not estimate_remaining_for_signalized:
        return records, count_entries

    # --- estimated entries (signalized junctions not in target list) ---
    # "signalized" = node type == "traffic_light" in the SUMO net.
    real_junction_ids = {r.junction_id for r in records
                         if r.junction_id and r.source == "real"}
    for node in net.getNodes():
        if node.getType() != "traffic_light":
            continue
        if node.getID() in real_junction_ids:
            continue
        edges_by_dir = incoming_edges_by_compass(net, node.getID())
        if not edges_by_dir:
            continue
        rec = IntersectionRecord(
            int_id=-1, junction_id=node.getID(), source="estimated",
            edge_ids_by_dir=edges_by_dir,
            notes=[f"default_volume_per_dir={DEFAULT_VOLUME_PER_DIR}"],
        )
        for d, edge_id in edges_by_dir.items():
            rec.volumes_by_dir[d] = DEFAULT_VOLUME_PER_DIR
            count_entries.append((edge_id, DEFAULT_VOLUME_PER_DIR,
                                  "estimated"))
        records.append(rec)

    return records, count_entries


def write_counts_xml(path: str, count_entries: list,
                     interval_begin: int = 0,
                     interval_end: int = 3600) -> None:
    """Write the routeSampler-compatible counts XML. Estimated entries
    get an XML comment for human readers; the attribute payload stays
    identical to the legacy format so routeSampler doesn't care."""
    lines = ['<data>',
             f'    <interval id="harvard" begin="{interval_begin}" '
             f'end="{interval_end}">']
    for edge_id, vol, source in count_entries:
        suffix = "  <!-- estimated -->" if source == "estimated" else ""
        lines.append(f'        <edge id="{edge_id}" '
                     f'entered="{vol}"/>{suffix}')
    lines.append('    </interval>')
    lines.append('</data>')
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_provenance_json(path: str, records: list, count_entries: list,
                          sources: dict) -> None:
    payload = {
        "schema_version": 1,
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "source_files": sources,
        "default_volume_per_dir": DEFAULT_VOLUME_PER_DIR,
        "summary": {
            "intersections_real": sum(1 for r in records
                                      if r.source == "real"
                                      and r.junction_id),
            "intersections_estimated": sum(1 for r in records
                                           if r.source == "estimated"),
            "intersections_match_failed": sum(1 for r in records
                                              if r.source == "real"
                                              and not r.junction_id),
            "edges_real": sum(1 for _, _, s in count_entries
                              if s == "real"),
            "edges_estimated": sum(1 for _, _, s in count_entries
                                   if s == "estimated"),
        },
        "intersections": [
            {
                "int_id": r.int_id,
                "junction_id": r.junction_id,
                "source": r.source,
                "ns_address": r.ns_address,
                "ew_address": r.ew_address,
                "volumes_by_dir": r.volumes_by_dir,
                "edge_ids_by_dir": r.edge_ids_by_dir,
                "notes": r.notes,
            }
            for r in records
        ],
        "edges": [
            {"edge_id": e, "volume": v, "source": s}
            for (e, v, s) in count_entries
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n",
                          encoding="utf-8")


def build(
    turn_ratio_xlsx: str,
    address_xlsx: str,
    net_file: str,
    out_counts_xml: str,
    out_provenance_json: str,
    target_int_ids: Optional[list] = None,
) -> dict:
    """Top-level entry point. Returns the provenance summary so callers
    can log it / fail a CI step on too many estimated edges."""
    target_int_ids = target_int_ids or list(DEFAULT_TARGET_INTS)
    records, count_entries = collect_records(
        turn_ratio_xlsx=turn_ratio_xlsx,
        address_xlsx=address_xlsx,
        net_file=net_file,
        target_int_ids=target_int_ids,
    )
    write_counts_xml(out_counts_xml, count_entries)
    sources = {
        "turn_ratios": os.path.abspath(turn_ratio_xlsx),
        "intersection_addresses": os.path.abspath(address_xlsx),
        "network": os.path.abspath(net_file),
    }
    write_provenance_json(out_provenance_json, records, count_entries,
                          sources)
    payload = json.loads(Path(out_provenance_json).read_text())
    return payload["summary"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--net", default="NE_8th_St_Corridor.net.xml")
    p.add_argument("--turn-ratios",
                   default="Real_intersection_data/Turn_ratio.xlsx")
    p.add_argument("--addresses",
                   default="Real_intersection_data/"
                           "Intersection_address.xlsx")
    p.add_argument("--out-counts",
                   default="real_world_counts_calibrated.xml")
    p.add_argument("--out-provenance",
                   default="ai/sumo_calibration/provenance.json")
    p.add_argument("--target-ints", nargs="*", type=int,
                   default=DEFAULT_TARGET_INTS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = build(
        turn_ratio_xlsx=args.turn_ratios,
        address_xlsx=args.addresses,
        net_file=args.net,
        out_counts_xml=args.out_counts,
        out_provenance_json=args.out_provenance,
        target_int_ids=args.target_ints,
    )
    print(f"\n=== build_calibrated_routes summary ===")
    print(f"  intersections_real           : "
          f"{summary['intersections_real']}")
    print(f"  intersections_estimated      : "
          f"{summary['intersections_estimated']}")
    print(f"  intersections_match_failed   : "
          f"{summary['intersections_match_failed']}")
    print(f"  edges_real                   : {summary['edges_real']}")
    print(f"  edges_estimated              : "
          f"{summary['edges_estimated']}")
    print(f"\nWrote {args.out_counts}")
    print(f"Wrote {args.out_provenance}")
    print(f"\nNext: drive routeSampler.py over the counts file -- see the "
          f"module docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Calibration provenance and what's still missing

This document is the honest receipt for `sim_calibrated.sumocfg`: what
data is real, what is estimated, and what the city-engagement track has
to deliver before any "calibrated SUMO" claim earns its scare quotes
back.

## What's real

**Turn-ratio counts (6 of 12 corridor intersections).** Source:
`Real_intersection_data/Turn_ratio.xlsx` (Harvard / Bellevue dataset).
Intersections covered: 7, 16, 21, 26, 27, 30 (the `DEFAULT_TARGET_INTS`
in `build_calibrated_routes.py`). Per-direction L/T/R volumes for a
single one-hour window.

**Intersection addresses (matched to OSM junctions).** Source:
`Real_intersection_data/Intersection_address.xlsx`. The
auto-matcher in `build_calibrated_routes.match_junction` aligns "NS"
and "EW" address strings to junctions in `NE_8th_St_Corridor.net.xml`
by normalized street names + compass headings of incident edges.

## What's estimated

**Turn ratios for the other 6 signalized corridor intersections.**
`build_calibrated_routes.collect_records(estimate_remaining_for_signalized=True)`
fills the gap by emitting `DEFAULT_VOLUME_PER_DIR = 300 veh/h/dir` per
direction. The number is an OSM-style lane-mix prior for a two-lane
minor arterial at midday in this part of Bellevue — within an order of
magnitude of the real entries (the real ones span 263..1792 veh/h on
the captured edges) but not from a measurement. Every estimated entry
is tagged `source: "estimated"` in the sidecar provenance JSON.

**Saturation flows.** Zero empirical measurements. The current Krauss
parameters in `vTypes_calibrated.add.xml` are HCM 6th Edition / SUMO
default midpoints: `tau=1.0, sigma=0.5, decel=4.5`. These yield a
nominal saturation flow of roughly 1800–1900 veh/h/lane, the HCM
midpoint for urban arterials. Without ground truth, we have no way to
say whether the corridor's actual saturation flow is closer to the
~1700 floor seen at high-truck Bellevue intersections or the ~2000
ceiling literature reports for ideal conditions.

**Car-following dynamics.** Same caveat: literature defaults, not a
fit. `calibrate_carfollow.py` is set up so that swapping in
empirical data is one CLI flag.

**Vehicle mix.** All vehicles are `passenger`. No trucks, no buses,
no two-wheelers. Real corridor mix is unknown — Bellevue truck % is
plausibly 5–15% on NE 8th St but no per-corridor count is in the
repo.

**Demand profile.** Single 1-hour `interval id="harvard" begin="0"
end="3600"` block. No AM peak / midday / PM peak separation, no
weekday-vs-weekend stratification, no seasonal variation. The V2
acceptance gate's `--vary-demand` flag (per `PLAN_V2.md` §1.2) builds
random multipliers around this single point — it doesn't replace it.

**Pedestrian and bicycle counts.** Not modeled. No bike lanes in the
.net.xml; no ped phases beyond what the OSM-derived signal programs
inherit. Affects realized saturation flow (ped calls extend cycle
length) but uncaptured.

**Origin-destination matrix.** Inferred by `routeSampler.py` from the
edge counts. Not measured. Multiple OD matrices fit any given edge
count set — the one routeSampler picks is whichever its objective
function lands on, and the V2 training will learn against that
particular plausible-but-unmeasured OD.

## City-engagement long pole (R3 from PLAN_V2.md §1.2)

The "calibrated" label only earns its scare quotes back when these
land. Listed in priority order.

| Priority | Item                                                       | Source to ask                              |
|----------|------------------------------------------------------------|--------------------------------------------|
| P0       | 24h directional counts at the 6 uncovered intersections    | City of Bellevue Traffic Engineering       |
| P0       | Current signal timing plans (.tim / Synchro PDFs)          | City of Bellevue Traffic Engineering       |
| P1       | Saturation-flow surveys (1 per cycle × 4 cycles × 2 sites) | Field survey; alternatively WSDOT TSMO     |
| P1       | Heavy-vehicle fractions                                    | WSDOT permanent traffic recorders          |
| P2       | AM / midday / PM peak count separation                     | WSDOT short-duration counts                |
| P2       | Pedestrian + bicycle counts (peak hours)                   | City of Bellevue Transportation Department |
| P3       | Origin-destination from a license-plate or Bluetooth study | Existing Bellevue B-Line ITS dataset       |

Until P0/P1 land, **Phase 1's acceptance gate stays pinned to relative
metrics** (`V2 beats V1 on the same calibrated env`, paired-seed 95% CI).
The absolute ceiling (−25% wait vs native) is reportable but always
with the "pending validation against city counts" caveat.

## How to use this

```bash
# Generate the counts XML + provenance JSON
python -m ai.sumo_calibration.build_calibrated_routes

# Generate the calibrated vTypes file (literature defaults today)
python -m ai.sumo_calibration.calibrate_carfollow

# Drive SUMO's routeSampler over the counts -> calibrated route file
python "$SUMO_HOME/tools/randomTrips.py" \
    -n NE_8th_St_Corridor.net.xml \
    -r route_pool_calibrated.rou.xml --fringe-factor 100
python "$SUMO_HOME/tools/routeSampler.py" \
    -r route_pool_calibrated.rou.xml \
    --edgedata-files real_world_counts_calibrated.xml \
    -o harvard_simulation_calibrated.rou.xml

# Smoke-test that V1 still wins on the calibrated env
# (Phase 1.1 sanity gate: V1 beats native by >=5pp on wait)
python ai/regression_test.py --sumo-cfg sim_calibrated.sumocfg
```

If `regression_test.py --sumo-cfg sim_calibrated.sumocfg` doesn't run
green, do not advance to Phase 1.2 — either the calibration eroded
V1's win (in which case the parameter choices need re-examination) or
V1 was over-fitting to the uncalibrated env (in which case V2 was
about to inherit the same bias, and catching it now is the whole point
of the calibration step).

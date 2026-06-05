"""Calibration pipeline: real-world turn ratios + Krauss fit -> sim_calibrated.

Three modules, three outputs:

  build_calibrated_routes  -> real_world_counts_calibrated.xml + provenance.json
  calibrate_carfollow      -> vTypes_calibrated.add.xml
  (driver / orchestration  -> sim_calibrated.sumocfg)

Provenance is the load-bearing concept: with 6/12 corridor intersections
having real turn-ratio data and zero having empirical saturation flow,
the calibrated sim is partially fictional. Every estimated input is
tagged so downstream eval reports can carry the caveat honestly.
"""

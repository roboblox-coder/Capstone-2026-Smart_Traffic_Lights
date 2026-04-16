"""
SUMO Simulation Launcher
========================
Single entry-point to select and run different simulation modes.
"""

import os
import sys
import subprocess
import xml.etree.ElementTree as ET

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMOCFG    = os.path.join(SCRIPT_DIR, "sim.sumocfg")

ROUTE_HARVARD   = "harvard_simulation.rou.xml"
ROUTE_SYNTHETIC = "synthetic.rou.xml"

# ── Helpers ──────────────────────────────────────────────────────────

def set_route_file(route_filename):
    """Rewrite sim.sumocfg to use the given route file."""
    tree = ET.parse(SUMOCFG)
    root = tree.getroot()
    route_elem = root.find(".//input/route-files")
    if route_elem is not None:
        route_elem.set("value", route_filename)
    tree.write(SUMOCFG, xml_declaration=True, encoding="UTF-8")
    print(f"   ✅ sim.sumocfg → route-files set to '{route_filename}'")


def run_script(script_name):
    """Run another Python script in the v2 directory."""
    script = os.path.join(SCRIPT_DIR, script_name)
    subprocess.run([sys.executable, script], cwd=SCRIPT_DIR)


def launch_sumo_gui(route_file=None):
    """Open SUMO-GUI with sim.sumocfg, optionally overriding the route file."""
    cmd = ["sumo-gui", "-c", SUMOCFG, "--tls.actuated.jam-threshold", "30"]
    if route_file:
        cmd += ["--route-files", route_file]
    subprocess.run(cmd, cwd=SCRIPT_DIR)
    if route_file:
        print(f"   ✅ Launched SUMO-GUI with route file override: '{route_file}'")

# ── Menu ─────────────────────────────────────────────────────────────

MENU = """
═══════════════════════════════════════════
       🚦  SUMO Simulation Launcher
═══════════════════════════════════════════

  1)  Harvard Real-Life Simulation
      → Generate routes from Harvard data, then open SUMO-GUI

  2)  Synthetic Traffic (GUI)
      → Run with synthetic routes in SUMO-GUI

  3)  Headless Data Extraction
      → Run headless SUMO, export FCD/queue → CSV

  4)  TraCI Simulation
       → Run via TraCI for programmatic control

  5)  WebSocket Simulation
       → TraCI + live WebSocket feed for Three.js frontend

  0)  Exit

═══════════════════════════════════════════"""


def main():
    while True:
        print(MENU)
        choice = input("  Select an option [0-5]: ").strip()

        if choice == "0":
            print("\n  👋 Goodbye!\n")
            break

        elif choice == "1":
            print("\n── Step 1/2: Generating routes from Harvard data ──")
            run_script("real_life_simulation.py")

            print("\n── Step 2/2: Launching SUMO-GUI ──")
            launch_sumo_gui(ROUTE_HARVARD)   # <-- pass the route file directly

        elif choice == "2":
            print("\n── Launching SUMO-GUI with synthetic routes ──")
            launch_sumo_gui(ROUTE_SYNTHETIC)   # <-- pass the route file directly

        elif choice == "3":
            print("\n── Running headless simulation + CSV export ──")
            # Ask which route file to use
            r = input("  Route file? [h]arvard (default) / [s]ynthetic: ").strip().lower()
            if r.startswith("s"):
                set_route_file(ROUTE_SYNTHETIC)
            else:
                set_route_file(ROUTE_HARVARD)
            run_script("run_auto_sim.py")

        elif choice == "4":
            print("\n── Running TraCI simulation ──")
            r = input("  Route file? [h]arvard (default) / [s]ynthetic: ").strip().lower()
            if r.startswith("s"):
                set_route_file(ROUTE_SYNTHETIC)
            else:
                set_route_file(ROUTE_HARVARD)
            run_script("run_traci_sim.py")

        elif choice == "5":
            print("\n── Running WebSocket Simulation ──")
            r = input("  Route file? [h]arvard (default) / [s]ynthetic: ").strip().lower()
            if r.startswith("s"):
                set_route_file(ROUTE_SYNTHETIC)
            else:
                set_route_file(ROUTE_HARVARD)
            run_script("run_websocket_sim.py")

        else:
            print("\n  ⚠️  Invalid choice, try again.")


if __name__ == "__main__":
    main()

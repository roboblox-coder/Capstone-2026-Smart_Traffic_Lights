import os
import sys
import subprocess
import csv
import math
import pandas as pd

# --- CONFIGURATION ---
NET_FILE = "NE_8th_St_Corridor.net.xml"
COUNTS_FILE = "Real_intersection_data/Turn_ratio.xlsx"
ADDRESS_FILE = "Real_intersection_data/Intersection_address.xlsx"
OUTPUT_COUNTS = "real_world_counts.xml"
OUTPUT_ROUTES = "harvard_simulation.rou.xml"

# Target Intersections in Bellevue
TARGET_INTS = [7, 16, 21, 27, 26, 30]

# Auto-detect SUMO Tools
SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Users\Duke3\AppData\Local\Programs\Python\Python313\Lib\site-packages\sumo")
RANDOM_TRIPS = os.path.join(SUMO_HOME, "tools", "randomTrips.py")
ROUTE_SAMPLER = os.path.join(SUMO_HOME, "tools", "routeSampler.py")

try:
    import sumolib
except ImportError:
    sys.exit("❌ Error: 'sumolib' not found. Run: pip install sumolib")

# --- HELPER FUNCTIONS ---
def normalize_name(name):
    """Aggressively simplifies names to match 'Way Northeast' style maps"""
    if not name or pd.isna(name): return ""
    n = str(name).lower()
    # Remove symbols and common suffixes to find the core street name
    for word in ["†", "street", "st", "avenue", "ave", "way", "wy", "ne", "northeast"]:
        n = n.replace(word, "")
    return n.strip()

def get_compass_dir(angle):
    if 315 <= angle or angle < 45: return "NB" 
    elif 45 <= angle < 135: return "EB"
    elif 135 <= angle < 225: return "SB"
    elif 225 <= angle < 315: return "WB"
    return None

def load_excel_data(file_path):
    """Loads XLSX and cleans headers/values to avoid whitespace errors"""
    try:
        df = pd.read_excel(file_path)
        df.columns = [str(col).strip() for col in df.columns]
        data = df.to_dict(orient='records')
        cleaned_data = []
        for row in data:
            cleaned_row = {str(k).strip(): (str(v).strip() if pd.notna(v) else "") for k, v in row.items()}
            cleaned_data.append(cleaned_row)
        return cleaned_data
    except Exception as e:
        sys.exit(f"❌ Error reading Excel file {file_path}: {e}")

# --- 1. LOAD DATA ---
print(f"🌍 Loading Network: {NET_FILE}...")
net = sumolib.net.readNet(NET_FILE)

print("📊 Loading Harvard Data from Excel...")
address_data = load_excel_data(ADDRESS_FILE)
addresses = {}
for row in address_data:
    if 'Int#' in row and row['Int#']:
        try:
            int_id = int(float(row['Int#']))
            addresses[int_id] = row
        except (ValueError, TypeError): pass

traffic_data = []
count_data = load_excel_data(COUNTS_FILE)
for row in count_data:
    if 'Int#' in row and row['Int#']:
        try:
            int_id = int(float(row['Int#']))
            if int_id in TARGET_INTS:
                if int_id in addresses:
                    row.update(addresses[int_id])
                    traffic_data.append(row)
        except (ValueError, TypeError): pass

print(f"✅ Loaded {len(traffic_data)} target intersections.")

# --- 2. MAP TO SUMO ---
xml_edges = []
print("\n--- 🔍 AUTO-MATCHING JUNCTIONS ---")

for row in traffic_data:
    int_id = int(float(row['Int#']))
    ns_name = row.get('NS Address', 'Unknown')
    ew_name = row.get('EW Address', 'Unknown')
    print(f"\n📍 Processing Int #{int_id}: {ns_name} & {ew_name}")

    found_junction = None
    candidates = []
    
    norm_ns = normalize_name(ns_name)
    norm_ew = normalize_name(ew_name)

    for node in net.getNodes():
        # Check all edges connected to this junction
        edges = node.getIncoming() + node.getOutgoing()
        edge_names = [normalize_name(e.getName()) for e in edges]
        
        # Match if both normalized street names are found in the junction's edges
        if any(norm_ns in en for en in edge_names) and any(norm_ew in en for en in edge_names):
            candidates.append(node)

    if len(candidates) >= 1:
        found_junction = candidates[0]
        print(f"   🔹 Matched Node: {found_junction.getID()}")
    else:
        print(f"   ❌ Auto-match failed for {ns_name} & {ew_name}")
        jid = input(f"   👉 Enter Junction ID from Netedit (Inspect Mode): ").strip()
        if jid: 
            try: found_junction = net.getNode(jid)
            except: print("      Invalid ID.")

    if found_junction:
        dir_map = {}
        for edge in found_junction.getIncoming():
            shape = edge.getShape()
            if len(shape) < 2: continue
            p1, p2 = shape[-2], shape[-1]
            angle = (math.degrees(math.atan2(p2[0]-p1[0], p2[1]-p1[1])) + 360) % 360
            d = get_compass_dir(angle)
            if d: dir_map[d] = edge.getID()

        for d in ["NB", "SB", "EB", "WB"]:
            vol = 0
            for turn in ["L", "T", "R"]:
                key = f"{d}_{turn}"
                if key in row and row[key]:
                    try: vol += int(float(row[key]))
                    except (ValueError, TypeError): pass
            
            if vol > 0 and d in dir_map:
                xml_edges.append(f'        <edge id="{dir_map[d]}" entered="{vol}"/>')

# --- 3. WRITE XML & RUN TOOLS ---
with open(OUTPUT_COUNTS, "w") as f:
    f.write('<data>\n    <interval id="harvard" begin="0" end="3600">\n')
    f.write("\n".join(xml_edges))
    f.write('\n    </interval>\n</data>')

print("\n🚀 Generating Routes...")
try:
    subprocess.run(["python", RANDOM_TRIPS, "-n", NET_FILE, "-r", "route_pool.rou.xml", "--fringe-factor", "100"], check=True)
    subprocess.run(["python", ROUTE_SAMPLER, "-r", "route_pool.rou.xml", "--edgedata-files", OUTPUT_COUNTS, "-o", OUTPUT_ROUTES], check=True)
    print("\n🎉 DONE! Simulation routes generated.")
except subprocess.CalledProcessError as e:
    print(f"❌ Error running SUMO tools: {e}")
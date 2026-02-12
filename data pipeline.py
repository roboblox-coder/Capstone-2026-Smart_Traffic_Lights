#Cory Maccini
import pandas as pd
import numpy as np
import pygame
import random
import sys


#Part 1: Data Ingestion (Traffic Generator)
#Logic derived from sources containing real-world traffic data and signal timing principles, including:
#1. Bellevue Traffic Volume Data (2023) - City of Bellevue Transportation Department
#2. Signal Timing Guidelines - Federal Highway Administration (FHWA)            
def generate_traffic_data():
    """Generates synthetic traffic data based on Bellevue intersections."""
    np.random.seed(42)
    NUM_SAMPLES = 1000  

    #[cite_start]Intersection profiles [cite: 1, 2, 3, 4, 5]
    intersections = {
        'NE 8th and 148th': {'type': 'Major Arterial', 'cycle_range': (110, 140), 'adt': 55000, 'lanes': 4},
        'NE 8th and 140th': {'type': 'Major Arterial', 'cycle_range': (100, 140), 'adt': 42000, 'lanes': 3},
        'NE 4th and Bellevue Way': {'type': 'Urban Principal', 'cycle_range': (90, 120), 'adt': 35000, 'lanes': 3},
        '140th Ave NE and NE 24th': {'type': 'Neighborhood Collector', 'cycle_range': (60, 90), 'adt': 18000, 'lanes': 2}
    }

    data = []

    for _ in range(NUM_SAMPLES):
        loc_name, profile = list(intersections.items())[np.random.randint(0, len(intersections))]
        hour = np.random.randint(6, 22)
        is_peak = (7 <= hour <= 9) or (16 <= hour <= 18)
        
        #Volume Calculation
        vol_factor = np.random.uniform(0.9, 1.2) if is_peak else np.random.uniform(0.4, 0.7)
        hourly_vol = (profile['adt'] * 0.1) * vol_factor 
        arrival_rate_vph = hourly_vol / profile['lanes'] 
        lambda_arrival = arrival_rate_vph / 3600.0 
        saturation_flow = 1900 / 3600.0                   
        
        #Signal Timing
        cycle_length = np.random.randint(*profile['cycle_range'])
        green_ratio = np.random.uniform(0.5, 0.75)
        
        #Using Webster's Delay Calculation
        x = lambda_arrival / (saturation_flow * green_ratio)
        x = min(x, 0.98) 
        
        term1_numerator = (cycle_length * (1 - green_ratio)**2)
        term1_denominator = (2 * (1 - (lambda_arrival / saturation_flow)))
        term1 = (term1_numerator / term1_denominator) if term1_denominator > 0 else (cycle_length * 0.5)
        
        term2 = (x**2) / (2 * lambda_arrival * (1 - x)) if x > 0 else 0
        total_delay = term1 + term2
        
        actual_wait = total_delay * np.random.uniform(0.9, 1.3)

        data.append({
            'Intersection ID': loc_name,
            'Hour of Day': hour,
            'Is Peak Hour': is_peak,
            'Vehicles Per Hour': int(hourly_vol),
            'Est Wait Time Sec': round(actual_wait, 1)
        })

    return pd.DataFrame(data)


#Part 2: Data Transmition

def get_simulation_config(df, intersection_name, target_hour=None):
    subset = df[df['Intersection ID'] == intersection_name]
    
    if target_hour is None:
        scenario = subset[subset['Is Peak Hour'] == True].iloc[0]
    else:
        scenario = subset[subset['Hour of Day'] == target_hour].iloc[0]

    #Convert Vehicles Per Hour to Millisecond Interval
    vph = scenario['Vehicles Per Hour']
    spawn_interval_ms = (3600 / vph) * 1000 if vph > 0 else 5000
    
    #Convert Wait Time to Milliseconds
    wait_time_ms = scenario['Est Wait Time Sec'] * 1000

    print(f"--- SIMULATION CONFIGURATION ---")
    print(f"Intersection: {scenario['Intersection ID']}")
    print(f"Real Volume: {vph} Vehicles/Hour")
    print(f"Real Est Wait: {scenario['Est Wait Time Sec']} seconds")
    print("--------------------------------\n")
    
    return spawn_interval_ms, wait_time_ms


#Part 3: Visual Simulation (With Debugging added in(Although this was Nikolaus' part that I decided to mesh in))

def run_visual_simulation(spawn_interval, base_wait_time):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("AI Traffic Simulation Pipeline (Debug Mode)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    #Assets & Setup
    player_image = pygame.Surface((50, 50))
    player_image.fill((255, 0, 0)) 
    
    #Barriers
    B_WIDTH, B_HEIGHT = 225, 140
    barriers = [
        pygame.Rect(0, 0, B_WIDTH, B_HEIGHT),
        pygame.Rect(800 - B_WIDTH, 0, B_WIDTH, B_HEIGHT),
        pygame.Rect(0, 600 - B_HEIGHT, B_WIDTH, B_HEIGHT),
        pygame.Rect(800 - B_WIDTH, 600 - B_HEIGHT, B_WIDTH, B_HEIGHT)
    ]
    
    #Road Lines
    white_lines = [
        ((B_WIDTH, B_HEIGHT), (400, B_HEIGHT)), 
        ((800 - B_WIDTH, B_HEIGHT), (800 - B_WIDTH, 300)),
        ((800 - B_WIDTH, 600 - B_HEIGHT), (400, 600 - B_HEIGHT)),
        ((B_WIDTH, 600 - B_HEIGHT), (B_WIDTH, 300))
    ]

    #Spawn Point Definitions
    spawn_points = [
        {"pos": [400, 600], "vel": [0, -1], "target": 450, "axis": 1},
        {"pos": [400, 0],   "vel": [0, 1],  "target": 150, "axis": 1},
        {"pos": [800, 300], "vel": [-1, 0], "target": 600, "axis": 0},
        {"pos": [0, 300],   "vel": [1, 0],  "target": 200, "axis": 0}
    ]

    cars = []
    CAR_SPEED = 5
    last_spawn_time = pygame.time.get_ticks()
    
    running = True
    print("\nStarting Simulation Loop...")
    
    while running:
        current_time = pygame.time.get_ticks()
        
        #     DEBUGGING OUTPUT    
        #This will print the active car count and time every frame, overwriting the line
        sys.stdout.write(f"\rDEBUG: Time {current_time}ms | Active Cars: {len(cars)} | FPS: {int(clock.get_fps())}")
        sys.stdout.flush()
        # _______________________

        #Automated Spawning Logic
        if current_time - last_spawn_time > spawn_interval:
            sp_data = random.choice(spawn_points)
            new_rect = pygame.Rect(sp_data["pos"][0]-25, sp_data["pos"][1]-25, 50, 50)
            
            individual_wait = base_wait_time * random.uniform(0.9, 1.1)
            
            cars.append({
                "rect": new_rect,
                "vel": sp_data["vel"],
                "target": sp_data["target"],
                "axis": sp_data["axis"],
                "state": "moving",
                "wait_total": individual_wait,
                "wait_start": 0
            })
            last_spawn_time = current_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        #Car Movement & State Logic
        for p in cars[:]:
            if p["state"] == "moving":
                p["rect"].x += p["vel"][0] * CAR_SPEED
                p["rect"].y += p["vel"][1] * CAR_SPEED
                
                pos = p["rect"].x if p["axis"] == 0 else p["rect"].y
                if (p["vel"][0] + p["vel"][1] > 0 and pos >= p["target"]) or \
                   (p["vel"][0] + p["vel"][1] < 0 and pos <= p["target"]):
                    p["state"] = "waiting"
                    p["wait_start"] = current_time

            elif p["state"] == "waiting":
                if current_time - p["wait_start"] >= p["wait_total"]:
                    p["state"] = "exiting"

            elif p["state"] == "exiting":
                p["rect"].x += p["vel"][0] * CAR_SPEED
                p["rect"].y += p["vel"][1] * CAR_SPEED
                if not screen.get_rect().colliderect(p["rect"]):
                    cars.remove(p)

        #Rendering
        screen.fill((50, 50, 50)) 
        
        for wall in barriers:
            pygame.draw.rect(screen, (34, 139, 34), wall) 
            
        for start, end in white_lines:
            pygame.draw.line(screen, (255, 255, 255), start, end, 10)

        for c in cars:
            color = (0, 0, 255) 
            if c["state"] == "waiting": color = (255, 0, 0)
            if c["state"] == "exiting": color = (0, 255, 0)
            pygame.draw.rect(screen, color, c["rect"])

        #On-Screen Stats
        debug_text = font.render(f"Cars: {len(cars)}", True, (255, 255, 255))
        screen.blit(debug_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("\nSimulation Closed.")

#Main Execution

if __name__ == "__main__":
    traffic_df = generate_traffic_data()
    spawn_ms, wait_ms = get_simulation_config(traffic_df, 'NE 8th and 148th')
    run_visual_simulation(spawn_ms, wait_ms) 
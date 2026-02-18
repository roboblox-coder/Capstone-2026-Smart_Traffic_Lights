import traci
import pygame
import sys

# 1. Initialize PyGame
pygame.init()
screen = pygame.display.set_mode((1200, 400)) # Wide for a linear corridor
clock = pygame.time.Clock()
SCALE = 2.0  # Pixels per meter

# 2. Start SUMO in the background (no gui)
sumo_cmd = ["sumo", "-c", "your_config.sumocfg"]
traci.start(sumo_cmd)

running = True
while running:
    # Handle PyGame Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 3. Advance SUMO and Get Data
    traci.simulationStep()
    vehicles = traci.vehicle.getIDList()
    
    # 4. Rendering
    screen.fill((50, 50, 50)) # Asphalt color
    
    for veh_id in vehicles:
        x, y = traci.vehicle.getPosition(veh_id)
        # Convert to PyGame coordinates
        px = int(x * SCALE)
        py = int(200 - (y * SCALE)) # Centered vertically
        
        # Draw vehicle as a simple rectangle or circle
        pygame.draw.rect(screen, (255, 0, 0), (px, py, 10, 5))

    # 5. Draw Intersections (Static or Dynamic)
    # You can get traffic light states via traci.trafficlight.getPhase()
    
    pygame.display.flip()
    clock.tick(60) # Sync PyGame to 60fps

traci.close()
pygame.quit()
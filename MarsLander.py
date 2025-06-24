import matplotlib.pyplot as plt
import numpy as np

from World import World
from Bot import Bot, BotPrograms

GRAVITY = 3.711  # Mars gravity in m/s^2
SURFACE = [(0,100),(1000,500),(1500,1500),(3000,1000),(4000,150),(5500,150),(6999,800)]
WORLD_SIZE = (7000,3000)
INITIAL_BOT_POSITION = (2500,2699)
INITIAL_BOT_VELOCITY = (0,0)
INITIAL_BOT_ANGLE = 0
INITIAL_BOT_POWER = 0
INITIAL_BOT_FUEL = 5501

REDUCTION = 200
DISPLAY_RATE = 100

if __name__ == "__main__":
    # Create a world object with Mars gravity and surface
    mars = World(WORLD_SIZE, GRAVITY, SURFACE)

    plt.ion()
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_xlim(0, WORLD_SIZE[0])
    ax.set_ylim(0, WORLD_SIZE[1])

    # Plot the surface
    surface = mars.get_surface()
    #ax.plot(surface[:, 0], surface[:, 1], label="Surface", color="orange")

    lander = Bot(position=INITIAL_BOT_POSITION, velocity=INITIAL_BOT_VELOCITY, 
                 angle=INITIAL_BOT_ANGLE, power=INITIAL_BOT_POWER, fuel=INITIAL_BOT_FUEL)
    mars.place_bot(lander)
    trace = np.array(lander.position)
    bot_programs = BotPrograms.generate_random_programs(lander)
    step = 0
    
    while (True):
        step += 1
        emulation_results = mars.emulate_bot_programs(bot_programs)
        bot_programs.calculate_scores(emulation_results)
        best = np.argmax(bot_programs._scores)
        trajectories = emulation_results.bot_position
        
        if (step % REDUCTION == 0):
            # Determine actions to execute
            mars.execute(bot_programs.angle_change[best, 0], bot_programs.power_change[best, 0])
            if emulation_results.termination_step[best] == 0:
                break
            bot_programs.progress(lander)
            trace = np.vstack((trace, lander.position))
        
        if (step % DISPLAY_RATE == 1):
            ax.clear()
            ax.set_xlim(0, WORLD_SIZE[0])
            ax.set_ylim(0, WORLD_SIZE[1])
            ax.plot(surface[:, 0], surface[:, 1], label="Surface", color="orange")
            for trajectory in trajectories:
                ax.plot(trajectory[:,0],trajectory[:,1],color='red')
            if bot_programs._scores[best] >= 100:
                ax.plot(trajectories[best,:,0], trajectories[best,:,1], color = 'green')
            ax.plot(trace[...,0], trace[...,1], color = 'blue')
            plt.pause(0.1)
        
        bot_programs = bot_programs.evolve()
    plt.ioff()
    print(bot_programs._scores[best])
    print(emulation_results.bot_velocity[best, emulation_results.termination_step[best]])
    print(emulation_results.bot_fuel[best, emulation_results.termination_step[best]])
    plt.show()
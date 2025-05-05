import matplotlib.pyplot as plt
import numpy as np

from World import World
from Bot import Bot, BotState

MARS_GRAVITY = 3.711  # Mars gravity in m/s^2
SURFACE = np.array([[0,100],[1000,500],[1500,1500],[3000,1000],[4000,150],[5500,150],[6999,800]])

if __name__ == "__main__":
    # Create a world object with Mars gravity and surface
    mars = World(MARS_GRAVITY, SURFACE)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_ylim(0, 3000)

    # Plot the surface
    surface = mars.get_surface()
    ax.plot(surface[:, 0], surface[:, 1], label="Surface", color="orange")

    lander = Bot(position=(2500, 2699), velocity=(0, 0), angle=0, power=0, fuel=5501)
    mars.place_bot(lander)

    plt.show()
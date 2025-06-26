import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from bot import Bot, BotPrograms
from config import DisplayParams as Display
from config import EnvParams
from config import GeneticAlgorithmParams as GA
from world import World


class MarsLanderEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.MultiDiscrete(
            [
                EnvParams.MAX_ANGLE_CHANGE - EnvParams.MIN_ANGLE_CHANGE + 1,
                EnvParams.MAX_POWER_CHANGE - EnvParams.MIN_POWER_CHANGE + 1,
            ],
            start=[EnvParams.MIN_ANGLE_CHANGE, EnvParams.MIN_POWER_CHANGE],
        )
        self.observation_space = gym.spaces.Dict({})


if __name__ == "__main__":
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, EnvParams.WORLD_SIZE[0])
    ax.set_ylim(0, EnvParams.WORLD_SIZE[1])

    lander = Bot(
        position=EnvParams.INITIAL_BOT_POSITION,
        velocity=EnvParams.INITIAL_BOT_VELOCITY,
        angle=EnvParams.INITIAL_BOT_ANGLE,
        power=EnvParams.INITIAL_BOT_POWER,
        fuel=EnvParams.INITIAL_BOT_FUEL,
    )
    # Create a world object with Mars gravity and surface
    mars = World(EnvParams.WORLD_SIZE, EnvParams.GRAVITY, EnvParams.SURFACE, lander)

    # Plot the surface
    surface = mars.get_surface()
    # ax.plot(surface[:, 0], surface[:, 1], label="Surface", color="orange")

    trace = np.array(lander.position)
    bot_programs = BotPrograms.generate_random_programs(lander)
    step = 0

    while True:
        step += 1
        emulation_results = mars.emulate_bot_programs(bot_programs)
        bot_programs.calculate_scores(emulation_results)
        best = np.argmax(bot_programs._scores)
        trajectories = emulation_results.bot_position

        if step % GA.REDUCTION == 0:
            # Determine actions to execute
            mars.execute(
                bot_programs.angle_change[best, 0], bot_programs.power_change[best, 0]
            )
            if emulation_results.termination_step[best] == 0:
                break
            bot_programs.progress(lander)
            trace = np.vstack((trace, lander.position))

        if step % Display.DISPLAY_RATE == 1:
            ax.clear()
            ax.set_xlim(0, EnvParams.WORLD_SIZE[0])
            ax.set_ylim(0, EnvParams.WORLD_SIZE[1])
            ax.plot(surface[:, 0], surface[:, 1], label="Surface", color="orange")
            for trajectory in trajectories:
                ax.plot(trajectory[:, 0], trajectory[:, 1], color="red")
            if bot_programs._scores[best] >= 100:
                ax.plot(
                    trajectories[best, :, 0], trajectories[best, :, 1], color="green"
                )
            ax.plot(trace[..., 0], trace[..., 1], color="blue")
            plt.pause(0.1)

        bot_programs = bot_programs.evolve()
    plt.ioff()
    print(bot_programs._scores[best])
    print(
        emulation_results.bot_velocity[best, emulation_results.termination_step[best]]
    )
    print(emulation_results.bot_fuel[best, emulation_results.termination_step[best]])
    plt.show()

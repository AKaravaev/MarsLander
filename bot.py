from enum import Enum

import numpy as np

from config import EnvParams as Env
from config import GeneticAlgorithmParams as GA


class BotStatus(Enum):
    PENDING = 0
    OUT_OF_BOUNDS = 1
    SURFACE_COLLISION = 2
    LANDING = 3
    # LANDING_SUCCESS = 4


class Bot:

    def __init__(self, position, velocity, angle, power, fuel):
        self.position = position
        self.velocity = velocity
        self.angle = angle
        self.power = power
        self.fuel = fuel


class BotPrograms:
    def __init__(self, bot, programs):
        self._programs = programs
        self._bot = bot
        self.calculate_bot_actions(self._bot)
        self._scores = np.repeat(0, self.num)

    @property
    def num(self):
        return self._programs.shape[0]

    @property
    def len(self):
        return self._programs.shape[1]

    @property
    def angle_change(self):
        return self._programs[..., 0]

    @property
    def power_change(self):
        return self._programs[..., 1]

    @property
    def angle(self):
        return self._actions[..., 0]

    @property
    def power(self):
        return self._actions[..., 1]

    @staticmethod
    def generate_random_programs(
        bot, num=GA.PROGRAM_POPULATION_SIZE, len=GA.PROGRAM_LENGTH
    ):
        rng = np.random.default_rng()
        angle_change = rng.integers(
            low=Env.MIN_ANGLE_CHANGE,
            high=Env.MAX_ANGLE_CHANGE,
            size=(num, len),
            endpoint=True,
        )
        power_change = rng.integers(
            low=Env.MIN_POWER_CHANGE,
            high=Env.MAX_POWER_CHANGE,
            size=(num, len),
            endpoint=True,
        )
        return BotPrograms(bot, np.stack((angle_change, power_change), axis=2))

    def calculate_bot_actions(self, bot):
        # Calculate commands based on the program
        angle = bot.angle + np.cumsum(self.angle_change, axis=1)
        power = bot.power + np.cumsum(self.power_change, axis=1)
        # Apply restrictions
        angle[angle < Env.MIN_ANGLE] = Env.MIN_ANGLE
        angle[angle > Env.MAX_ANGLE] = Env.MAX_ANGLE
        power[power < Env.MIN_POWER] = Env.MIN_POWER
        power[power > Env.MAX_POWER] = Env.MAX_POWER
        self._actions = np.stack((angle, power), axis=2)

    def calculate_scores(self, bot_states):
        # Default score is 0
        self._scores = np.zeros(bot_states.len)

        # Score collision based on how far we are from the target
        collisions = bot_states.bot_status == BotStatus.SURFACE_COLLISION
        self._scores[collisions] = GA.LANDING_FOUND_SCORE * (
            1 - bot_states.goal_distance[collisions]
        )

        # Identify landings
        landings = bot_states.bot_status == BotStatus.LANDING

        # Find which landings were successful
        rows = np.arange(bot_states.len)
        landing_velocity = bot_states.bot_velocity[rows, bot_states.termination_step]
        landing_angle = self.angle[rows, bot_states.termination_step]
        landing_success_criteria = np.hstack(
            (Env.MAXIMUM_LANDING_VELOCITY, Env.MAXIMUM_LANDING_ANGLE)
        )
        bot_landing_state = np.hstack((landing_velocity, landing_angle[:, None]))
        landing_successes = landings & np.all(
            np.abs(bot_landing_state) <= landing_success_criteria, axis=1
        )
        landing_fails = landings & ~landing_successes

        # Score fails based on how far off we are from target velocity & angle
        landing_penalties = (
            np.sum(
                np.tanh(
                    np.abs(bot_landing_state[landing_fails]) / landing_success_criteria
                    - 1
                ),
                axis=1,
            )
            / 3
        )
        self._scores[landing_fails] = (
            GA.LANDING_FOUND_SCORE + GA.LANDING_SUCCESS_SCORE * (1 - landing_penalties)
        )

        # Score full successeses
        landing_fuel = bot_states.bot_fuel[rows, bot_states.termination_step]
        self._scores[landing_successes] = (
            GA.LANDING_FOUND_SCORE
            + GA.LANDING_SUCCESS_SCORE
            + GA.FUEL_EFFICIENCY_SCORE
            * landing_fuel[landing_successes]
            / self._bot.fuel
        )

    def evolve(self):
        sort_order = np.argsort(-self._scores)
        # Preserve the elite
        elite_size = int(GA.PROGRAM_POPULATION_SIZE * GA.ELITISM)
        elite = self._programs[sort_order[:elite_size], ...]
        # Create reproduction pool
        rng = np.random.default_rng()
        scores = self._scores[sort_order]
        probabilities = np.cumsum(scores / np.sum(scores))
        reproduction_pool_size = int(
            GA.PROGRAM_POPULATION_SIZE * GA.REPRODUCTION_SELECTION
        )
        # Reproduction pool should be at least 2
        if reproduction_pool_size < 2:
            reproduction_pool_size = 2
        reproduction_pool = sort_order[
            np.argmax(
                rng.random(reproduction_pool_size)[:, None] < probabilities, axis=1
            )
        ]
        # Select parents from the reproduction pool
        children_num = GA.PROGRAM_POPULATION_SIZE - elite_size
        n = children_num
        parents = np.empty((n, 2), dtype=int)
        replace = np.repeat(True, n)
        while n:
            parents[replace, :] = rng.choice(reproduction_pool, size=(n, 2))
            replace[replace] = parents[replace, 0] == parents[replace, 1]
            n = np.count_nonzero(replace)
        parents = self._programs[parents, ...]
        betas = (
            (1 + 2 * GA.CROSSOVER_BETA) * rng.random(size=(parents.shape[0], self.len))
            - GA.CROSSOVER_BETA
        )[..., None]
        programs = (
            parents[:, 0, ...] * betas + parents[:, 1, ...] * (1 - betas)
        ).astype(int)
        # Mutate angle
        mutation = rng.random((children_num, self.len)) < GA.MUTATION_RATE
        programs[..., 0][mutation] = rng.integers(
            low=Env.MIN_ANGLE_CHANGE,
            high=Env.MAX_ANGLE_CHANGE,
            size=np.count_nonzero(mutation),
            endpoint=True,
        )
        # Mutate angle
        mutation = rng.random((children_num, self.len)) < GA.MUTATION_RATE
        programs[..., 1][mutation] = rng.integers(
            low=Env.MIN_POWER_CHANGE,
            high=Env.MAX_POWER_CHANGE,
            size=np.count_nonzero(mutation),
            endpoint=True,
        )
        programs = np.vstack((elite, programs))
        return BotPrograms(self._bot, programs)

    def progress(self, bot):
        self._bot = bot
        np.delete(self._programs, 0, axis=1)
        np.delete(self._actions, 0, axis=1)


class BotStates:
    def __init__(
        self,
        bot_pos,
        bot_vel,
        bot_acc,
        bot_fuel,
        bot_status,
        termination_step,
        goal_distance,
    ):
        self.bot_position = bot_pos
        self.bot_velocity = bot_vel
        self.bot_acceleration = bot_acc
        self.bot_fuel = bot_fuel
        self.bot_status = bot_status
        self.termination_step = termination_step
        self.goal_distance = goal_distance

    @property
    def len(self):
        return len(self.bot_position)

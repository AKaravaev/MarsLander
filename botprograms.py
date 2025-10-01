import numpy as np

from config import EnvConfig as Env


class BotPrograms:
    @property
    def num(self):
        return self._programs.shape[0]

    @property
    def len(self):
        return self._programs.shape[1]

    @property
    def relative_change(self):
        return self._programs

    @property
    def angle_change(self):
        return self._programs[..., 0]

    @property
    def scores(self):
        return self._scores

    @scores.setter
    def scores(self, value):
        self._scores = value

    @property
    def power_change(self):
        return self._programs[..., 1]

    def __init__(self, num, len):
        self._step = 0
        # self.calculate_bot_actions(self._bot)
        # Generate random program
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
        self._programs = np.stack((angle_change, power_change), axis=2)
        self._scores = np.zeros(self.num)

    def evolve(self, generations):
        pass

    def extract_best_action(self):
        i = np.argmax(self._scores)
        self._step += 1
        angle, power = self._programs[i][0]
        self._programs = np.delete(self._programs, 0, axis=1)
        return angle, power


"""
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

"""

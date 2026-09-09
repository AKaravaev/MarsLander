from abc import ABC, abstractmethod

import numpy as np

from config import EnvConfig, GAConfig
from env import EnvStatus, MarsLanderEnv
from genalgo import Chromosomes
from npgeom import distance


class Policy(ABC):
    def __init__(self, env: MarsLanderEnv, refresh=None):
        self.env = env
        self.refresh = refresh
        self.reset()

    @abstractmethod
    def action(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class RandomPolicy(Policy):
    def action(self):
        return self.env.action_space.sample()


class GAPolicy(Policy):

    def reset(self):
        self.inputs = self.env.generate_random_inputs(
            (GAConfig.POPULATION_SIZE, GAConfig.PROGRAM_LENGTH)
        )
        self.best_input_id = 0

    def action(self):
        if self.inputs.shape[1] == 0:
            return self.env.action_space.sample()
        bot_states, term_status, term_step = self.env.emulate_inputs(self.inputs)
        scores = self.calculate_scores(bot_states, term_status, term_step)
        self.best_input_id = np.argmax(scores).astype(int)
        self.env.set_best_path(self.best_input_id)
        for i in range(GAConfig.GENERATIONS_NUM):
            genes = Chromosomes(self.inputs, scores, self.env.generate_random_inputs)
            self.inputs = genes.evolve()
            bot_states, term_status, term_step = self.env.emulate_inputs(self.inputs)
            scores = self.calculate_scores(bot_states, term_status, term_step)
            self.best_input_id = np.argmax(scores).astype(int)
            self.env.set_best_path(self.best_input_id)
            if self.refresh:
                self.refresh()
        best_input = self.inputs[self.best_input_id]
        angle, power = best_input[0]
        if term_step[self.best_input_id] == 1:
            angle = -self.env.bot.angle
        self.inputs = self.env.normalize_inputs(self.inputs)
        self.inputs = np.delete(self.inputs, 0, axis=1)
        return angle, power

    def calculate_scores(self, bot_states, term_status, term_step):
        program_num = bot_states.position.shape[0]
        scores = np.zeros(program_num)
        is_crash = term_status == EnvStatus.BOT_CRASHED.value
        is_landing_crash = term_status == EnvStatus.BOT_LANDING_CRASH.value
        is_landing = term_status == EnvStatus.BOT_LANDED.value

        # In case of crashes, we need to identify crash point distance to the landing
        if np.any(is_crash):
            # col_dist = geom.point_to_segment_dist_2d(
            #    col_point[is_not_landing], self.mars.landing[0], self.mars.landing[1]
            # )
            crash_point = np.take_along_axis(
                bot_states.position[is_crash], term_step[is_crash, None, None], axis=1
            ).squeeze(axis=1)
            crash_dist = np.min(
                np.abs(
                    np.stack(
                        [
                            crash_point[:, 0] - self.env.mars.landing[0, 0],
                            crash_point[:, 0] - self.env.mars.landing[1, 0],
                        ],
                        axis=1,
                    )
                ),
                axis=1,
            )
            # Reward for crashes is proportional to the distance
            """
            scores[is_crash] = (
                1 - np.minimum(crash_dist, self.size[0]) / self.size[0]
            ) * Rewards.LANDING_FOUND
            """
            scores[is_crash] = GAConfig.LANDING_DISTANCE_REWARD * (
                1 - np.tanh(2 * crash_dist / self.env.size[0])
            )

        if np.any(is_landing_crash):
            max_land_vel = np.asarray(EnvConfig.MAXIMUM_LANDING_VELOCITY)

            landing_vel = np.abs(
                np.take_along_axis(
                    bot_states.velocity[is_landing_crash],
                    term_step[is_landing_crash, None, None],
                    axis=1,
                ).squeeze(axis=1)
            )

            land_vel_excess = distance(
                np.maximum(landing_vel - max_land_vel, 0) / max_land_vel, (0, 0)
            )

            scores[is_landing_crash] = (
                GAConfig.LANDING_FOUND_REWARD
                + GAConfig.LANDING_SUCCESS_REWARD * (1 - np.tanh(land_vel_excess))
            )

        if np.any(is_landing):

            landing_fuel = np.take_along_axis(
                bot_states.fuel[is_landing], term_step[is_landing, None], axis=1
            ).squeeze(axis=1)
            scores[is_landing] = (
                GAConfig.LANDING_FOUND_REWARD
                + GAConfig.LANDING_SUCCESS_REWARD
                + (landing_fuel)
            )

        return scores

import random
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import npgeom as geom
from bot import Bot
from config import EnvConfig as Conf
from surface import Surface


class EnvStatus(Enum):
    RUNNING = 0
    BOT_LANDED = 1
    BOT_CRASHED = 2
    BOT_LANDING_CRASH = 3
    BOT_OUT_OF_BOUNDS = 4
    BOT_OUT_OF_FUEL = 5


class MarsLanderEnv(gym.Env):
    # metadata = {"render_modes": ["human"]}

    @property
    def terminated(self):
        return self.status != EnvStatus.RUNNING

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.size = np.asarray(Conf.WORLD_SIZE)
        self.action_space = spaces.MultiDiscrete(
            [
                2 * Conf.MAX_ANGLE_CHANGE + 1,
                2 * Conf.MAX_POWER_CHANGE + 1,
            ],
            start=[-Conf.MAX_ANGLE_CHANGE, -Conf.MAX_POWER_CHANGE],
        )
        self.observation_space = spaces.Dict(
            {
                "surface": spaces.Box(
                    low=0,
                    high=1,
                    shape=(Conf.MAX_SURFACE_LEN, 2),
                ),
                "landing": spaces.Box(low=0, high=1, shape=(2, 2)),
                "bot": spaces.Dict(
                    {
                        "position": spaces.Box(low=0, high=1, shape=(2,)),
                        "velocity": spaces.Box(low=-1, high=1, shape=(2,)),
                        "angle": spaces.Box(low=-1, high=1),
                        "thrust": spaces.Box(low=0, high=1),
                        "fuel": spaces.Box(low=0, high=1),
                    }
                ),
            }
        )
        # self.reset()

    def _get_obs(self):
        return {
            "surface": self.mars.surface / self.size,
            "landing": self.mars.landing / self.size,
            "bot": {
                "position": self.bot.position / self.size,
                "velocity": self.bot.velocity / self.size,
                "angle": self.bot.angle / Conf.MAX_ANGLE,
                "power": self.bot.power / Conf.MAX_POWER,
            },
        }

    def _get_info(self):
        return

    def _check_term_conditions(self):
        pos = [i.position for i in self.bot.history[-2:]]
        step, segment, col_point = self.mars.detect_collisions(pos)
        # If we've collided with the surface
        if step:
            if segment == self.mars._landing_index:
                # Checking landing speed
                if np.all(
                    np.abs(self.bot.velocity)
                    <= np.asarray(Conf.MAXIMUM_LANDING_VELOCITY)
                ) and (self.bot.angle == 0):
                    self.status = EnvStatus.BOT_LANDED
                else:
                    self.status = EnvStatus.BOT_CRASHED
            else:
                # If collision was outside of the landing site
                self.status = EnvStatus.BOT_CRASHED

        elif geom.is_point_out_of_rect(self.bot.position, self.size):
            self.status = EnvStatus.BOT_OUT_OF_BOUNDS
        elif self.bot.fuel < 0:
            self.status = EnvStatus.BOT_OUT_OF_FUEL

    def generate_random_inputs(self, size):
        rng = np.random.default_rng()
        angle_change = rng.integers(
            low=-Conf.MAX_ANGLE_CHANGE,
            high=Conf.MAX_ANGLE_CHANGE,
            size=size,
            endpoint=True,
        )
        power_change = rng.integers(
            low=-Conf.MAX_POWER_CHANGE,
            high=Conf.MAX_POWER_CHANGE,
            size=size,
            endpoint=True,
        )
        inputs = np.stack((angle_change, power_change), axis=-1)
        return inputs

    def load_initial_state(self, surface, bot, add_randomness=True):
        if add_randomness and random.random() < 0.5:
            is_flipped = True
        else:
            is_flipped = False
        if is_flipped:
            surface = np.asarray(surface)
            surface[:, 0] = self.size[0] - surface[:, 0]
            surface = surface[::-1]
            bot["position"] = (self.size[0] - bot["position"][0], bot["position"][1])
            bot["velocity"] = (-bot["velocity"][0], bot["velocity"][1])
            bot["angle"] *= -1
        self.mars = Surface(surface)
        self.bot = Bot(**bot)

    def reset(self):
        self.iter = 0
        self.status = EnvStatus.RUNNING
        self.truncated = False
        self.reward = 0
        self.load_initial_state(**random.choice(Conf.TEST_CASES))
        self.best_path_id = -1
        return self._get_obs(), self._get_info()

    def set_best_path(self, id: int):
        self.best_path_id = id

    def step(self, action):
        self.iter += 1
        angle_change, power_change = action
        self.bot.execute(angle_change, power_change)
        self._check_term_conditions()
        return (
            self._get_obs(),
            self.reward,
            self.terminated,
            self.truncated,
            self._get_info(),
        )

    def render(self):
        pass

    def convert_inputs_to_actions(self, inputs):
        actions = (self.bot.angle, self.bot.power) + np.cumsum(inputs, axis=1)
        np.clip(actions[:, :, 0], -Conf.MAX_ANGLE, Conf.MAX_ANGLE, actions[:, :, 0])
        np.clip(actions[:, :, 1], 0, Conf.MAX_POWER, actions[:, :, 1])
        return actions

    def convert_actions_to_inputs(self, actions):
        return np.insert(
            np.diff(actions, axis=1),
            0,
            actions[:, 0, :] - (self.bot.angle, self.bot.power),
            axis=1,
        )

    def normalize_inputs(self, inputs):
        actions = self.convert_inputs_to_actions(inputs)
        return self.convert_actions_to_inputs(actions)

    def emulate_inputs(self, inputs):
        actions = self.convert_inputs_to_actions(inputs)
        return self.emulate_actions(actions)

    def emulate_actions(self, actions):
        actions_num = actions.shape[1]
        program_num = actions.shape[0]

        bot_states = self.bot.emulate(actions)

        # Checking termination conditions
        # Is bot out of bounds?
        is_oob = geom.is_point_out_of_rect(bot_states.position, self.size)
        oob_step = np.argmax(is_oob, axis=-1)
        # Is bot out of fuel?
        is_oof = bot_states.fuel < 0
        oof_step = np.argmax(is_oof, axis=-1)
        # Is there a surface collision?
        col_step, col_segment, col_point = self.mars.detect_collisions(
            bot_states.position
        )
        # Now finding which conditions was triggered earlier
        cond_term_step = np.stack((oob_step, oof_step, col_step), axis=1)
        # If conditions wasn't triggered, then assuming maximum possible step
        cond_term_step[cond_term_step == 0] = actions_num + 1
        term_cond = np.argmin(cond_term_step, axis=-1)
        term_step = np.min(cond_term_step, axis=-1)

        term_status = np.repeat(EnvStatus.RUNNING.value, program_num)
        term_status[(term_step <= actions_num) & (term_cond == 0)] = (
            EnvStatus.BOT_OUT_OF_BOUNDS.value
        )
        term_status[term_cond == 1] = EnvStatus.BOT_OUT_OF_FUEL.value

        # Determine which traces ended in collission
        is_col = term_cond == 2
        # Update traces to terminate at a collision point
        bot_states.position[is_col, term_step[is_col]] = col_point[is_col]

        # Separate collisions with landing segment from other crashes
        is_landing_col = is_col & (col_segment == self.mars._landing_index)
        is_not_landing_col = is_col & (~is_landing_col)
        term_status[is_not_landing_col] = EnvStatus.BOT_CRASHED.value
        term_status[is_landing_col] = EnvStatus.BOT_LANDING_CRASH.value

        # If landing is found, let's check the landing speed
        if np.any(is_landing_col):
            # landing_step = col_step[is_landing_col] - 1
            landing_vel = np.abs(
                np.take_along_axis(
                    bot_states.velocity[is_landing_col],
                    col_step[is_landing_col][:, None, None],
                    axis=1,
                ).squeeze(axis=1)
            )
            max_land_vel = np.asarray(Conf.MAXIMUM_LANDING_VELOCITY)
            is_landing_success = np.all(landing_vel <= max_land_vel, axis=-1)

            term_status[np.where(is_landing_col)[0][is_landing_success]] = (
                EnvStatus.BOT_LANDED.value
            )

        return bot_states, term_status, term_step


"""
        # terminal_velocities = np.abs(velocities[rows, termination_step,:])
        # is_valid_terminal_velocity = (terminal_velocities[:,0] <=
        #   self.maximum_landing_velocity[0]) & \
        #   (terminal_velocities[:,1] <= self.maximum_landing_velocity[1])
        # bot_status[landings & is_valid_terminal_velocity] = BotStatus.LANDING_SUCCESS

        return bot_status, termination_step, goal_distance
"""

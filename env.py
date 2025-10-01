from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import npgeom as geom
from bot import Bot
from config import EnvConfig as Conf
from config import Rewards
from surface import Surface


class EnvStatus(Enum):
    RUNNING = 0
    BOT_LANDED = 1
    BOT_CRASHED = 2
    BOT_OUT_OF_BOUNDS = 3
    BOT_OUT_OF_FUEL = 4


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
                Conf.MAX_ANGLE_CHANGE - Conf.MIN_ANGLE_CHANGE + 1,
                Conf.MAX_POWER_CHANGE - Conf.MIN_POWER_CHANGE + 1,
            ],
            start=[Conf.MIN_ANGLE_CHANGE, Conf.MIN_POWER_CHANGE],
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
        self.reset()

    def _get_obs(self):
        return {
            "surface": self.mars.surface / self.size,
            "landing": self.mars.landing / self.size,
            "bot": {
                "position": self.bot.position / self.size,
                "velocity": self.bot.velocity / self.size,
                "angle": self.bot.angle / max(abs(Conf.MAX_ANGLE), abs(Conf.MIN_ANGLE)),
                "power": self.bot.power / max(abs(Conf.MAX_POWER), abs(Conf.MIN_POWER)),
            },
        }

    def _get_info(self):
        return

    def _check_conditions(self):
        step, segment, col_point = self.mars.detect_collisions(
            self.bot.history.position[-2:, ...]
        )
        # If we've collided with the surface
        if step:
            if segment == self.mars._landing_index:
                # Checking landing speed
                if np.all(
                    np.abs(self.bot.history.velocity[-1, ...])
                    < np.asarray(Conf.MAXIMUM_LANDING_VELOCITY)
                ) and (self.bot.history.angle[-1, ...] == 0):
                    self.status = EnvStatus.BOT_LANDED
                    self.reward = Rewards.LANDING_SUCCESS
                else:
                    self.status = EnvStatus.BOT_CRASHED
            else:
                # If collision was outside of the landing site
                self.status = EnvStatus.BOT_CRASHED

        elif geom.is_point_out_of_rect(self.bot.position, self.size):
            self.status = EnvStatus.BOT_OUT_OF_BOUNDS
        elif self.bot.fuel < 0:
            self.status = EnvStatus.BOT_OUT_OF_FUEL

    def generate_random_actions(self, size):
        rng = np.random.default_rng()
        angle_change = rng.integers(
            low=Conf.MIN_ANGLE_CHANGE,
            high=Conf.MAX_ANGLE_CHANGE,
            size=size,
            endpoint=True,
        )
        power_change = rng.integers(
            low=Conf.MIN_POWER_CHANGE,
            high=Conf.MAX_POWER_CHANGE,
            size=size,
            endpoint=True,
        )
        actions = np.stack((angle_change, power_change), axis=-1)
        return actions

    def reset(self):
        self.iter = 0
        self.status = EnvStatus.RUNNING
        self.truncated = False
        self.reward = 0
        self.mars = Surface(Conf.SURFACE)
        self.bot = Bot(
            position=Conf.INITIAL_BOT_POSITION,
            velocity=Conf.INITIAL_BOT_VELOCITY,
            angle=Conf.INITIAL_BOT_ANGLE,
            power=Conf.INITIAL_BOT_POWER,
            fuel=Conf.INITIAL_BOT_FUEL,
        )
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.iter += 1
        angle_change, power_change = action
        self.bot.execute(angle_change, power_change)
        self._check_conditions()
        return (
            self._get_obs(),
            self.reward,
            self.terminated,
            self.truncated,
            self._get_info(),
        )

    def render(self):
        pass

    def emulate_programs(self, programs):
        program_length = programs.shape[1]
        program_num = programs.shape[0]
        actions = (self.bot.angle, self.bot.power) + np.cumsum(programs, axis=1)
        np.clip(actions[..., 0], Conf.MIN_ANGLE, Conf.MAX_ANGLE, actions[..., 0])
        np.clip(actions[..., 1], Conf.MIN_POWER, Conf.MAX_POWER, actions[..., 1])
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
        cond_term_step[cond_term_step == 0] = program_length + 1
        term_cond = np.argmin(cond_term_step, axis=-1)
        term_step = np.min(cond_term_step, axis=-1)

        # Calculating scores

        # If not a collision, the score is zero
        scores = np.zeros(program_num)
        # Determine which traces ended in collission
        is_col = term_cond == 2
        # Separate collisions with landing segment from other crashes
        is_landing = is_col & (col_segment == self.mars._landing_index)
        is_not_landing = is_col & (~is_landing)

        # In case of crashes, we need to identify crash point
        # and it's distance to the landing segment
        col_dist = geom.point_to_segment_dist_2d(
            col_point[is_not_landing], self.mars.landing[0], self.mars.landing[1]
        )

        # Reward for crashes is proportional to the distance
        """
        scores[is_crash] = (
            1 - np.minimum(col_dist, self.size[0]) / self.size[0]
        ) * Rewards.LANDING_FOUND
        """
        scores[is_not_landing] = Rewards.LANDING_FOUND * (
            1 - np.tanh(2 * col_dist / self.size[0])
        )
        # If landing is found, let's check the landing speed
        landing_step = col_step[is_landing] - 1
        landing_vel = np.abs(
            np.take_along_axis(
                bot_states.velocity[is_landing],
                landing_step[:, np.newaxis, np.newaxis],
                axis=-2,
            ).squeeze(axis=-2)
        )
        landing_fuel = np.take_along_axis(
            bot_states.fuel[is_landing], landing_step[:, np.newaxis], axis=-1
        ).squeeze(axis=-1)

        max_land_vel = np.asarray(Conf.MAXIMUM_LANDING_VELOCITY)

        land_vel_excess = geom.distance(
            np.maximum(landing_vel - max_land_vel, 0) / max_land_vel, (0, 0)
        )

        is_landing_success = np.all(landing_vel <= max_land_vel, axis=-1)

        scores[is_landing] = np.where(
            is_landing_success,
            Rewards.LANDING_FOUND + Rewards.LANDING_SUCCESS + (landing_fuel / 100),
            Rewards.LANDING_SUCCESS * (1 - np.tanh(land_vel_excess))
            + Rewards.LANDING_FOUND,
        )

        return scores, term_step


"""
        # terminal_velocities = np.abs(velocities[rows, termination_step,:])
        # is_valid_terminal_velocity = (terminal_velocities[:,0] <=
        #   self.maximum_landing_velocity[0]) & \
        #   (terminal_velocities[:,1] <= self.maximum_landing_velocity[1])
        # bot_status[landings & is_valid_terminal_velocity] = BotStatus.LANDING_SUCCESS

        return bot_status, termination_step, goal_distance
"""

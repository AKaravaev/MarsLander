import math
from typing import NamedTuple

import numpy as np

from config import EnvConfig as Env


class BotStates(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    fuel: np.ndarray
    angle: np.ndarray
    power: np.ndarray


class Bot:

    def __init__(self, position, velocity, angle, power, fuel):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.acceleration = np.array((0, 0))
        self.angle = angle
        self.power = power
        self.fuel = fuel
        self.history = BotStates(
            position=self.position[np.newaxis, :],
            velocity=self.velocity[np.newaxis, :],
            acceleration=self.acceleration[np.newaxis, :],
            fuel=np.array(self.fuel),
            angle=np.array(self.angle),
            power=np.array(self.power),
        )
        self.traces = None

    @property
    def x(self):
        return self.position[0]

    @x.setter
    def x(self, value):
        self.position[0] = value

    @property
    def y(self):
        return self.position[1]

    @y.setter
    def y(self, value):
        self.position[1] = value

    @property
    def vx(self):
        return self.velocity[0]

    @vx.setter
    def vx(self, value):
        self.velocity[0] = value

    @property
    def vy(self):
        return self.velocity[1]

    @vy.setter
    def vy(self, value):
        self.velocity[1] = value

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = max(Env.MIN_ANGLE, min(Env.MAX_ANGLE, value))

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, value):
        self._power = max(Env.MIN_POWER, min(Env.MAX_POWER, value))

    def _add_current_state_to_history(self):
        self.history = BotStates(
            position=np.vstack((self.history.position, self.position)),
            velocity=np.vstack((self.history.velocity, self.velocity)),
            acceleration=np.vstack((self.history.acceleration, self.acceleration)),
            fuel=np.vstack((self.history.fuel, self.fuel)),
            angle=np.vstack((self.history.angle, self.angle)),
            power=np.vstack((self.history.power, self.power)),
        )

    def execute(self, angle_change, power_change):
        self.angle += angle_change
        self.power += power_change

        # Calculating new state
        a = np.array(
            (
                math.sin(math.radians(self.angle)) * self.power,
                math.cos(math.radians(self.angle)) * self.power - Env.GRAVITY,
            )
        )
        self.position += self.velocity + a / 2
        self.velocity += a
        self.fuel -= self.power
        self._add_current_state_to_history()

    def emulate(self, actions):
        # Vectorized version of execute for simultaneous emulation
        # of multiple actions, actions are in the form [program#, step#, [angle, power]]

        emulations_num = len(actions)
        angle = np.radians(actions[..., 0])
        power = actions[..., 1]

        # Calculate trajectory based on the bot actions
        # Start by determining acceleration along each axis
        acc = np.stack(
            (np.sin(angle) * power, np.cos(angle) * power - Env.GRAVITY), axis=-1
        )

        power = actions[..., 1]
        # Calculating velocities at each point
        initial_vel = np.broadcast_to(self.velocity, (emulations_num, 1, 2))
        vel = np.concatenate(
            (initial_vel, self.velocity + np.cumsum(acc, axis=1)), axis=1
        )

        # Calculating resulting positions
        pos_change = vel[:, :-1, :] + acc / 2
        initial_pos = np.broadcast_to(self.position, (emulations_num, 1, 2))
        pos = np.concatenate(
            (initial_pos, self.position + np.cumsum(pos_change, axis=1)), axis=1
        )

        # Calculating fuel
        fuel = np.concatenate(
            (
                np.repeat(self.fuel, emulations_num)[:, None],
                self.fuel - np.cumsum(power, axis=1),
            ),
            axis=1,
        )

        self.traces = BotStates(
            position=pos,
            velocity=vel,
            acceleration=acc,
            fuel=fuel,
            angle=actions[..., 0],
            power=actions[..., 1],
        )

        return self.traces

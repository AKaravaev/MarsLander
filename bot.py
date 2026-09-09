import math
from typing import NamedTuple

import numpy as np

from config import EnvConfig as Env


class BotStates(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray
    fuel: np.ndarray
    angle: np.ndarray
    power: np.ndarray


class BotState(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray
    fuel: int
    angle: int
    power: int


class Bot:

    def __init__(self, position, velocity, angle, power, fuel):
        self.position = position
        self.velocity = np.asarray(velocity)
        self.angle = angle
        self.power = power
        self.fuel = fuel
        self.history = []
        self._add_current_state_to_history()
        self.traces = None

    @property
    def position(self):
        return self._position.copy()

    @position.setter
    def position(self, value):
        self._position: np.ndarray = np.array(value, dtype=np.float64)

    @property
    def velocity(self):
        return self._velocity.copy()

    @velocity.setter
    def velocity(self, value):
        self._velocity: np.ndarray = np.array(value, dtype=np.float64)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle: int = max(-Env.MAX_ANGLE, min(Env.MAX_ANGLE, value))

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, value):
        self._power: int = max(0, min(Env.MAX_POWER, value))

    @property
    def fuel(self):
        return self._fuel

    @fuel.setter
    def fuel(self, value):
        self._fuel: int = max(value, 0)

    def _add_current_state_to_history(self):
        self.history.append(
            BotState(
                position=self.position,
                velocity=self.velocity,
                fuel=self.fuel,
                angle=self.angle,
                power=self.power,
            )
        )
        """
        self.history = BotStates(
            position=np.vstack((self.history.position, self.position)),
            velocity=np.vstack((self.history.velocity, self.velocity)),
            acceleration=np.vstack((self.history.acceleration, self.acceleration)),
            fuel=np.vstack((self.history.fuel, self.fuel)),
            angle=np.vstack((self.history.angle, self.angle)),
            power=np.vstack((self.history.power, self.power)),
        )
        """

    def execute(self, angle_change, power_change):
        self.angle += angle_change
        self.power += power_change

        # Calculating new state
        rad_angle = math.radians(self.angle)
        acc = np.array(
            (
                math.sin(rad_angle) * self.power,
                math.cos(rad_angle) * self.power - Env.GRAVITY,
            )
        )
        self._position += self._velocity + acc / 2
        self._velocity += acc
        self._fuel -= self.power
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
            fuel=fuel,
            angle=actions[..., 0],
            power=actions[..., 1],
        )

        return self.traces

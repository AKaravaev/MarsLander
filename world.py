import math
from typing import cast

import numpy as np
import numpy.typing as npt

from bot import BotStates, BotStatus
from config import EnvParams as Env


class World:
    def __init__(self, size, g, surface, bot):
        self.size = size
        self.g = g
        self.surface = np.array(surface)
        self.surface_vectors = np.diff(self.surface, axis=0)
        self.landing_zone_index = None
        self.landing_zone_index = self.get_landing_zone_index()
        self.bot = bot

    def identify_terminal_state(self, trajectories):
        # Check out of bounds
        is_out_of_bounds = (
            (trajectories[:, 1:, 0] < 0)
            | (trajectories[:, 1:, 1] < 0)
            | (trajectories[:, 1:, 1] >= self.size[1])
            | (trajectories[:, 1:, 0] >= self.size[0])
        )

        # Now checking for surface collisions
        # Collecting line segments point together
        traj_vect = np.diff(trajectories, axis=1)[..., None, :]
        traj_p1 = trajectories[:, :-1, None, :]
        traj_p2 = trajectories[:, 1:, None, :]
        surf_p1 = self.surface[:-1,]
        surf_p2 = self.surface[1:,]

        # Calculating triplets' orientation
        o1 = self.surface_vectors[..., 1] * (
            traj_p1[..., 0] - surf_p1[..., 0]
        ) - self.surface_vectors[..., 0] * (traj_p1[..., 1] - surf_p1[..., 1])
        o2 = self.surface_vectors[..., 1] * (
            traj_p2[..., 0] - surf_p1[..., 0]
        ) - self.surface_vectors[..., 0] * (traj_p2[..., 1] - surf_p1[..., 1])
        o3 = traj_vect[..., 1] * (surf_p1[..., 0] - traj_p1[..., 0]) - traj_vect[
            ..., 0
        ] * (surf_p1[..., 1] - traj_p1[..., 1])
        o4 = traj_vect[..., 1] * (surf_p2[..., 0] - traj_p1[..., 0]) - traj_vect[
            ..., 0
        ] * (surf_p2[..., 1] - traj_p1[..., 1])

        # Calculating if intersection occurs for each trajectory and surface segment
        is_surface_segment_collision = (np.sign(o1) != np.sign(o2)) & (
            np.sign(o3) != np.sign(o4)
        )
        # Flag surface collision for a trajectory if any surface segment (ex. landing)
        # is intersected
        is_rough_surface_collision = cast(
            npt.NDArray[np.bool],
            np.delete(
                is_surface_segment_collision, self.get_landing_zone_index(), axis=2
            ).any(axis=2),
        )
        # Flag landing site collisions
        is_landing_site_collision = is_surface_segment_collision[
            ..., self.get_landing_zone_index()
        ]
        # Identify trajectory points where any of the termination conditions is observed
        is_termination_condition = np.stack(
            (is_out_of_bounds, is_rough_surface_collision, is_landing_site_collision),
            axis=2,
        )
        # Identify the earliest step for which one of the termination
        # conditions is triggered
        is_terminated = np.any(is_termination_condition, axis=2)
        termination_step = np.argmax(is_terminated, axis=1)
        # termination_point = trajectories[rows, termination_step, :]

        # Find which termination was triggered
        rows = np.arange(len(trajectories))
        # Default is pending
        bot_status = np.repeat(
            cast(npt.ArrayLike, [BotStatus.PENDING]), len(trajectories)
        )
        goal_distance = np.repeat(np.nan, len(trajectories))

        # Out of bounds
        bot_status[is_out_of_bounds[rows, termination_step]] = (
            BotStatus.OUT_OF_BOUNDS.value
        )

        # Collisions
        collisions = is_rough_surface_collision[rows, termination_step]
        collision_rows = rows[collisions]
        collision_step = termination_step[collisions]
        collision_segment = np.argmax(
            is_surface_segment_collision[collision_rows, collision_step, :], axis=1
        )
        # Identify intersection points
        traj_vect = traj_vect.squeeze(axis=2)
        termination_points = (
            surf_p1[collision_segment, :]
            - (
                o3[collision_rows, collision_step, collision_segment]
                / (
                    self.surface_vectors[collision_segment, 0]
                    * traj_vect[collision_rows, collision_step, 1]
                    - self.surface_vectors[collision_segment, 1]
                    * traj_vect[collision_rows, collision_step, 0]
                )
            )[:, None]
            * self.surface_vectors[collision_segment]
        )
        bot_status[collisions] = BotStatus.SURFACE_COLLISION
        # Calculate the distance to the nearest landing zone end point
        landing_zone = np.vstack(
            (surf_p1[self.landing_zone_index], surf_p2[self.landing_zone_index])
        )
        goal_distance[collision_rows] = np.sqrt(
            np.min(
                np.sum(
                    np.square(termination_points[:, None, :] - landing_zone), axis=2
                ),
                axis=1,
            )
        )
        # Normalize distance to [0:1] using tanh
        goal_distance = np.tanh(2 * goal_distance / self.size[0])
        # Landings
        landings = is_landing_site_collision[rows, termination_step] & ~collisions
        bot_status[landings] = BotStatus.LANDING
        goal_distance[landings] = 0

        # terminal_velocities = np.abs(velocities[rows, termination_step,:])
        # is_valid_terminal_velocity = (terminal_velocities[:,0] <=
        #   self.maximum_landing_velocity[0]) & \
        #   (terminal_velocities[:,1] <= self.maximum_landing_velocity[1])
        # bot_status[landings & is_valid_terminal_velocity] = BotStatus.LANDING_SUCCESS

        return bot_status, termination_step, goal_distance

    def get_surface(self):
        return self.surface

    def get_landing_zone_index(self):
        # If a landing zone is already defined, return it to avoid recalculating
        if self.landing_zone_index is not None:
            return self.landing_zone_index
        # Find the landing zone by checking the surface points
        # The landing zone is defined as the first point where the y stays the same
        py = self.surface[0][1]
        for i, y in enumerate(self.surface[1:, 1], start=1):
            if y == py:
                return i - 1
            py = y
        # If no landing zone is found, return -1
        return -1

    def emulate_bot_programs(self, programs):
        angle = np.radians(programs.angle)
        power = programs.power

        # Calculate trajectory based on the bot actions
        acc = np.stack((np.sin(angle) * power, np.cos(angle) * power - self.g), axis=2)
        initial_vel = np.broadcast_to(self.bot.velocity, (programs.num, 1, 2))
        vel = np.concatenate(
            (initial_vel, self.bot.velocity + np.cumsum(acc, axis=1)), axis=1
        )
        pos_change = vel[:, :-1, :] + acc / 2
        initial_pos = np.broadcast_to(self.bot.position, (programs.num, 1, 2))
        pos = np.concatenate(
            (initial_pos, self.bot.position + np.cumsum(pos_change, axis=1)), axis=1
        )
        fuel = np.concatenate(
            (
                np.repeat(self.bot.fuel, programs.num)[:, None],
                self.bot.fuel - np.cumsum(power, axis=1),
            ),
            axis=1,
        )

        bot_status, termination_step, goal_distance = self.identify_terminal_state(pos)

        return BotStates(
            pos, vel, acc, fuel, bot_status, termination_step, goal_distance
        )

    def execute(self, angle_change, power_change):
        self.bot.angle += angle_change
        self.bot.power += power_change
        # Applying restrictions
        if self.bot.angle > Env.MAX_ANGLE:
            self.bot.angle = Env.MAX_ANGLE
        elif self.bot.angle < Env.MIN_ANGLE:
            self.bot.angle = Env.MIN_ANGLE
        if self.bot.power > Env.MAX_POWER:
            self.bot.power = Env.MAX_POWER
        elif self.bot.power < Env.MIN_POWER:
            self.bot.power = Env.MIN_POWER

        self.bot.fuel -= self.bot.power
        acc = (
            math.sin(math.radians(self.bot.angle)) * self.bot.power,
            math.cos(math.radians(self.bot.angle)) * self.bot.power - self.g,
        )
        self.bot.position = (
            self.bot.position[0] + self.bot.velocity[0] + acc[0] / 2,
            self.bot.position[1] + self.bot.velocity[1] + acc[1] / 2,
        )
        self.bot.velocity = (
            self.bot.velocity[0] + acc[0],
            self.bot.velocity[1] + acc[1],
        )

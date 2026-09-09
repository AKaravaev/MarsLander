import numpy as np

import npgeom as geom
from config import EnvConfig as Env


class Surface:
    def __init__(self, surface):
        self.surface = surface

    @property
    def surface(self):
        return self._surface

    @property
    def landing_index(self):
        return self._landing_index

    @property
    def landing(self):
        return self._landing

    @surface.setter
    def surface(self, value):
        value = np.asarray(value, dtype=float)
        self._surface = value
        if value.ndim != 2:
            raise ValueError(
                f"The surface should be a 2-d array of nodes."
                f" Got {value.ndim}-d array instead"
            )
        if value.shape[0] > Env.MAX_SURFACE_LEN:
            raise ValueError(
                f"Maximum surface is {Env.MAX_SURFACE_LEN} points."
                f" Got {value.shape[0]} points instead"
            )
        if value.shape[1] != 2:
            raise ValueError(
                f"Surface should be an array of 2d points."
                f" Got {value.shape[1]}-d points instaed"
            )
        # Find landing segment
        landing = None
        for i in range(len(value) - 1):
            if value[i][1] == value[i + 1][1]:
                landing = [value[i], value[i + 1]]
                break
        if not landing:
            raise ValueError("No flat landing identified")
        self._landing_index = i
        self._landing = np.array(landing)

        # Normalize the surface to contain exactly Env.MAX_SURFACE_LEN points
        while value.shape[0] < Env.MAX_SURFACE_LEN:
            # Find index of the longest segment
            i = np.argmax(geom.distance(value[:-1], value[1:]))
            # Find segment middle point
            p = (value[i] + value[i + 1]) / 2
            # Insert intermediary point into surface array
            value = np.insert(value, i + 1, p, axis=0)
        self._normalized_surface = value

    def detect_collisions(self, traces):
        # traces[num of traces; steps in a trace; [x,y]]
        # Returns collision step (0 if no collision) and collision segment
        traces = np.asarray(traces)
        if traces.ndim < 2 or traces.ndim > 3:
            raise ValueError(
                f"Traces should have 2 or 3 dimensions. Got {traces.ndim} instead"
            )
        if traces.ndim == 2:
            traces = np.expand_dims(traces, axis=0)
        if traces.shape[-1] != 2:
            raise ValueError(f"Expected 2 coordinates. Got {traces.shape[-1]} instead")
        if traces.shape[-2] < 2:
            raise ValueError("Trace should have at least 2 steps")

        # Determine which trace segments intersect with surface segments
        is_intersect = geom.is_segment_intersect_2d(
            traces[..., :-1, np.newaxis, :],
            traces[..., 1:, np.newaxis, :],
            self.surface[..., :-1, :],
            self.surface[..., 1:, :],
        )
        # For each trace steps determine if there is an intersection
        # with any of the surface segments
        is_collision_step = is_intersect.any(axis=-1)
        is_collision_trace = is_collision_step.any(axis=-1)

        # If trace intersects with the surface, determine collision step
        # Collision step = 0, if there is no collision
        collision_step = is_collision_step.argmax(axis=-1)

        # Determine which surface segment has trace intersected
        collision_segment = (
            np.take_along_axis(
                is_intersect, np.expand_dims(collision_step, axis=(-1, -2)), axis=-2
            )
            .squeeze(axis=-2)
            .argmax(axis=-1)
        )

        # Calculate correct collisions step , or 0 if no collision
        collision_step[is_collision_trace] += 1

        # Now we can find exact intersection point
        collision_trace_segment = np.take_along_axis(
            traces,
            np.expand_dims(
                np.stack((collision_step - 1, collision_step), axis=1), axis=-1
            ),
            axis=-2,
        )
        collision_surface_segment = self.surface[
            np.stack((collision_segment, collision_segment + 1), axis=1)
        ]

        collision_point = geom.find_segment_intersect_2d(
            collision_trace_segment[:, 0, :],
            collision_trace_segment[:, 1, :],
            collision_surface_segment[:, 0, :],
            collision_surface_segment[:, 1, :],
        )

        return collision_step, collision_segment, collision_point

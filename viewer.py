import numpy as np
import pygame
from pygame.math import Vector2

from config import DisplayConfig as Display
from env import EnvStatus, MarsLanderEnv


class MarsLanderViewer:
    def __init__(self, env: MarsLanderEnv):
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(Display.SCREEN_SIZE)
        self.font = pygame.font.SysFont(Display.FONT_NAME, Display.FONT_SIZE)
        self.env = env
        self.background = None
        self.bot_sprite = None

    def _convert_coords(self, points):
        # Scale to convert coordinates from env to screen
        scale = np.asarray(Display.SCREEN_SIZE) / np.asarray(self.env.size)
        scaled_points = points * scale
        # Invert y axis
        scaled_points[..., 1] = Display.SCREEN_SIZE[1] - scaled_points[..., 1]
        return scaled_points

    def draw_background(self):
        # Draw the background if there is no saved background
        if not self.background:
            surface = self._convert_coords(self.env.mars.surface)
            self.background = pygame.Surface(self.screen.get_size())  # .convert()
            pygame.draw.lines(
                self.background, Display.SURFACE_COLOR, False, surface.tolist()
            )
        # Put background on screen
        self.screen.blit(self.background, (0, 0))

    def draw_bot(self):
        bot_pos = Vector2(*self._convert_coords(self.env.bot.position))
        # If there is no bot image yet, draw for the first time
        if not self.bot_sprite:
            sz = Display.BOT_SIZE
            self.bot_sprite = pygame.Surface((sz, sz)).convert()
            shape = [
                (sz / 4 - 1, 0),
                (sz * 3 / 4 - 1, 0),
                (sz - 1, sz - 1),
                (0, sz - 1),
            ]
            pygame.draw.lines(
                self.bot_sprite,
                Display.BOT_COLOR,
                True,
                shape,
            )
        rotated_bot_sprite = pygame.transform.rotate(
            self.bot_sprite, -self.env.bot.angle
        )
        # Calculate offset vector from bot's bottom center to top left of the image rect
        # by calculating the rotated vector from bot's bottom center to image center
        # and adding vector from image center to the image top left
        bot_offset_vector = Vector2(0, -self.bot_sprite.get_height() / 2).rotate(
            self.env.bot.angle
        ) - Vector2(
            rotated_bot_sprite.get_width() / 2, rotated_bot_sprite.get_height() / 2
        )
        self.screen.blit(
            rotated_bot_sprite,
            bot_pos + bot_offset_vector,
        )
        if len(self.env.bot.history.position) >= 2:
            pygame.draw.lines(
                self.screen,
                Display.PATH_COLOR,
                closed=False,
                points=self._convert_coords(self.env.bot.history.position).tolist(),
            )
        self._draw_power(bot_pos)
        # Draw traces
        if self.env.bot.traces:
            traces = self._convert_coords(self.env.bot.traces.position)
            for trace in traces:
                pygame.draw.lines(
                    self.screen,
                    Display.TRACES_COLOR,
                    closed=False,
                    points=trace,
                )
        if self.env.status == EnvStatus.BOT_CRASHED:
            self._draw_crash_marker(bot_pos)

    def _draw_crash_marker(self, pos):
        pygame.draw.line(
            self.screen,
            Display.CRASH_MARKER_COLOR,
            pos - Vector2(Display.CRASH_MARKER_SIZE, Display.CRASH_MARKER_SIZE) / 2,
            pos + Vector2(Display.CRASH_MARKER_SIZE, Display.CRASH_MARKER_SIZE) / 2,
            width=Display.CRASH_MARKER_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            Display.CRASH_MARKER_COLOR,
            pos - Vector2(Display.CRASH_MARKER_SIZE, -Display.CRASH_MARKER_SIZE) / 2,
            pos + Vector2(Display.CRASH_MARKER_SIZE, -Display.CRASH_MARKER_SIZE) / 2,
            width=Display.CRASH_MARKER_WIDTH,
        )

    def _draw_power(self, pos):
        pygame.draw.line(
            self.screen,
            Display.POWER_VECTOR_COLOR,
            pos,
            pos
            + Vector2(0, Display.POWER_VECTOR_LENGTH * self.env.bot.power).rotate(
                self.env.bot.angle
            ),
        )

    def show_info(self):
        power_info = self.font.render(
            f"Power: {self.env.bot.power}", True, Display.FONT_COLOR
        )
        angle_info = self.font.render(
            f"Angle: {self.env.bot.angle}", True, Display.FONT_COLOR
        )
        fuel_info = self.font.render(
            f"Fuel: {self.env.bot.fuel}", True, Display.FONT_COLOR
        )
        vel_info = self.font.render(
            f"Velocity: {self.env.bot.velocity}", True, Display.FONT_COLOR
        )
        self.screen.blit(fuel_info, (0, 0))
        self.screen.blit(angle_info, (0, Display.INFO_SPACING))
        self.screen.blit(power_info, (0, Display.INFO_SPACING * 2))
        self.screen.blit(vel_info, (0, Display.INFO_SPACING * 3))

    def draw_frame(self):
        self.draw_background()
        self.draw_bot()
        self.show_info()
        pygame.display.flip()
        pygame.display.flip()

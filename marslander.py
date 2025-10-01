import sys

import pygame

from config import DisplayConfig as Display
from env import MarsLanderEnv
from policy import GAPolicy as Policy
from viewer import MarsLanderViewer

if __name__ == "__main__":
    running = True
    pygame.init()
    clock = pygame.time.Clock()
    env = MarsLanderEnv()
    viewer = MarsLanderViewer(env)
    obs, info = env.reset()
    viewer.draw_frame()
    policy = Policy(env)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            elif event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_q:
                        quit()
                    case pygame.K_p:
                        running = not running
                    case pygame.K_r:
                        policy.reset()
                        env.reset()
                        running = True

        if running:
            action = policy.action()
            obs, reward, terminated, trunctated, info = env.step(action)

        viewer.draw_frame()
        if terminated or trunctated:
            running = False
            # policy.reset()
            # env.reset()
        clock.tick(Display.FPS)


def quit():
    env.close()
    pygame.quit()
    sys.exit()

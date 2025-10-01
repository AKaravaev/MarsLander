from abc import ABC, abstractmethod

import numpy as np

from config import GeneticAlgorithmConfig as GA
from genalgo import Chromosomes


class Policy(ABC):
    def __init__(self, env):
        self.env = env
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
        self.programs = self.env.generate_random_actions(
            (GA.PROGRAM_POPULATION_SIZE, GA.PROGRAM_LENGTH)
        )

    def action(self):
        if self.programs.shape[1] == 0:
            return self.env.action_space.sample()
        for i in range(GA.GENERATIONS_NUM):
            scores, term_step = self.env.emulate_programs(self.programs)
            genes = Chromosomes(self.programs, scores, self.env.generate_random_actions)
            self.programs = genes.evolve()
        best_program_id = np.argmax(scores)
        best_program = self.programs[best_program_id]
        angle, power = best_program[0]
        if term_step[best_program_id] == 1:
            angle = -self.env.bot.angle
        self.programs = np.delete(self.programs, 0, axis=1)
        return angle, power

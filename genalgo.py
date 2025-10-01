import numpy as np
from numpy.typing import ArrayLike

from config import GeneticAlgorithmConfig as Conf


class Chromosomes:
    def __init__(self, chromos: ArrayLike, scores: ArrayLike, gene_generator):
        chromos, scores = map(np.asarray, (chromos, scores))
        if chromos.ndim != 3:
            raise ValueError("Chromosome should have 3 dimensions")
        if chromos.shape[0] < 2:
            raise ValueError("Population size should be at least 2")
        if chromos.shape[1] < 1:
            raise ValueError("Chromosome should have at least one gene")
        self._chromos = chromos
        self._scores = scores
        self.gene_generator = gene_generator

    def evolve(self):
        population_size = self._chromos.shape[0]
        chromo_len = self._chromos.shape[1]
        sort_order = np.argsort(-self._scores)
        # Preserve the elite
        elite_num = int(population_size * Conf.ELITE_PROPORTION)
        elite = self._chromos[sort_order[:elite_num], ...]
        # Create reproduction pool
        rng = np.random.default_rng()
        scores = self._scores[sort_order]
        # Probability to select adult for reproduction is proportional to the score
        probs = np.cumsum(scores / np.sum(scores))
        reproduction_pool_size = int(
            population_size * Conf.REPRODUCTION_POOL_PROPORTION
        )
        reproduction_pool = sort_order[
            np.argmax(rng.random(reproduction_pool_size)[:, None] < probs, axis=1)
        ]
        # Select parents from the reproduction pool
        children_num = population_size - elite_num
        n = children_num
        parents = np.empty((n, 2), dtype=int)
        replace = np.repeat(True, n)

        # If both parents are the same, we will be repacing with a new pair
        # And keep repeating until all pairs are different
        while n:
            new_choice = rng.choice(reproduction_pool, size=(n, 2))
            parents[replace, :] = new_choice
            replace[replace] = parents[replace, 0] == parents[replace, 1]
            n = np.count_nonzero(replace)
            # If the parents are the same for all the children,
            # maybe we only have one parent available,
            # so let's stop trying
            if n == children_num:
                break

        parents = self._chromos[parents, ...]

        # Betas are chosen for each gene
        # Betas control how much of the gene each parent contributes
        betas = (
            (1 + 2 * Conf.CROSSOVER_BETA) * rng.random(size=(children_num, chromo_len))
            - np.float64(Conf.CROSSOVER_BETA)
        )[..., None]

        children = (
            parents[:, 0, ...] * betas + parents[:, 1, ...] * (1 - betas)
        ).astype(int)

        # Mutate genes
        mutations = children[
            rng.random((children_num, chromo_len)) < Conf.MUTATION_RATE
        ]
        mutations = self.gene_generator(mutations.shape[:-1])

        programs = np.vstack((elite, children))
        return programs

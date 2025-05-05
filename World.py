from Bot import Bot

class World:
    def __init__(self, g, surface):
        self.g = g
        self.surface = surface
        self.landing_zone = None
        self.landing_zone = self.get_landing_zone()
    
    def get_surface(self):
        return self.surface
    
    def get_landing_zone(self):
        # If a landing zone is already defined, return it to avoid recalculating
        if self.landing_zone is not None:
            return self.landing_zone
        # Find the landing zone by checking the surface points
        # The landing zone is defined as the first point where the y stays the same
        py = self.surface[0][1]
        for i, y in enumerate(self.surface[1:,1], start=1):
            if y == py:
                return self.surface[i-1:i+1, :]
            py = y
        # If no landing zone is found, return the first two points as a fallback
        return self.surface[0:2, :]
    
    def place_bot(self, bot):
        self.bot = bot
    
    def emulate(self):
        np.random.Generator.integers(low=0, high=15, size=2, endpoint=True)
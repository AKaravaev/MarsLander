class Bot:
    def __init__(self, position, velocity, angle, power, fuel):
        self.position = position
        self.velocity = velocity
        self.angle = angle
        self.power = power
        self.fuel = fuel

    def __str__(self):
        return f"position={self.position}, velocity={self.velocity}, angle={self.angle}, power={self.power}, fuel={self.fuel}"

    def __repr__(self):
        return self.__str__()

class BotAction:
    def __init__(self, angle, power):
        self.angle = angle
        self.power = power

    def __str__(self):
        return f"BotAction(angle={self.angle}, power={self.power})"

    def __repr__(self):
        return self.__str__()

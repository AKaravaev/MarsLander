class DisplayConfig:
    SCREEN_SIZE = (700, 300)
    FPS = 30
    SURFACE_COLOR = "orange"
    BOT_COLOR = "white"
    PATH_COLOR = "red"
    TRACES_COLOR = "green"
    BEST_TRACE_COLOR = "blue"
    POWER_VECTOR_COLOR = "red"
    POWER_VECTOR_LENGTH = 5
    CRASH_MARKER_SIZE = 10
    CRASH_MARKER_COLOR = "red"
    CRASH_MARKER_WIDTH = 5
    INFO_SPACING = 15
    BOT_SIZE = 20
    FONT_NAME = "arial"
    FONT_SIZE = 16
    FONT_COLOR = "white"


class EnvConfig:
    GRAVITY = 3.711  # Mars gravity in m/s^2
    TEST_CASES = (
        # Episode II
        {
            # Easy on the right
            "surface": (
                (0, 100),
                (1000, 500),
                (1500, 1500),
                (3000, 1000),
                (4000, 150),
                (5500, 150),
                (6999, 800),
            ),
            "bot": {
                "position": (2500, 2700),
                "velocity": (0, 0),
                "fuel": 550,
                "angle": 0,
                "power": 0,
            },
        },
        {
            # Initial speed, correct side
            "surface": (
                (0, 100),
                (1000, 500),
                (1500, 100),
                (3000, 100),
                (3500, 500),
                (3700, 200),
                (5000, 1500),
                (5800, 300),
                (6000, 1000),
                (6999, 2000),
            ),
            "bot": {
                "position": (6500, 2800),
                "velocity": (-100, 0),
                "fuel": 600,
                "angle": 90,
                "power": 0,
            },
        },
        {
            # Initial speed, wrong side
            "surface": (
                (0, 100),
                (1000, 500),
                (1500, 1500),
                (3000, 1000),
                (4000, 150),
                (5500, 150),
                (6999, 800),
            ),
            "bot": {
                "position": (6500, 2800),
                "velocity": (-90, 0),
                "fuel": 750,
                "angle": 90,
                "power": 0,
            },
        },
        {
            # Deep canyon
            "surface": (
                (0, 1000),
                (300, 1500),
                (350, 1400),
                (500, 2000),
                (800, 1800),
                (1000, 2500),
                (1200, 2100),
                (1500, 2400),
                (2000, 1000),
                (2200, 500),
                (2500, 100),
                (2900, 800),
                (3000, 500),
                (3200, 1000),
                (3500, 2000),
                (3800, 800),
                (4000, 200),
                (5000, 200),
                (5500, 1500),
                (6999, 2800),
            ),
            "bot": {
                "position": (500, 2700),
                "velocity": (100, 0),
                "fuel": 800,
                "angle": -90,
                "power": 0,
            },
        },
        {
            # High ground
            "surface": (
                (0, 1000),
                (300, 1500),
                (350, 1400),
                (500, 2100),
                (1500, 2100),
                (2000, 200),
                (2500, 500),
                (2900, 300),
                (3000, 200),
                (3200, 1000),
                (3500, 500),
                (3800, 800),
                (4000, 200),
                (4200, 800),
                (4800, 600),
                (5000, 1200),
                (5500, 900),
                (6000, 500),
                (6500, 300),
                (6999, 500),
            ),
            "bot": {
                "position": (6500, 2700),
                "velocity": (-50, 0),
                "fuel": 1500,
                "angle": 90,
                "power": 0,
            },
        },
    )

    WORLD_SIZE = (7000, 3000)
    MAX_SURFACE_LEN = 30
    INITIAL_BOT_POSITION = (2500.0, 2699.0)
    INITIAL_BOT_VELOCITY = (0.0, 0.0)
    INITIAL_BOT_ANGLE = 0
    INITIAL_BOT_POWER = 0
    INITIAL_BOT_FUEL = 5501
    MAXIMUM_LANDING_VELOCITY = (20, 40)
    MAXIMUM_LANDING_ANGLE = 15
    MAX_ANGLE = 90
    MAX_POWER = 4
    MAX_ANGLE_CHANGE = 15
    MAX_POWER_CHANGE = 1


class GAConfig:
    PROGRAM_LENGTH = 150
    POPULATION_SIZE = 100
    GENERATIONS_NUM = 20
    ELITE_PROPORTION = 0.1
    REPRODUCTION_POOL_PROPORTION = 0.25
    CROSSOVER_BETA = 0.25
    MUTATION_RATE = 0.05
    LANDING_DISTANCE_REWARD = 100
    LANDING_FOUND_REWARD = 500
    LANDING_SUCCESS_REWARD = 1000

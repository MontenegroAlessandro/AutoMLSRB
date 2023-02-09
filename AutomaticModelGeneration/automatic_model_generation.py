# Here we implement different versions of Algorithm Selection Modules


# Class defining the Arm concept
class Arm:
    def __init__(self, model=None, tuner=None):
        assert model is not None, "[Error] No Model specified."
        self.model = model

        assert tuner is not None, "[Error] No Tuner specified."
        self.tuner = tuner


# Class implementing Algorithm Selection via Stochastic Rising Bandits
class StochasticRB:
    def __init__(self):
        pass

    def learn(self):
        pass

    def save_results(self):
        pass

    def reset(self):
        pass
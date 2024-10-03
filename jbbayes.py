### Jonathan Bostock

class LeastSquaresLinear():

    def __init__(self, x_data, y_data, seed=198534):

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        x = (x_data - np.mean(x_data)) / np.std(x_data)
        y = (y_data - np.mean(y_data)) / np.std(y_data)

        self.points = np.stack([x,y])

        self.generator = np.random.default_rng(seed=seed)

    def mcmc(self):

        lines = [self.generator.normal(size=(3))]

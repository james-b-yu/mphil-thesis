
# taken from HDM paper
class InstanceStandardize:
    def __init__(self):
        pass

    def fit(self, dataset):
        self.mean = dataset.mean([1,2]).unsqueeze(-1).unsqueeze(-1)
        self.std = dataset.std([1,2]).unsqueeze(-1).unsqueeze(-1)

    def transform(self, dataset):
        y = dataset
        y = y - self.mean
        y = y / self.std
        return y

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

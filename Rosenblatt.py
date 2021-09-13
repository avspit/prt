import numpy as np

'''
Персептрон Розенблатта.
Решение задач классификации для линейно-разделимых классов
'''
class Rosenblatt():

    def __init__(self, alfa=0.1, epochs=10):
        self.alfa = alfa
        self.epochs = epochs

    def fit(self, X, T):
        self.w = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.epochs):
            errors = 0
            for coords, classes in zip(X, T):
                update = self.alfa * (classes - self.predict(coords))
                self.w[1:] += update * coords
                self.w[0] += update
                errors += int(update != 0.0)
                print(_)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

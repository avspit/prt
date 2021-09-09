import numpy as np
import matplotlib.pyplot as plt

'''
Персептрон Розенблатта.
Решение задач классификации для линейно-разделимых классов
'''
class Rosenblatt():

    def __init__(self):
        # Веса модели
        self.weights = None
        # Количество ошибок на каждой стадии
        self.errors_ = list()

        self.bias = 0.0

    def H(self, x):
        return np.heaviside(x, 1).astype(np.int)

    def decision_function(self, coords):
        scores = coords.dot(self.weights) + self.bias
        return scores

    def predict(self, coords):
        return self.H(self.decision_function(coords))

    def calculate(self, coords, classes, maxiter=100):
        self.weights = np.zeros((coords.shape[1],))
        errors = list()
        for _ in range(maxiter):
            for crd, clz in zip(coords, classes):
                error = clz - self.predict(crd.reshape((1, -1)))
                if error != 0:
                    # --> Update the weights and bias.
                    self.weights += error * crd

            errors.append(abs(classes - self.predict(coords)).sum())

            if errors[-1] == 0:
                break

        self.errors_ = np.asarray(errors)

        return self

def init_data():
    coords = np.array([[0,0],[1,0],[0,1],[1,1]])
    classes = np.array([0,1,0,1])
    return coords, classes

def decision(model, x):
    return -(model.weights[0] * x + model.bias)/model.weights[1]

def main():
    coords, classes = init_data()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=classes,
        cmap="RdYlGn",
        s=40,
        edgecolors="k",
        alpha=0.5,
    )
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    model = Rosenblatt()
    model.calculate(coords, classes)

    x = np.linspace(*ax.get_xlim())

    ax.plot(x, decision(model, x), color="royalblue")

    ax.set_xlim(x.min(), x.max())

    plt.show()

if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt
from Rosenblatt import Rosenblatt

def decision(model, x):
    return -(model.weights[0] * x)/model.weights[1]

if __name__ == '__main__':
    perceptron = Rosenblatt(alfa=0.1, epochs=10)
    coordinates = np.array([[0,0], [1,0], [0,1], [1,1]])
    classes = np.array([0, 1, 0, 1])
    perceptron.fit(coordinates, classes)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.scatter(
        coordinates[:, 0], coordinates[:, 1],
        c=classes,
        cmap="RdYlGn",
        s=40,
        alpha=0.5,
    )
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    x = np.linspace(*ax.get_xlim(), num=4)

    ax.plot(x, decision(perceptron, x), color="royalblue")

    ax.set_xlim(x.min(), x.max())

    plt.show()


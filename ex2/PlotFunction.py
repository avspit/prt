import numpy as np
import matplotlib.pyplot as plt

class PlotFunctions():
    """
    Построение функций
    """

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def plot1(self):
        """
        Построение sin(x) на интервале [0,1]
        """
        x = np.linspace(0, np.pi, 100)
        y = np.sin(x)
        plt.subplot(2, 4, 1)
        plt.title('sin(x)')
        plt.plot(x, y)

    def plot2(self):
        """
        Построение f(x) = sigmoid(x)
        """
        x = np.linspace(-5, 5, 100)
        plt.subplot(2, 4, 2)
        plt.title('f(x) = sigmoid(x)')
        plt.plot(x, self.sigmoid(x))

    def plot3(self):
        """
        Построение -f(x+6)*6 - 6
        """
        x = np.linspace(-5, 5, 100)
        plt.subplot(2, 4, 3)
        plt.title('-f(x+6)*6-6')
        plt.plot(x, -(self.sigmoid(x)+6)*6 - 6)

    def plot4(self):
        """
        Построение 2f(x+3)+1
        """
        x = np.linspace(-5, 5, 100)
        plt.subplot(2, 4, 4)
        plt.title('2f(x+3)+1')
        plt.plot(x, 2*(self.sigmoid(x)+3)+1)

    def plot5(self):
        """
        Построение одного графика из двух
        """
        x = np.linspace(-5, 5, 100)
        plt.subplot(2, 4, 5)
        plt.title('Два в одном')
        plt.plot(x-5, self.sigmoid(x))
        plt.plot(x+5, self.sigmoid(-x))

    def plot6(self):
        """
        Аппроксимация sin(x)
        """
        x = np.linspace(-5, 5, 20)
        y = np.sin(x)
        t = np.polyfit(x, y, 7)
        f = np.poly1d(t)
        plt.subplot(2, 4, 6)
        plt.title('Аппроксимация sin(x)')
        plt.plot(x, y, 'o', x, f(x), 'r')

def main():
    plt.rcParams['figure.figsize'] = [15, 5]
    test = PlotFunctions()
    test.plot1()
    test.plot2()
    test.plot3()
    test.plot4()
    test.plot5()
    test.plot6()
    plt.show()

if __name__ == '__main__':
    main()

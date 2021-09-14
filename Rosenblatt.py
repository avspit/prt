import numpy as np

class Rosenblatt():
    """
    Персептрон Розенблатта.
    Решение задач классификации для линейно-разделимых классов
    """

    def __init__(self, alfa=0.1, epochs=10):
        """
        Инициализация

        Параметры
        ---------
        alfa : Темп обучения
        epochs : Количество проходов (эпох)
        """

        self.alfa = alfa
        self.epochs = epochs

    def fit(self, X, T):
        """
        Подгонка модели, используя тренировочные данные

        Параметры
        ---------
        X : Тренировочные данные
        T : Классы тренировочных данных
        """

        # Веса модели
        self.weights = np.zeros(1 + X.shape[1])
        # Количество ошибок в проходе
        self.errors = []

        for _ in range(self.epochs):
            errors = 0
            for coords, classes in zip(X, T):
                update = self.alfa * (classes - self.predict(coords))
                self.weights[1:] += update * coords
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        """
            Расчет коэффициента F(w*x)

            Параметры
            ---------
            X : Тренировочные данные
        """
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        """
            Расчет e

            Параметры
            ---------
            X : Тренировочные данные
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

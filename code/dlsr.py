import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import StandardScaler
from enum import Enum
import itertools

def animate_logistic_regression(train_datas, house_enum, feature_x, feature_y):
    # On garde seulement deux features + house
    datas = train_datas[[feature_x, feature_y, "Hogwarts House"]].copy()
    datas = datas.dropna()

    X = datas[[feature_x, feature_y]].values
    y = (datas["Hogwarts House"] == house_enum.value).astype(int).values

    # Ajout de la colonne de biais
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Initialisation de theta
    theta = np.zeros(X.shape[1])
    learning_rate = 0.3
    iterations = 4000

    thetas = []  # on sauvegarde theta à chaque étape pour l'animation

    # Entraînement + collecte des thetas
    for i in range(iterations):
        gradient = compute_gradient(X, y, theta)
        theta = theta - learning_rate * gradient
        if i % 2 == 0:
            thetas.append(theta.copy())

    # Création de l'animation
    fig, ax = plt.subplots()

    x_vals = X[:, 1]
    y_vals = X[:, 2]

    # points
    ax.scatter(x_vals[y == 0], y_vals[y == 0], color="grey", label="others", alpha=0.2)
    if house_enum.name == "Ravenclaw":
        ax.scatter(x_vals[y == 1], y_vals[y == 1], color="blue", label=house_enum.name)
    elif house_enum.name == "Slytherin":
        ax.scatter(x_vals[y == 1], y_vals[y == 1], color="green", label=house_enum.name)
    elif house_enum.name == "Gryffindor":
        ax.scatter(x_vals[y == 1], y_vals[y == 1], color="red", label=house_enum.name)
    elif house_enum.name == "Hufflepuff":
        ax.scatter(x_vals[y == 1], y_vals[y == 1], color="yellow", label=house_enum.name)
    

    #line, = ax.plot([], [], 'k--', label="Frontière")
    line, = ax.plot([], [], color="black", linewidth=2, label="Frontière")

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f"Évolution de la frontière - {house_enum.name}")
    ax.legend()

    def decision_boundary(theta, x_range):
        # Equation : theta0 + theta1*x1 + theta2*x2 = 0 -> x2 = -(theta0 + theta1*x1)/theta2
        return -(theta[0] + theta[1]*x_range) / theta[2]

    def update(i):
        x_vals_range = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_vals_range = decision_boundary(thetas[i], x_vals_range)
        line.set_data(x_vals_range, y_vals_range)
        return line,


    ani = animation.FuncAnimation(fig, update, frames=len(thetas), interval=100, blit=True, repeat=False)
    plt.show()


class Houses(Enum):
    Ravenclaw = 0
    Slytherin = 1
    Gryffindor = 2
    Hufflepuff = 3


def fill_nan_by_house_mean(train_datas):
    # Sélectionner les colonnes numériques (pour le calcul des moyennes)
    colonnes_numeriques = train_datas.select_dtypes(include='number').columns

    # Grouper par maison et remplacer les NaN par la moyenne de chaque groupe
    for colonne in colonnes_numeriques:
        train_datas[colonne] = train_datas.groupby('Hogwarts House')[colonne].transform(lambda x: x.fillna(x.mean()))

    return train_datas


def init():
    # Load and clean
    train_datas = pd.read_csv("../datasets/dataset_train.csv")
    train_datas.dropna(how='all', inplace=True)
    train_datas = fill_nan_by_house_mean(train_datas)
    train_datas.drop(["Index", "First Name", "Last Name", "Birthday", "Best Hand"], axis=1, inplace=True)

    # normalize
    colonnes_numeriques = train_datas.select_dtypes(include=np.number)
    train_datas[colonnes_numeriques.columns] = (colonnes_numeriques - colonnes_numeriques.mean()) / colonnes_numeriques.std() # TODO modifier les mean() std()

    # convert house string value to numerical value (ref House(Enum))
    train_datas["Hogwarts House"] = train_datas["Hogwarts House"].map({house.name: house.value for house in Houses})

    return train_datas


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = -1/m * np.sum(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8))
    return cost


def compute_gradient(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    gradient = np.dot(X.T, (h - y)) / m
    return gradient


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000):
    #cost_history = []
    for i in range(iterations):
        gradient = compute_gradient(X, y, theta)
        theta = theta - learning_rate * gradient
        cost = compute_cost(X, y, theta)
        #cost_history.append(cost)
        # Optionnel: Affiche le coût toutes les N itérations pour suivre la convergence
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost}")
    return theta#, cost_history


def logistic_regression_house(train_datas, house_enum):
    # data séparation
    X = train_datas.drop(["Hogwarts House"], axis=1).values # FEATURES (suppr rep for trainning)
    y = (train_datas["Hogwarts House"] == house_enum.value).astype(int).values # np.array maison cible == 1 others = 0 (boolean).astype(int) True False -> 1 - 0

    # Ajouter la colonne de biais dans X, plutôt que de le traiter à part, permet :
    # Une écriture compacte et vectorisée du modèle.
    # Une mise à jour unifiée de tous les paramètres lors de la descente de gradient.
    # Une simplification du code en évitant de gérer séparément le biais.
    # C'est donc une pratique standard en machine learning, qui rend les calculs plus simples et plus élégants sans introduire de perte de généralité ou de précision.
    X = np.hstack((np.ones((X.shape[0], 1)), X)) # ajoute les biais à X (hstack = horizontal concat)
    np.set_printoptions(threshold=np.inf)

    # Initialisation des paramètres theta (dimension : nombre de features + 1)
    theta = np.zeros(X.shape[1])

    # Appliquer la descente de gradient
    learning_rate = 0.05
    iterations = 4000
    #theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)
    theta = gradient_descent(X, y, theta, learning_rate, iterations)

    return theta


def schemas(train_datas):
    colonnes_numeriques = train_datas.select_dtypes(include=np.number).drop("Hogwarts House", axis=1).columns
    # Pour chaque combinaison unique de deux colonnes
    for feature_x, feature_y in itertools.combinations(colonnes_numeriques, 2):
        print(f"🧪 {feature_x} vs {feature_y}")
        print("Ravenclaw :")
        animate_logistic_regression(train_datas, Houses.Ravenclaw, feature_x, feature_y)
        print("Slytherin :")
        animate_logistic_regression(train_datas, Houses.Slytherin, feature_x, feature_y)
        print("Gryffindor :")
        animate_logistic_regression(train_datas, Houses.Gryffindor, feature_x, feature_y)
        print("Hufflepuff :")
        animate_logistic_regression(train_datas, Houses.Hufflepuff, feature_x, feature_y)


def main():
    train_datas = init()
    # TODO Select les bon paramètres pour entrainer chaque house
    # TODO donc -> statistique par maisons par matièresen plus du pairplot
    theta_ravenclaw = logistic_regression_house(train_datas, Houses.Ravenclaw)
    theta_slytherin = logistic_regression_house(train_datas, Houses.Slytherin)
    theta_gryffindor = logistic_regression_house(train_datas, Houses.Gryffindor)
    theta_hufflepuff = logistic_regression_house(train_datas, Houses.Hufflepuff)

    schemas(train_datas)


if __name__ == "__main__":
    main()

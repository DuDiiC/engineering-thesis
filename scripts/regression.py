import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

from joblib import dump

training_set_path = '../data/training_set/'
models_save_path = '../data/models/regression/'
results_save_path = '../results/regression/'

data = pd.read_pickle(training_set_path + 'training_set.pkl')
X = data.drop(['new_watchers'], axis=1)
y = data['new_watchers']
s = StandardScaler()
X = s.fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=42)

def regression_summary(model, model_name, model_details, X_train, X_test, y_train, y_test):
    print(f'\nModel: {model_name}\n{model_details}')

    model.fit(X_train, y_train)
    dump(model, f'{models_save_path}{model_name} - {model_details}.joblib'.replace('\n', ' '))

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    train_s = model.score(X_train, y_train)
    test_s = model.score(X_test, y_test)

    print(f'Średni błąd kwadratowy na danych testowych: {mse}')
    print(f'R^2 dla danych uczących: {train_s}')
    print(f'R^2 dla danych testowych: {test_s}')

    return y_pred, mse, train_s, test_s

def compare_model_versions(x_num, y_num, y, y_preds, names, title,
                           save_path, size_x=15, size_y=15):
    fig = plt.figure(figsize=(size_x, size_y))
    fig.subplots_adjust(hspace=0.40, wspace=0.40)
    fig.suptitle(title, fontsize=20)
    for y_pred, name, i in zip(y_preds, names, range(1, x_num*y_num+1)):
        ax = fig.add_subplot(x_num, y_num, i)
        p = Polynomial.fit(y.values, y_pred, 1)

        ax.scatter(y.values, y_pred, marker=".")
        ax.plot(*p.linspace(), color='r')
        ax.plot(y.values, y.values, color='g')

        ax.set_xlabel('Wartości rzeczywiste', fontsize=12)
        ax.set_ylabel('Wartości przewidywane', fontsize=12)
        ax.set_title(name, fontsize=12)
    plt.savefig(save_path)

def compare_mses(mses, names, title, save_path, size_x=20, size_y=10):
    barDim = 0.45
    plt.figure(figsize=(size_x, size_y))
    plt.rcParams.update({'font.size': 10})

    r = np.arange(len(names))
    plt.barh(r, mses, height=barDim)
    plt.xlabel('Średni błąd kwadratowy', fontsize=15)
    plt.yticks([label for label in range(len(names))], names)
    plt.title(title, fontsize=20)
    plt.savefig(save_path)

def compare_scores(train_scores, test_scores, names, title,
                   save_path, size_x=20, size_y=10):
    barDim = 0.45
    plt.figure(figsize=(size_x, size_y))
    plt.rcParams.update({'font.size': 10})

    r1 = np.arange(len(names))
    r2 = [x + barDim for x in r1]
    plt.barh(r1, train_scores, height=barDim,
             edgecolor='white', label='dane treningowe')
    plt.barh(r2, test_scores, height=barDim,
             edgecolor='white', label='dane testowe')
    plt.xlabel('Współczynnik determinacji $R^2$', fontsize=15)
    plt.yticks([r +  barDim/2 for r in range(len(names))], names)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=12)
    plt.savefig(save_path)

best_model_names = []
best_models_mses = []
best_models_train_scores = []
best_models_test_scores = []

# random forest model

models = [
    RandomForestRegressor(n_jobs=6, max_depth=3, n_estimators=100, random_state=42),
    RandomForestRegressor(n_jobs=6, max_depth=3, n_estimators=200, random_state=42),
    RandomForestRegressor(n_jobs=6, max_depth=3, n_estimators=400, random_state=42),
    RandomForestRegressor(n_jobs=6, max_depth=4, n_estimators=100, random_state=42),
    RandomForestRegressor(n_jobs=6, max_depth=4, n_estimators=200, random_state=42),
    RandomForestRegressor(n_jobs=6, max_depth=4, n_estimators=400, random_state=42),
    RandomForestRegressor(n_jobs=6, max_depth=5, n_estimators=100, random_state=42),
    RandomForestRegressor(n_jobs=6, max_depth=5, n_estimators=200, random_state=42),
    RandomForestRegressor(n_jobs=6, max_depth=5, n_estimators=400, random_state=42),
]

names = [
    'głębokość=3\nliczba drzew=100',
    'głębokość=3\nliczba drzew=200',
    'głębokość=3\nliczba drzew=400',
    'głębokość=4\nliczba drzew=100',
    'głębokość=4\nliczba drzew=200',
    'głębokość=4\nliczba drzew=400',
    'głębokość=5\nliczba drzew=100',
    'głębokość=5\nliczba drzew=200',
    'głębokość=5\nliczba drzew=400',
]

y_preds = []
mean_squared_errors = []
train_scores = []
test_scores = []

for forest, name in zip(models, names):
    y_pred, mse, train_s, test_s = \
        regression_summary(forest, 'Las losowy', name,
                           X_train, X_test, y_train, y_test)
    y_preds.append(y_pred)
    mean_squared_errors.append(mse)
    train_scores.append(train_s)
    test_scores.append(test_s)

compare_model_versions(3, 3, y_test, y_preds, names, 'RandomForestRegressor',
                      results_save_path + 'las losowy - porównanie parametrów.jpg')

compare_mses(mean_squared_errors, names, 'RandomForestRegressor',
              results_save_path + 'las losowy - MSE.jpg')

compare_scores(train_scores, test_scores, names, 'RandomForestRegressor',
              results_save_path + 'las losowy - R^2.jpg')

best_model_names.append('Las losowy')
best_models_mses.append(mean_squared_errors[5])
best_models_train_scores.append(train_scores[5])
best_models_test_scores.append(test_scores[5])

# decision tree model

models = [
    DecisionTreeRegressor(max_depth=2, random_state=42),
    DecisionTreeRegressor(max_depth=3, random_state=42),
    DecisionTreeRegressor(max_depth=4, random_state=42),
    DecisionTreeRegressor(max_depth=5, random_state=42),
]

names = [
    'głebokość=2',
    'głębokość=3',
    'głębokość=4',
    'głębokość=5',
]

y_preds = []
mean_squared_errors = []
train_scores = []
test_scores = []

for tree, name in zip(models, names):
    y_pred, mse, train_s, test_s = \
        regression_summary(tree, 'Drzewo decyzyjne', name, X_train, X_test, y_train, y_test)
    y_preds.append(y_pred)
    mean_squared_errors.append(mse)
    train_scores.append(train_s)
    test_scores.append(test_s)

compare_model_versions(2, 2, y_test, y_preds, names, 'DecisionTreeRegressor',
                      results_save_path + 'drzewo decyzyjne - porównanie parametrów.jpg', 15, 15)

compare_mses(mean_squared_errors, names, 'DecisionTreeRegressor',
              results_save_path + 'drzewo decyzyjne - MSE.jpg', 12, 5)

compare_scores(train_scores, test_scores, names, 'DecisionTreeRegressor',
              results_save_path + 'drzewo decyzyjne - R^2.jpg', 12, 5)

best_model_names.append('Drzewo decyzyjne')
best_models_mses.append(mean_squared_errors[2])
best_models_train_scores.append(train_scores[2])
best_models_test_scores.append(test_scores[2])

# gradient boosting model

models = [
    GradientBoostingRegressor(n_estimators=32, random_state=42),
    GradientBoostingRegressor(n_estimators=32, max_depth=2, random_state=42),
    GradientBoostingRegressor(n_estimators=64, max_depth=2, random_state=42),
    GradientBoostingRegressor(n_estimators=128, max_depth=2, random_state=42),
]

names = [
    'liczba drzew=32\ngłębokość=3',
    'liczba drzew=32\ngłębokość=2',
    'liczba drzew=64\ngłębokość=2',
    'liczba drzew=128\ngłębokość=2',
]

y_preds = []
mean_squared_errors = []
train_scores = []
test_scores = []

for gradient_boost, name in zip(models, names):
    y_pred, mse, train_s, test_s = \
        regression_summary(gradient_boost, 'Wzmocnienie gradientowe', name, X_train, X_test, y_train, y_test)
    y_preds.append(y_pred)
    mean_squared_errors.append(mse)
    train_scores.append(train_s)
    test_scores.append(test_s)

compare_model_versions(2, 2, y_test, y_preds, names, 'GradientBoostingRegressor',
                      results_save_path + 'wzmocnienie gradientowe - porównanie parametrów.jpg', 15, 15)

compare_mses(mean_squared_errors, names, 'GradientBoostingRegressor',
              results_save_path + 'wzmocnienie gradientowe - MSE.jpg', 12, 5)

compare_scores(train_scores, test_scores, names, 'GradientBoostingRegressor',
              results_save_path + 'wzmocnienie gradientowe - R^2.jpg', 12, 5)

best_model_names.append('Wzmocnienie\ngradientowe')
best_models_mses.append(mean_squared_errors[2])
best_models_train_scores.append(train_scores[2])
best_models_test_scores.append(test_scores[2])

# comparision

compare_mses(best_models_mses, best_model_names,
             'Porównanie najlepszych wersji modeli regresji',
              results_save_path + 'porównanie modeli - MSE.jpg', 12, 5)

compare_scores(best_models_train_scores, best_models_test_scores, best_model_names,
               'Porównanie najlepszych wersji modeli regresji',
               results_save_path + 'porównanie modeli - R^2.jpg', 12, 5)

# ---------------------------------------

# bad model example - linear regression

models = [
    LinearRegression(),
]

names = [
    'LinearRegression',
]

y_preds = []
mean_squared_errors = []
train_scores = []
test_scores = []

for model, name in zip(models, names):
    y_pred, mse, train_s, test_s = \
        regression_summary(model, 'Regresja liniowa', name, X_train, X_test, y_train, y_test)
    y_preds.append(y_pred)
    mean_squared_errors.append(mse)
    train_scores.append(train_s)
    test_scores.append(test_s)

compare_model_versions(1, 1, y_test, y_preds, names, 'Inne modele',
                      results_save_path + 'regresja liniowa - błędny model.jpg', 8, 8)
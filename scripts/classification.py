import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, \
                            confusion_matrix, plot_confusion_matrix

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import matplotlib.pyplot as plt

from joblib import dump

training_set_path = '../data/training_set/'
models_save_path = '../data/models/classification/'
results_save_path = '../results/classification/'

def visualize_data_sampling(x_num, y_num, X_trains, y_trains, group_names,
                            data_labels, save_path, size_x=20, size_y=20):
    fig = plt.figure(figsize=(size_x, size_y))
    fig.subplots_adjust(hspace=0.10, wspace=0.1)

    for X, y, data_label, i in zip(X_trains, y_trains, data_labels, range(1, x_num*y_num+1)):
        ax = fig.add_subplot(x_num, y_num, i)
        counter = Counter(y)
        print(f'{data_label} - {counter}')
        for label, _ in counter.items():
            row_ix = np.where(y == label)[0]
            ax.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        ax.legend(group_names)
        ax.set_title(f'Rozkład danych - {data_label}', fontsize=15)

    plt.savefig(save_path)

def classification_summary(model, name, data_label, labels,
                           X_train, X_test, y_train, y_test, save_path):
    model.fit(X_train, y_train)
    dump(model, f'{models_save_path}{name} - {data_label}.joblib')

    print(f'\nModel: {name}\nDane: {data_label}')
    y_pred = model.predict(X_test).astype(int)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    print(classification_report(y_test, y_pred, target_names=labels))
    return y_pred, cm

def compare_confusion_matrices(x_num, y_num, model_list, X_test, y_test, names,
                               model_name, data_labels, save_path, size_x=20, size_y=20):
    fig = plt.figure(figsize=(size_x, size_y))
    fig.subplots_adjust(hspace=0.10, wspace=0.25)
    for model, data_label, i in zip(model_list, data_labels, range(1, x_num*y_num+1)):
        ax = fig.add_subplot(x_num, y_num, i)
        plot_confusion_matrix(model, X_test, y_test,
                          normalize='true',
                          display_labels=names,
                          cmap=plt.cm.Blues,
                          ax=ax)
        ax.set_xlabel('Wartość przewidywana', fontsize=15)
        ax.set_ylabel('Wartość rzeczywista', fontsize=15)
        ax.set_title(f'{model_name}\n{data_label}', fontsize=12)
    plt.savefig(save_path)

def compare_classification_models(confusion_matrices, model_names, group_names,
                                  save_path, size_x=20, size_y=10):
    diagonal_values = []
    for matrix in confusion_matrices:
        diagonal_per_matrix = []
        for i in range(len(matrix)):
            diagonal_per_matrix.append(matrix[i][i])
        diagonal_values.append(diagonal_per_matrix[:])

    barDim = 0.25

    r = []
    r.append(np.arange(len(diagonal_values[0])))
    for i in range(1, len(diagonal_values)):
        r.append([x + barDim for x in r[i-1]])

    plt.figure(figsize=(size_x, size_y))

    for r_, values, model in zip(r, diagonal_values, model_names):
        plt.bar(r_, values, width=barDim, edgecolor='white', label=model)

    plt.xlabel('Klasa klasyfikacji', fontsize=15)
    plt.xticks([r_ + barDim for r_ in range(len(diagonal_values[0]))], group_names)

    plt.title('Porównanie skuteczności modeli klasyfikacji\n' +
              'dla poszczególnych klas',
              fontsize=20)
    plt.legend(fontsize=12)
    plt.savefig(save_path)

# prepare data

data = pd.read_pickle(training_set_path + 'training_set.pkl')

bins = (-np.inf, 0, 15, 50, 100, np.inf)
group_names = ['1.brak', '2.[1-15)', '3.[15-50)', '4.[50-100)', '5.[100+]']
data['new_watchers'] = pd.cut(data['new_watchers'], bins=bins, labels=group_names)
data['new_watchers'].value_counts()

labels = LabelEncoder()
data['new_watchers'] = labels.fit_transform(data['new_watchers'])

X = data.drop(['new_watchers'], axis=1)
y = data['new_watchers']

s = StandardScaler()
X = s.fit_transform(X)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=42)

X_trains = []
y_trains = []
data_labels = []

X_trains.append(X_train)
y_trains.append(y_train)
data_labels.append('brak usprawnień')

oversample = SMOTE(random_state=42)
X_train_oversample, y_train_oversample = \
    oversample.fit_resample(X_train, y_train)

X_trains.append(X_train_oversample)
y_trains.append(y_train_oversample)
data_labels.append('oversampling')

undersample = RandomUnderSampler(random_state=42)
X_train_undersample, y_train_undersample = \
    undersample.fit_resample(X_train, y_train)

X_trains.append(X_train_undersample)
y_trains.append(y_train_undersample)
data_labels.append('undersampling')

n_samples = y_train.value_counts().nlargest(2).iloc[1] * 10

oversample = SMOTE(sampling_strategy={
                        1: n_samples,
                        2: n_samples,
                        3: n_samples,
                        4: n_samples,
                    }, random_state=42)
X_train_mix, y_train_mix = \
        oversample.fit_resample(X_train, y_train)

undersample = RandomUnderSampler(random_state=42)
X_train_mix, y_train_mix = \
    undersample.fit_resample(X_train_mix, y_train_mix)

X_trains.append(X_train_mix)
y_trains.append(y_train_mix)
data_labels.append('kombinacja')

visualize_data_sampling(2, 2, X_trains, y_trains, group_names, data_labels,
                        results_save_path + 'rozkład danych.jpg')

# training models

models = [
    [
        DecisionTreeClassifier(random_state=42),
        DecisionTreeClassifier(random_state=42),
        DecisionTreeClassifier(random_state=42),
        DecisionTreeClassifier(random_state=42),
    ],
    [
        RandomForestClassifier(n_jobs=6, random_state=42),
        RandomForestClassifier(n_jobs=6, random_state=42),
        RandomForestClassifier(n_jobs=6, random_state=42),
        RandomForestClassifier(n_jobs=6, random_state=42),
    ],
    [
        HistGradientBoostingClassifier(max_depth=4, random_state=42),
        HistGradientBoostingClassifier(max_depth=4, random_state=42),
        HistGradientBoostingClassifier(max_depth=4, random_state=42),
        HistGradientBoostingClassifier(max_depth=4, random_state=42),
    ]
]

names = [
    'Drzewo decyzyjne',
    'Las losowy',
    'Wzmocnienie gradientowe',
]

y_preds = []
confusion_matrices = []

for model_list, name in zip(models, names):
    y_pred_list = []
    confusion_matrice_list = []
    for model, x, y, data_label in zip(model_list, X_trains, y_trains, data_labels):
        y_pred, cm = \
            classification_summary(model, name, data_label, group_names,
                x, X_test, y, y_test,
                f'{results_save_path}{name} {data_label} - macierz.jpg')
        y_pred_list.append(y_pred)
        confusion_matrice_list.append(cm)
    y_preds.append(y_pred_list[:])
    confusion_matrices.append(confusion_matrice_list[:])

# results visualization

best_models_confusion_matrices = []

for model_list, name, confusion_matrices_list in zip(models, names, confusion_matrices):

    compare_confusion_matrices(2, 2, model_list, X_test, y_test,
                                group_names, name, data_labels,
                                f'{results_save_path}macierze koincydencji - {name}.png')
    # najlepszy jest zawsze z over + under
    best_models_confusion_matrices.append(confusion_matrices_list[len(confusion_matrices_list)-1])

compare_classification_models(best_models_confusion_matrices, names, group_names,
                              results_save_path + 'porównanie skuteczności modeli klasyfikacji.png')
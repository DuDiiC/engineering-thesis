### Projekt rozwijany w ramach pracy inżynierskiej

# Porównanie metod uczenia maszynowego na przykładzie predykcji popularności projektów open source na platformie Github.

## Użyte technologie

| Technologia | wersja |
| :---: | :---: |
| Python | 3.7.6 |
| Conda | 4.9.2 |
| IPython | 7.12.0 |
| Pandas | 1.1.1 |
| Scikit-learn | 0.21.3 |
| Matplotlib | 3.1.1 |
| SQLAlchemy | 1.3.13 |
| Imbalanced-learn | 0.7.0 |

### Wartość przewidywana:

**Liczba nowych obserwujących w miesiącu** - dla regresji przewiduję konkretną wartość, dla klasyfikacji przewiduję należenie do jednej ze zdefiniowanych klas:

| Klasa | Liczba nowych obserwujących w miesiącu |
| :---: | :---: |
| brak | `0` |
| bardzo mało | `[1; 20)` |
| mało | `[20; 50)` |
| dużo | `[50; 100)` |
| bardzo dużo | `100+` |

## Użyte modele uczenia maszynowego:

| REGRESJA | KLASYFIKACJA |
| :---: | :---: |
| `DecisionTreeRegressor` | `DecisionTreeClassifier` |
| `RandomForestRegressor` | `RandomForestClassifier` |
| `GradientBoostingRegressor` | `GradientBoostingClassifier` |

## Uzyskane wyniki

### Regresja

![MSE](results/regression/porównanie%20modeli%20-%20MSE.jpg)
![R^2](results/regression/porównanie%20modeli%20-%20R%5E2.jpg)

### Klasyfikacja

![MATRICES](results/classification/porównanie%20skuteczności%20modeli%20klasyfikacji.png)

## Odtworzenie procesu uczenia z pominięciem etapu pobrania danych z bazy:

- do katalogu `data/from_db/` rozpakować archiwum z plikami *.pkl* do pobrania pod adresem https://www-users.mat.umk.pl/~maciejdudek/from_db.zip (dostępne z sieci UMK)
- przy pomocy odpowiedniego narzędzia uruchomić skrypty pythonowe w katalogu `notebooks/` w następującej kolejności:

  - *data_mining.ipynb*
  - *data_preparing.ipynb*
  - *regression.ipynb*
  - *classification.ipynb*

Możliwe, że niezbędne będzie wcześniejsze utworzenie struktury katalogu `data/`.

**Dodatkowo pobranie pełnego archiwum zawierającego wypełnione danymi katalogi `data/` oraz `results/` możliwe jest pod adresem:** https://www-users.mat.umk.pl/~maciejdudek/implementation_with_data.zip
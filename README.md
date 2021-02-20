# Comparison of machine learning methods based on the prediction of the popularity of open source projects on the GitHub platform

## Conspect

The theoretical part describes the basics of machine learning and selected supervised learning algorithms: decision trees, random forests and gradient boosting. The training set from the GHTorrent website was also described.

In the practical part, on the example of predicting the popularity of projects on the GitHub platform, the full process related to working on a machine learning project was carried out – data mining, preparation of learning data, training of selected models and analysis of the obtained results. The machine learning models and their performance in the studied case were also compared.

Implementation was done using the Python programming language and its popular libraries, mainly scikit-learn and pandas.

keywords: *machine learning, supervised learning, predictions, Python, pandas, scikit-learn, , GtiHub, GHTorrent*

## Technologies

|  | version |
| :---: | :---: |
| Python | 3.7.6 |
| Conda | 4.9.2 |
| IPython | 7.12.0 |
| Pandas | 1.1.1 |
| Scikit-learn | 0.21.3 |
| Matplotlib | 3.1.1 |
| SQLAlchemy | 1.3.13 |
| Imbalanced-learn | 0.7.0 |

### Predicted values:

**The number of new stars in the given month**

- for regression predicting a specific value
- for classification, predicting one of the predefined classes:

| class | the number of new stars in the given month |
| :---: | :---: |
| 0 | `0` |
| 1 | `[1; 20)` |
| 2 | `[20; 50)` |
| 3 | `[50; 100)` |
| 4 | `100+` |

## Machine learning models:

| REGRESSION | CLASSIFICATION |
| :---: | :---: |
| `DecisionTreeRegressor` | `DecisionTreeClassifier` |
| `RandomForestRegressor` | `RandomForestClassifier` |
| `GradientBoostingRegressor` | `GradientBoostingClassifier` |

## Results

### Regression

![MSE](results/regression/porównanie%20modeli%20-%20MSE.jpg)
![R^2](results/regression/porównanie%20modeli%20-%20R%5E2.jpg)

### Classification

![MATRICES](results/classification/porównanie%20skuteczności%20modeli%20klasyfikacji.png)

---

*For more details, please contact me by [e-mail](mailto:maciejdudekdev@gmail.com).*
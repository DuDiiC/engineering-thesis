> Readme jest nieaktualne, w trakcie przygotowywania. Sam proces implementacyjny został zakończony, istotne są jedynie pliki w głównym katalogu `notebooks`.

### Aplikacja rozwijana w ramach pracy inżynierskiej o roboczym tytule:

# Wykorzystanie metod uczenia maszynowego na przykładzie predykcji popularności projektów open source na platformie GitHub

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

## DOCELOWY DATAFRAME DO UCZENIA

### Grupowanie danych w szeregi czasowe

Wartości poszczególnych danych grupowane są ze względu na ID repozytorium i a wartości sumowane w odniesieniu dla każdego miesiąca.

- w przypadku obserwujących, commitów i issues operacje będą bardzo podobne:

  - pobranie danych do `DataFrame`
  - dołożenie kolumn miesiąca (`month`) i roku (`year`) na podstawie kolumny `created_at`
  - pogrupowanie danych ze względu na ID repozytorium, którego dotyczy, miesiąc oraz rok
  - wyliczenie łącznej sumy poszczególnych elementów dla każdego z projektów

- w przypadku pull requestów, najpierw potrzebne jest dołączenie informacji o historii, a następnie rozbicie na pull requesty zmergowane i odrzucone oraz poszczególne operacje

- dla komentarzy commitów i issues potrzebne jest najpierw dołączenie informacji, którego projektu dotyczą

Dodatkowo po procesie grupowania zmienione zostaje nazewnictwo kolumn, aby je ustandaryzować i łatwiej zmergować:

- ID repozytorium przechowywane w kolumnie `project_id`
- ilość nowych danych przechowywana w kolumnie `new_(nazwa_elementu)`
- łączna ilość danych od początku projektu przechowywana w kolumnie `total_(nazwa_elementu)`

| parametr | opis |
| ---: | :--- |
| `project_id` | id projektu |
| `months_from_create` | oznacza wartość, który jest to miesiąc od utworzenia projektu |
| `year` | rok wyciągnięty z daty dla danych |
| `month` | miesiąc wyciągnięty z daty dla danych |
| `language` | język programowania projektu |
| `new_commits` | liczba nowych commitów w miesiącu |
| `total_commits` | liczba wszystkich commitów od początku projektu |
| `unique_committers` | liczba unikalnych commitujących w miesiącu |
| `total_unique_committers` | liczba wszystkich unikalnych commitujących od początku projektu |
| `new_commit_comments` | liczba nowych komentarzy commitów w miesiącu |
| `total_commit_comments` | liczba wszystkich komentarzy commitów od początku projektu |
| `new_issues` | liczba nowych issues w miesiącu |
| `total_issues` | liczba wszystkich issues od początku projektu |
| `new_issue_comments` | liczba nowych kommentarzy commitów w miesiącu |
| `total_issue_comments` | liczba wszystkich komentarzy issues od początku projektu |
| `new_opened_pull_requests_to_merge` | liczba nowootwartych pull requestów w miesiącu, które będą zmergowane |
| `new_merged_pull_requests` | liczba zmergowanych pull requestów w miesiącu |  |
| `new_closed_merged_pull_requests` | liczba zamkniętych pull requestów w miesiącu, które zostały zmergowane |
| `total_merged_pull_requests` | liczba wszystkich zmergowanych pull requestów od początku projektu |
| `new_opened_pull_requests_to_discard` | liczba nowootwartych pull requestów w miesiącu, które nie będą zmergowane |
| `new_closed_unmerged_pull_requests` | liczba zamkniętych pull requestów w miesiącu, które nie będą zmergowane |
| `total_unmerged_pull_requests` | liczba wszystich niezmergowanych pull requestów od początku projektu, które zostały zamknięte |
| `new_pull_request_comments` | liczba komentarzy pull requestów w miesiącu |
| `total_pull_request_comments` | liczba wszystkich komentarzy pull requestów od początku projektu |
| `total_watchers` | łączna liczba obserwujących od początku projektu (nie licząc aktualnego miesiąca) |

### Wartość przewidywana

`new_watchers` - liczba nowych obserwujących w miesiącu

Dla regresji, przewiduję konkretną wartość, dla klasyfikacji przewiduję należenie do jednej ze zdefiniowanych klas:

| Klasa | Liczba nowych obserwujących w miesiącu |
| :---: | :---: |
| brak | `0` |
| bardzo mało | `[1; 20)` |
| mało | `[20; 50)` |
| dużo | `[50; 100)` |
| bardzo dużo | `100+` |

## Użyte algorytmy uczenia maszynowego:

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

# TMP

---
---
---

## Struktura projektu

Struktura została oparta w dużej mierze na artykule na portalu [medium.com](https://medium.com/swlh/how-to-structure-a-python-based-data-science-project-a-short-tutorial-for-beginners-7e00bff14f56) autorstwa Misha Berrien, została jednak odpowiednio uproszczona ze względu na niski poziom skomplikowania tworzonej aplikacji. Drugi alternatywny (niewykorzystany model) znajduje się w [tym artykule](https://towardsdatascience.com/manage-your-data-science-project-structure-in-early-stage-95f91d4d0600).

Struktura przedstawia się następująco:

- katalog `data` zawiera dane w formacie `.pkl` przygotowane do wczytywania bezpośrednio do `DataFrame`, w szczególności poszczególne katalogi:

  - `01_data_from_db` - dane poszczególnych tabel pobrane z analizowanej bazy danych, po wstępnym oczyszczeniu i usunięciu nadmiarowych danych
  - ...,
  - ...

- katalog `notebooks` zawiera notatniki `*.ipynb`, które używane są w celu eksploracji, podejmowania prób i konstruowania finalnych skryptów, które zostaną utrwalone w końcowej wersji aplikacji,

- katalog `src` jest **pakietem** w rozumieniu pythona, zawiera skrypty właściwej wersji projektu, przy czym szczegółowo:

  - `01_data_mining` - skrypty do czytania/pisania danych z bazy
  - `02_data_cleaning` - skrypty do czyszczenia danych, ich transformacje do danych pośrednich (usunięcie błędnych danych, zmi ana typów, optymalizacja rozmiaru itd.)
  - `03_data_processing` - skrypty przekształcające dane pośrednie w dane wejściowe dla uczenia (głównie grupowanie w szeregi czasowe), podział na dane uczące i testowe

    > //TODO: przyjrzenie się jakie dane przyjmują algorytmy uczące w scikit-learn)

  - `04_modelling` - skrypty do trenowania modeli przy użyciu poszczególnych algorytmów
  - `05_model_evaluation` - sprawdzanie działania modelu na danych testowych
  - `06_reporting` skrypty do tworzenia tabel i wykresów z raportami zbiorczymi dotyczącymi wyników przeprowadzonego eksperymentu
  - `utils` - skrypty pomocnicze używane w całym projekcie, zebrane zbiorczo
  - plik `config.py` zawierające stałe, ścieżki i inne zmienne używane w całym projekcie

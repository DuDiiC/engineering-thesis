### Aplikacja rozwijana w ramach pracy inżynierskiej o roboczym tytule:

# Wykorzystanie metod uczenia maszynowego na przykładzie predykcji popularności projektów open source na platformie GitHub

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


## Czyszczenie danych

Czyli które informacje z bazy danych zostają z poszczególnych tabel.

### Projekty:

| `project_id` | `name` | `language` | `created_at` |
| :---: | :---: | :---: | :---: |
| id projektu | nazwa projektu | użyty język programowania | data utworzenia |

> może dodać:
>   - `forked_from` (czy to nowy projekt, czy forkowany)
>   - `description` (co by to miało dać?)
>   - `owner_id` - można by sprawdzać, czy w udziela się w komentarzach commitów/issues/pull requestów

### Commity:

| `commit_id` | `committer_id` | `project_id` | `created_at` |
| :---: | :---: | :---: | :---: |
| id commita | id autora commita | id projektu | data utworzenia |

### Komentarze commitów

| `commit_comment_id` | `commit_id` | `body` | `created_at` |
| :---: | :---: | :---: | :---: |
| id komentarza | id commita | treść komentarza | data utworzenia |

> może dodać id komentującego (sprawdzanie czy jest właścicielem repo)

### Issues

| `issue_id` | `project_id` | `created_at` |
| :---: | :---: | :---: |
| id issue | id projektu | data utworzenia |

### Komentarze issues

| `issue_id` | `comment_id` | `created_at` |
| :---: | :---: | :---: |
| id issue | id komentarza | data utworzenia |

> może dodać id komentującego (sprawdzanie czy jest właścicielem repo)

### Pull requesty

| `pull_request_id` | `project_id` | `merged` |
| :---: | :---: | :---: |
| id pull requesta | id projektu | czy został zmergowany |

> może dodać id użytkownika wykonującego pull request (liczba unikalnych użytkowników na miesiąc/sprawdzanie czy jest właściwielem czy nie?)

### Historia pull requestów

| `pull_request_history_id` | `pull_request_id` | `created_at` | `action` |
| :---: | :---: | :---: | :---: |
| id wpisu historii | id pull requesta | data utworzenia | akcja (`opened`/`closed`/`merged`) |

### Komentarze pull requestów

| `pull_request_id` | `comment_id` | `body` | `created_at` |
| :---: | :---: | :---: | :---: |
| id pull requesta | id komentarza | treść komentarza | data utworzenia |

> może dodać id użytkownika piszącego komentarz (liczba unikalnych użytkowników na miesiac/sprawdzenie czy jest właścicielem?)

### Pull requesty z historią

| `pull_request_id` | `project_id` | `merged` | `pull_request_history_id` | `created_at` | `action` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| id pull requesta | id projektu | czy został zmergowany | id wpisu historii | data utworzenia wpisu | akcja (`opened`/`closed`/`merged`) |

### Obserwujący (gwiazdkujący)

| `project_id` | `created_at` |
| :---: | :---: |
| id projektu | data utworzenia |

## DOCELOWY DATAFRAME DO UCZENIA

| parametr | opis | status |
| :---: | :---: | :---: |
| `project_id` | id projektu | <span style="color:green">IMPLEMENTED</span> |
| `year` | rok wyciągnięty z daty | <span style="color:green">IMPLEMENTED</span> |
| `month` | miesiąc wyciągnięty z daty | <span style="color:green">IMPLEMENTED</span> |
| `language` | język programowania projektu | <span style="color:yellow">NOT_IMPLEMENTED</span> |
| `new_commits` | liczba nowych commitów w miesiącu | <span style="color:yellow">NOT_IMPLEMENTED</span> |
| `total_commits` (?) | liczba wszystkich commitów od początku projektu | <span style="color:blue">NEEDED?</span>
| `new_unique_commiters` | liczba unikalnych commitujących w miesiącu | <span style="color:yellow">NOT_IMPLEMENTED</span> |
| `total_unique_commiters` (?) | liczba wszystkich unikalnych commitujących od początku projektu | <span style="color:blue">NEEDED?</span> |
| `new_commit_comments` | liczba nowych komentarzy commitów w miesiącu (to może jakoś rozdzielić ze słowami kluczowymi?) | <span style="color:red">do przemyślenia</span> |
| `total_commit_comments` (?) | liczba wszystkich komentarzy commitów od początku projektu | <span style="color:blue">NEEDED?</span> |
| `new_issues` | liczba nowych issues w miesiącu | <span style="color:yellow">NOT_IMPLEMENTED</span> |
| `total_issues` (?) | liczba wszystkich issues od początku projektu | <span style="color:blue">NEEDED?</span>
| `new_issue_comments` | liczba nowych kommentarzy commitów w miesiącu | <span style="color:yellow">NOT_IMPLEMENTED</span> |
| `total_issue_comments` (?) | liczba wszystkich komentarzy issues od początku projektu | <span style="color:blue">NEEDED?</span> |
| `new_opened_merged_pull_requests` | liczba nowootwartych pull requestów w miesiącu, które będą zmergowane | <span style="color:yellow">NOT_IMPLEMENTED</span> |
| `new_merged_pull_requests` | liczba zmergowanych pull requestów w miesiącu | <span style="color:yellow">NOT_IMPLEMENTED</span> |
| `new_closed_merged_pull_requests` | liczba zamkniętych pull requestów w miesiącu, które zostały zmergowane | <span style="color:yellow">NOT_IMPLEMENTED</span> |
| `total_merged_pull_requests` (?) | liczba wszystkich zmergowanych pull requestów od początku projektu | <span style="color:blue">NEEDED?</span> |
| `new_opened_unmerged_pull_requests` | liczba nowootwartych pull requestów w miesiącu, które nie będą zmergowane | <span style="color:yellow">NOT_IMPLEMENTED</span> |
| `new_closed_unmerged_pull_requests` | liczba zamkniętych pull requestów w miesiącu, które nie będą zmergowane | <span style="color:yellow">NOT_IMPLEMENTED</span> |
| `total_unmerged_pull_requests` (?) | liczba wszystich niezmergowanych pull requestów od początku projektu, które zostały zamknięte | <span style="color:blue">NEEDED?</span> |
| `new_pull_request_comments` | liczba komentarzy pull requestów w miesiącu (to może jakoś rozdzielić ze słowami kluczowymi?) | <span style="color:red">do przemyślenia</span> |
| `total_pull_request_comments` (?) | liczba wszystkich komentarzy pull requestów od początku projektu | <span style="color:blue">NEEDED?</span> |
| `total_watchers` | łączna liczba obserwujących od początku projektu (nie licząc aktualnego miesiąca) | <span style="color:yellow">NOT_IMPLEMENTED</span>

### Wartość przewidywana
`new_watchers` - liczba nowych obserwujących w miesiącu <span style="color:yellow">NOT_IMPLEMENTED</span>

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
"""
Dokumentacja dla programu rekomendacji cen telewizorów.

Ten program wykorzystuje logikę rozmytą do rekomendacji cen telewizorów
na podstawie trzech atrybutów: rozmiaru ekranu, rozdzielczości i częstotliwości odświeżania.

Autorzy: Wiktor Krieger i Sebastian Augustyniak

Użycie:
1. Uruchom program, który wczytuje dane z pliku Excel (telewizory_ceneo.xlsx znajdujący się w tym samym katalogu).
2. Tworzy trzy zmienne rozmyte: 'rozmiar_ekranu', 'rozdzielczosc' i 'czestotliwosc_odswiezania'.
3. Określa trzy zbiory rozmyte dla każdej z tych zmiennych.
4. Tworzy trzy zmienne rozmyte dla ceny.
5. Określa trzy reguły rozmyte, które łączą zmienne wejściowe z ceną.
6. Tworzy system sterowania rekomendacji ceny.
7. Pozwala użytkownikowi wprowadzić dane wejściowe: rozmiar ekranu, rozdzielczość w pionie i częstotliwość odświeżania.
8. Oblicza rekomendowaną cenę telewizora na podstawie podanych danych wejściowych.
9. Wyświetla wynik w postaci rekomendowanej ceny telewizora.
10. Tworzy wykresy dla zmiennych rozmytych i wyświetla je.

Przykład użycia:
1. Uruchom program i podaj wymagane dane wejściowe.
2. Program obliczy rekomendowaną cenę telewizora na podstawie podanych danych.

Uwagi:
Niestety za późno zabraliśmy się za projekt w związku z czym nie udało nam się problemu związanego
z tym że wszystkie wyniki wachają się w granicach 4000 zł, możliwe że zawiniło pobranie nieaktualnych danych
z ceneo albo nieumiejętne użycie logiki rozmytej

Wymagane biblioteki:
- pandas
- scikit-fuzzy
- skfuzzy.control
- matplotlib.pyplot
"""

import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

df = pd.read_excel('telewizory_ceneo.xlsx')

rozmiar_ekranu = ctrl.Antecedent(universe=(df['Rozmiar ekranu'].min(), df['Rozmiar ekranu'].max()),
                                 label='Rozmiar ekranu')
rozdzielczosc = ctrl.Antecedent(universe=(df['Rozdzielczość'].min(), df['Rozdzielczość'].max()), label='Rozdzielczość')
czestotliwosc_odswiezania = ctrl.Antecedent(
    universe=(df['Częstotliwość odświeżania'].min(), df['Częstotliwość odświeżania'].max()),
    label='Częstotliwość odświeżania')
cena = ctrl.Consequent(universe=(df['Cena'].min(), df['Cena'].max()), label='Cena')

rozmiar_ekranu['mały'] = fuzz.trimf(rozmiar_ekranu.universe, [0, 20, 40])
rozmiar_ekranu['średni'] = fuzz.trimf(rozmiar_ekranu.universe, [30, 50, 70])
rozmiar_ekranu['duży'] = fuzz.trimf(rozmiar_ekranu.universe, [60, 80, 100])

rozdzielczosc['niska'] = fuzz.trimf(rozdzielczosc.universe, [0, 720, 1080])
rozdzielczosc['średnia'] = fuzz.trimf(rozdzielczosc.universe, [960, 1080, 2160])
rozdzielczosc['wysoka'] = fuzz.trimf(rozdzielczosc.universe, [1920, 3840, 3840])

czestotliwosc_odswiezania['niska'] = fuzz.trimf(czestotliwosc_odswiezania.universe, [0, 30, 50])
czestotliwosc_odswiezania['średnia'] = fuzz.trimf(czestotliwosc_odswiezania.universe, [45, 60, 70])
czestotliwosc_odswiezania['wysoka'] = fuzz.trimf(czestotliwosc_odswiezania.universe, [65, 120, 240])

cena['niska'] = fuzz.trimf(cena.universe, [0, 2000, 3000])
cena['średnia'] = fuzz.trimf(cena.universe, [2500, 4000, 6000])
cena['wysoka'] = fuzz.trimf(cena.universe, [5500, 8000, 10000])

regula1 = ctrl.Rule(rozmiar_ekranu['mały'] | rozdzielczosc['niska'] | czestotliwosc_odswiezania['niska'], cena['niska'])
regula2 = ctrl.Rule(rozmiar_ekranu['średni'] & rozdzielczosc['średnia'] & czestotliwosc_odswiezania['średnia'],
                    cena['średnia'])
regula3 = ctrl.Rule(rozmiar_ekranu['duży'] & (rozdzielczosc['wysoka'] | czestotliwosc_odswiezania['wysoka']),
                    cena['wysoka'])

rekomendacja_ceny = ctrl.ControlSystem([regula1, regula2, regula3])
rekomendacja_ceny_ctrl = ctrl.ControlSystemSimulation(rekomendacja_ceny)

rozmiar_ekranu_input = float(input("Podaj rozmiar ekranu (w calach): "))
rozdzielczosc_input = int(input("Podaj rozdzielczość pionową (720, 1080, 2160, 3840 itp.): "))
czestotliwosc_odswiezania_input = int(input("Podaj częstotliwość odświeżania (Hz): "))

rekomendacja_ceny_ctrl.input['Rozmiar ekranu'] = rozmiar_ekranu_input
rekomendacja_ceny_ctrl.input['Rozdzielczość'] = rozdzielczosc_input
rekomendacja_ceny_ctrl.input['Częstotliwość odświeżania'] = czestotliwosc_odswiezania_input

rekomendacja_ceny_ctrl.compute()

print("Rekomendowana cena telewizora: {:.2f} zł".format(rekomendacja_ceny_ctrl.output['Cena']))

rozmiar_ekranu.view()
rozdzielczosc.view()
czestotliwosc_odswiezania.view()
cena.view()

plt.show()

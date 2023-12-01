"""
Abalone Rings Prediction Program

Autorzy: Wiktor Krieger & Sebastian Augustyniak

Opis problemu:
Program ten ma na celu przewidywanie liczby pierścieni u małży na podstawie różnych cech fizycznych

Wyjaśnienie kroków:

1. Wczytywanie danych z pliku CSV ('abalone.data.csv').
2. Przygotowanie danych:
   - Wybierane są odpowiednie cechy
   - Kategoria płci jest przekształcana za pomocą kodowania one-hot (pd.get_dummies).
   - Brakujące dane są uzupełniane średnimi wartościami (SimpleImputer).

3. Podział danych na zestawy treningowe i testowe (train_test_split).
4. Uczenie modelu drzewa decyzyjnego:
   - Uczenie modelu drzewa decyzyjnego z ustawieniami parametrów.
   - Przewidywanie na zbiorze testowym.
   - Ocena dokładności i generowanie raportu klasyfikacji.

5. Skalowanie danych i uczenie modelu SVM:
   - Skalowanie danych za pomocą StandardScaler.
   - Podział danych skalowanych na zestawy treningowe i testowe.
   - Uczenie modelu SVM z ustawieniami parametrów.
   - Przewidywanie na zbiorze testowym.
   - Ocena dokładności i generowanie raportu klasyfikacji.

Instrukcja użycia:
   - Upewnij się, że wymagane biblioteki (pandas, sklearn) są zainstalowane.
   - Upewnij się, że plik 'abalone.data.csv' jest dostępny.
   - Uruchom program w środowisku Python.
   - Program wypisze dokładność oraz raport klasyfikacji dla obu modeli: Drzewa Decyzyjnego i SVM.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

abalone_data = pd.read_csv('abalone.data.csv')

features = abalone_data[['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']]
labels = abalone_data['Rings']

features = pd.get_dummies(features, columns=['Sex'], drop_first=True)

imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_imputed, labels, test_size=0.2, random_state=42)

decision_tree_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=5, min_samples_leaf=5)
decision_tree_model.fit(X_train, y_train)

predictions_tree = decision_tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, predictions_tree)
print(f'Decision Tree Accuracy: {accuracy_tree:.2f}')
print('\nDecision Tree Classification Report:')
print(classification_report(y_test, predictions_tree, zero_division=1))

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

predictions_svm = svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, predictions_svm)
print(f'SVM Accuracy: {accuracy_svm:.2f}')
print('\nSVM Classification Report:')

print(classification_report(y_test, predictions_svm, zero_division=1))

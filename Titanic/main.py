"""
Prognozowanie przeżycia na Titanicu

Autorzy: Sebastian Augustyniak & Wiktor Krieger

Opis problemu:
Program wykorzystuje modele uczenia maszynowego, takie jak Klasyfikator Drzewa Decyzyjnego i Maszyna Wektorów Nośnych (SVM),
do przewidzenia przeżycia pasażerów na pokładzie Titanica na podstawie określonych cech.

Wyjaśnienie kroków:
1. Program wczytuje zbiór danych Titanic z określonego adresu URL
2. Wybiera istotne cechy ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked') oraz zmienną docelową ('Survived').
3. Przetwarzanie danych:
   - Koduje kategoryczne cechy ('Sex' i 'Embarked') za pomocą pd.get_dummies.
   - Uzupełnia brakujące wartości w zestawie danych, korzystając z strategii średniej.

4. Dzieli zestaw danych na zbiory treningowe i testowe za pomocą train_test_split.
5. Model Drzewa Decyzyjnego:
   - Inicjalizuje klasyfikator DecisionTreeClassifier.
   - Dopasowuje model do danych treningowych.
   - Przewiduje przeżycie na danych testowych.
   - Oblicza i drukuje dokładność oraz raport klasyfikacji dla modelu Drzewa Decyzyjnego.

6. Standaryzacja danych do uzupełnionych cech za pomocą StandardScaler.

7. Model SVM:
   - Inicjalizuje model Maszyny Wektorów Nośnych (SVM) z jądrem liniowym.
   - Dopasowuje model SVM do standaryzowanych danych treningowych.
   - Przewiduje przeżycie na standaryzowanych danych testowych.
   - Oblicza i drukuje dokładność oraz raport klasyfikacji dla modelu SVM.


Instrukcje użytkowania:
   - Upewnij się, że wymagane biblioteki (pandas, sklearn) są zainstalowane.
   - Skopiuj i uruchom cały skrypt w środowisku Python.
   - Program wypisze dokładność oraz raport klasyfikacji dla obu modeli: Drzewa Decyzyjnego i SVM.
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_data = pd.read_csv(url)

features = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
label = titanic_data['Survived']
features = pd.get_dummies(features, columns=['Sex', 'Embarked'], drop_first=True)

imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features_imputed, label, test_size=0.2, random_state=42)
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

predictions_tree = decision_tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, predictions_tree)
print(f'Decision Tree Accuracy: {accuracy_tree:.2f}')
print('\nDecision Tree Classification Report:')
print(classification_report(y_test, predictions_tree))

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(features_scaled, label, test_size=0.2, random_state=42)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

predictions_svm = svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, predictions_svm)
print(f'SVM Accuracy: {accuracy_svm:.2f}')
print('\nSVM Classification Report:')
print(classification_report(y_test, predictions_svm))

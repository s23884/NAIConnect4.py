"""
System Rekomendacji Filmów

Program dostarcza rekomendacji filmów na podstawie ocen użytkownika oraz preferencji
innych użytkowników

Autorzy: Wiktor Krieger i Sebastian Augustyniak

Definicja funkcji:
1. Funkcja euclidean_distance oblicza odległość euklidesową
   między dwoma użytkownikami na podstawie ocen filmów,
   i wyświetla dystanse między innymi użytkownikami
2. Funkcja recommend_movies udziela rekomendacji filmów
   oraz filmów do unikania dla danego użytkownika.
3. Główna funkcja main() wczytuje nazwę użytkownika,
   wczytuje dane z pliku Excel i wyświetla rekomendacje.

Przykład Użycia:
Uruchomienie programu spowoduje pytanie o nazwę użytkownika, a następnie dostarczy
rekomendacje filmów oraz filmy do unikania. Program działa w pętli, pozwalając użytkownikowi
kontynuować wprowadzanie nazw użytkownika lub zakończyć program poprzez naciśnięcie Enter.

Uwagi:
- Program jest prostym systemem rekomendacji
  i wyniki mogą być zaburzone między innymi przez złożony gust grupy badawczej
- Założono, że dane o ocenach znajdują się w pliku Excel (filmy.xlsx) z odpowiednim formatowaniem.
- W momencie przed publikacją znalazłem błąd, dla Sebastiana Augustyniaka (ostatni w tabeli)
  nie podaje żadnych filmów
- Program nie został też zaktualizowany o 3 ostatnio dopisanych użytkowników

Potrzebne Zasoby:
- Plik Excel (filmy.xlsx) z danymi o ocenach filmów w tym samym katalogu co plik z kodem main.py.
- Biblioteka openpyxl do obsługi plików Excel.
- Biblioteka numpy do operacji numerycznych.

"""

import openpyxl
import numpy as np


def euclidean_distances(target_user_ratings, all_users_ratings):
    distances = {}

    for user_name, other_user_ratings in all_users_ratings.items():
        # Sprawdź, które filmy oba użytkownicy ocenili
        common_movies = set(target_user_ratings.keys()).intersection(set(other_user_ratings.keys()))

        if common_movies:
            # Oblicz różnice między ocenami filmów
            differences = [target_user_ratings[movie] - other_user_ratings[movie] for movie in common_movies]

            distance = np.linalg.norm(differences)
            distances[user_name] = distance
        else:
            distances[user_name] = float('inf')

    print(f"Dystanse euklidesowe dla użytkownika {distances}")

    return distances


def recommend_movies(user_data, all_users, all_movies, num_interesting=5, num_avoid=5):
    user_distances = euclidean_distances(user_data, all_users)

    sorted_users = sorted(user_distances, key=user_distances.get)

    selected_users = sorted_users[:num_interesting + num_avoid]

    recommended_movies = set()
    avoid_movies = set()

    for movie in all_movies:
        if movie not in user_data:
            # Sprawdź, czy użytkownik już ocenił ten film
            if movie not in recommended_movies and movie not in avoid_movies:
                # Sprawdź, czy sąsiedzi ocenili ten film
                neighbor_ratings = [all_users[user][movie] for user in selected_users if movie in all_users[user]]
                if neighbor_ratings:
                    # Oblicz średnią ocenę filmu przez sąsiadów
                    average_rating = np.mean(neighbor_ratings)

                    # Dodaj film do propozycji lub listy do unikania w zależności od oceny
                    if len(recommended_movies) < num_interesting and average_rating > 6.5:
                        recommended_movies.add(movie)
                    elif len(avoid_movies) < num_avoid and average_rating < 5.5:
                        avoid_movies.add(movie)

    return recommended_movies, avoid_movies


def main():
    while True:
        workbook = openpyxl.load_workbook('filmy.xlsx')
        sheet = workbook.active

        film_ratings = {}

        for row in sheet.iter_rows(values_only=True):
            person = row[0]

            films_and_ratings = {row[i]: int(row[i + 1]) for i in range(1, len(row), 2) if row[i + 1] is not None}

            film_ratings[person] = films_and_ratings

        all_movies = set([row[i] for i in range(1, len(sheet[1]), 2) if sheet[1][i] is not None])

        user_name = input("\nPodaj nazwę użytkownika (naciśnij Enter, aby zakończyć): ")

        if not user_name:
            break

        if user_name in film_ratings:
            user_data = film_ratings[user_name]

            recommended_movies, avoid_movies = recommend_movies(user_data, film_ratings, all_movies)

            recommended_movies = list(recommended_movies)[:5]
            print("\nPropozycje filmów, które warto obejrzeć:")
            print(', '.join(recommended_movies))
            print("\nPropozycje filmów, których warto unikać:")
            print(', '.join(avoid_movies))
        else:
            print(f"\nUżytkownik o nazwie {user_name} nie istnieje.")

        workbook.close()


if __name__ == "__main__":
    main()

import argparse
import json
import numpy as np
from compute_scores import pearson_score
from collaborative_filtering import find_similar_users

# Визначимо функцію для парсингу вхідних аргументів. У разі єдиним вхідним аргументом є ім'я користувача.
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Знайдіть рекомендації фільмів для зазначеного користувача')
    parser.add_argument('--user', dest='user', required=True, help='Ім’я користувача')
    return parser

# Визначимо функцію, яка отримуватиме рекомендації для зазначеного користувача.
# Якщо інформація про вказаного користувача відсутня у наборі даних, генерується виняток.
def get_recommendations(dataset, input_user):
    if input_user not in dataset:
        raise TypeError('Не знайдено ' + input_user + ' у наборі даних')

    overall_scores = {}
    similarity_scores = {}

    # Обчислимо оцінки подібності між зазначеним користувачем та всіма іншими користувачами у наборі даних.
    for user in [x for x in dataset if x != input_user]:
        similarity_score = pearson_score(dataset, input_user, user)

        if similarity_score <= 0:
            continue

        # Відфільтрований список фільмів, не оцінених введеним користувачем
        filtered_list = [x for x in dataset[user] if x not in dataset[input_user] or dataset[input_user][x] == 0]

        # Обчислення зважених оцінок для фільмів у відфільтрованому списку на основі подібності
        for item in filtered_list:
            overall_scores.update({item: dataset[user][item] * similarity_score})
            similarity_scores.update({item: similarity_score})

    # Якщо немає відповідних фільмів, повертаємо повідомлення
    if len(overall_scores) == 0:
        return ['Неможливо надати рекомендації']

    # Нормалізуємо оцінки на підставі зважених оцінок
    movie_scores = np.array([[score / similarity_scores[item], item] for item, score in overall_scores.items()])

    # Сортування за спаданням
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]

    # Вилучення рекомендацій фільмів
    movie_recommendations = [movie for _, movie in movie_scores]
    return movie_recommendations

# Основна функція для аналізу вхідних аргументів та отримання імені зазначеного користувача.
if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    # Завантажимо дані з файлу ratings.json, в якому містяться імена користувачів та рейтинги фільмів
    ratings_file = 'ratings.json'
    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    # Отримуємо рекомендації фільмів для зазначеного користувача
    print('\nРекомендації для користувача ' + user + ':\n')
    movies = get_recommendations(data, user)
    for i, movie in enumerate(movies):
        print(f"{i + 1}. {movie}")


import numpy as np

# Задаю seed для відтворюваності результатів рандому
np.random.seed(100)

# 1. Знайти в датасеті таргет. Завантажити датасет без таргета
# Таргет у датасеті Iris — це 5-та колонка (текстові назви класів).
# Завантажую лише перші 4 колонки (індекси 0, 1, 2, 3).
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# Використовую np.genfromtxt для прямого завантаження
arr = np.genfromtxt(url, delimiter=',', usecols=(0, 1, 2, 3))

# 2. Перетворюю колонки, що залишились в 2D масив (аби впевнитись)
print("Завдання 2:")
print(f"Кількість вимірів (ndim): {arr.ndim}")
print(f"Форма масиву (shape): {arr.shape}")
# Якщо ndim == 2, то це вже 2D масив.

# 3. Порахував mean, median, standard deviation для 1-ї колонки (індекс 0)
col_1 = arr[:, 0]
mean_val = np.mean(col_1)
median_val = np.median(col_1)
std_val = np.std(col_1)
print("\nЗавдання 3 (Статистика 1-ї колонки):")
print(f"Mean: {mean_val:.2f}, Median: {median_val:.2f}, Std: {std_val:.2f}")

# 4. Вставив 20 значень np.nan на випадкові позиції в масиві
# Щоб гарантувати 20 УНІКАЛЬНИХ позицій, використовую np.random.choice з replace=False
# Шукаю індекси для плоского (1D) масиву
random_indices = np.random.choice(arr.size, 20, replace=False)

# ravel() повертає view (посилання на оригінал), тому зміни вплинуть на arr
arr.ravel()[random_indices] = np.nan

# 5. Знаходжу позиції вставлених значень np.nan в 1-й колонці (індекс 0)
# np.where повертає індекси, де умова True
nan_positions_col1 = np.where(np.isnan(arr[:, 0]))[0]
print("\nЗавдання 5:")
print(f"Індекси рядків з NaN у 1-й колонці: {nan_positions_col1}")

# 6. Замінюю всі значення np.nan на 0
arr[np.isnan(arr)] = 0

# 7. Відфільтровую масив (3-тя колонка > 1.5 та 1-ша колонка < 5.0) і зберігаю
# 3-тя колонка — це індекс 2, 1-ша — індекс 0.
condition = (arr[:, 2] > 1.5) & (arr[:, 0] < 5.0)
filtered_arr = arr[condition]
print("\nЗавдання 7:")
print(f"Розмір відфільтрованого масиву: {filtered_arr.shape}")

# 8. Розбиваю масив по вертикалі на 2 рівні частини
# (Працюємо з основним масивом arr. Він має 150 рядків, тому розіб'ється рівно по 75)
part1, part2 = np.vsplit(arr, 2)

# 9. Відсортовую обидва масиви по 1-й колонці: 1-й за збільшенням, 2-й за зменшенням
# argsort() повертає індекси, які б відсортували масив
part1_sorted = part1[part1[:, 0].argsort()]
# Для сортування за зменшенням просто розвертаю індекси [::-1]
part2_sorted = part2[part2[:, 0].argsort()[::-1]]

# 10. Збираю обидва масиви в одне ціле
combined_arr = np.vstack((part1_sorted, part2_sorted))

# 11. Рахую всі унікальні значення в масиві та виводжу їх разом із кількістю
# Використовую return_counts=True
unique_vals, counts = np.unique(combined_arr, return_counts=True)
print("\nЗавдання 11:")
print(f"Унікальні значення:\n{unique_vals}")
print(f"Їх кількість відповідно:\n{counts}")

# 12. Знаходжу найбільш часто повторюване значення в масиві
# np.argmax знаходить індекс максимального елемента в масиві counts
most_frequent_index = np.argmax(counts)
most_frequent_value = unique_vals[most_frequent_index]
print("\nЗавдання 12:")
print(f"Найбільш часто повторюване значення: {most_frequent_value} (зустрічається {counts[most_frequent_index]} разів)")

# 13. Пишу функцію, яка б множила всі значення < середнього на 2, інші ділила на 4
def modify_column(column):
    col_mean = np.mean(column)
    # np.where - ідеальний векторизований інструмент: np.where(умова, якщо_True, якщо_False)
    modified_col = np.where(column < col_mean, column * 2, column / 4)
    return modified_col

# 14. Застосовую отриману функцію до 3-ї колонки (індекс 2)
# Змінюю колонку в даному зібраному масиві
combined_arr[:, 2] = modify_column(combined_arr[:, 2])

print("\nЗавдання 14 виконано успішно. Оновлена 3-тя колонка готова.")

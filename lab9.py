import unittest
import numpy as np
from typing import Union, Sequence


def create_even_mask(array: Union[np.ndarray, Sequence[int]]) -> bool:
    """
    Создает булеву маску для четных чисел в массиве.

    """
    return np.asarray(array) % 2 == 0


def longest_even_sequence(array: Union[np.ndarray, Sequence[int]]) -> int:
    """
    Находит длину самой длинной последовательности чётных чисел в массиве.
    """
    # Преобразуем входной массив в numpy array
    array = np.asarray(array)

    # Если массив пустой, то возвращаем 0.
    if array.size == 0:
        return 0

    # Создаем boolean маску, где True означает, что число в массиве чётное, иначе False.
    even_mask = create_even_mask(array)

    # Если в массиве нет ни одного чётного числа, возвращаем 0.
    if not np.any(even_mask):
        return 0

    # Находим индексы, где меняется последовательность чётности (с True на False или наоборот).
    # Для этого вычисляем разницу между соседними элементами маски (приведя её к int) и находим
    # индексы, где эта разница не равна 0
    diff_indices = np.where(np.diff(even_mask.astype(int)))[0] + 1

    # Добавляем индексы начала (0) и конца массива, чтобы учесть все последовательности.
    all_indices = np.concatenate(([0], diff_indices, [len(array)]))

    # Вычисляем длины всех последовательностей (и чётных, и нечётных).
    sequence_lengths = np.diff(all_indices)

    # Выбираем только длины последовательностей, которые начинаются с чётного числа.
    even_sequence_lengths = sequence_lengths[even_mask[all_indices[:-1]]]

    # Если есть хотя бы одна последовательность чётных чисел, возвращаем длину самой длинной, иначе 0.
    if len(even_sequence_lengths) > 0:
        return np.max(even_sequence_lengths)
    else:
        return 0


class TestLongestEvenSequence(unittest.TestCase):
    def test_empty_array(self):
        # Тест с пустым массивом.
        self.assertEqual(longest_even_sequence(np.array([])), 0)

    def test_no_even_numbers(self):
        # Тест с массивом, не содержащим чётных чисел.
        self.assertEqual(longest_even_sequence(np.array([1, 3, 5, 7])), 0)

    def test_single_even_number(self):
        # Тест с массивом, содержащим только одно чётное число.
        self.assertEqual(longest_even_sequence(np.array([2])), 1)

    def test_single_odd_number(self):
        # Тест с массивом, содержащим только одно нечётное число.
        self.assertEqual(longest_even_sequence(np.array([1])), 0)

    def test_sequence_of_even_numbers(self):
        # Тест с массивом, содержащим последовательность чётных чисел.
        self.assertEqual(longest_even_sequence(np.array([2, 4, 6, 8, 10])), 5)

    def test_sequence_of_even_negative_numbers(self):
        # Тест с массивом, содержащим последовательность отрицательных чётных чисел.
        self.assertEqual(longest_even_sequence(np.array([-2, -4, -6, -8, -10])), 5)

    def test_mixed_even_and_odd(self):
        # Тест с массивом, содержащим смешанные чётные и нечётные числа.
        self.assertEqual(
            longest_even_sequence(np.array([1, 2, 4, 6, 1, 3, 8, 10, 12, 1, 2, 4, 1, 5])), 3
        )

    def test_even_at_start(self):
        # Тест с массивом, где чётные числа находятся в начале.
        self.assertEqual(longest_even_sequence(np.array([2, 4, 1, 3])), 2)

    def test_even_at_end(self):
        # Тест с массивом, где чётные числа находятся в конце.
        self.assertEqual(longest_even_sequence(np.array([1, 3, 2, 4])), 2)

    def test_multiple_even_sequences(
            self,
    ):  # Тест с массивом, где есть несколько последовательностей чётных чисел.
        self.assertEqual(longest_even_sequence(np.array([1, 2, 4, 1, 2, 1, 2, 4, 6])), 3)

    def test_equal_even_numbers(self):
        # Тест с массивом, где есть последовательность одинаковых чётных чисел.
        self.assertEqual(longest_even_sequence(np.array([2, 2, 1, 4, 4, 4])), 3)

    def test_only_even_sequence_with_single_odd(self):
        # Тест с массивом где есть одна последовательность четных и один нечетный элемент
        self.assertEqual(longest_even_sequence(np.array([2, 2, 1])), 2)

    def test_single_even_sequence_with_single_odd(self):
        # Тест с массивом где есть один четный элемент в последовательности, окруженной нечетными элементами
        self.assertEqual(longest_even_sequence(np.array([1, 2, 1])), 1)

    def test_odd_even_odd(self):
        # Тест с массивом вида [нечёт, чёт, нечёт].
        self.assertEqual(longest_even_sequence(np.array([1, 2, 1])), 1)

    def test_negative_odd_numbers(self):
        # Тест с массивом содержащим отрицательные нечетные числа.
        self.assertEqual(longest_even_sequence(np.array([-1, -3, -5, -7])), 0)

    def test_long_even_sequence(self):
        # Тест с длинной последовательностью из 10 чётных чисел.
        self.assertEqual(longest_even_sequence(np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])), 10)

    def test_negative_long_even_sequence(self):
        # Тест с длинной последовательностью из 10 чётных отрицательных чисел.
        self.assertEqual(
            longest_even_sequence(
                np.array([-200, -180, -160, -140, -120, -80, -60, -40, -20, -10])
            ),
            10,
        )

    def test_large_positive_even(self):
        # Тест с большим массивом четных положительных чисел
        large_even_positive = np.arange(2, 2002, 2)
        self.assertEqual(longest_even_sequence(large_even_positive), 1000)

    def test_large_negative_even(self):
        # Тест с большим массивом четных отрицательных чисел
        large_even_negative = np.arange(-2, -2002, -2)
        self.assertEqual(longest_even_sequence(large_even_negative), 1000)

    def test_large_mixed(self):
        # Тест с большим массивом смешанных чисел (поочередно четное и нечетное)
        large_mixed = np.array([i if i % 2 == 0 else 1 for i in range(2000)])
        self.assertEqual(longest_even_sequence(large_mixed), 1)


if __name__ == "__main__":
    input_str = input("Введите числа через пробел: ")
    # Преобразуем введенную строку в список целых чисел.
    input_list = [int(x) for x in input_str.split()]
    # Создаем NumPy массив из списка.
    input_arr = np.array(input_list, dtype=int)
    # Вызываем функцию для вычисления самой длинной последовательности чётных чисел.
    result = longest_even_sequence(input_arr)
    # Выводим результат.
    print(f"Самая длинная последовательность четных чисел в введенном массиве: {result}")

    unittest.main()

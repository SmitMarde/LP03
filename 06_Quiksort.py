
import random

def partition(arr, low, high, steps_counter):
    steps = 0
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        steps += 1  # Count each comparison

        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]

    steps_counter[0] += steps
    return i + 1


def random_partition(arr, low, high, steps_counter):
    rand_index = random.randint(low, high)
    arr[rand_index], arr[high] = arr[high], arr[rand_index]

    return partition(arr, low, high, steps_counter)


def quick_sort_det(arr, low, high, steps_counter):
    if low < high:
        pi = partition(arr, low, high, steps_counter)

        quick_sort_det(arr, low, pi - 1, steps_counter)
        quick_sort_det(arr, pi + 1, high, steps_counter)


def quick_sort_rand(arr, low, high, steps_counter):
    if low < high:
        pi = random_partition(arr, low, high, steps_counter)

        quick_sort_rand(arr, low, pi - 1, steps_counter)
        quick_sort_rand(arr, pi + 1, high, steps_counter)


if __name__ == "__main__":
    try:
        n = int(input("Enter number of elements: "))

        if n < 0:
            print("Number of elements cannot be negative.")
        else:
            elements = list(map(
                int, input("Enter elements separated by spaces: ").split()
            ))

            if len(elements) != n:
                print(f"Error: Expected {n} elements, but received {len(elements)}.")
            else:
                arr1 = elements.copy()
                arr2 = elements.copy()

                det_counter = [0]
                rand_counter = [0]

                quick_sort_det(arr1, 0, n - 1, det_counter)
                quick_sort_rand(arr2, 0, n - 1, rand_counter)

                print("\nDeterministic QuickSort:", *arr1)
                print("Steps (approx comparisons):", det_counter[0])

                print("\nRandomized QuickSort:", *arr2)
                print("Steps (approx comparisons):", rand_counter[0])

    except ValueError:
        print("Error: Please enter valid integers.")



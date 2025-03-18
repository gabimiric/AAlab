import time
import matplotlib.pyplot as plt
import random
import sys
sys.setrecursionlimit(10**6)  # Increase recursion limit to 1,000,000

PAUSE = 0.1 # Global pause for visualization

# QuickSort
def partition(array, low, high):
    pivot = array[high]  # Choose the last element as pivot
    i = low - 1  # Pointer for smaller elements

    for j in range(low, high):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]  # Swap

    array[i + 1], array[high] = array[high], array[i + 1]  # Place pivot correctly
    return i + 1


def quickSort(array, start, end):
    if start < end:
        piv = partition(array, start, end)
        quickSort(array, start, piv - 1)  # Sort left part
        quickSort(array, piv + 1, end)  # Sort right part

# MergeSort
def merge(arr, l, m, r):
    L = arr[l:m + 1]  # Left half
    R = arr[m + 1:r + 1]  # Right half

    i = j = 0
    k = l

    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < len(L):  # Copy remaining elements of L
        arr[k] = L[i]
        i += 1
        k += 1

    while j < len(R):  # Copy remaining elements of R
        arr[k] = R[j]
        j += 1
        k += 1

def mergeSort(arr, l, r):
    if l < r:
        m = l + (r - l) // 2  # Middle index
        mergeSort(arr, l, m)  # Sort left half
        mergeSort(arr, m + 1, r)  # Sort right half
        merge(arr, l, m, r)  # Merge sorted halves

# HeapSort
def heapify(arr, n, i):
    largest = i  # Assume root is largest
    l = 2 * i + 1  # Left child
    r = 2 * i + 2  # Right child

    if l < n and arr[l] > arr[largest]:
        largest = l  # Left child is larger

    if r < n and arr[r] > arr[largest]:
        largest = r  # Right child is larger

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Swap with largest
        heapify(arr, n, largest)  # Recursively heapify the affected subtree

def countingSort(arr):
    max_val = max(arr)
    count = [0] * (max_val + 1)

    while len(arr) > 0:
        num = arr.pop(0)
        count[num] += 1

    for i in range(len(count)):
        while count[i] > 0:
            arr.append(i)
            count[i] -= 1

def heapSort(arr):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):  # Build max heap
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):  # Extract elements one by one
        arr[i], arr[0] = arr[0], arr[i]  # Move max to end
        heapify(arr, i, 0)  # Restore heap property

def measure_time(arr):
    methods = ["quickSort", "mergeSort", "heapSort", "countSort"]
    times = {}
    for method in methods:
        arrCopy = arr[:]
        start = time.perf_counter()
        if method == "quickSort":
            quickSort(arrCopy, 0, len(arrCopy)-1)
        elif method == "mergeSort":
            mergeSort(arrCopy, 0, len(arrCopy)-1)
        elif method == "heapSort":
            heapSort(arrCopy)
        elif method == "countSort":
            countingSort(arrCopy)
        times[method] = (time.perf_counter() - start) * 1000  # milliseconds
    return times

arrays = [
    random.sample(range(0, 1000), 100),

    random.sample(range(0, 1000), 1000),

    random.sample(range(0, 10000), 10000),

    random.sample(range(0, 100000), 100000),

    random.sample(range(0, 200000), 200000)
]

# Array names for printing purposes
array_names = ["100 Elements", "1K Elements", "10K Elements", "100K Elements", "200K Elements"]

# Print the header
print(f"{'Elements':<15}{'QuickSort (ms)':<20}{'MergeSort (ms)':<20}{'HeapSort (ms)':<20}{'CountingSort (ms)'}")

quick_sort_times = []
merge_sort_times = []
heap_sort_times = []
counting_sort_times = []

# Measure time for each array and print the results
for i, arr in enumerate(arrays):
    times = measure_time(arr)
    print(f"{array_names[i]:<15}{times['quickSort']:<20.4f}{times['mergeSort']:<20.4f}{times['heapSort']:<20.4f}{times['countSort']:<.4f}")
    quick_sort_times.append(times["quickSort"])
    merge_sort_times.append(times["mergeSort"])
    heap_sort_times.append(times["heapSort"])
    counting_sort_times.append(times["countSort"])

# Plotting
x_values = [100, 1000, 10000, 100000, 200000]
plt.figure(figsize=(10, 6))
plt.plot(x_values, quick_sort_times, marker='o', linestyle='-', label="QuickSort")
plt.plot(x_values, merge_sort_times, marker='s', linestyle='-', label="MergeSort")
plt.plot(x_values, heap_sort_times, marker='^', linestyle='-', label="HeapSort")
plt.plot(x_values, counting_sort_times, marker='d', linestyle='-', label="CountingSort")

# Labels and title
plt.xlabel("Number of Elements")
plt.ylabel("Time (ms)")
plt.title("Sorting Algorithm Efficiency")
plt.xscale("log")  # Log scale for better visualization
plt.yscale("log")  # Log scale if times vary significantly
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show plot
plt.show()

# Sorted test arrays
sorted_ascending = list(range(10000))  # Sorted in ascending order
sorted_descending = list(range(9999, -1, -1))  # Sorted in descending order
almost_sorted_ascending = sorted_ascending[:]
almost_sorted_descending = sorted_descending[:]

# Introduce slight randomness to make them "almost sorted"
for _ in range(10):  # Swap 10 random pairs
    i, j = random.sample(range(10000), 2)
    almost_sorted_ascending[i], almost_sorted_ascending[j] = almost_sorted_ascending[j], almost_sorted_ascending[i]
    almost_sorted_descending[i], almost_sorted_descending[j] = almost_sorted_descending[j], almost_sorted_descending[i]

# New array list and names
new_arrays = [sorted_ascending, sorted_descending, almost_sorted_ascending, almost_sorted_descending]
new_array_names = ["Sorted Asc", "Sorted Desc", "Almost Sorted Asc", "Almost Sorted Desc"]

# Measure time for the new arrays
quick_sort_times_new = []
merge_sort_times_new = []
heap_sort_times_new = []
counting_sort_times_new = []

# Print the header for the new table
print(f"\n{'Array Type (10K)':<20}{'QuickSort (ms)':<20}{'MergeSort (ms)':<20}{'HeapSort (ms)':<20}{'CountingSort (ms)'}")

# Measure and print results for the new arrays
for i, arr in enumerate(new_arrays):
    times = measure_time(arr)
    print(f"{new_array_names[i]:<20}{times['quickSort']:<20.4f}{times['mergeSort']:<20.4f}{times['heapSort']:<20.4f}{times['countSort']:<.4f}")
    quick_sort_times_new.append(times["quickSort"])
    merge_sort_times_new.append(times["mergeSort"])
    heap_sort_times_new.append(times["heapSort"])
    counting_sort_times_new.append(times["countSort"])

# Plotting the new results
plt.figure(figsize=(10, 6))
plt.plot(new_array_names, quick_sort_times_new, marker='o', linestyle='-', label="QuickSort")
plt.plot(new_array_names, merge_sort_times_new, marker='s', linestyle='-', label="MergeSort")
plt.plot(new_array_names, heap_sort_times_new, marker='^', linestyle='-', label="HeapSort")
plt.plot(new_array_names, counting_sort_times_new, marker='d', linestyle='-', label="CountingSort")

# Labels and title
plt.xlabel("Array Type")
plt.ylabel("Time (ms)")
plt.title("Sorting Algorithm Efficiency on Special Arrays")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show plot
plt.show()

import tkinter as tk
from tkinter import simpledialog

# Sorting visualization

def visualize_quick_sort(arr):
    plt.figure(figsize=(10, 6))
    quick_sort_display(arr)
    plt.show()

def visualize_merge_sort(arr):
    plt.figure(figsize=(10, 6))
    merge_sort_display(arr)
    plt.show()

def visualize_heap_sort(arr):
    plt.figure(figsize=(10, 6))
    heap_sort_display(arr)
    plt.show()

def visualize_counting_sort(arr):
    counting_sort_display(arr)
    plt.show()

def quick_sort_display(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot = partition(arr, low, high)
        plt.clf()
        plt.bar(range(len(arr)), arr, color='blue')
        plt.pause(PAUSE)
        quick_sort_display(arr, low, pivot - 1)
        quick_sort_display(arr, pivot + 1, high)

def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1

def merge_sort_display(arr, l=0, r=None):
    if r is None:
        r = len(arr) - 1
    if l < r:
        m = (l + r) // 2
        merge_sort_display(arr, l, m)
        merge_sort_display(arr, m + 1, r)
        merge(arr, l, m, r)
        plt.clf()
        plt.bar(range(len(arr)), arr, color='blue')
        plt.pause(PAUSE)

def merge(arr, l, m, r):
    left = arr[l:m + 1]
    right = arr[m + 1:r + 1]
    i = j = 0
    k = l
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1

def heap_sort_display(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        plt.clf()
        plt.bar(range(len(arr)), arr, color='blue')
        plt.pause(PAUSE)
        heapify(arr, i, 0)

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def counting_sort_display(arr):
    max_val = max(arr)
    count = [0] * (max_val + 1)

    for num in arr:
        count[num] += 1

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    index = 0
    for i in range(len(count)):
        while count[i] > 0:
            arr[index] = i
            index += 1
            count[i] -= 1

            axes[0].cla()
            axes[0].bar(range(len(arr)), arr, color='blue')
            axes[0].set_title("Sorted Array")

            axes[1].cla()
            axes[1].bar(range(len(count)), count, color='red')
            axes[1].set_title("Count Array")

            plt.pause(PAUSE)

# Prompt for sorting visualization

def select_sorting():
    root = tk.Tk()
    root.withdraw()

    while True:
        sort_choice = simpledialog.askstring("Sorting Visualization",
                                             "Choose a sorting algorithm: Quick, Merge, Heap, Counting or type 'exit' to quit")

        if not sort_choice or sort_choice.lower() == "exit":
            break

        arr = [random.randint(0,1000) for _ in range(200)]

        if sort_choice.lower() == "quick":
            visualize_quick_sort(arr)
        elif sort_choice.lower() == "merge":
            visualize_merge_sort(arr)
        elif sort_choice.lower() == "heap":
            visualize_heap_sort(arr)
        elif sort_choice.lower() == "counting":
            visualize_counting_sort(arr)
        else:
            print("Invalid selection.")

select_sorting()

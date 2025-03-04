import time
import matplotlib.pyplot as plt
import random

from sympy import false

snapshots = []

def captureSnapshot(arr):
    snapshots.append(arr[:])

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


def quickSort(array, start, end, snap = false):
    if start < end:
        piv = partition(array, start, end)
        if (snap) : captureSnapshot(array)
        quickSort(array, start, piv - 1)  # Sort left part
        quickSort(array, piv + 1, end)  # Sort right part

# MergeSort
def merge(arr, l, m, r, snap):
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
        if (snap) : captureSnapshot(arr[:])

    while i < len(L):  # Copy remaining elements of L
        arr[k] = L[i]
        i += 1
        k += 1
        if (snap): captureSnapshot(arr[:])

    while j < len(R):  # Copy remaining elements of R
        arr[k] = R[j]
        j += 1
        k += 1
        if (snap): captureSnapshot(arr[:])


def mergeSort(arr, l, r, snap = false):
    if l < r:
        m = l + (r - l) // 2  # Middle index
        mergeSort(arr, l, m)  # Sort left half
        mergeSort(arr, m + 1, r)  # Sort right half
        merge(arr, l, m, r, snap)  # Merge sorted halves

# HeapSort
def heapify(arr, n, i, snap):
    largest = i  # Assume root is largest
    l = 2 * i + 1  # Left child
    r = 2 * i + 2  # Right child

    if l < n and arr[l] > arr[largest]:
        largest = l  # Left child is larger

    if r < n and arr[r] > arr[largest]:
        largest = r  # Right child is larger

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Swap with largest
        if (snap) : captureSnapshot(arr[:])
        heapify(arr, n, largest, snap)  # Recursively heapify the affected subtree


def heapSort(arr, snap=false):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):  # Build max heap
        heapify(arr, n, i, snap)

    for i in range(n - 1, 0, -1):  # Extract elements one by one
        arr[i], arr[0] = arr[0], arr[i]  # Move max to end
        heapify(arr, i, 0, snap)  # Restore heap property

def measure_time(arr):
    methods = ["quickSort", "mergeSort", "heapSort"]
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
        times[method] = (time.perf_counter() - start) * 1000  # milliseconds
    return times

arrays = [
    [37, 5, 92, 16, 81, 24, 64, 49, 12, 78, 55, 89, 1, 30, 7, 100, 42, 68, 91, 21,
     44, 99, 15, 58, 83, 26, 36, 66, 20, 72, 31, 77, 63, 47, 41, 28, 14, 98, 40, 70,
     67, 62, 50, 80, 51, 43, 73, 69, 56, 11, 23, 9, 85, 39, 34, 2, 29, 59, 53, 82,
     32, 22, 65, 18, 4, 54, 60, 10, 17, 33, 74, 8, 57, 75, 76, 71, 18, 48, 93, 45,
     79, 86, 94, 13, 95, 90, 46, 19, 96, 84, 97, 63, 25, 27, 61, 52, 100, 37, 55],

    [1, 5, 7, 12, 16, 21, 24, 30, 37, 42, 49, 55, 64, 68, 78, 81, 89, 91, 92, 100,
     2, 3, 4, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 22, 23, 25, 26, 28,
     29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53,
     54, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76,
     77, 79, 80, 82, 83, 84, 85, 86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 98, 99],

    [100, 92, 91, 89, 81, 78, 72, 68, 64, 61, 55, 53, 48, 44, 42, 37, 35, 30, 27, 21,
     93, 84, 70, 66, 59, 51, 45, 56, 39, 41, 73, 80, 23, 18, 25, 36, 79, 32, 22, 49,
     67, 69, 63, 50, 30, 58, 76, 63, 44, 38, 10, 5, 17, 57, 86, 1, 4, 8, 29, 75,
     33, 85, 11, 43, 12, 14, 52, 16, 65, 46, 53, 37, 34, 20, 24, 40, 47, 55, 2, 3,
     71, 19, 77, 62, 65, 28, 74, 9, 13, 62, 26, 31, 82, 60, 54, 15, 68, 48, 36, 66],

    random.sample(range(0, 1000), 1000)
]

# Randomly generated array
print(arrays[3])

# Array names for printing purposes
array_names = ["1st Array", "2nd Array", "3rd Array", "Random Array"]

# Print the header
print(f"{'Array Name':<15}{'QuickSort (ms)':<20}{'MergeSort (ms)':<20}{'HeapSort (ms)'}")

# Measure time for each array and print the results
for i, arr in enumerate(arrays):
    times = measure_time(arr)
    print(f"{array_names[i]:<15}{times['quickSort']:<20.4f}{times['mergeSort']:<20.4f}{times['heapSort']:<.4f}")


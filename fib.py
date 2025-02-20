import time
import matplotlib.pyplot as plt


# Fibonacci methods
def fibonacci_polynomial(n):
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b


def fib_fast_doubling(n):
    def fib_pair(n):
        if n == 0:
            return (0, 1)
        else:
            a, b = fib_pair(n // 2)
            c = a * (2 * b - a)
            d = a * a + b * b
            if n % 2 == 0:
                return (c, d)
            else:
                return (d, c + d)

    return fib_pair(n)[0]


def fibonacci_continued_fraction(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result = 1 + 1 / result
    return round(result)


# Function to measure time for each Fibonacci method
def measure_time(n_values):
    times_poly = []
    times_fast_doubling = []
    times_continued_fraction = []

    print("{:<15}{:<15}".format("Method", " ".join([f"{n:<9}" for n in n_values])))
    methods = ["Polynomial", "Fast Doubling", "Continued Fraction"]

    for method in methods:
        if method == "Polynomial":
            times = []
            for n in n_values:
                start = time.perf_counter()  # Renamed to avoid 'time' conflict
                fibonacci_polynomial(n)
                elapsed_time = (time.perf_counter() - start) * 1000
                times.append(elapsed_time)
            print("{:<15}".format("Polynomial"), end="")
            for time_taken in times:
                print(f"{time_taken:<10.5f}", end="")
            print()
            times_poly = times

        elif method == "Fast Doubling":
            times = []
            for n in n_values:
                start = time.perf_counter()  # Renamed to avoid 'time' conflict
                fib_fast_doubling(n)
                elapsed_time = (time.perf_counter() - start) * 1000
                times.append(elapsed_time)
            print("{:<15}".format("Fast Doubling"), end="")
            for time_taken in times:
                print(f"{time_taken:<10.5f}", end="")
            print()
            times_fast_doubling = times

        elif method == "Continued Fraction":
            times = []
            for n in n_values:
                start = time.perf_counter()  # Renamed to avoid 'time' conflict
                fibonacci_continued_fraction(n)
                elapsed_time = (time.perf_counter() - start) * 1000
                times.append(elapsed_time)
            print("{:<15}".format("Cont Fraction"), end="")
            for time_taken in times:
                print(f"{time_taken:<10.5f}", end="")
            print()
            times_continued_fraction = times

    return times_poly, times_fast_doubling, times_continued_fraction


# Values for Fibonacci calculation
n_values = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849]

# Measure the time for each n
times_poly, times_fast_doubling, times_continued_fraction = measure_time(n_values)

# Plot for Polynomial
plt.figure(figsize=(10, 5))
plt.plot(n_values, times_poly, marker='o', color='b', linestyle='-', markersize=5)
plt.xlabel('Nth Fibonacci Term')
plt.ylabel('Time (ms)')
plt.title('Polynomial Method: Time vs Nth Fibonacci Term')
plt.grid(True)
plt.show()

# Plot for Fast Doubling
plt.figure(figsize=(10, 5))
plt.plot(n_values, times_fast_doubling, marker='s', color='r', linestyle='-', markersize=5)
plt.xlabel('Nth Fibonacci Term')
plt.ylabel('Time (ms)')
plt.title('Fast Doubling Method: Time vs Nth Fibonacci Term')
plt.grid(True)
plt.show()

# Plot for Continued Fraction
plt.figure(figsize=(10, 5))
plt.plot(n_values, times_continued_fraction, marker='^', color='g', linestyle='-', markersize=5)
plt.xlabel('Nth Fibonacci Term')
plt.ylabel('Time (ms)')
plt.title('Continued Fraction Method: Time vs Nth Fibonacci Term')
plt.grid(True)
plt.show()

# Plot all methods together
plt.plot(n_values, times_poly, marker='o', color='b', linestyle='-', markersize=5, label="Polynomial")
plt.plot(n_values, times_fast_doubling, marker='s', color='r', linestyle='-', markersize=5, label="Fast Doubling")
plt.plot(n_values, times_continued_fraction, marker='^', color='g', linestyle='-', markersize=5, label="Continued Fraction")

plt.xlabel('Nth Fibonacci Term')
plt.ylabel('Time (ms)')
plt.title('Time to Compute Fibonacci Terms Using Different Methods')
plt.legend()
plt.grid(True)
plt.show()

import time
from functools import lru_cache


# Example function with built-in caching
@lru_cache(maxsize=None)  # None means unlimited cache size
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# Using perf_counter for more precise timing
def measure_time(func, *args):
    start = time.perf_counter()  # More precise than time.time()
    result = func(*args)
    end = time.perf_counter()
    return result, end - start


# Demonstrate caching effect
print("Calculating fibonacci(300) without cache:")
result, time_taken = measure_time(fibonacci, 300)
print(f"Result: {result}")
print(f"Time: {time_taken:.10f} seconds")  # Show more decimal places

print("\nCalculating fibonacci(300) with cache:")
result, time_taken = measure_time(fibonacci, 300)
print(f"Time: {time_taken:.10f} seconds")

# Clear cache to show difference
fibonacci.cache_clear()  # Using built-in method to clear cache

print("\nCalculating fibonacci(300) after clearing cache:")
result, time_taken = measure_time(fibonacci, 300)
print(f"Time: {time_taken:.10f} seconds")

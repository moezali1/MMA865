import functools
import time


# Define a simple caching decorator
def cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = str(args) + str(kwargs)
        if cache_key not in wrapper.cache:
            wrapper.cache[cache_key] = func(*args, **kwargs)
        return wrapper.cache[cache_key]

    wrapper.cache = {}
    return wrapper


# Example function with caching
@cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# Function to measure execution time
def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start


# Demonstrate caching effect
print("Calculating fibonacci(30) without cache:")
result, time_taken = measure_time(fibonacci, 300)
print(f"Result: {result}, Time: {time_taken:.6f} seconds")

print("\nCalculating fibonacci(30) with cache:")
result, time_taken = measure_time(fibonacci, 300)
print(f"Result: {result}, Time: {time_taken:.6f} seconds")

# Clear cache to show difference
fibonacci.cache.clear()

print("\nCalculating fibonacci(30) after clearing cache:")
result, time_taken = measure_time(fibonacci, 300)
print(f"Result: {result}, Time: {time_taken:.6f} seconds")

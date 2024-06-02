import time
import threading
from rich.console import Console
from time import sleep

load = False
def debug(func,*args, **kwargs):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        print(f" ========== Calling {func.__name__} ==========")
        result =  func(*args, **kwargs)
        end = time.perf_counter()
        print(f"********** Agent {func.__name__} took {end - start:.4f} seconds **********")
        return result
    return wrapper



# Example function to demonstrate the decorator
@debug
def example_function(seconds):

    time.sleep(seconds)
    return "Function complete!"


# Run the example function
example_function(3)

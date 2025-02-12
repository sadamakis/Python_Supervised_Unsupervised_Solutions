import time

# COPIED
def time_function(func):
    def inner_time(*args, **kwargs):
        begin = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {round(end-begin, 2)} seconds")
        return out
    return inner_time
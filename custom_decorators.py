import time
from logging import Logger
import os
import functools
import inspect


def timeit(logger: Logger = None, print_time: bool = False, return_val: bool = False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            file_name = os.path.basename(func.__code__.co_filename)
            message = f"{func.__name__} (file: {file_name}) executed in {elapsed_time:.4f} seconds"
            
            if logger:
                logger.info(message)
            if print_time:
                print(message)
                
            if return_val:
                return result, elapsed_time
            return result
        return wrapper
    return decorator

def log_call(logger: Logger = None, log_params: list = None, hide_res: bool = False):
    """
    Dekorator, który loguje wywołanie funkcji.
    
    :param logger: opcjonalny obiekt loggera; jeśli None, używany jest print.
    :param log_params: opcjonalna lista nazw parametrów, które mają zostać wypisane.
                       Jeśli None, wypisze wszystkie parametry.

    Daj go nad timeit, jeśli mieszasz dekoratory, bo coś tam logger potem zły plik pokazuje, odpalić się odpali,
    ale w logach będzie pokazywać plik jako customer_decorator.py, zamiast faktycznie plik, w którym siedzi udekorowana przez ciebie metoda.
    Nie chciało mi się szukac przyczyny, szybki workaround pomógł, no dziękuję za wypowiedź
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            # Mapowanie argumentów na nazwy parametrów przy użyciu inspect.signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Wybieramy, które parametry wypisać
            if log_params is None:
                params_to_log = dict(bound_args.arguments)
            else:
                params_to_log = {k: v for k, v in bound_args.arguments.items() if k in log_params}
            
            start_msg = f"start {func_name} with parameters: {params_to_log}"
            if logger:
                logger.info(start_msg)
            else:
                print(start_msg)
            
            # Wywołanie funkcji
            result = func(*args, **kwargs)
            if not hide_res:
                end_msg = f"end {func_name} with result: {result}"
            else:
                end_msg = f"end {func_name}"
            if logger:
                logger.info(end_msg)
            else:
                print(end_msg)
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    @timeit(print_time=True, return_val=True)
    def sample_function(n):
        total = 0
        for i in range(n):
            total += i
        return total

    res, exec_time = sample_function(1000000)
    print(f"{res=}")
    print(f"{exec_time=}")
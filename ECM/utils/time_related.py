import time


def timer(solver_func):
    """
    Timer function is intended to be a decorator function that takes in any solver function and calculates the solver
    solution time. It then displays the solution time.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        sol = solver_func(*args, **kwargs)
        print(f"Solver execution time: {time.time() - start_time}s")
        return sol
    return wrapper
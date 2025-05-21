
def log_function_call(func):
    def wrapper(a, b):
        print(f"입력된 매개변수 a : {a}, b : {b}, 반환값 : {func(a,b)}")
        return func(a,b)
    return wrapper


@log_function_call
def add(a, b):
    return a+b

add(1,2)

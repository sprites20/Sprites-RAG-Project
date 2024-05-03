def print_nth_fibonacci(n):
    if n <= 0:
        print("Invalid input. Index must be greater than 0.")
        return
    elif n == 1:
        fibonacci_number = 0
    elif n == 2:
        fibonacci_number = 1
    else:
        a, b = 0, 1
        for _ in range(2, n):
            a, b = b, a + b
        fibonacci_number = b
    
    print(f"The {n}th Fibonacci number is: {fibonacci_number}")

def print_nth_fibonacci_negative(n):
    if n <= 0:
        print("Invalid input. Index must be greater than 0.")
        return
    elif n == 1:
        fibonacci_number = 0
    elif n == 2:
        fibonacci_number = -1
    else:
        a, b = 0, 1
        for _ in range(2, n):
            a, b = b, a + b
        fibonacci_number = -b

    print(f"The {n}th Fibonacci number (negative) is: {fibonacci_number}")


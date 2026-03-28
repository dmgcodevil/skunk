def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


def main() -> None:
    limit = 35
    total = 0
    for i in range(limit):
        total += fib(i)
    print(total)


if __name__ == "__main__":
    main()

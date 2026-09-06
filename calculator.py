def calculator(expression):

    try:

        allowed = set(
            "0123456789+-*/().% "
        )

        if not all(
            char in allowed
            for char in expression
        ):

            return (
                "Invalid mathematical expression."
            )

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {}
        )

        return str(result)

    except Exception as e:

        return (
            "Calculator error: "
            + str(e)
        )


if __name__ == "__main__":

    print(
        "Calculator Tool"
    )

    expression = input(
        "Enter calculation: "
    )

    print(
        "Answer:",
        calculator(expression)
    )
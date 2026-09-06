def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    Use this tool when the user asks for a calculation.
    """

    try:
        allowed = "0123456789+-*/().% "

        if not all(
            char in allowed
            for char in expression
        ):
            return "Invalid mathematical expression."

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"
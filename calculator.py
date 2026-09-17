import ast
import operator

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _calculate_node(node):

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Invalid number.")

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("Invalid operator.")

        operand = _calculate_node(node.operand)
        return _ALLOWED_OPERATORS[operator_type](operand)

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)

        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("Invalid operator.")

        left = _calculate_node(node.left)
        right = _calculate_node(node.right)

        if operator_type is ast.Pow and abs(right) > 100:
            raise ValueError("Power value is too large.")

        return _ALLOWED_OPERATORS[operator_type](left, right)

    raise ValueError("Invalid mathematical expression.")


def calculator(expression):

    try:

        if expression is None:
            return "Invalid mathematical expression."

        expression = str(expression).strip()

        if not expression:
            return "Invalid mathematical expression."

        prefixes = [
            "calculate ",
            "calc ",
            "what is ",
            "what's ",
            "solve ",
            "compute ",
            "find "
        ]

        expression_lower = expression.lower()

        for prefix in prefixes:
            if expression_lower.startswith(prefix):
                expression = expression[len(prefix):].strip()
                break

        expression = (
            expression
            .replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
            .replace("–", "-")
            .replace("^", "**")
        )

        expression = expression.replace(",", "")

        tree = ast.parse(expression, mode="eval")

        result = _calculate_node(tree.body)

        if isinstance(result, float):

            if result.is_integer():
                return str(int(result))

            return str(round(result, 10))

        return str(result)

    except ZeroDivisionError:
        return "Calculator error: Division by zero."

    except SyntaxError:
        return "Invalid mathematical expression."

    except Exception:
        return "Invalid mathematical expression."


if __name__ == "__main__":

    print("Calculator Tool")
    print("----------------")

    while True:

        expression = input(
            "\nEnter calculation (or type exit): "
        )

        if expression.lower() == "exit":
            break

        print("Answer:", calculator(expression))

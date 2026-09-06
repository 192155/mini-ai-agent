from calculator import calculator
from web_search import web_search


def run_calculator(expression):
    """
    Mathematical calculations perform karta hai.
    """
    return calculator(expression)


def run_web_search(query):
    """
    Internet par current information search karta hai.
    """
    return web_search(query)


def get_available_tools():
    """
    Agent ke available tools return karta hai.
    """

    return {
        "calculator": {
            "description": (
                "Perform mathematical calculations. "
                "Use this for arithmetic expressions."
            ),
            "function": run_calculator
        },

        "web_search": {
            "description": (
                "Search the internet for current or "
                "up-to-date information."
            ),
            "function": run_web_search
        }
    }


def run_tool(tool_name, arguments):
    """
    Selected tool ko execute karta hai.
    """

    tools = get_available_tools()

    if tool_name not in tools:
        return f"Unknown tool: {tool_name}"

    tool_function = tools[tool_name]["function"]

    try:

        if tool_name == "calculator":

            expression = arguments.get(
                "expression",
                ""
            )

            return tool_function(
                expression
            )

        if tool_name == "web_search":

            query = arguments.get(
                "query",
                ""
            )

            return tool_function(
                query
            )

        return "Tool arguments invalid."

    except Exception as e:

        return (
            f"Tool execution failed: {str(e)}"
        )


if __name__ == "__main__":

    print("\n==============================")
    print("       AGENT TOOLS TEST")
    print("==============================")

    tools = get_available_tools()

    print("\nAvailable tools:")

    for name in tools:
        print(f"- {name}")

    print("\nCalculator test:")

    result = run_tool(
        "calculator",
        {
            "expression": "25 * 40"
        }
    )

    print("25 * 40 =", result)

    print("\nTool system working.")
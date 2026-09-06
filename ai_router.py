import re


# =========================================================
# LOCAL AI ROUTER
# =========================================================

def local_route(question):

    q = question.lower().strip()


    # =====================================================
    # CALCULATOR
    # =====================================================

    math_chars = set(
        "0123456789+-*/().% "
    )

    if q and all(
        char in math_chars
        for char in q
    ):

        return "calculator"


    calculator_keywords = [
        "calculate",
        "calculator",
        "solve",
        "percentage",
        "percent",
        "multiply",
        "multiplication",
        "divide",
        "division",
        "addition",
        "subtraction"
    ]


    if any(
        keyword in q
        for keyword in calculator_keywords
    ):

        if re.search(
            r"\d+\s*[\+\-\*\/\%]\s*\d+",
            q
        ):

            return "calculator"


    # =====================================================
    # PDF / NOTES ROUTING
    # =====================================================
    #
    # IMPORTANT:
    # Normal study questions should NOT automatically
    # go to PDF.
    #
    # PDF is selected only when the user clearly asks
    # about their notes / uploaded documents / PDF.
    # =====================================================

    pdf_keywords = [

        "according to my notes",

        "according to the notes",

        "from my notes",

        "in my notes",

        "my notes",

        "my pdf",

        "my document",

        "my documents",

        "uploaded pdf",

        "uploaded document",

        "uploaded documents",

        "according to the pdf",

        "according to my pdf",

        "from the pdf",

        "from my pdf",

        "in the pdf",

        "in my pdf",

        "from the document",

        "from my document",

        "in the document",

        "in my document",

        "study material",

        "study notes",

        "class notes",

        "lecture notes",

        "university notes",

        "notes according",

        "pdf according",

        "pdf notes",

        "pdf file",

        "uploaded file"
    ]


    if any(
        keyword in q
        for keyword in pdf_keywords
    ):

        return "pdf"


    # =====================================================
    # WEB SEARCH
    # =====================================================

    web_keywords = [

        "latest",

        "current",

        "today",

        "news",

        "recent",

        "live",

        "right now",

        "this week",

        "this month",

        "latest version",

        "current version",

        "latest price",

        "current price",

        "weather",

        "who won",

        "score today",

        "what happened today",

        "recent update",

        "new update"
    ]


    if any(
        keyword in q
        for keyword in web_keywords
    ):

        return "web"


    # =====================================================
    # GENERAL AI
    # =====================================================

    return "general"


# =========================================================
# MAIN ROUTER
# =========================================================

def route_question(question):

    if not question:

        return "general"


    tool = local_route(
        question
    )


    print(
        f"Local router selected: {tool}"
    )


    return tool


# =========================================================
# TOOL DISPLAY NAMES
# =========================================================

def get_tool_display_name(tool):

    names = {

        "calculator":
            "🧮 Calculator",

        "pdf":
            "📚 PDF Knowledge",

        "web":
            "🌐 Web Search",

        "general":
            "🤖 Gemini AI"

    }


    return names.get(
        tool,
        "🤖 Gemini AI"
    )
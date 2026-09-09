import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

def web_search(query):
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return "Web search API key is missing."

    tavily = TavilyClient(api_key=api_key)

    response = tavily.search(
        query=query,
        max_results=5
    )

    results = []

    for result in response.get("results", []):
        results.append(
            f"Title: {result.get('title')}\n"
            f"Content: {result.get('content')}\n"
            f"URL: {result.get('url')}"
        )

    return "\n\n".join(results)
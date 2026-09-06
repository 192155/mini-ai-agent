import os

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


def web_search(query):

    api_key = os.getenv(
        "TAVILY_API_KEY"
    )

    if not api_key:

        return (
            "Tavily API key is not configured."
        )

    try:

        client = TavilyClient(
            api_key=api_key
        )

        response = client.search(
            query=query,
            max_results=3,
            search_depth="basic"
        )

        results = response.get(
            "results",
            []
        )

        if not results:

            return (
                "No web search results found."
            )

        formatted = []

        for result in results:

            title = result.get(
                "title",
                "No title"
            )

            content = result.get(
                "content",
                ""
            )

            url = result.get(
                "url",
                ""
            )

            content = content[:1500]

            formatted.append(
                f"Title: {title}\n"
                f"Content: {content}\n"
                f"URL: {url}"
            )

        return "\n\n".join(
            formatted
        )

    except Exception as e:

        return (
            "Web search failed:\n"
            + str(e)
        )
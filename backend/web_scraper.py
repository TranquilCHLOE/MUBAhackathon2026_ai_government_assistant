"""Web scraping with Firecrawl – for real-time claim verification."""

import os
import re
import json
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")

# Initialize the client (if API key exists)
firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY) if FIRECRAWL_API_KEY else None


def is_url(text: str) -> bool:
    """Check if the input looks like a URL."""
    url_pattern = re.compile(r'^https?://[^\s]+$')
    return bool(url_pattern.match(text.strip()))


def parse_firecrawl_response(response) -> list:
    """
    Parse Firecrawl's response into a list of results.
    Handles: list, dict with 'data' key, SearchData object, or string with 'web=' format.
    """
    # --- 1. If it's already a list, return it ---
    if isinstance(response, list):
        return response
    
    # --- 2. If it's a dict with 'data' or 'results' ---
    if isinstance(response, dict):
        return response.get('data', response.get('results', []))
    
    # --- 3. If it's a SearchData object (has 'web' attribute) ---
    if hasattr(response, 'web'):
        # The 'web' attribute is a list of SearchResultWeb objects
        web_results = getattr(response, 'web', [])
        if web_results:
            # Convert each SearchResultWeb to a dict
            results = []
            for item in web_results:
                if hasattr(item, 'url'):
                    result = {'url': item.url}
                    if hasattr(item, 'title'):
                        result['title'] = item.title
                    if hasattr(item, 'description'):
                        result['description'] = item.description
                    results.append(result)
            return results
    
    # --- 4. If it's a string with "web=..." format ---
    if isinstance(response, str):
        # Look for 'web=[' pattern
        match = re.search(r'web=\[(.*?)\]$', response, re.DOTALL)
        if match:
            try:
                # Try to extract URLs using regex
                url_matches = re.findall(r"url='(https?://[^']+)'", response)
                title_matches = re.findall(r"title='([^']+)'", response)
                
                results = []
                for i, url in enumerate(url_matches):
                    title = title_matches[i] if i < len(title_matches) else "No title"
                    results.append({'url': url, 'title': title})
                return results
            except:
                pass
    
    # --- 5. Try to get 'data' attribute if it exists ---
    if hasattr(response, 'data'):
        return getattr(response, 'data', [])
    
    return []


def _get_field(obj, field: str, default=None):
    """Read a field from either a dict or a pydantic-style object
    (newer firecrawl-py returns Document objects, not dicts, from
    scrape_url/search — this works with both)."""
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def fetch_web_content(query_or_url: str, is_search: bool = False) -> str:
    """
    Fetches content from the web using Firecrawl.
    
    Args:
        query_or_url: A URL to scrape OR a search query.
        is_search: If True, performs a web search and scrapes the top result.
    
    Returns:
        Clean text content (Markdown or plain text).
    """
    if not firecrawl_app:
        return "Firecrawl API key not configured. Please set FIRECRAWL_API_KEY in .env"

    try:
        if is_search:
            # --- Perform the search ---
            search_response = firecrawl_app.search(query_or_url, limit=1)
            
            print("=" * 60)
            print(f"🔍 Response type: {type(search_response)}")
            
            # Debug: Print the object's attributes if it's a SearchData
            if hasattr(search_response, '__dict__'):
                print(f"🔍 Object attributes: {dir(search_response)}")
                if hasattr(search_response, 'web'):
                    print(f"🔍 web attribute: {search_response.web}")
            print("=" * 60)

            # --- Parse the response using our helper ---
            results_list = parse_firecrawl_response(search_response)

            if results_list and len(results_list) > 0:
                top_result = results_list[0]
                
                # Try multiple possible URL field names
                top_url = (
                    top_result.get('url') if isinstance(top_result, dict) else 
                    getattr(top_result, 'url', None)
                )
                
                if top_url:
                    print(f"🌐 Scraping search result URL: {top_url}")
                    # Scrape the full content of that URL
                    scrape_result = firecrawl_app.scrape_url(top_url)
                    content = (
                        _get_field(scrape_result, 'markdown')
                        or _get_field(scrape_result, 'content')
                        or ""
                    )
                    return f"Source: {top_url}\n\n{content[:3000]}"
            
            return "No search results found."
        
        else:
            # --- Scrape a direct URL ---
            print(f"🔗 Scraping direct URL: {query_or_url}")
            scrape_result = firecrawl_app.scrape_url(query_or_url)
            content = (
                _get_field(scrape_result, 'markdown')
                or _get_field(scrape_result, 'content')
                or ""
            )
            return f"Source: {query_or_url}\n\n{content[:3000]}"
    
    except Exception as e:
        print(f"❌ Firecrawl exception: {e}")
        return f"Error fetching web content: {str(e)}"
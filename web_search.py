import requests
import re
import urllib.parse
import html


HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
}


def _clean(value):
    return re.sub(r"<.*?>", "", html.unescape(value or "")).strip()


def _parse_lite(html_text, max_results):
    parsed = []
    blocks = html_text.split("class='result-link'")
    for block in blocks[1:]:
        link_match = re.search(r"href=['\"]([^'\"]+)['\"]>(.*?)</a>", block, re.DOTALL)
        if not link_match:
            continue
        href = link_match.group(1)
        uddg_match = re.search(r"uddg=([^&]+)", href)
        url = urllib.parse.unquote(uddg_match.group(1)) if uddg_match else href
        snippet_match = re.search(r"class='result-snippet'[^>]*>(.*?)</td>", block, re.DOTALL)
        parsed.append({
            "title": _clean(link_match.group(2)),
            "url": url,
            "snippet": _clean(snippet_match.group(1)) if snippet_match else "",
        })
    return parsed[:max_results]


def _parse_bing(html_text, max_results):
    parsed = []
    for block in html_text.split('<li class="b_algo"')[1:]:
        link_match = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
        if not link_match:
            continue
        snippet_match = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL)
        parsed.append({
            "title": _clean(link_match.group(2)),
            "url": html.unescape(link_match.group(1)),
            "snippet": _clean(snippet_match.group(1)) if snippet_match else "",
        })
    return parsed[:max_results]

def search_web_ddg(query, max_results=4):
    """
    Search the web using DuckDuckGo HTML search.
    Returns a list of dicts: [{'title': str, 'url': str, 'snippet': str}]
    """
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=8)
        if res.status_code != 200:
            # The HTML endpoint occasionally returns 202; Lite is a reliable fallback.
            lite_url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
            res = requests.get(lite_url, headers=HEADERS, timeout=8)
            if res.status_code in (200, 202):
                lite_results = _parse_lite(res.text, max_results)
                if lite_results:
                    return lite_results

            bing_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
            bing = requests.get(bing_url, headers=HEADERS, timeout=8)
            if bing.status_code == 200:
                return _parse_bing(bing.text, max_results)
            print(f"Web search returned statuses DDG={res.status_code}, Bing={bing.status_code}")
            return []
            
        html = res.text
        # Split page by result wrapper class
        blocks = html.split('class="result results_links results_links_deep web-result ')
        
        parsed = []
        for block in blocks[1:]:  # first block is header HTML
            if 'badge--ad' in block:
                continue
                
            # Extract title and URL
            title_match = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
            if not title_match:
                continue
                
            href = title_match.group(1)
            title = _clean(title_match.group(2))
            
            # Extract real URL from redirection wrapper
            parsed_url = href
            uddg_match = re.search(r'uddg=([^&]+)', href)
            if uddg_match:
                parsed_url = urllib.parse.unquote(uddg_match.group(1))
                
            # Extract snippet
            snippet = ""
            snippet_match = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL)
            if snippet_match:
                snippet = _clean(snippet_match.group(1))
                
            parsed.append({
                "title": title,
                "url": parsed_url,
                "snippet": snippet
            })
            
        if parsed:
            return parsed[:max_results]

        bing_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
        bing = requests.get(bing_url, headers=HEADERS, timeout=8)
        return _parse_bing(bing.text, max_results) if bing.status_code == 200 else []
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
        return []

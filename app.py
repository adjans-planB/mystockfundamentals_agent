"""
Portfolio Research Agent
========================
Flask service deployed on Railway.
Receives portfolio context from n8n, runs an agentic news research loop
using the Anthropic tool-use API, returns a structured HTML morning briefing.

Endpoint: POST /analyse
Payload:  { "ig_positions": [...], "roc_positions": [...], "calendar": {...} }
Returns:  { "report_html": "...", "report_text": "...", "stocks_covered": [...] }
"""

import os
import json
import logging
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYDNEY = ZoneInfo("Australia/Sydney")
MODEL  = "claude-opus-4-5"

RAPIDAPI_KEY     = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HEADERS = {
    "x-rapidapi-key":  RAPIDAPI_KEY,
    "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com",
}

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_news",
        "description": (
            "Search Yahoo Finance for recent news articles about a stock. "
            "Returns a list of articles with title, URL, source, and published date. "
            "Call this first for each stock to discover what news is available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "ASX ticker with .AX suffix e.g. CBA.AX, BHP.AX — or US ticker e.g. AAPL"
                },
                "company_name": {
                    "type": "string",
                    "description": "Company name for supplementary search context"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "fetch_article",
        "description": (
            "Fetch and read the full text of a news article from its URL. "
            "Use this after search_news to read important stories in full. "
            "Prioritise articles about earnings, guidance, dividends, management changes, "
            "regulatory issues, or macro events affecting the stock."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL of the article to read"
                },
                "ticker": {
                    "type": "string",
                    "description": "Ticker this article relates to (for logging)"
                }
            },
            "required": ["url"]
        }
    }
]

# ── Tool implementations ──────────────────────────────────────────────────────

def search_news(ticker: str, company_name: str = "") -> str:
    """Search Yahoo Finance news via RapidAPI."""
    try:
        url    = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/news"
        params = {"tickers": ticker, "type": "ALL"}
        resp   = requests.get(url, headers=RAPIDAPI_HEADERS, params=params, timeout=10)

        if resp.status_code != 200:
            log.warning(f"News search for {ticker} returned {resp.status_code}")
            return json.dumps({"error": f"API returned {resp.status_code}", "articles": []})

        data     = resp.json()
        articles = data.get("body", [])

        results = []
        for a in articles[:8]:
            results.append({
                "title":     a.get("title", ""),
                "url":       a.get("url", a.get("link", "")),
                "source":    a.get("source", a.get("publisher", "")),
                "published": a.get("pubDate", a.get("published", "")),
            })

        log.info(f"  search_news({ticker}): {len(results)} articles")
        return json.dumps({"ticker": ticker, "articles": results})

    except Exception as e:
        log.error(f"search_news error for {ticker}: {e}")
        return json.dumps({"error": str(e), "articles": []})


def fetch_article(url: str, ticker: str = "") -> str:
    """Fetch and extract readable text from an article URL."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=12, allow_redirects=True)

        if resp.status_code != 200:
            return json.dumps({"error": f"HTTP {resp.status_code}", "content": ""})

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "iframe", "noscript"]):
            tag.decompose()

        # Try semantic article containers first
        article = (
            soup.find("article")
            or soup.find("div", class_=re.compile(r"article|content|story|body", re.I))
            or soup.find("main")
        )
        text = (article or soup).get_text(separator=" ", strip=True)

        # Truncate to keep context window sane
        if len(text) > 3000:
            text = text[:3000] + "… [truncated]"

        log.info(f"  fetch_article({ticker}, {url[:60]}): {len(text)} chars")
        return json.dumps({"url": url, "content": text})

    except Exception as e:
        log.error(f"fetch_article error: {e}")
        return json.dumps({"error": str(e), "content": ""})


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "search_news":
        return search_news(**inputs)
    if name == "fetch_article":
        return fetch_article(**inputs)
    return json.dumps({"error": f"Unknown tool: {name}"})


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    today = datetime.now(SYDNEY).strftime("%A %d %B %Y")
    return f"""You are a professional financial analyst preparing a daily morning briefing for an ASX trader.
Today is {today} (Sydney time).

The trader runs TWO portfolios:

1. ROC PORTFOLIO (Primary Strategy)
   Nick Radge / TheChartist momentum system. Stocks ranked by 200-day Rate of Change,
   ATR-based position sizing, monthly rebalance on the last trading day of each month.
   Each position shows: ticker, shares, entry_price, entry_value, roc_pct (rank score), weight_pct.
   KEY FLAG: Identify any ⚠️ news that could cause a stock to lose momentum or drop out of
   the top 40 ROC ranking — profit warnings, structural decline, earnings misses, sector rotation.

2. IG CFD POSITIONS (Secondary)
   Shorter-term CFD positions. Each shows: symbol, quantity, opening_price, running_profit_loss.
   Flag upcoming earnings/dividend dates from the calendar section.

RESEARCH PROCESS:
- Call search_news for EVERY stock across both portfolios
- Call fetch_article for the 2-3 most material articles per stock
- Prioritise: earnings results, guidance changes, dividends, M&A, regulatory events,
  macro themes affecting the sector

REPORT FORMAT — produce clean HTML:

<h2>🌅 Morning Briefing — {today}</h2>

<h3>📈 ROC Portfolio</h3>
Table or list: for each position show ticker | entry $ | weight% | ROC rank
Then 2-4 sentence news summary. Flag ⚠️ thesis risks.

<h3>💼 IG CFD Positions</h3>
For each: ticker | P&L | upcoming events from calendar
Then news summary.

<h3>🌍 Market & Macro</h3>
2-3 sentences on XJO/ASX200 context and any macro themes.

<h3>⚡ Key Actions Today</h3>
3-5 bullet points — concrete things to act on or watch closely.

Be direct and concise. One line is enough for stocks with no material news.
Do not pad or invent commentary."""


def build_user_prompt(payload: dict) -> str:
    ig   = payload.get("ig_positions", [])
    roc  = payload.get("roc_positions", [])
    cal  = payload.get("calendar", {})

    all_tickers = sorted(set(
        [p.get("symbol", p.get("ticker", "")) for p in ig] +
        [p.get("ticker", "") for p in roc]
    ))
    all_tickers = [t for t in all_tickers if t]

    return f"""Research all stocks below and produce the morning briefing.

=== ROC PORTFOLIO (Primary — {len(roc)} positions) ===
{json.dumps(roc, indent=2) if roc else "No active ROC positions."}

=== IG CFD POSITIONS ({len(ig)} positions) ===
{json.dumps(ig, indent=2) if ig else "No active IG CFD positions."}

=== EARNINGS & DIVIDENDS CALENDAR ===
{json.dumps(cal, indent=2) if cal else "No calendar data provided."}

=== TICKERS TO RESEARCH ({len(all_tickers)} total) ===
{', '.join(all_tickers)}

For each ticker: call search_news first, then fetch_article for material stories.
After all research is complete, write the full HTML briefing."""


# ── Agentic loop ──────────────────────────────────────────────────────────────

def run_agent(payload: dict) -> dict:
    messages = [{"role": "user", "content": build_user_prompt(payload)}]

    max_iterations = 60   # safety cap — large portfolios need more tool calls
    iteration      = 0

    log.info(
        f"Agent starting — "
        f"{len(payload.get('roc_positions', []))} ROC positions, "
        f"{len(payload.get('ig_positions', []))} IG positions"
    )

    while iteration < max_iterations:
        iteration += 1
        log.info(f"Iteration {iteration}")

        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=build_system_prompt(),
            tools=TOOLS,
            messages=messages,
        )

        log.info(f"  stop={response.stop_reason} blocks={[b.type for b in response.content]}")

        # Done — pull final text
        if response.stop_reason == "end_turn":
            final = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            log.info(f"Agent complete in {iteration} iterations")
            return {"report": final, "iterations": iteration}

        # Process tool calls
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            results = []
            for block in response.content:
                if block.type == "tool_use":
                    log.info(f"  {block.name}({str(block.input)[:100]})")
                    result = dispatch_tool(block.name, block.input)
                    results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })
                    time.sleep(0.25)   # gentle rate limiting

            messages.append({"role": "user", "content": results})
            continue

        log.warning(f"Unexpected stop_reason: {response.stop_reason}")
        break

    # Hit cap — return whatever we have
    log.warning(f"Agent hit iteration cap ({max_iterations})")
    last_text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    return {
        "report":     last_text or "Agent reached iteration limit without completing.",
        "iterations": iteration
    }


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now(SYDNEY).isoformat()})


@app.route("/analyse", methods=["POST"])
def analyse():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "Empty payload"}), 400

        log.info(
            f"POST /analyse — "
            f"ig={len(payload.get('ig_positions', []))}, "
            f"roc={len(payload.get('roc_positions', []))}"
        )

        result = run_agent(payload)
        report = result["report"]

        # Strip HTML for plain-text version
        plain = re.sub(r"<[^>]+>", "", report).strip()

        stocks_covered = sorted(set(
            [p.get("symbol", p.get("ticker", "")) for p in payload.get("ig_positions", [])] +
            [p.get("ticker", "")                   for p in payload.get("roc_positions", [])]
        ))

        return jsonify({
            "report_html":    report,
            "report_text":    plain,
            "stocks_covered": [t for t in stocks_covered if t],
            "iterations":     result["iterations"],
            "generated_at":   datetime.now(SYDNEY).isoformat(),
        })

    except Exception as e:
        log.exception("Error in /analyse")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

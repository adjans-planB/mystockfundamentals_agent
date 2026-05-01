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
MODEL  = "claude-sonnet-4-6"

RAPIDAPI_KEY     = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HEADERS = {
    "x-rapidapi-key":  RAPIDAPI_KEY,
    "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com",
}

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

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

        # Truncate to keep token usage manageable
        if len(text) > 1500:
            text = text[:1500] + "… [truncated]"

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

def build_system_prompt(headlines_pre_loaded: bool = False) -> str:
    today = datetime.now(SYDNEY).strftime("%A %d %B %Y")

    if headlines_pre_loaded:
        research_instructions = """RESEARCH PROCESS:
Headlines from 8 sources (ASX Announcements, Yahoo Finance, SMH, Stockhead, Rask Media,
Investing.com, Investing.com Fundamental, BetaShares) have already been gathered and are
provided in the payload for each IG position.

For ROC positions (no pre-loaded headlines): call search_news to find articles, then
fetch_article for material ones.

For IG positions with headlines: DO NOT call search_news — the headlines are already there.
Instead, scan the provided headlines and call fetch_article ONLY for URLs that look material
(earnings results, guidance, profit warnings, dividends, M&A, regulatory). Skip headlines
that are generic market commentary or clearly irrelevant."""
    else:
        research_instructions = """RESEARCH PROCESS:
- Call search_news for EVERY stock across both portfolios
- Call fetch_article for the 2-3 most material articles per stock
- Prioritise: earnings results, guidance changes, dividends, M&A, regulatory events,
  macro themes affecting the sector"""

    return f"""You are a professional financial analyst preparing a daily morning briefing for an ASX trader.
Today is {today} (Sydney time).

The trader runs TWO portfolios:

1. ROC PORTFOLIO (Primary Strategy — read carefully)
   Nick Radge / TheChartist momentum system. Stocks ranked by 200-day Rate of Change (ROC),
   ATR-based position sizing, monthly rebalance on the last trading day of each month.

   Each held position includes: ticker, rank (by portfolio weight), entry_price, current_price,
   pnl_pct (live P&L %), entry_value, roc_pct (200d ROC score), weight_pct.

   ROC POSITION RULES:
   - For positions WITH material news: write 2-4 sentences covering the news and its impact
   - For positions with NO material news: write exactly ONE line:
     "Momentum intact — [sector] exposure. Rank #{{rank}} in portfolio."
     DO NOT write "No material news found" — it adds no value.
   - Flag ⚠️ ONLY for: profit warnings, earnings misses, guidance cuts, structural decline,
     or rank drop of 3+ positions vs yesterday

   SECTOR CONCENTRATION: After listing all positions, add a short paragraph noting any sector
   representing >25% combined weight. Flag correlated drawdown risk if multiple sectors
   are moving together (e.g. gold + lithium = commodity cycle correlation).

2. ROC TOP 40 UNIVERSE (Opportunity Analysis)
   The full top 40 ranked stocks are provided with today's rank, yesterday's rank, and
   whether each stock is currently held.

   ROC MOVERS ANALYSIS — the payload includes roc_movers with these fields:
   - held_rank_changes: ALL held positions with rank data sorted by magnitude of change.
     Show every held position's rank movement, e.g. "LTR #4→#2 (▲2)" or "NHC #6→#9 (▼3)".
     ⚠️ Flag any held position falling 3+ ranks as momentum concern.
   - new_entries: Stocks NEW to top 40 today not currently held — potential rebalance buys.
     For each, note sector and whether it complements or duplicates existing exposure.
   - exits: Stocks that FELL OUT of top 40 today. If 3 stocks entered, 3 exited — name them.
     If any held position is in exits (held_exits), this is CRITICAL — flag immediately.
   - at_risk: Held positions ranked 31-40 IN THE TOP 40 UNIVERSE (not portfolio rank).
     Rank here means their position within the top 40 ROC stocks — rank 31-40 means they
     are near the boundary of the universe and could drop out before the next rebalance.
     Portfolio rank (#1-20 in held positions) is different from universe rank (#1-40).

3. IG CFD POSITIONS (Secondary)
   Shorter-term CFD positions. Each shows: symbol, quantity, opening_price, current_price,
   rs_data (relative strength vs XJO), has_price_sensitive (ASX price-sensitive announcement flag).
   Flag upcoming earnings/dividend dates from the calendar section.

{research_instructions}

REPORT FORMAT — produce clean HTML:

<h2>🌅 Morning Briefing — {today}</h2>

<h3>📈 ROC Portfolio ({today})</h3>
RANKING SYSTEM — ONE RANK ONLY:
All ranks use a single system: stocks ordered by 200-day ROC score descending across the full
tracked universe. Rank #1 = highest ROC score. Your 20 held stocks each have a rank within
this list — e.g. PLS at #2 means it has the 2nd highest ROC of all tracked ASX stocks today.
This rank changes daily as ROC scores update. A stock dropping from #2 to #5 means 3 stocks
have overtaken it in ROC momentum. The top 40 is just the top 40 stocks in this same list.
Use this rank everywhere — in position headers, in the movers section, in the at-risk section.

For each position use this exact structure — each stock MUST be wrapped in its own div:

<div style="margin-bottom: 12px; padding: 8px 0; border-bottom: 1px solid #eee;">
<p style="margin: 0 0 4px 0;"><b>TICKER</b> | ROC Rank #[rank] | Entry $X → Current $Y | P&L: +/-Z% | Weight: W%</p>
<p style="margin: 0; color: #444;">One line or 2-4 sentence news summary here. Flag ⚠️ thesis risks.</p>
</div>

After all positions, add:
<h4>🏭 Sector Concentration</h4>
One paragraph on portfolio sector exposure and correlation risk.

<h3>📊 ROC Top 40 Movers</h3>
Use this exact structure:

<p><b>📊 ROC Rank Changes (held positions only, 1=highest ROC in full universe):</b></p>
<ul style="margin: 4px 0 12px 0;">
  <li>TICKER — #X → #Y (▲N or ▼N) — note only if change ≥ 2, otherwise list as one line</li>
</ul>
List all held positions that have rank data, sorted by magnitude of change (biggest movers first).
Use rank_today and rank_yesterday from held_rank_changes — these are ROC ranks in the full universe.
Flag ⚠️ on any position falling 3+ ranks (momentum fading).
Flag 🚀 on any position rising 3+ ranks (momentum accelerating).

<p><b>🆕 New Entrants (Not Held — Rebalance Candidates):</b></p>
<ul style="margin: 4px 0 12px 0;">
  <li>TICKER — Rank #N @ $X — sector, and whether it duplicates existing exposure</li>
</ul>

<p><b>👋 Exits from Top 40:</b></p>
<ul style="margin: 4px 0 12px 0;">
  <li>TICKER — was Rank #N — dropped out today</li>
</ul>
If a held position appears here, flag ⚠️ CRITICAL — position dropped out of universe.

<p><b>⚠️ Held Positions in Universe At-Risk Zone (Universe Rank 31-40):</b></p>
<ul style="margin: 4px 0 12px 0;">
  <li>TICKER — Universe Rank #N — could drop out of top 40 before next rebalance</li>
</ul>
Use the at_risk field from roc_movers. If none, write "None today."

<h3>💼 IG CFD Positions</h3>
For each position use this exact structure:

<div style="margin-bottom: 12px; padding: 8px 0; border-bottom: 1px solid #eee;">
<p style="margin: 0 0 4px 0;"><b>TICKER</b> | Current $X | P&L: +/-Y% | RS: [trend] | [Earnings/Dividend date if applicable]</p>
<p style="margin: 0; color: #444;">News summary. <b>Bold price-sensitive ASX announcements.</b></p>
</div>

<h3>🌍 Market & Macro</h3>
THIS SECTION IS MANDATORY — always include it even if brief.
Use market_context data: report overnight moves for S&P 500, Nasdaq, ASX 200 with exact figures.
Include breadth stats (advancers/decliners, % above MA50).
Note top RS movers from asx200_top_rs.
Note any macro themes affecting both books (commodities, USD/AUD, rates, geopolitics).

<h3>⚡ Key Actions Today</h3>
3-5 concrete bullet points — things to act on or monitor today specifically.
Include rebalance countdown if within 10 trading days.

Be direct and professional. No padding. Numbers over adjectives."""


def build_user_prompt(payload: dict) -> str:
    ig                   = payload.get("ig_positions", [])
    roc                  = payload.get("roc_positions", [])
    roc_top40            = payload.get("roc_top40", [])
    roc_movers           = payload.get("roc_movers", {})
    cal                  = payload.get("calendar", {})
    mkt                  = payload.get("market_context", {})
    headlines_pre_loaded = payload.get("headlines_pre_loaded", False)

    all_tickers = sorted(set(
        [p.get("symbol", p.get("ticker", "")) for p in ig] +
        [p.get("ticker", "") for p in roc]
    ))
    all_tickers = [t for t in all_tickers if t]

    research_note = (
        "Headlines are PRE-LOADED for IG positions — do NOT call search_news for those. "
        "Call fetch_article for any material URLs in the headlines. "
        "For ROC positions, call search_news as normal (no pre-loaded headlines)."
        if headlines_pre_loaded
        else "Call search_news then fetch_article for each ticker."
    )

    # Build rebalance countdown
    from datetime import date
    import calendar as cal_mod
    today_d = date.today()
    # Last trading day of current month (approximate — last weekday)
    last_day = date(today_d.year, today_d.month,
                    cal_mod.monthrange(today_d.year, today_d.month)[1])
    while last_day.weekday() > 4:
        last_day = date(last_day.year, last_day.month, last_day.day - 1)
    trading_days_left = sum(
        1 for d in range((last_day - today_d).days + 1)
        if date.fromordinal(today_d.toordinal() + d).weekday() < 5
    )

    return f"""Research all stocks and produce the morning briefing.

NOTE: {research_note}

=== ROC PORTFOLIO — {len(roc)} HELD POSITIONS ===
(includes rank, current_price, pnl_pct vs entry — use these in the report)
{json.dumps(roc, indent=2) if roc else "No active ROC positions."}

=== ROC TOP 40 UNIVERSE (full ranked list with rank changes) ===
{json.dumps(roc_top40, indent=2) if roc_top40 else "Top 40 data not available."}

=== ROC MOVERS TODAY ===
{json.dumps(roc_movers, indent=2) if roc_movers else "No mover data available."}

=== IG CFD POSITIONS — {len(ig)} positions (headlines pre-loaded) ===
{json.dumps(ig, indent=2) if ig else "No active IG CFD positions."}

=== EARNINGS & DIVIDENDS CALENDAR ===
{cal.get('calendar_text') or json.dumps(cal, indent=2) if cal else "No calendar data provided."}

=== MARKET CONTEXT ===
{json.dumps(mkt, indent=2) if mkt else "No market context provided."}

=== REBALANCE COUNTDOWN ===
Next ROC rebalance: {last_day.strftime("%d %B %Y")} ({trading_days_left} trading days away)

=== ALL TICKERS TO RESEARCH ({len(all_tickers)} total) ===
{', '.join(all_tickers)}

After all research, write the full HTML briefing."""


# ── Agentic loop ──────────────────────────────────────────────────────────────

def run_agent(payload: dict) -> dict:
    headlines_pre_loaded = payload.get("headlines_pre_loaded", False)
    messages  = [{"role": "user", "content": build_user_prompt(payload)}]

    max_iterations      = 60
    iteration           = 0
    news_cache: dict[str, list] = {}   # ticker -> list of article dicts
    total_input_tokens  = 0
    total_output_tokens = 0

    log.info(
        f"Agent starting — "
        f"{len(payload.get('roc_positions', []))} ROC positions, "
        f"{len(payload.get('ig_positions', []))} IG positions, "
        f"headlines_pre_loaded={headlines_pre_loaded}"
    )

    while iteration < max_iterations:
        iteration += 1
        log.info(f"Iteration {iteration}")

        # Retry on rate limit with backoff
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=8192,
                    system=build_system_prompt(headlines_pre_loaded),
                    tools=TOOLS,
                    messages=messages,
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < 2:
                    wait = 60 * (attempt + 1)
                    log.warning(f"Rate limit hit — waiting {wait}s before retry {attempt+2}/3")
                    time.sleep(wait)
                else:
                    raise

        # Track token usage
        total_input_tokens  += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        log.info(
            f"  stop={response.stop_reason} blocks={[b.type for b in response.content]} "
            f"tokens=({response.usage.input_tokens}in/{response.usage.output_tokens}out)"
        )

        if response.stop_reason == "end_turn":
            final = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            # Calculate cost
            sonnet_cost = (total_input_tokens * 3 + total_output_tokens * 15) / 1_000_000
            opus_cost   = (total_input_tokens * 15 + total_output_tokens * 75) / 1_000_000
            log.info(
                f"Agent complete in {iteration} iterations — "
                f"tokens: {total_input_tokens:,} in / {total_output_tokens:,} out — "
                f"cost: ${sonnet_cost:.4f} USD (Sonnet) / ${opus_cost:.4f} USD (Opus equiv)"
            )
            return {
                "report":               final,
                "iterations":           iteration,
                "news_cache":           news_cache,
                "total_input_tokens":   total_input_tokens,
                "total_output_tokens":  total_output_tokens,
                "estimated_cost_usd":   round(sonnet_cost, 4),
            }

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            results = []
            for block in response.content:
                if block.type == "tool_use":
                    log.info(f"  {block.name}({str(block.input)[:100]})")
                    result = dispatch_tool(block.name, block.input)

                    # Cache articles returned by search_news for write-back later
                    if block.name == "search_news":
                        try:
                            parsed  = json.loads(result)
                            ticker  = block.input.get("ticker", "")
                            articles = parsed.get("articles", [])
                            if ticker and articles:
                                news_cache.setdefault(ticker, [])
                                news_cache[ticker].extend(articles)
                        except Exception:
                            pass

                    results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })
                    time.sleep(0.25)

            messages.append({"role": "user", "content": results})
            continue

        log.warning(f"Unexpected stop_reason: {response.stop_reason}")
        break

    log.warning(f"Agent hit iteration cap ({max_iterations})")
    last_text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    return {
        "report":     last_text or "Agent reached iteration limit without completing.",
        "iterations": iteration,
        "news_cache": news_cache,
    }


# ── Supabase write-back ───────────────────────────────────────────────────────

def save_to_stock_insights(
    stocks_covered: list[str],
    report_html: str,
    payload: dict,
    news_cache: dict,
) -> None:
    """
    Upsert one row per stock into stock_insights.
    news_cache: dict of ticker -> list of article dicts collected during the run.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.warning("SUPABASE_URL/KEY not set — skipping stock_insights write-back")
        return

    today      = datetime.now(SYDNEY).date().isoformat()
    ig_map     = {p.get("symbol", p.get("ticker", "")): p for p in payload.get("ig_positions", [])}
    roc_map    = {p.get("ticker", ""): p                  for p in payload.get("roc_positions", [])}
    cal        = payload.get("calendar", {})

    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "resolution=merge-duplicates",  # upsert behaviour
    }

    rows = []
    for ticker in stocks_covered:
        articles  = news_cache.get(ticker, [])
        headlines = [a.get("title", "") for a in articles if a.get("title")]
        urls      = [a.get("url", "")   for a in articles if a.get("url")]

        # Build key_metrics from whichever portfolio has this ticker
        ig_pos  = ig_map.get(ticker, {})
        roc_pos = roc_map.get(ticker, {})
        key_metrics = {}
        if ig_pos:
            key_metrics["running_pnl"]   = ig_pos.get("running_profit_loss")
            key_metrics["opening_price"] = ig_pos.get("opening_price")
            key_metrics["quantity"]      = ig_pos.get("quantity")
            key_metrics["portfolio"]     = "IG_CFD"
        if roc_pos:
            key_metrics["entry_price"] = roc_pos.get("entry_price")
            key_metrics["entry_value"] = roc_pos.get("entry_value")
            key_metrics["weight_pct"]  = roc_pos.get("weight_pct")
            key_metrics["roc_pct"]     = roc_pos.get("roc_pct")
            key_metrics["portfolio"]   = key_metrics.get("portfolio", "") + " ROC"

        # Pull upcoming events for this ticker from calendar
        upcoming = {}
        for e in cal.get("upcoming_earnings", []):
            if e.get("ticker") == ticker or e.get("symbol") == ticker:
                upcoming["earnings"] = e
        for d in cal.get("upcoming_dividends", []):
            if d.get("ticker") == ticker or d.get("symbol") == ticker:
                upcoming["dividend"] = d

        rows.append({
            "symbol":          ticker,
            "report_date":     today,
            "headlines":       headlines,
            "key_metrics":     key_metrics,
            "upcoming_events": upcoming,
            "ai_summary":      report_html,   # full report stored on each row; query by date to retrieve
            "source_links":    urls,
        })

    log.info(f"stock_insights: {len(rows)} rows to write, stocks={stocks_covered[:5]}")
    log.info(f"stock_insights: SUPABASE_URL={'set' if SUPABASE_URL else 'MISSING'}, KEY={'set' if SUPABASE_KEY else 'MISSING'}")

    if not rows:
        log.warning("stock_insights: rows is empty — skipping")
        return

    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/stock_insights",
            headers=headers,
            json=rows,
            timeout=15,
        )
        log.info(f"stock_insights response: {resp.status_code} — {resp.text[:300]}")
        if resp.status_code in (200, 201):
            log.info(f"stock_insights: wrote {len(rows)} rows for {today}")
        elif resp.status_code == 409:
            # Conflict on unique key — try plain insert without upsert header
            headers_plain = {k: v for k, v in headers.items() if k != "Prefer"}
            resp2 = requests.post(
                f"{SUPABASE_URL}/rest/v1/stock_insights",
                headers=headers_plain,
                json=rows,
                timeout=15,
            )
            log.info(f"stock_insights plain insert: {resp2.status_code} — {resp2.text[:200]}")
        else:
            log.warning(f"stock_insights write failed: {resp.status_code} {resp.text[:500]}")
    except Exception as e:
        log.error(f"stock_insights write error: {e}")


# ── Flask routes ──────────────────────────────────────────────────────────────


@app.route("/debug", methods=["POST"])
def debug():
    """Echo back what we receive — helps diagnose n8n body format issues."""
    raw = request.get_data(as_text=True)
    content_type = request.content_type
    try:
        parsed = request.get_json(force=True)
        parse_ok = True
        parse_type = type(parsed).__name__
    except Exception as e:
        parsed = None
        parse_ok = False
        parse_type = str(e)
    return jsonify({
        "content_type": content_type,
        "raw_length": len(raw),
        "raw_preview": raw[:500],
        "parse_ok": parse_ok,
        "parse_type": parse_type,
        "parsed_keys": list(parsed.keys()) if isinstance(parsed, dict) else (
            f"array of {len(parsed)}" if isinstance(parsed, list) else None
        ),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now(SYDNEY).isoformat()})


@app.route("/analyse", methods=["POST"])
def analyse():
    try:
        import json as _json
        # Read raw body and parse manually — handles all n8n encoding variants
        raw = request.get_data(as_text=True)
        if not raw or not raw.strip():
            return jsonify({"error": "Empty body"}), 400
        try:
            payload = _json.loads(raw)
        except Exception as e:
            return jsonify({"error": f"JSON parse failed: {e}", "raw_preview": raw[:200]}), 400
        # Unwrap double-encoded string
        if isinstance(payload, str):
            payload = _json.loads(payload)
        # Unwrap array wrapper
        if isinstance(payload, list):
            payload = payload[0]

        log.info(
            f"POST /analyse — "
            f"ig={len(payload.get('ig_positions', []))}, "
            f"roc={len(payload.get('roc_positions', []))}"
        )

        result = run_agent(payload)
        report = result["report"]

        # Strip any agent preamble before the first HTML tag
        html_start = report.find('<')
        if html_start > 0:
            report = report[html_start:]

        # Strip HTML for plain-text version
        plain = re.sub(r"<[^>]+>", "", report).strip()

        stocks_covered = sorted(set(
            [p.get("symbol", p.get("ticker", "")) for p in payload.get("ig_positions", [])] +
            [p.get("ticker", "")                   for p in payload.get("roc_positions", [])]
        ))
        stocks_covered = [t for t in stocks_covered if t]

        # Write per-stock rows to stock_insights for historical archive
        save_to_stock_insights(
            stocks_covered=stocks_covered,
            report_html=report,
            payload=payload,
            news_cache=result.get("news_cache", {}),
        )

        cost_usd = result.get("estimated_cost_usd", 0)
        return jsonify({
            "report_html":         report,
            "report_text":         plain,
            "stocks_covered":      stocks_covered,
            "iterations":          result["iterations"],
            "total_input_tokens":  result.get("total_input_tokens", 0),
            "total_output_tokens": result.get("total_output_tokens", 0),
            "estimated_cost_usd":  cost_usd,
            "generated_at":        datetime.now(SYDNEY).isoformat(),
        })

    except Exception as e:
        log.exception("Error in /analyse")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

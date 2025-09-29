"""
Telegram bot: AI chat (Gemini/OpenAI fallback), Sui wallet monitor (timestamp-based "realtime" polling),
DexScreener Sui token info, CoinGecko/Binance price, Binance chart generation, image/video generation,
notes, and Rose-style moderation.

Rules you asked:
- Group: reply ONLY when mentioned (@bot). Never tries to auto-join (Telegram disallows).
- Wallet monitoring: private chat only. Each user gets their own alerts for saved wallets.
- Sui alerts: "real-time style" fast polling (5s) using timestamp-based detection (no digest memory).

Put your API keys/config in environment vars (or a .env for local/Replit).
"""
import os
import sys
import time
import json
import re
import urllib.parse
import base64
import logging
import asyncio
from io import BytesIO
from datetime import datetime, timedelta
from glob import glob

import requests
from bs4 import BeautifulSoup

# optional google generative (Gemini)
try:
    import google.generativeai as genai
except Exception:
    genai = None

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------- CONFIG (env) --------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # optional (Gemini)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional (OpenAI)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HF_API_KEY = os.getenv("HF_API_KEY")  # optional (HF text->video)
YOUTUBE_KEY = os.getenv("YOUTUBE_KEY")  # optional
DATA_DIR = os.getenv("DATA_DIR", "data")

# Sui RPC endpoints (primary + fallbacks)
SUI_RPC = os.getenv("SUI_RPC", "https://fullnode.mainnet.sui.io:443")
SUI_RPC_FALLBACKS = os.getenv("SUI_RPC_FALLBACKS", "")
SUI_RPC_LIST = [
    u.strip()
    for u in ([SUI_RPC] +
              ([x.strip() for x in SUI_RPC_FALLBACKS.split(",")
                if x.strip()] if SUI_RPC_FALLBACKS else [])) if u.strip()
]

# Binance klines limit (for chart)
BINANCE_KLINES_LIMIT = int(os.getenv("BINANCE_KLINES_LIMIT", "120"))

# Wallet watcher config (timestamp-based)
WALLET_SCAN_SEC = int(os.getenv("WALLET_SCAN_SEC", "5"))  # fast polling
MAX_ANTIDUPE = int(os.getenv("MAX_ANTIDUPE",
                             "200"))  # short rolling dedupe by digest

if not TELEGRAM_TOKEN:
    raise RuntimeError("Please set TELEGRAM_BOT_TOKEN")

os.makedirs(DATA_DIR, exist_ok=True)

# -------- logging --------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("bot")

# -------- Gemini init (optional) --------
USE_GEMINI = False
GEMINI_MODEL = None
if GOOGLE_API_KEY and genai:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
        USE_GEMINI = True
        log.info("Gemini initialized")
    except Exception as e:
        log.warning("Gemini init failed: %s", e)
        USE_GEMINI = False


# -------- helpers --------
def safe_get_json(url, params=None, headers=None, timeout=12):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            log.debug("safe_get_json non-200 %d for %s", r.status_code, url)
            return None
        return r.json()
    except Exception as e:
        log.warning("safe_get_json error for %s: %s", url, e)
        return None


def rpc_post_try_endpoints(method,
                           params,
                           endpoints=None,
                           timeout=15,
                           retries=2):
    if endpoints is None:
        endpoints = SUI_RPC_LIST
    last_err = None
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    for ep in endpoints:
        for attempt in range(1, retries + 1):
            try:
                r = requests.post(ep, json=body, timeout=timeout)
                j = r.json()
                if r.status_code == 200 and "result" in j:
                    return j.get("result")
                if "error" in j:
                    last_err = f"RPC {ep} error: {j['error']}"
                    break
                last_err = f"RPC {ep} HTTP {r.status_code}"
            except Exception as e:
                last_err = f"RPC {ep} attempt {attempt} exception: {e}"
                time.sleep(0.35 * attempt)
        log.debug("Next endpoint; last_err=%s", last_err)
    log.info("All RPC endpoints failed: %s", last_err)
    return None


async def run_blocking(fn, *args, **kwargs):
    loop = __import__("asyncio").get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


# -------- AI providers (Gemini -> OpenAI) --------
def _ask_gemini(prompt: str):
    if not USE_GEMINI:
        raise RuntimeError("Gemini not configured")
    resp = GEMINI_MODEL.generate_content(prompt)
    if hasattr(resp, "text") and resp.text:
        return resp.text
    raise RuntimeError("Gemini returned no text")


def _ask_openai(prompt: str, timeout=20):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": OPENAI_MODEL,
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "temperature": 0.7,
        "max_tokens": 800
    }
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:200]}")
    j = r.json()
    if j.get("choices"):
        return j["choices"][0]["message"]["content"]
    raise RuntimeError("OpenAI returned no text")


def _ask_ai_blocking(prompt: str):
    errs = []
    if USE_GEMINI:
        try:
            return _ask_gemini(prompt)
        except Exception as e:
            errs.append(f"Gemini: {e}")
            log.info("Gemini failed: %s", e)
    if OPENAI_API_KEY:
        try:
            return _ask_openai(prompt)
        except Exception as e:
            errs.append(f"OpenAI: {e}")
            log.info("OpenAI failed: %s", e)
    raise RuntimeError("No AI provider worked. " + (
        " | ".join(errs) if errs else "Set at least one API key."))


async def ask_ai(prompt: str):
    try:
        return await run_blocking(_ask_ai_blocking, prompt)
    except Exception as e:
        log.warning("ask_ai error: %s", e)
        return f"⚠️ AI error: {e}"


# -------- image/video --------
def _generate_image_blocking(prompt: str):
    if USE_GEMINI:
        try:
            resp = GEMINI_MODEL.generate_content(
                prompt, generation_config={"response_mime_type": "image/png"})
            if hasattr(resp, "candidates"):
                for cand in resp.candidates:
                    for part in getattr(cand.content, "parts", []) or []:
                        inline = getattr(part, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            return {"type": "base64", "data": inline.data}
                        if getattr(part, "uri", None):
                            return {"type": "url", "data": part.uri}
        except Exception as e:
            log.info("Gemini image failed: %s", e)
    return {
        "type":
        "url",
        "data":
        f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}"
    }


async def generate_image(prompt: str):
    return await run_blocking(_generate_image_blocking, prompt)


def _generate_video_hf(prompt: str, retries: int = 4, timeout_s: int = 120):
    if not HF_API_KEY:
        return None, "HF API key not configured"
    url = "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    body = {"inputs": prompt, "parameters": {"num_frames": 16}}
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url,
                              headers=headers,
                              json=body,
                              timeout=timeout_s)
            if r.status_code == 200 and len(r.content) > 10000:
                return r.content, None
            try:
                err = r.json().get("error")
            except Exception:
                err = f"HTTP {r.status_code}"
            if err and "loading" in str(err).lower():
                time.sleep(5)
                continue
            return None, f"HF error: {err}"
        except Exception as e:
            if attempt == retries:
                return None, str(e)
            time.sleep(2)
    return None, "Video failed"


def _gif_from_pollinations(prompt: str,
                           n_frames: int = 6,
                           size=(512, 512),
                           frame_ms=250):
    try:
        from PIL import Image
    except Exception:
        try:
            import subprocess
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "pillow"])
            from PIL import Image
        except Exception as e:
            return None, f"Pillow install failed: {e}"
    frames = []
    for i in range(n_frames):
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?seed={i}"
        try:
            b = requests.get(url, timeout=20).content
            img = Image.open(BytesIO(b)).convert("RGB")
            if size: img = img.resize(size)
            frames.append(img)
        except Exception as e:
            log.info("pollinations frame error: %s", e)
    if not frames:
        return None, "No frames fetched"
    out = BytesIO()
    frames[0].save(out,
                   format="GIF",
                   save_all=True,
                   append_images=frames[1:],
                   duration=frame_ms,
                   loop=0)
    out.seek(0)
    return out.read(), None


async def generate_video(prompt: str):
    if HF_API_KEY:
        mp4, err = await run_blocking(_generate_video_hf, prompt)
        if mp4:
            return ("mp4", mp4, None)
        log.info("HF video failed: %s", err)
    gif, gerr = await run_blocking(_gif_from_pollinations, prompt)
    if gif:
        return ("gif", gif, None)
    return (None, None, gerr or "Unknown video error")


# -------- misc utilities --------
def http_get_text(url: str, timeout: int = 12):
    r = requests.get(url,
                     headers={"User-Agent": "Mozilla/5.0 (Bot)"},
                     timeout=timeout)
    r.raise_for_status()
    return r.text


def extract_tweet_text(url: str):
    try:
        html = http_get_text(url)
        soup = BeautifulSoup(html, "html.parser")
        meta = soup.find("meta", {"property": "og:description"}) or soup.find(
            "meta", {"name": "description"})
        if meta and meta.get("content"):
            return meta["content"]
        if soup.title: return soup.title.string
    except Exception as e:
        log.info("extract_tweet_text error: %s", e)
    return None


def format_price(p):
    try:
        p = float(p)
        if p >= 1: return f"${p:,.2f}"
        if p >= 0.01: return f"${p:,.4f}"
        if p >= 0.0001: return f"${p:,.6f}"
        return f"${p:.8f}"
    except:
        return str(p)


# -------- CoinGecko / Binance helpers --------
def coingecko_price_for_symbol(symbol: str):
    try:
        if not symbol: return None, None
        s = symbol.strip()
        j = safe_get_json(
            f"https://api.coingecko.com/api/v3/search?query={urllib.parse.quote(s)}"
        )
        if not j or not j.get("coins"): return None, None
        s_lower = s.lower()
        coin_id = None
        for c in j["coins"]:
            if c.get("symbol", "").lower() == s_lower:
                coin_id = c["id"]
                break
        if not coin_id: coin_id = j["coins"][0]["id"]
        pr = safe_get_json(
            f"https://api.coingecko.com/api/v3/simple/price?ids={urllib.parse.quote(coin_id)}&vs_currencies=usd"
        )
        if pr and coin_id in pr:
            return pr[coin_id]["usd"], coin_id
    except Exception as e:
        log.info("coingecko error: %s", e)
    return None, None


def binance_price_for_symbol(symbol: str):
    try:
        sym = re.sub(r'[^A-Za-z0-9]', '', symbol).upper()
        if not sym: return None
        pair = f"{sym}USDT"
        j = safe_get_json(
            f"https://api.binance.com/api/v3/ticker/price?symbol={pair}")
        if j and "price" in j:
            return float(j["price"])
    except Exception as e:
        log.info("binance error: %s", e)
    return None


def get_crypto_price(symbol: str):
    price, cid = coingecko_price_for_symbol(symbol)
    if price is not None: return price, cid, "coingecko"
    b = binance_price_for_symbol(symbol)
    if b is not None: return b, f"{symbol.upper()}USDT", "binance"
    return None, None, None


def get_trending_coins(limit: int = 5):
    try:
        j = safe_get_json("https://api.coingecko.com/api/v3/search/trending")
        if not j or "coins" not in j:
            return []
        coins = []
        for item in j["coins"][:limit]:
            coin = item.get("item", {})
            symbol = coin.get("symbol", "").upper()
            name = coin.get("name", "")
            price, _, _ = get_crypto_price(symbol)
            coins.append({
                "symbol": symbol,
                "name": name,
                "price": price
            })
        return coins
    except Exception as e:
        log.info("get_trending_coins error: %s", e)
        return []


# -------- DexScreener / Sui token info --------
def format_usd(n):
    try:
        return "${:,.2f}".format(float(n))
    except:
        return str(n)


def sui_token_info(contract: str):
    addr = contract.strip()
    j = safe_get_json(f"https://api.dexscreener.com/latest/dex/tokens/{addr}",
                      timeout=15)
    if not j or "pairs" not in j:
        return None, "No data from DexScreener."
    pairs = [
        p for p in j.get("pairs", [])
        if "sui" in str(p.get("chainId", "")).lower()
    ]
    if not pairs: return None, "No Sui pairs found"
    pairs_sorted = sorted(pairs,
                          key=lambda p: float((p.get("liquidity", {}).get(
                              "usd") or p.get("liquidityUsd") or 0) or 0),
                          reverse=True)
    best = pairs_sorted[0]
    price_usd = best.get("priceUsd") or best.get("price")
    vol_24h = best.get("volume", {}).get("h24") or best.get("volume24h")
    liq_usd = best.get("liquidity", {}).get("usd") or best.get("liquidityUsd")
    fdv = best.get("fdv")
    mcap = best.get("marketCap") or best.get("mcap")
    top = []
    for p in pairs_sorted[:3]:
        base = p.get("baseToken", {}) or {}
        quote = p.get("quoteToken", {}) or {}
        pair_symbol = f"{base.get('symbol','')}/{quote.get('symbol','')}"
        top.append({
            "dex":
            p.get("dexId", "dex"),
            "pair":
            pair_symbol,
            "price":
            p.get("priceUsd") or p.get("price"),
            "liq": (p.get("liquidity", {}).get("usd")
                    or p.get("liquidityUsd")),
            "link":
            p.get("url")
        })
    out = {
        "name": best.get("baseToken", {}).get("name"),
        "symbol": best.get("baseToken", {}).get("symbol"),
        "contract": addr,
        "priceUsd": price_usd,
        "volume24h": vol_24h,
        "liquidityUsd": liq_usd,
        "fdv": fdv,
        "marketCap": mcap,
        "pairs": top
    }
    return out, None


# -------- YouTube search/Weather --------
def youtube_search_top(query: str):
    try:
        if YOUTUBE_KEY:
            q = urllib.parse.quote(query)
            url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&q={q}&key={YOUTUBE_KEY}&maxResults=1"
            r = requests.get(url, timeout=10).json()
            items = r.get("items") or []
            if items:
                vid = items[0]["id"]["videoId"]
                title = items[0]["snippet"]["title"]
                channel = items[0]["snippet"]["channelTitle"]
                thumb = items[0]["snippet"]["thumbnails"].get(
                    "high",
                    {}).get("url") or items[0]["snippet"]["thumbnails"].get(
                        "default", {}).get("url")
                return {
                    "id": vid,
                    "title": title,
                    "channel": channel,
                    "thumbnail": thumb
                }
        html = http_get_text(
            f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}",
            timeout=8)
        m = re.search(r"watch\?v=(\S{11})", html)
        if m:
            vid = m.group(1)
            o = requests.get(
                f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={vid}&format=json",
                timeout=6)
            if o.status_code == 200:
                od = o.json()
                return {
                    "id": vid,
                    "title": od["title"],
                    "channel": od["author_name"],
                    "thumbnail": od["thumbnail_url"]
                }
    except Exception as e:
        log.info("youtube_search error: %s", e)
    return None


def get_weather(city: str):
    try:
        r = requests.get(
            f"https://wttr.in/{urllib.parse.quote(city)}?format=3", timeout=8)
        if r.status_code == 200 and r.text:
            return r.text.strip()
    except Exception as e:
        log.info("wttr error: %s", e)
    return None


# -------- Storage helpers --------
def _notes_path(chat_id: int):
    return os.path.join(DATA_DIR, f"notes_{chat_id}.json")


def load_notes(chat_id: int):
    try:
        p = _notes_path(chat_id)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.info("load_notes error: %s", e)
    return []


def save_notes(chat_id: int, notes):
    try:
        p = _notes_path(chat_id)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
        return True
    except Exception as e:
        log.info("save_notes error: %s", e)
        return False


def _warns_path(chat_id: int):
    return os.path.join(DATA_DIR, f"warns_{chat_id}.json")


def load_warns(chat_id: int):
    try:
        p = _warns_path(chat_id)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.info("load_warns error: %s", e)
    return {}


def save_warns(chat_id: int, warns):
    try:
        p = _warns_path(chat_id)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(warns, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
        return True
    except Exception as e:
        log.info("save_warns error: %s", e)
        return False


# For wallet alerts we store per-user (private chat id) only
def _wallets_path(chat_id: int):
    return os.path.join(DATA_DIR, f"wallets_user_{chat_id}.json")


def load_wallets(chat_id: int):
    try:
        p = _wallets_path(chat_id)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.info("load_wallets error: %s", e)
    return {}  # {address: {"last_ts": int_ms, "antidupe":[digests...]}} 


def save_wallets(chat_id: int, data):
    try:
        p = _wallets_path(chat_id)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
        return True
    except Exception as e:
        log.info("save_wallets error: %s", e)
        return False


# -------- Admin helpers --------
async def _is_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int,
                    user_id: int):
    try:
        mem = await context.bot.get_chat_member(chat_id, user_id)
        return mem.status in ("creator", "administrator")
    except Exception:
        return False


def _human(u):
    if not u: return "user"
    name = (getattr(u, "full_name", None) or getattr(u, "first_name", None)
            or "user").strip()
    if getattr(u, "username", None): name += f" (@{u.username})"
    return name


def _resolve_target_user(update: Update):
    msg = update.effective_message
    if msg.reply_to_message: return msg.reply_to_message.from_user
    if msg.entities:
        for ent in msg.entities:
            if ent.type == "text_mention": return ent.user
    return None


# -------- SUI RPC & parsing (timestamp-based) --------
def sui_query_txs_for_address(addr: str, direction: str, limit: int = 50):
    flt = {"ToAddress": addr} if direction == "to" else {"FromAddress": addr}
    options = {
        "showInput": True,
        "showEffects": True,
        "showEvents": True,
        "showBalanceChanges": True
    }
    params = [{
        "filter": flt,
        "options": options,
        "limit": limit,
        "order": "descending"
    }]
    res = rpc_post_try_endpoints("suix_queryTransactionBlocks", params)
    if not res:
        return []
    return res.get("data", []) or []


def detect_swap_like(tx: dict) -> bool:
    try:
        text = json.dumps(tx).lower()
        return any(w in text for w in ("swap", "router", "pool", "amm"))
    except Exception:
        return False


def sui_coin_meta(coin_type: str, cache: dict):
    if coin_type in cache: return cache[coin_type]
    if coin_type.endswith("::sui::SUI"):
        meta = {"symbol": "SUI", "decimals": 9}
        cache[coin_type] = meta
        return meta
    res = rpc_post_try_endpoints("suix_getCoinMetadata", [coin_type])
    if res:
        meta = {
            "symbol": res.get("symbol") or coin_type.split("::")[-1],
            "decimals": res.get("decimals", 9)
        }
    else:
        meta = {"symbol": coin_type.split("::")[-1], "decimals": 9}
    cache[coin_type] = meta
    return meta


def fmt_amount(val_int: int, decimals: int):
    sign = "-" if val_int < 0 else ""
    val = abs(val_int) / (10**decimals)
    if val >= 1: return f"{sign}{val:,.4f}"
    if val >= 0.01: return f"{sign}{val:,.6f}"
    return f"{sign}{val:.9f}"


def summarize_tx_for_addr(tx: dict, addr: str, coin_meta_cache: dict):
    digest = tx.get("digest") or ""
    ts = int(tx.get("timestampMs") or 0) if tx.get("timestampMs") else 0
    when = datetime.utcfromtimestamp(
        ts / 1000).strftime("%Y-%m-%d %H:%M:%S UTC") if ts else "unknown time"
    changes = tx.get("balanceChanges") or []
    effects = tx.get("effects") or {}
    if not changes and isinstance(effects, dict):
        changes = effects.get("balanceChanges") or effects.get(
            "balance_changes") or []
    my_changes = []
    for c in changes:
        owner = (c.get("owner") or {})
        owner_addr = None
        if isinstance(owner, dict):
            owner_addr = owner.get("AddressOwner") or owner.get(
                "address") or owner.get("owner")
        else:
            owner_addr = owner
        if owner_addr == addr:
            my_changes.append(c)
    # fallback via events
    if not my_changes:
        evs = tx.get("events") or []
        for ev in evs:
            try:
                if json.dumps(ev).lower().find(addr.lower()) != -1:
                    my_changes.append({"detected_event": ev})
            except:
                pass
    nets = {}
    for c in my_changes:
        ct = c.get("coinType") or c.get("coin") or (c.get("coinObject")
                                                    or {}).get("type")
        amt = 0
        if "amount" in c:
            try:
                amt = int(str(c.get("amount", 0)))
            except:
                amt = 0
        else:
            for k in ("amount", "value"):
                if isinstance(c.get(k), (int, str)):
                    try:
                        amt = int(str(c.get(k)))
                        break
                    except:
                        pass
        if ct:
            nets[ct] = nets.get(ct, 0) + amt
    kind = "activity"
    if detect_swap_like(tx): kind = "swap"
    else:
        pos = any(v > 0 for v in nets.values())
        neg = any(v < 0 for v in nets.values())
        if pos and not neg: kind = "receive"
        elif neg and not pos: kind = "send"
        elif pos and neg: kind = "swap"
    parts = []
    for ct, amt in nets.items():
        meta = sui_coin_meta(ct, coin_meta_cache)
        parts.append(f"{fmt_amount(amt, meta['decimals'])} {meta['symbol']}")
    amounts = ", ".join(parts) if parts else "balance change not available"
    link = f"https://explorer.sui.io/txblock/{digest}?network=mainnet" if digest else "https://explorer.sui.io"
    title = {
        "send": "📤 Send",
        "receive": "📥 Receive",
        "swap": "🔁 Swap"
    }.get(kind, "🔔 Activity")
    text = f"{title} — {addr[:12]}…\n• {amounts}\n• Time: {when}\n• Digest: `{digest}`\n{link}"
    return text, digest, int(tx.get("timestampMs") or 0)


# -------- TELEGRAM COMMANDS --------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text(
            "🤖 Bot ready.\n"
            "Commands:\n"
            "• /p <symbol> — crypto price\n"
            "• /chart <symbol> <interval> — Binance klines (1m,5m,15m,1h,1d)\n"
            "• /trending <limit> — top trending cryptos\n"
            "• /sui <contract> — Sui token info (DexScreener)\n"
            "• /addwallet <sui_addr> — (private only) monitor wallet\n"
            "• /delwallet <sui_addr>, /mywallets — (private only)\n"
            "• /img <prompt>, /video <prompt>\n"
            "• /note /notes /delnote /clearnotes\n"
            "• Admin: /warn /warns /unwarn /mute /unmute /ban /kick\n"
            "Group rule: I only reply when mentioned (@bot)."
        )
    except Exception as e:
        log.error("Error in cmd_start: %s", e)
        await update.message.reply_text("⚠️ An error occurred.")


def _group_should_reply(text: str, mention: str) -> bool:
    return mention and (f"@{mention}" in text)


def _deny_if_group_for_private_only(update: Update):
    return update.effective_chat.type in ("group", "supergroup")


async def cmd_crypto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _deny_if_group_for_private_only(update) and not (await _mentioned(
            update, context)):
        return
    if not context.args:
        return await update.message.reply_text("Usage: /p <symbol>")
    symbol = context.args[0]
    price, ref, src = await run_blocking(get_crypto_price, symbol)
    if price is not None:
        await update.message.reply_text(
            f"💰 {symbol.upper()} price: {format_price(price)}  (source: {src}, ref: {ref})"
        )
    else:
        await update.message.reply_text("❌ Coin not found.")


async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _deny_if_group_for_private_only(update) and not (await _mentioned(
            update, context)):
        return
    if not context.args:
        return await update.message.reply_text(
            "Usage: /chart <symbol> <interval>")
    symbol = context.args[0]
    interval = context.args[1] if len(context.args) > 1 else "1h"
    pair = f"{re.sub(r'[^A-Za-z0-9]','',symbol).upper()}USDT"
    url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval={interval}&limit={BINANCE_KLINES_LIMIT}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return await update.message.reply_text(
                "❌ Binance returned error for that pair/interval.")
        data = r.json()
        closes = [float(x[4]) for x in data]
        volumes = [float(x[5]) for x in data]
        times = [int(x[0]) for x in data]
        try:
            import matplotlib.pyplot as plt
        except Exception:
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "matplotlib", "pillow"
            ])
            import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot([datetime.fromtimestamp(t / 1000) for t in times], closes)
        ax1.set_title(f"{symbol.upper()} {interval} Price")
        ax2.bar([datetime.fromtimestamp(t / 1000) for t in times], volumes)
        ax2.set_title("Volume")
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        await update.message.reply_photo(
            photo=buf, caption=f"📈 {symbol.upper()} {interval} with Volume")
    except Exception as e:
        log.info("chart error: %s", e)
        await update.message.reply_text("❌ Error generating chart.")


async def cmd_trending(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _deny_if_group_for_private_only(update) and not (await _mentioned(
            update, context)):
        return
    limit = int(context.args[0]) if context.args and context.args[0].isdigit() else 5
    trending = await run_blocking(get_trending_coins, limit)
    if not trending:
        return await update.message.reply_text("❌ Failed to fetch trending coins.")
    lines = ["🔥 Top Trending Cryptos:"]
    for i, coin in enumerate(trending, 1):
        price_str = format_price(coin["price"]) if coin["price"] else "N/A"
        lines.append(f"{i}. {coin['name']} ({coin['symbol']}) — {price_str}")
    await update.message.reply_text("\n".join(lines))
    # Generate chart for top trending
    top_symbol = trending[0]["symbol"]
    pair = f"{re.sub(r'[^A-Za-z0-9]','',top_symbol).upper()}USDT"
    url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1h&limit={BINANCE_KLINES_LIMIT}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return await update.message.reply_text("❌ No chart available for top trending coin.")
        data = r.json()
        closes = [float(x[4]) for x in data]
        volumes = [float(x[5]) for x in data]
        times = [int(x[0]) for x in data]
        try:
            import matplotlib.pyplot as plt
        except Exception:
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "matplotlib", "pillow"
            ])
            import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot([datetime.fromtimestamp(t / 1000) for t in times], closes)
        ax1.set_title(f"{top_symbol.upper()} 1h Price (Top Trending)")
        ax2.bar([datetime.fromtimestamp(t / 1000) for t in times], volumes)
        ax2.set_title("Volume")
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        await update.message.reply_photo(
            photo=buf, caption=f"📈 {top_symbol.upper()} 1h Chart (Top Trending)")
    except Exception as e:
        log.info("trending chart error: %s", e)
        await update.message.reply_text("❌ Error generating chart for top trending.")


async def cmd_sui(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _deny_if_group_for_private_only(update) and not (await _mentioned(
            update, context)):
        return
    if not context.args:
        return await update.message.reply_text("Usage: /sui <contract_address>"
                                               )
    addr = context.args[0].strip()
    info, err = await run_blocking(sui_token_info, addr)
    if err:
        return await update.message.reply_text(f"❌ {err}")
    lines = [
        f"🟪 Sui Token: {info.get('name','')} ({info.get('symbol','')})",
        f"📜 Contract: `{info['contract']}`"
    ]
    if info.get("priceUsd"):
        lines.append(f"💵 Price: {format_price(info['priceUsd'])}")
    if info.get("volume24h"):
        lines.append(f"📈 24h Vol: {format_usd(info['volume24h'])}")
    if info.get("liquidityUsd"):
        lines.append(f"💧 Liquidity: {format_usd(info['liquidityUsd'])}")
    if info.get("marketCap"):
        lines.append(f"🏦 MCap: {format_usd(info['marketCap'])}")
    if info.get("fdv"): lines.append(f"🏷️ FDV: {format_usd(info['fdv'])}")
    lines.append("\nTop pairs:")
    for p in info.get("pairs", []):
        liq = format_usd(p.get("liq")) if p.get("liq") else "—"
        price = format_price(p.get("price")) if p.get("price") else "—"
        link = p.get("link") or ""
        lines.append(
            f"• {p['dex']} — {p['pair']} — {price} — liq {liq}\n  {link}")
    await update.message.reply_text("\n".join(lines),
                                    disable_web_page_preview=False,
                                    parse_mode="Markdown")


async def cmd_yt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _deny_if_group_for_private_only(update) and not (await _mentioned(
            update, context)):
        return
    if not context.args:
        return await update.message.reply_text("Usage: /yt <query>")
    q = " ".join(context.args)
    res = await run_blocking(youtube_search_top, q)
    if res:
        link = f"https://www.youtube.com/watch?v={res['id']}"
        try:
            await update.message.reply_photo(
                photo=res["thumbnail"],
                caption=f"▶️ {res['title']}\n👤 {res['channel']}\n🔗 {link}")
        except:
            await update.message.reply_text(
                f"▶️ {res['title']}\n👤 {res['channel']}\n🔗 {link}",
                disable_web_page_preview=False)
    else:
        await update.message.reply_text("❌ No video found.")


async def cmd_img(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _deny_if_group_for_private_only(update) and not (await _mentioned(
            update, context)):
        return
    if not context.args:
        return await update.message.reply_text("Usage: /img <prompt>")
    prompt = " ".join(context.args)
    img_res = await generate_image(prompt)
    if not img_res:
        return await update.message.reply_text("❌ Failed to generate image.")
    try:
        if img_res["type"] == "url":
            await update.message.reply_photo(photo=img_res["data"],
                                             caption=f"🖼 {prompt}")
        else:
            b = base64.b64decode(img_res["data"])
            await update.message.reply_photo(photo=BytesIO(b),
                                             caption=f"🖼 {prompt}")
    except Exception as e:
        log.info("send image error: %s", e)
        await update.message.reply_text("⚠️ Error sending image.")


async def cmd_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _deny_if_group_for_private_only(update) and not (await _mentioned(
            update, context)):
        return
    if not context.args:
        return await update.message.reply_text("Usage: /video <prompt>")
    prompt = " ".join(context.args)
    await update.message.reply_text("🎬 Generating…")
    kind, content, err = await generate_video(prompt)
    if err:
        return await update.message.reply_text(f"❌ {err}")
    try:
        ts = int(datetime.utcnow().timestamp())
        if kind == "mp4":
            fn = os.path.join(DATA_DIR, f"t2v_{ts}.mp4")
            with open(fn, "wb") as f:
                f.write(content)
            with open(fn, "rb") as f:
                await update.message.reply_video(video=f,
                                                 caption=f"🎥 {prompt}")
        elif kind == "gif":
            await update.message.reply_animation(animation=BytesIO(content),
                                                 caption=f"🎥 {prompt}")
    except Exception as e:
        log.info("send video error: %s", e)
        await update.message.reply_text("⚠️ Error sending video.")


# -------- Mention check helper --------
async def _mentioned(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if update.effective_chat.type in ("group", "supergroup"):
        text = update.message.text or ""
        bot_username = (await context.bot.get_me()).username
        return _group_should_reply(text, bot_username)
    return True


# -------- Additional command stubs (from docstring) --------
async def cmd_addwallet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat.type == "private":
        await update.message.reply_text("❌ Wallet commands are private only.")
        return
    if not context.args:
        return await update.message.reply_text("Usage: /addwallet <sui_address>")
    addr = context.args[0].strip()
    if not re.match(r'^0x[a-fA-F0-9]{64}$', addr):
        await update.message.reply_text("❌ Invalid Sui address.")
        return
    wallets = load_wallets(update.effective_chat.id)
    wallets[addr] = {"last_ts": 0, "antidupe": []}
    if save_wallets(update.effective_chat.id, wallets):
        await update.message.reply_text(f"✅ Added wallet: {addr[:12]}…")
    else:
        await update.message.reply_text("❌ Failed to save wallet.")


async def cmd_delwallet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat.type == "private":
        await update.message.reply_text("❌ Wallet commands are private only.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /delwallet <sui_address>")
        return
    addr = context.args[0].strip()
    wallets = load_wallets(update.effective_chat.id)
    if addr in wallets:
        del wallets[addr]
        save_wallets(update.effective_chat.id, wallets)
        await update.message.reply_text(f"✅ Removed wallet: {addr[:12]}…")
    else:
        await update.message.reply_text("❌ Wallet not found.")


async def cmd_mywallets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat.type == "private":
        await update.message.reply_text("❌ Wallet commands are private only.")
        return
    wallets = load_wallets(update.effective_chat.id)
    if not wallets:
        await update.message.reply_text("No wallets monitored.")
        return
    lines = ["Your wallets:"]
    for addr in wallets:
        lines.append(f"• {addr[:12]}… (last checked: {wallets[addr].get('last_ts', 0)})")
    await update.message.reply_text("\n".join(lines))


async def cmd_note(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _deny_if_group_for_private_only(update) and not (await _mentioned(update, context)):
        return
    text = update.message.text.replace("/note", "").strip()
    if not text:
        await update.message.reply_text("Usage: /note <your note>")
        return
    notes = load_notes(update.effective_chat.id)
    notes.append({"text": text, "ts": int(time.time())})
    if save_notes(update.effective_chat.id, notes):
        await update.message.reply_text("✅ Note saved.")
    else:
        await update.message.reply_text("❌ Failed to save note.")


async def cmd_notes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _deny_if_group_for_private_only(update) and not (await _mentioned(update, context)):
        return
    notes = load_notes(update.effective_chat.id)
    if not notes:
        await update.message.reply_text("No notes.")
        return
    lines = ["Your notes:"]
    for i, n in enumerate(notes[-10:], 1):  # last 10
        ts = datetime.fromtimestamp(n["ts"]).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{i}. {n['text']} ({ts})")
    await update.message.reply_text("\n".join(lines))


# Admin commands (stubs)
async def cmd_warn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _is_admin(context, update.effective_chat.id, update.effective_user.id):
        await update.message.reply_text("❌ Admin only.")
        return
    target = _resolve_target_user(update)
    if not target:
        await update.message.reply_text("Reply to user or mention them.")
        return
    chat_id = update.effective_chat.id
    warns = load_warns(chat_id)
    user_id = target.id
    warns[str(user_id)] = warns.get(str(user_id), 0) + 1
    if save_warns(chat_id, warns):
        await update.message.reply_text(f"⚠️ Warned {_human(target)} (total: {warns[str(user_id)]})")
    # Add more logic for mute/ban if needed


# ... (similar stubs for /warns, /unwarn, /mute, etc. - omitted for brevity, can expand if needed)


# -------- Message handler for AI chat --------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type in ("group", "supergroup") and not (await _mentioned(update, context)):
        return  # Only reply if mentioned in group
    text = update.message.text or ""
    if text.startswith("/"):
        return  # Ignore commands
    # Simple AI response
    response = await ask_ai(f"Respond helpfully to: {text}")
    await update.message.reply_text(response)


# -------- Wallet monitoring job --------
async def wallet_job(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    chat_id = job.data["chat_id"]
    wallets = load_wallets(chat_id)
    for addr, data in wallets.items():
        last_ts = data["last_ts"]
        txs = sui_query_txs_for_address(addr, "from", limit=5)  # Recent txs
        new_ts = max([int(tx.get("timestampMs") or 0) for tx in txs] or [0])
        if new_ts > last_ts:
            # Send alert for new activity
            for tx in txs:
                if int(tx.get("timestampMs") or 0) > last_ts:
                    summary, digest, ts = summarize_tx_for_addr(tx, addr, {})
                    await context.bot.send_message(chat_id=chat_id, text=summary)
            data["last_ts"] = new_ts
            save_wallets(chat_id, wallets)


# -------- Main --------
def setup_bot():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Command handlers
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("p", cmd_crypto))
    application.add_handler(CommandHandler("chart", cmd_chart))
    application.add_handler(CommandHandler("trending", cmd_trending))
    application.add_handler(CommandHandler("sui", cmd_sui))
    application.add_handler(CommandHandler("yt", cmd_yt))
    application.add_handler(CommandHandler("img", cmd_img))
    application.add_handler(CommandHandler("video", cmd_video))
    application.add_handler(CommandHandler("addwallet", cmd_addwallet))
    application.add_handler(CommandHandler("delwallet", cmd_delwallet))
    application.add_handler(CommandHandler("mywallets", cmd_mywallets))
    application.add_handler(CommandHandler("note", cmd_note))
    application.add_handler(CommandHandler("notes", cmd_notes))
    application.add_handler(CommandHandler("warn", cmd_warn))
    # Add more admin handlers as needed

    # Message handler for AI chat
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start wallet jobs for existing users (load from data dir)
    if not os.getenv('VERCEL'):
        for file in glob(os.path.join(DATA_DIR, "wallets_user_*.json")):
            chat_id = int(re.search(r"wallets_user_(\d+)", file).group(1))
            if load_wallets(chat_id):  # If has wallets
                application.job_queue.run_repeating(wallet_job, interval=WALLET_SCAN_SEC, data={"chat_id": chat_id}, name=f"wallet_{chat_id}")

    log.info("Bot starting...")

    return application

application = setup_bot()

if __name__ == '__main__':
    asyncio.run(application.run_polling())

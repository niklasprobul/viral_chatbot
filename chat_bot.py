#!/usr/bin/env python3
# Viral live-chat simulation bot - Admin DM control with global batched sending

import os
import json
import re
import logging
import random
import asyncio
import pathlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from html import escape as h
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest, RetryAfter, TimedOut
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

log = logging.getLogger("viral")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def load_secrets(base: pathlib.Path) -> Dict[str, object]:
    """
    Load secrets from data/secrets.json.
    Returns secrets as dict.
    If something is missing, raise RuntimeError.
    """
    path = base / "secrets.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError(f"Missing secrets.json at {path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}")

    token = raw.get("token")
    admin_id = raw.get("admin_id")
    chat_id = raw.get("chat_id")

    if not token or not isinstance(token, str):
        raise RuntimeError("secrets.json must contain string field 'token'")
    if not isinstance(admin_id, int):
        raise RuntimeError("secrets.json must contain integer field 'admin_id'")
    if not isinstance(chat_id, int):
        raise RuntimeError("secrets.json must contain integer field 'chat_id'")

    return raw

# ---------------- Configuration ----------------

DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "data"))
secrets = load_secrets(DATA_DIR)

# Bot token
TOKEN = secrets.get("token", "")

# Admin user id
ADMIN_ID = secrets.get("admin_id", 1234)

# Supergroup chat id and topic ids
CHAT_ID = secrets.get("chat_id", 1234)
TOPICS = {
    "chat": 6,
    "donations": 2,
    "top": 9,
    "maya": 4,
}

SPECIAL_USER_NAME = "ulitmaunum"

# Fallback mood settings (used if mood_config.json is missing or incomplete)
DEFAULT_MOOD_INTERVAL = {
    "angry": 0.5,
    "bored": 1.6,
    "neutral": 1.0,
    "good": 0.8,
    "excited": 0.3,
    "shocked": 0.3,
}

DEFAULT_DONATION_RANGES = {
    "angry": (1.0, 2.0),
    "bored": (1.0, 5.0),
    "neutral": (5.0, 20.0),
    "good": (5.0, 50.0),
    "excited": (10.0, 100.0),
    "shocked": (20.0, 200.0),
}

# -1 means bleed viewers/subs, +1 means gain viewers/subs
DEFAULT_AUDIENCE_TREND = {
    "angry": -1,
    "bored": -1,
    "neutral": 1,
    "good": 1,
    "excited": 1,
    "shocked": 1,
}

BASE_INTERVAL = 2.0
# to prevent 429 too many requests errors
RATE_GAP = 1.2

# ---------------- Utility helpers ----------------


def hb(s: str) -> str:
    return f"<b>{h(s)}</b>"


def format_thousand_dot(n: int) -> str:
    s = str(max(0, n))
    parts = []
    while len(s) > 3:
        parts.append(s[-3:])
        s = s[:-3]
    parts.append(s)
    parts.reverse()
    return ".".join(parts)


def money_fmt_int(x: int) -> str:
    return f"{x} €"


_send_lock = asyncio.Lock()
_last_send_ts: float = 0.0


async def throttled_call(coro_func, *args, **kwargs):
    global _last_send_ts, RATE_GAP
    async with _send_lock:
        now = asyncio.get_running_loop().time()
        gap = now - _last_send_ts
        if gap < RATE_GAP:
            await asyncio.sleep(RATE_GAP - gap)
        try:
            msg = await coro_func(*args, **kwargs)
            _last_send_ts = asyncio.get_running_loop().time()
            return msg
        except RetryAfter as e:
            wait_s = float(getattr(e, "retry_after", 1)) + 0.5
            log.warning("RetryAfter %.2fs - backing off", wait_s)
            await asyncio.sleep(wait_s)
            msg = await coro_func(*args, **kwargs)
            _last_send_ts = asyncio.get_running_loop().time()
            return msg
        except TimedOut:
            log.warning("TimedOut - retrying once after 1.0s")
            await asyncio.sleep(1.0)
            msg = await coro_func(*args, **kwargs)
            _last_send_ts = asyncio.get_running_loop().time()
            return msg


async def send_html(bot, chat_id: int, text: str, *, thread_id: Optional[int] = None, notify: bool = False):
    async def op(**kw):
        return await bot.send_message(**kw)

    try:
        return await throttled_call(
            op,
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
            message_thread_id=thread_id,
            disable_notification=not notify,
        )
    except BadRequest as e:
        if "thread" in str(e).lower():
            try:
                return await throttled_call(
                    op,
                    chat_id=chat_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    disable_notification=not notify,
                )
            except BadRequest:
                pass
        # fallback plain text
        return await throttled_call(
            op,
            chat_id=chat_id,
            text=re.sub(r"<[^>]+>", "", text),
            disable_notification=not notify,
        )


async def edit_html(bot, chat_id: int, message_id: int, text: str):
    async def op(**kw):
        return await bot.edit_message_text(**kw)

    return await throttled_call(
        op,
        chat_id=chat_id,
        message_id=message_id,
        text=text,
        parse_mode=ParseMode.HTML,
    )


# ---------------- Data loading helpers ----------------


def load_lines(path: pathlib.Path) -> List[str]:
    try:
        if path.exists():
            return [
                ln.strip()
                for ln in path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
    except Exception as e:
        log.warning("Failed to load %s: %s", path, e)
    return []


def load_usernames(base: pathlib.Path) -> List[str]:
    names = load_lines(base / "usernames.txt")
    if names:
        return names
    return [
        "Watcher42",
        "FoggyLens",
        "CreakyDoor",
        "HexBug",
        "Lighthouser",
        "CthuluChow",
        "CryptWatch",
        "MorgueFiles",
        "EchoJoint",
        "NightShift",
        "HushPeeper",
        "BatteryLow",
    ]


def fallback_prompts() -> Dict[str, List[str]]:
    return {
        "angry": [
            "That camera angle is a crime.",
            "Why is no one calling the cops already.",
            "Stop ignoring the chat.",
        ],
        "bored": [
            "Yawn. Any action soon.",
            "This is dragging on.",
            "Play something spooky at least.",
        ],
        "neutral": [
            "Chat checking in.",
            "Audio ok for anyone else.",
            "What is that in the corner.",
        ],
        "good": [
            "This is getting good.",
            "Nice flashlight work.",
            "That clue was slick.",
        ],
        "excited": [
            "LET'S GO.",
            "Do not go in there.",
            "Run run run.",
        ],
        "shocked": [
            "Did you see that.",
            "Door moved by itself.",
            "This is messed up.",
        ],
    }


def load_prompts(base: pathlib.Path, moods: List[str], locations: List[str]):
    """
    Load per-mood and per-location prompt pools from text files.
    Fallbacks are used if no file is found.
    """
    fb = fallback_prompts()

    mood_prompts: Dict[str, List[str]] = {}
    for m in moods:
        lines = load_lines(base / f"prompts_{m}.txt")
        if lines:
            mood_prompts[m] = lines
        else:
            mood_prompts[m] = fb.get(m, ["chat checking in"])

    location_prompts: Dict[str, List[str]] = {}
    for loc in locations:
        lines = load_lines(base / f"prompts_{loc}.txt")
        if lines:
            location_prompts[loc] = lines
        else:
            location_prompts[loc] = []

    return mood_prompts, location_prompts


def load_json(path: pathlib.Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to load %s: %s", path, e)
    return default


# ---------------- Config readers ----------------


@dataclass
class LocationScript:
    audience_hooks: List[str] = field(default_factory=list)
    maya_lines: List[str] = field(default_factory=list)
    donations: List[Dict] = field(default_factory=list)
    donation_level: float = 0.10
    subs_per_minute: float = 0.0
    viewers_per_minute: float = 0.0


def read_location_scripts(base: pathlib.Path) -> Tuple[Dict[str, LocationScript], List[str]]:
    """
    Load location_scripts.json.
    Also returns the ordered list of available locations.
    """
    raw = load_json(base / "location_scripts.json", {})
    loc_scripts: Dict[str, LocationScript] = {}
    loc_list: List[str] = []

    for loc, cfg in raw.items():
        loc_list.append(loc)
        loc_scripts[loc] = LocationScript(
            audience_hooks=list(cfg.get("audience_hooks", cfg.get("evil_lines", []))),
            maya_lines=list(cfg.get("maya_lines", [])),
            donations=list(cfg.get("donations", [])),
            donation_level=float(cfg.get("donation_level", 0.10)),
            subs_per_minute=float(cfg.get("subs_per_minute", 0.0)),
            viewers_per_minute=float(cfg.get("viewers_per_minute", 0.0)),
        )

    return loc_scripts, loc_list


def read_mood_config(base: pathlib.Path):
    """
    Load mood_config.json which defines:
      - moods and their order
      - mood_interval multiplier per mood
      - donation_range [min,max] per mood
      - audience_trend (+1 or -1) per mood
    Fallback to DEFAULT_* if file is missing.
    """
    raw = load_json(base / "mood_config.json", {})
    moods_section = raw.get("moods", {})

    if not isinstance(moods_section, dict) or not moods_section:
        moods_section = {}
        for m in DEFAULT_MOOD_INTERVAL.keys():
            moods_section[m] = {
                "mood_interval": DEFAULT_MOOD_INTERVAL[m],
                "donation_range": list(DEFAULT_DONATION_RANGES.get(m, (5.0, 20.0))),
                "audience_trend": DEFAULT_AUDIENCE_TREND.get(m, 1),
            }

    moods_list: List[str] = []
    mood_interval: Dict[str, float] = {}
    donation_ranges: Dict[str, Tuple[float, float]] = {}
    audience_trend: Dict[str, int] = {}

    for m, cfg in moods_section.items():
        moods_list.append(m)
        if not isinstance(cfg, dict):
            cfg = {}

        mood_interval[m] = float(cfg.get("mood_interval", DEFAULT_MOOD_INTERVAL.get(m, 1.0)))

        dr = cfg.get("donation_range", DEFAULT_DONATION_RANGES.get(m, (5.0, 20.0)))
        if not (isinstance(dr, (list, tuple)) and len(dr) >= 2):
            dr = DEFAULT_DONATION_RANGES.get(m, (5.0, 20.0))
        lo, hi = float(dr[0]), float(dr[1])
        donation_ranges[m] = (lo, hi)

        audience_trend[m] = int(cfg.get("audience_trend", DEFAULT_AUDIENCE_TREND.get(m, 1)))

    return moods_list, mood_interval, donation_ranges, audience_trend


# Audience config is unchanged
def load_audience_config(st: "State"):
    cfg = load_json(st.base / "audience_config.json", {})
    subs_start = cfg.get("subscribers_start")
    subs_plateau = cfg.get("subscribers_plateau")
    viewers_start = cfg.get("viewers_start")
    viewers_plateau = cfg.get("viewers_plateau")

    if isinstance(subs_start, (int, float)):
        st.subscribers = int(subs_start)
    if isinstance(subs_plateau, (int, float)):
        st.plateau_subs = int(subs_plateau)
    if isinstance(viewers_start, (int, float)):
        st.viewers = int(viewers_start)
    if isinstance(viewers_plateau, (int, float)):
        st.plateau_viewers = int(viewers_plateau)


# ---------------- Models ----------------


@dataclass
class State:
    base: pathlib.Path

    moods: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)

    mood_prompts: Dict[str, List[str]] = field(default_factory=dict)
    location_prompts: Dict[str, List[str]] = field(default_factory=dict)
    usernames: List[str] = field(default_factory=list)

    mood_interval: Dict[str, float] = field(default_factory=dict)
    donation_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    audience_trend: Dict[str, int] = field(default_factory=dict)

    mood: str = "neutral"
    location: str = "stream_start"
    paused: bool = True

    base_interval: float = BASE_INTERVAL
    hype: float = 1.0

    loc_donation_level: float = 0.10
    donation_meter: float = 0.0

    hook_remaining: List[str] = field(default_factory=list)
    maya_remaining: List[str] = field(default_factory=list)
    donation_window_remaining: int = 0
    pred_donations_remaining: List[Dict] = field(default_factory=list)

    top_donors: Dict[str, int] = field(default_factory=dict)
    leaderboard_msg_id: Optional[int] = None

    subscribers: int = 950_011
    plateau_subs: int = 998_655
    viewers: int = 412_000
    plateau_viewers: int = 1_200_000
    loc_subs_rate: float = 0.0
    loc_viewers_rate: float = 0.0

    awaiting_maya_text: bool = False
    awaiting_special_text: bool = False

    awaiting_donation_name: bool = False
    awaiting_donation_amount: bool = False
    pending_donation_name: str = ""

    awaiting_sub_burst: bool = False
    awaiting_viewer_burst: bool = False

    awaiting_spam_text: bool = False
    spam_active: bool = False
    spam_message_base: str = ""
    spam_ticks_left: int = 0

    job_name: str = "chat_loop"
    job_top: str = "top_refresh"
    job_drain: str = "drain_outbox"

    loc_change_ts: float = field(default_factory=lambda: time.time())


# ---------------- Globals that depend on State ----------------

LOCATION_SCRIPTS: Dict[str, LocationScript] = {}

# placeholder STATE. build_app() will overwrite with a fully loaded one.
STATE: State = State(base=DATA_DIR)

OUTBOX: Dict[str, List[str]] = {"chat": [], "donations": [], "maya": []}
CHANNEL_ORDER = ["maya", "donations", "chat"]
_rr_index = 0


# ---------------- State setup and refresh ----------------


def reset_location_windows(st: State):
    ls = LOCATION_SCRIPTS.get(st.location, LocationScript())

    st.hook_remaining = list(ls.audience_hooks)
    st.maya_remaining = list(ls.maya_lines)
    st.donation_window_remaining = 1
    st.pred_donations_remaining = list(ls.donations)
    st.loc_donation_level = ls.donation_level
    st.loc_subs_rate = ls.subs_per_minute
    st.loc_viewers_rate = ls.viewers_per_minute

    # finale instantly clamps subs/viewers
    if st.location == "finale":
        st.subscribers = 1
        st.viewers = 1

    st.loc_change_ts = time.time()

    log.info(
        "Location %s loaded, hooks %d, maya %d, scripted_donations %d, "
        "don_lvl %.2f, subs/min %.2f, viewers/min %.2f",
        st.location,
        len(st.hook_remaining),
        len(st.maya_remaining),
        len(st.pred_donations_remaining),
        st.loc_donation_level,
        st.loc_subs_rate,
        st.loc_viewers_rate,
    )


def init_state() -> State:
    base = DATA_DIR

    moods_list, mood_interval, donation_ranges, audience_trend = read_mood_config(base)
    loc_scripts, loc_list = read_location_scripts(base)
    mood_prompts, location_prompts = load_prompts(base, moods_list, loc_list)
    usernames = load_usernames(base)

    st = State(
        base=base,
        moods=moods_list,
        locations=loc_list,
        mood_prompts=mood_prompts,
        location_prompts=location_prompts,
        usernames=usernames,
        mood_interval=mood_interval,
        donation_ranges=donation_ranges,
        audience_trend=audience_trend,
        mood=moods_list[0] if moods_list else "neutral",
        location=loc_list[0] if loc_list else "stream_start",
    )

    # Load audience config
    load_audience_config(st)

    # publish LOCATION_SCRIPTS global
    global LOCATION_SCRIPTS
    LOCATION_SCRIPTS = loc_scripts

    reset_location_windows(st)
    return st


def refresh_configs(st: State, app: Application):
    """
    Re-read all configs (mood_config.json, location_scripts.json,
    audience_config.json, prompt files, usernames).
    Keeps leaderboard and donors.
    """
    moods_list, mood_interval, donation_ranges, audience_trend = read_mood_config(st.base)
    loc_scripts, loc_list = read_location_scripts(st.base)
    mood_prompts, location_prompts = load_prompts(st.base, moods_list, loc_list)
    usernames = load_usernames(st.base)

    st.moods = moods_list
    st.locations = loc_list
    st.mood_prompts = mood_prompts
    st.location_prompts = location_prompts
    st.usernames = usernames
    st.mood_interval = mood_interval
    st.donation_ranges = donation_ranges
    st.audience_trend = audience_trend

    # Update globals
    global LOCATION_SCRIPTS
    LOCATION_SCRIPTS = loc_scripts

    # Reload audience targets
    load_audience_config(st)

    # ensure current mood/location are valid
    if st.mood not in st.moods and st.moods:
        st.mood = st.moods[0]
    if st.location not in st.locations and st.locations:
        st.location = st.locations[0]

    reset_location_windows(st)

    # reschedule chat loop if running, so tick interval reflects new mood config
    if not st.paused:
        schedule_chat(app)


# ---------------- Queue helpers ----------------


def enqueue(channel: str, line: str):
    if channel not in OUTBOX:
        OUTBOX[channel] = []
    OUTBOX[channel].append(line)


def dequeue_one_chunk(channel: str, max_len: int = 3800) -> Optional[str]:
    buf = []
    total = 0
    q = OUTBOX.get(channel) or []
    while q:
        peek = q[0]
        add_len = len(peek) + (1 if buf else 0)
        if total + add_len > max_len:
            break
        buf.append(peek)
        total += add_len
        q.pop(0)
    if not buf:
        return None
    return "\n".join(buf)


# ---------------- Content generation ----------------


def pick_chat_line(st: State) -> str:
    mood_pool = st.mood_prompts.get(st.mood) or st.mood_prompts.get("neutral", [])
    if not mood_pool:
        mood_pool = ["chat checking in"]

    loc_pool = st.location_prompts.get(st.location, [])

    if st.location == "stream_start" and loc_pool:
        pool = loc_pool
    else:
        pool = loc_pool + mood_pool

    if not pool:
        pool = mood_pool

    return random.choice(pool)


def sample_amount_int(st: State) -> int:
    lo, hi = st.donation_ranges.get(st.mood, (5.0, 20.0))
    x = random.betavariate(1.2, 3.0)  # left skew
    amt = lo + x * (hi - lo)
    return max(int(round(amt)), int(lo))


def pick_donor_name(st: State) -> str:
    # VIP donor name chance 1 percent
    if random.random() < 0.01:
        return SPECIAL_USER_NAME
    pool = list(st.usernames) if st.usernames else []
    if not pool:
        pool = [
            SPECIAL_USER_NAME,
            "GhostFan42",
            "HexBug",
            "Lighthouser",
            "CthuluChow",
            "CryptWatch",
            "MorgueFiles",
        ]
    return random.choice(pool)


# ----- Spam emission -----


def build_spam_variant(base_msg: str) -> str:
    # 50 percent ALL CAPS
    msg = base_msg.upper() if random.random() < 0.5 else base_msg
    # 20 percent each ending variant
    endings = ["", "!", " !!!", " !!", " !?!"]
    msg += random.choice(endings)
    return msg


async def emit_spam(context: CallbackContext):
    st = STATE
    if not st.spam_active:
        return
    if st.spam_ticks_left <= 0:
        st.spam_active = False
        return

    st.spam_ticks_left -= 1

    user = random.choice(st.usernames) if st.usernames else "Viewer"
    spam_line = build_spam_variant(st.spam_message_base)
    enqueue("chat", hb(user) + ": " + h(spam_line))

    if st.spam_ticks_left <= 0:
        st.spam_active = False


# ----- Maya auto lines -----


async def emit_maya(context: CallbackContext):
    st = STATE
    # no Maya auto in finale
    if st.location == "finale":
        return
    if not st.maya_remaining:
        return
    now = time.time()
    seconds_in_location = now - st.loc_change_ts
    if seconds_in_location <= 300:
        raw_line = st.maya_remaining.pop(0)
        maya_text = hb("Maya") + ": " + h(raw_line)
        enqueue("maya", maya_text)


# ----- Viewer chat + hook lines -----


async def emit_chat(context: CallbackContext):
    st = STATE

    # finale blocks automatic chat entirely
    if st.location == "finale":
        return

    now = time.time()
    seconds_in_location = now - st.loc_change_ts

    # audience_hooks first 5 min, one per tick, username disguised as viewer
    if st.hook_remaining and seconds_in_location <= 300:
        raw_line = st.hook_remaining.pop(0)
        hook_user = random.choice(st.usernames) if st.usernames else "Viewer"
        hook_text = hb(hook_user) + ": " + h(raw_line)
        enqueue("chat", hook_text)

    # normal viewer chat line
    line = pick_chat_line(st)
    user = random.choice(st.usernames) if st.usernames else "Viewer"
    enqueue("chat", hb(user) + ": " + h(line))


# ----- Donations -----


async def donate_and_update(donor: str, amount_eur_int: int, note: str):
    st = STATE
    amount_eur_int = int(amount_eur_int)
    st.top_donors[donor] = st.top_donors.get(donor, 0) + amount_eur_int
    text = hb(donor) + " donated " + hb(money_fmt_int(amount_eur_int))
    if note:
        text += " - " + h(note)
    enqueue("donations", text)


async def emit_donations(context: CallbackContext):
    st = STATE
    sent_one = False

    # scripted donations always fire even in finale
    if st.pred_donations_remaining and not sent_one:
        if st.donation_window_remaining <= 0:
            d = st.pred_donations_remaining.pop(0)
            donor = d.get("donor") or pick_donor_name(st)
            amt_int = max(int(round(float(d.get("amount", 25.0)))), 0)
            await donate_and_update(donor, amt_int, d.get("note", ""))
            st.donation_window_remaining = 1
            sent_one = True
        else:
            # attempt early 25 percent
            if random.random() < 0.25:
                d = st.pred_donations_remaining.pop(0)
                donor = d.get("donor") or pick_donor_name(st)
                amt_int = max(int(round(float(d.get("amount", 25.0)))), 0)
                await donate_and_update(donor, amt_int, d.get("note", ""))
                st.donation_window_remaining = 1
                sent_one = True
            else:
                st.donation_window_remaining -= 1

    # finale blocks organic/random after scripted
    if st.location == "finale":
        return

    # organic donation roll
    if not sent_one:
        st.donation_meter += st.loc_donation_level * max(st.hype, 0.1)
        if st.donation_meter >= 1.0:
            st.donation_meter -= 1.0
            donor = pick_donor_name(st)
            amt_int = sample_amount_int(st)
            note = pick_chat_line(st)
            await donate_and_update(donor, amt_int, note)


# ---------------- Audience simulation ----------------


def update_subscribers(st: State):
    # Finale logic: lock subs to 1
    if st.location == "finale":
        st.subscribers = 1
        return

    base_rate = st.loc_subs_rate
    if base_rate is None or base_rate == 0:
        base_rate = 100.0

    delta_raw = base_rate * random.uniform(0.5, 1.5)
    delta = int(round(delta_raw))

    # mood-driven trend: negative bleeds, positive gains
    trend = st.audience_trend.get(st.mood, 1)
    if trend < 0:
        delta = -delta

    # plateau rule unless crypt or finale
    if st.subscribers >= st.plateau_subs and st.location not in ("crypt", "finale"):
        if delta > 0:
            delta = -delta

    new_val = st.subscribers + delta
    if new_val < 1:
        new_val = 1
    st.subscribers = new_val
    log.info("Subscribers %+d -> %d", delta, st.subscribers)


def update_viewers(st: State):
    # Finale logic: lock viewers to 1
    if st.location == "finale":
        st.viewers = 1
        return

    base_rate = st.loc_viewers_rate
    if base_rate is None or base_rate == 0:
        base_rate = 5000.0

    delta_raw = base_rate * random.uniform(0.5, 1.5)
    delta = int(round(delta_raw))

    # mood-driven trend: negative bleeds, positive gains
    trend = st.audience_trend.get(st.mood, 1)
    if trend < 0:
        delta = -delta

    # plateau unless crypt or finale
    if st.viewers >= st.plateau_viewers and st.location not in ("crypt", "finale"):
        if delta > 0:
            delta = -delta

    new_val = st.viewers + delta
    if new_val < 1:
        new_val = 1
    st.viewers = new_val
    log.info("Viewers %+d -> %d", delta, st.viewers)


# ---------------- Leaderboard rendering ----------------


def render_top_text(st: State) -> str:
    subs_line = "Subscribers: " + hb(format_thousand_dot(st.subscribers))
    viewers_line = "Viewers: " + hb(format_thousand_dot(st.viewers))

    if not st.top_donors:
        lines = [subs_line, viewers_line, "", "No donations yet."]
        total_sum = 0
    else:
        rows = sorted(
            st.top_donors.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:10]
        lines = [subs_line, viewers_line, "", hb("Top donors")]
        for i, (name, total) in enumerate(rows, 1):
            lines.append(f"{i}. {hb(h(name))}: {hb(money_fmt_int(total))}")
        total_sum = sum(st.top_donors.values())
    lines.append("")
    lines.append("Total: " + hb(money_fmt_int(total_sum)))
    return "\n".join(lines)


async def ensure_leaderboard_message(bot) -> int:
    st = STATE
    if st.leaderboard_msg_id is not None:
        return st.leaderboard_msg_id
    msg = await send_html(
        bot,
        CHAT_ID,
        render_top_text(st),
        thread_id=TOPICS.get("top"),
    )
    st.leaderboard_msg_id = msg.message_id
    return st.leaderboard_msg_id


async def leaderboard_tick(context: CallbackContext):
    bot = context.application.bot
    st = STATE

    # Finale stays clamped
    if st.location == "finale":
        st.subscribers = 1
        st.viewers = 1
    else:
        update_subscribers(st)
        update_viewers(st)

    # We'll render once and reuse
    new_text = render_top_text(st)

    # Make sure we have a leaderboard message
    try:
        msg_id = await ensure_leaderboard_message(bot)
    except BadRequest as e:
        log.warning("Failed to ensure leaderboard message: %s", e)
        new_msg = await send_html(
            bot,
            CHAT_ID,
            new_text,
            thread_id=TOPICS.get("top"),
        )
        st.leaderboard_msg_id = new_msg.message_id
        msg_id = st.leaderboard_msg_id

    # Try to edit the existing leaderboard message in place
    try:
        await edit_html(bot, CHAT_ID, msg_id, new_text)
    except BadRequest as e:
        text_l = str(e).lower()
        if "message is not modified" in text_l:
            # Nothing changed, that's fine. Do NOT create a new message.
            return
        # Real failure (deleted thread, etc). Recreate fresh message.
        log.warning("Leaderboard edit failed: %s - recreating", e)
        new_msg = await send_html(
            bot,
            CHAT_ID,
            new_text,
            thread_id=TOPICS.get("top"),
        )
        st.leaderboard_msg_id = new_msg.message_id


# ---------------- Scheduling loops ----------------


async def chat_tick(context: CallbackContext):
    st = STATE
    if st.paused:
        return

    # Spam first so it's loud
    await emit_spam(context)

    # Maya auto lines (first 5 minutes per location)
    await emit_maya(context)

    # Viewer chat (hooks + normal)
    await emit_chat(context)

    # Donations (scripted always, organic unless finale)
    await emit_donations(context)


def calc_tick_interval(st: State) -> float:
    mult = st.mood_interval.get(st.mood, 1.0)
    eff = st.base_interval * mult / max(st.hype, 0.1)
    return max(1.2, eff)


def schedule_chat(app: Application):
    st = STATE
    for job in app.job_queue.get_jobs_by_name(st.job_name):
        job.schedule_removal()

    interval = calc_tick_interval(st)

    app.job_queue.run_repeating(
        chat_tick,
        interval=interval,
        first=1.0,
        name=st.job_name,
        job_kwargs={
            "max_instances": 1,
            "coalesce": True,
            "misfire_grace_time": 30,
        },
    )
    log.info(
        "Scheduled chat loop interval %.2f s (mood=%s hype=%.1f loc=%s)",
        interval,
        st.mood,
        st.hype,
        st.location,
    )


def unschedule_chat(app: Application):
    st = STATE
    for job in app.job_queue.get_jobs_by_name(st.job_name):
        job.schedule_removal()
    log.info("Unscheduled chat loop")


async def drain_once(context: CallbackContext):
    bot = context.application.bot
    global _rr_index
    n = len(CHANNEL_ORDER)
    if n == 0:
        return
    for _ in range(n):
        ch = CHANNEL_ORDER[_rr_index]
        chunk = dequeue_one_chunk(ch)
        _rr_index = (_rr_index + 1) % n
        if chunk:
            thread_id = TOPICS.get(ch)
            await send_html(bot, CHAT_ID, chunk, thread_id=thread_id)


def schedule_drain(app: Application):
    st = STATE
    for job in app.job_queue.get_jobs_by_name(st.job_drain):
        job.schedule_removal()
    app.job_queue.run_repeating(
        drain_once,
        interval=RATE_GAP,
        first=1.0,
        name=st.job_drain,
        job_kwargs={
            "max_instances": 1,
            "coalesce": True,
            "misfire_grace_time": 10,
        },
    )
    log.info("Scheduled global drain every %.2f s", RATE_GAP)


def reschedule_drain(app: Application):
    schedule_drain(app)


def schedule_top_refresh(app: Application):
    st = STATE
    for job in app.job_queue.get_jobs_by_name(st.job_top):
        job.schedule_removal()
    app.job_queue.run_repeating(
        leaderboard_tick,
        interval=60.0,
        first=10.0,
        name=st.job_top,
        job_kwargs={
            "max_instances": 1,
            "coalesce": True,
            "misfire_grace_time": 60,
        },
    )
    log.info("Scheduled leaderboard refresh every 60 seconds")


# ---------------- Admin panel and commands ----------------


HELP_TEXT = (
    hb("Viral control")
    + "\n"
    + "Admin DM only. Panel controls mood, location, hype, pause/resume, Maya, Special, Donation, Spam, Bursts, Refresh Board.\n"
    + hb("Commands")
    + "\n"
    + "/panel - show admin panel\n"
    + "/mood <name> - set mood\n"
    + "/location <name> - set location\n"
    + "/pause - pause the loop\n"
    + "/resume - resume the loop\n"
    + "/say_maya <text> - speak as Maya\n"
    + "/say_special <text> - speak as Special user\n"
    + "/set_interval <seconds> - change base message interval\n"
    + "/set_rate_gap <seconds> - change min seconds between sends\n"
    + "/set_donation_range <mood> <min> <max> - set amount range for mood\n"
    + "/reload_configs - reload all configs\n"
    + "/top - init/update leaderboard"
)


def panel_markup(st: State) -> InlineKeyboardMarkup:
    # moods in rows of 2
    mood_rows: List[List[InlineKeyboardButton]] = []
    for i in range(0, len(st.moods), 2):
        chunk = st.moods[i:i + 2]
        mood_rows.append([
            InlineKeyboardButton(
                text=(("✓ " if st.mood == m else "") + m),
                callback_data=f"mood:{m}",
            )
            for m in chunk
        ])

    # locations in rows of 3
    loc_rows: List[List[InlineKeyboardButton]] = []
    for i in range(0, len(st.locations), 3):
        chunk = st.locations[i:i + 3]
        loc_rows.append([
            InlineKeyboardButton(
                text=(("✓ " if st.location == loc else "") + loc),
                callback_data=f"loc:{loc}",
            )
            for loc in chunk
        ])

    controls_main = [
        [
            InlineKeyboardButton(
                text=("Resume" if st.paused else "Pause"),
                callback_data="toggle_pause",
            ),
            InlineKeyboardButton(
                text="Send Maya",
                callback_data="prompt_maya",
            ),
            InlineKeyboardButton(
                text="Send Special",
                callback_data="prompt_special",
            ),
        ],
        [
            InlineKeyboardButton(
                text="Send Donation",
                callback_data="prompt_donation",
            ),
            InlineKeyboardButton(
                text="Burst Subs",
                callback_data="burst_subs",
            ),
            InlineKeyboardButton(
                text="Burst Viewers",
                callback_data="burst_viewers",
            ),
        ],
        [
            InlineKeyboardButton(
                text="Spam",
                callback_data="prompt_spam",
            ),
            InlineKeyboardButton(
                text="Reload configs",
                callback_data="reload_cfg",
            ),
            InlineKeyboardButton(
                text="Init leaderboard",
                callback_data="post_top",
            ),
        ],
        [
            InlineKeyboardButton(
                text="Refresh Board",
                callback_data="refresh_top",
            ),
            InlineKeyboardButton(
                text="Hype -",
                callback_data="hype:-",
            ),
            InlineKeyboardButton(
                text="Hype +",
                callback_data="hype:+",
            ),
        ],
    ]

    kb = mood_rows + loc_rows + controls_main
    return InlineKeyboardMarkup(kb)


def preview_tick_interval(st: State) -> float:
    return calc_tick_interval(st)


def panel_header(st: State, pad: int = 0) -> str:
    lo, hi = st.donation_ranges.get(st.mood, (5.0, 20.0))
    eff_interval = preview_tick_interval(st)
    return (
        f'{"-"*pad} {hb("Admin panel")} {"-"*pad}'
        + f"\nMood: {hb(st.mood)}"
        + f"\nLocation: {hb(st.location)}"
        + f"\nDonation range: {hb(f'{int(lo)}-{int(hi)} €')}"
        + f"\nSpeed: {hb(f'{eff_interval:.2f} s tick')}  Hype: {hb(f'{st.hype:.1f}')}"
        + f"\nPaused: {hb(str(st.paused))}"
    )

ADMIN_ONLY = filters.ChatType.PRIVATE & filters.User(user_id=ADMIN_ID)
ADMIN_TEXT_ONLY = ADMIN_ONLY & filters.TEXT & ~filters.COMMAND


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        hb("Viral bot ready")
        + "\nDefault mode is silent. Use the panel to Resume when you want to start streaming.",
        parse_mode=ParseMode.HTML,
    )
    await panel_cmd(update, context)


async def help_cmd(update: Update, context: CallbackContext):
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.HTML)


async def panel_cmd(update: Update, context: CallbackContext):
    await update.message.reply_text(
        panel_header(STATE, pad=40),
        parse_mode=ParseMode.HTML,
        reply_markup=panel_markup(STATE),
    )


async def mood_cmd(update: Update, context: CallbackContext):
    if not context.args:
        await update.message.reply_text(
            h("Usage: /mood <name>"),
            parse_mode=ParseMode.HTML,
        )
        return
    m = context.args[0].strip().lower()
    if m not in STATE.moods:
        await update.message.reply_text(
            h(f"Unknown mood {m}"),
            parse_mode=ParseMode.HTML,
        )
        return
    STATE.mood = m
    if not STATE.paused:
        schedule_chat(context.application)
    await update.message.reply_text(
        h(f"Mood set to {m}"),
        parse_mode=ParseMode.HTML,
    )


async def location_cmd(update: Update, context: CallbackContext):
    if not context.args:
        await update.message.reply_text(
            h("Usage: /location <name>"),
            parse_mode=ParseMode.HTML,
        )
        return
    loc = context.args[0].strip().lower()
    if loc not in STATE.locations:
        await update.message.reply_text(
            h(f"Unknown location {loc}"),
            parse_mode=ParseMode.HTML,
        )
        return
    STATE.location = loc
    reset_location_windows(STATE)
    if not STATE.paused:
        schedule_chat(context.application)
    await update.message.reply_text(
        h(f"Location set to {loc}"),
        parse_mode=ParseMode.HTML,
    )


async def pause_cmd(update: Update, context: CallbackContext):
    STATE.paused = True
    unschedule_chat(context.application)
    await update.message.reply_text(
        h("Paused. No messages will be sent."),
        parse_mode=ParseMode.HTML,
    )


async def resume_cmd(update: Update, context: CallbackContext):
    STATE.paused = False
    schedule_chat(context.application)
    await update.message.reply_text(
        h("Resumed. Streaming started."),
        parse_mode=ParseMode.HTML,
    )


async def say_maya(update: Update, context: CallbackContext):
    if not context.args:
        await update.message.reply_text(
            h("Usage: /say_maya <text>"),
            parse_mode=ParseMode.HTML,
        )
        return
    msg = " ".join(context.args)
    enqueue("maya", hb("Maya") + ": " + h(msg))
    await update.message.reply_text(
        h("Queued."),
        parse_mode=ParseMode.HTML,
    )


async def say_special(update: Update, context: CallbackContext):
    if not context.args:
        await update.message.reply_text(
            h("Usage: /say_special <text>"),
            parse_mode=ParseMode.HTML,
        )
        return
    msg = " ".join(context.args)
    enqueue("chat", hb(SPECIAL_USER_NAME) + ": " + h(msg))
    await update.message.reply_text(
        h("Queued."),
        parse_mode=ParseMode.HTML,
    )


async def on_admin_text(update: Update, context: CallbackContext):
    st = STATE
    txt = update.message.text.strip()

    # Spam setup
    if st.awaiting_spam_text:
        st.awaiting_spam_text = False
        st.spam_active = True
        st.spam_message_base = txt
        st.spam_ticks_left = 20
        await update.message.reply_text(
            h("Spam armed for 20 ticks."),
            parse_mode=ParseMode.HTML,
        )
        return

    # Donation step 1
    if st.awaiting_donation_name:
        st.pending_donation_name = txt
        st.awaiting_donation_name = False
        st.awaiting_donation_amount = True
        await update.message.reply_text(
            h("Got donor name. Now send the donation amount in EUR (integer)."),
            parse_mode=ParseMode.HTML,
        )
        return

    # Donation step 2
    if st.awaiting_donation_amount:
        try:
            amt_int = int(round(float(txt)))
        except ValueError:
            await update.message.reply_text(
                h("Amount must be a number."),
                parse_mode=ParseMode.HTML,
            )
            return
        donor = st.pending_donation_name or pick_donor_name(st)
        note = pick_chat_line(st)
        await donate_and_update(donor, amt_int, note)

        st.awaiting_donation_amount = False
        st.pending_donation_name = ""
        await update.message.reply_text(
            h("Donation queued."),
            parse_mode=ParseMode.HTML,
        )
        return

    # Burst subs
    if st.awaiting_sub_burst:
        st.awaiting_sub_burst = False
        try:
            delta = int(round(float(txt)))
        except ValueError:
            await update.message.reply_text(
                h("Burst must be a number."),
                parse_mode=ParseMode.HTML,
            )
            return
        st.subscribers += delta
        if st.subscribers < 1:
            st.subscribers = 1
        await update.message.reply_text(
            h(f"Subscribers adjusted to {st.subscribers}."),
            parse_mode=ParseMode.HTML,
        )
        return

    # Burst viewers
    if st.awaiting_viewer_burst:
        st.awaiting_viewer_burst = False
        try:
            delta = int(round(float(txt)))
        except ValueError:
            await update.message.reply_text(
                h("Burst must be a number."),
                parse_mode=ParseMode.HTML,
            )
            return
        st.viewers += delta
        if st.viewers < 1:
            st.viewers = 1
        await update.message.reply_text(
            h(f"Viewers adjusted to {st.viewers}."),
            parse_mode=ParseMode.HTML,
        )
        return

    # Maya manual
    if st.awaiting_maya_text:
        msg = txt
        enqueue("maya", hb("Maya") + ": " + h(msg))
        st.awaiting_maya_text = False
        await update.message.reply_text(
            h("Queued Maya message."),
            parse_mode=ParseMode.HTML,
        )
        return

    # Special manual
    if st.awaiting_special_text:
        msg = txt
        enqueue("chat", hb(SPECIAL_USER_NAME) + ": " + h(msg))
        st.awaiting_special_text = False
        await update.message.reply_text(
            h("Queued Special message."),
            parse_mode=ParseMode.HTML,
        )
        return

    # fallback
    await update.message.reply_text(
        h("Use /panel for controls or commands like /mood, /location."),
        parse_mode=ParseMode.HTML,
    )


async def set_interval(update: Update, context: CallbackContext):
    if not context.args:
        await update.message.reply_text(
            h("Usage: /set_interval <seconds>"),
            parse_mode=ParseMode.HTML,
        )
        return
    try:
        sec = float(context.args[0])
    except ValueError:
        await update.message.reply_text(
            h("Value must be a number like 2.0"),
            parse_mode=ParseMode.HTML,
        )
        return
    STATE.base_interval = max(1.2, sec)
    if not STATE.paused:
        schedule_chat(context.application)
    await update.message.reply_text(
        h(f"Base interval set to {STATE.base_interval:.2f} s"),
        parse_mode=ParseMode.HTML,
    )


async def set_rate_gap(update: Update, context: CallbackContext):
    global RATE_GAP
    if not context.args:
        await update.message.reply_text(
            h("Usage: /set_rate_gap <seconds>"),
            parse_mode=ParseMode.HTML,
        )
        return
    try:
        sec = float(context.args[0])
    except ValueError:
        await update.message.reply_text(
            h("Value must be a number like 1.2"),
            parse_mode=ParseMode.HTML,
        )
        return
    RATE_GAP = max(1.0, sec)
    reschedule_drain(context.application)
    await update.message.reply_text(
        h(f"Min gap set to {RATE_GAP:.2f} s"),
        parse_mode=ParseMode.HTML,
    )


async def set_donation_range(update: Update, context: CallbackContext):
    if len(context.args) != 3:
        await update.message.reply_text(
            h("Usage: /set_donation_range <mood> <min> <max>"),
            parse_mode=ParseMode.HTML,
        )
        return
    m, lo, hi = context.args
    try:
        lo = float(lo)
        hi = float(hi)
    except ValueError:
        await update.message.reply_text(
            h("Min and max must be numbers"),
            parse_mode=ParseMode.HTML,
        )
        return
    if m not in STATE.moods:
        await update.message.reply_text(
            h(f"Unknown mood {m}"),
            parse_mode=ParseMode.HTML,
        )
        return
    if lo >= hi or lo < 0:
        await update.message.reply_text(
            h("Expected 0 <= min < max"),
            parse_mode=ParseMode.HTML,
        )
        return
    STATE.donation_ranges[m] = (lo, hi)
    await update.message.reply_text(
        h(f"Donation range for {m} set to [{int(lo)}-{int(hi)}] €"),
        parse_mode=ParseMode.HTML,
    )


async def reload_configs(update: Update, context: CallbackContext):
    refresh_configs(STATE, context.application)
    await update.message.reply_text(
        h("Reloaded configs, prompts, usernames, and reset windows."),
        parse_mode=ParseMode.HTML,
    )


async def post_top(update: Update, context: CallbackContext):
    await ensure_leaderboard_message(context.application.bot)
    await leaderboard_tick(context)
    await update.message.reply_text(
        h("Leaderboard initialized or updated."),
        parse_mode=ParseMode.HTML,
    )


async def on_panel(update: Update, context: CallbackContext):
    query = update.callback_query
    if query.from_user.id != ADMIN_ID:
        await query.answer("Unauthorized", show_alert=False)
        return

    data = query.data or ""
    handled = False

    if data.startswith("mood:"):
        m = data.split(":", 1)[1]
        if m in STATE.moods:
            STATE.mood = m
            if not STATE.paused:
                schedule_chat(context.application)
            handled = True

    elif data.startswith("loc:"):
        loc = data.split(":", 1)[1]
        if loc in STATE.locations:
            STATE.location = loc
            reset_location_windows(STATE)
            if not STATE.paused:
                schedule_chat(context.application)
            handled = True

    elif data == "toggle_pause":
        STATE.paused = not STATE.paused
        if STATE.paused:
            unschedule_chat(context.application)
        else:
            schedule_chat(context.application)
        handled = True

    elif data == "prompt_maya":
        STATE.awaiting_maya_text = True
        STATE.awaiting_special_text = False
        STATE.awaiting_donation_name = False
        STATE.awaiting_donation_amount = False
        STATE.awaiting_sub_burst = False
        STATE.awaiting_viewer_burst = False
        STATE.awaiting_spam_text = False
        STATE.pending_donation_name = ""
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=h("Reply with Maya text. It will be posted to the Maya channel."),
            parse_mode=ParseMode.HTML,
        )
        handled = True

    elif data == "prompt_special":
        STATE.awaiting_special_text = True
        STATE.awaiting_maya_text = False
        STATE.awaiting_donation_name = False
        STATE.awaiting_donation_amount = False
        STATE.awaiting_sub_burst = False
        STATE.awaiting_viewer_burst = False
        STATE.awaiting_spam_text = False
        STATE.pending_donation_name = ""
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=h(
                f"Reply with Special text. It will be posted to the chat channel as {SPECIAL_USER_NAME}."
            ),
            parse_mode=ParseMode.HTML,
        )
        handled = True

    elif data == "prompt_donation":
        STATE.awaiting_donation_name = True
        STATE.awaiting_donation_amount = False
        STATE.awaiting_maya_text = False
        STATE.awaiting_special_text = False
        STATE.awaiting_sub_burst = False
        STATE.awaiting_viewer_burst = False
        STATE.awaiting_spam_text = False
        STATE.pending_donation_name = ""
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=h("Send donor username now."),
            parse_mode=ParseMode.HTML,
        )
        handled = True

    elif data == "burst_subs":
        STATE.awaiting_sub_burst = True
        STATE.awaiting_viewer_burst = False
        STATE.awaiting_maya_text = False
        STATE.awaiting_special_text = False
        STATE.awaiting_donation_name = False
        STATE.awaiting_donation_amount = False
        STATE.awaiting_spam_text = False
        STATE.pending_donation_name = ""
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=h("Send number to add/remove subscribers (can be negative)."),
            parse_mode=ParseMode.HTML,
        )
        handled = True

    elif data == "burst_viewers":
        STATE.awaiting_viewer_burst = True
        STATE.awaiting_sub_burst = False
        STATE.awaiting_maya_text = False
        STATE.awaiting_special_text = False
        STATE.awaiting_donation_name = False
        STATE.awaiting_donation_amount = False
        STATE.awaiting_spam_text = False
        STATE.pending_donation_name = ""
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=h("Send number to add/remove viewers (can be negative)."),
            parse_mode=ParseMode.HTML,
        )
        handled = True

    elif data == "prompt_spam":
        STATE.awaiting_spam_text = True
        STATE.awaiting_sub_burst = False
        STATE.awaiting_viewer_burst = False
        STATE.awaiting_maya_text = False
        STATE.awaiting_special_text = False
        STATE.awaiting_donation_name = False
        STATE.awaiting_donation_amount = False
        STATE.pending_donation_name = ""
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=h("Send spam text now. It will be blasted for 20 ticks."),
            parse_mode=ParseMode.HTML,
        )
        handled = True

    elif data == "reload_cfg":
        refresh_configs(STATE, context.application)
        handled = True

    elif data == "post_top":
        await ensure_leaderboard_message(context.application.bot)
        await leaderboard_tick(context)
        handled = True

    elif data == "refresh_top":
        await leaderboard_tick(context)
        handled = True

    elif data.startswith("hype:"):
        direction = data.split(":", 1)[1]
        delta = 0.2 if direction == "+" else -0.2
        STATE.hype = max(
            0.2,
            min(5.0, round(STATE.hype + delta, 2))
        )
        if not STATE.paused:
            schedule_chat(context.application)
        handled = True

    if handled:
        await query.answer("Updated")
        try:
            await query.edit_message_text(
                panel_header(STATE, pad=40),
                parse_mode=ParseMode.HTML,
                reply_markup=panel_markup(STATE),
            )
        except BadRequest:
            await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=panel_header(STATE, pad=40),
                parse_mode=ParseMode.HTML,
                reply_markup=panel_markup(STATE),
            )
    else:
        await query.answer("No change")


# ---------------- App wiring ----------------


def build_app() -> Application:
    global STATE, LOCATION_SCRIPTS
    # overwrite placeholder STATE with a fully initialized one
    STATE = init_state()

    app = ApplicationBuilder().token(TOKEN).build()

    # Admin-only DM handlers
    app.add_handler(CommandHandler("start", start, filters=ADMIN_ONLY))
    app.add_handler(CommandHandler("help", help_cmd, filters=ADMIN_ONLY))
    app.add_handler(CommandHandler("panel", panel_cmd, filters=ADMIN_ONLY))

    app.add_handler(CommandHandler("mood", mood_cmd, filters=ADMIN_ONLY))
    app.add_handler(CommandHandler("location", location_cmd, filters=ADMIN_ONLY))
    app.add_handler(CommandHandler("pause", pause_cmd, filters=ADMIN_ONLY))
    app.add_handler(CommandHandler("resume", resume_cmd, filters=ADMIN_ONLY))

    app.add_handler(CommandHandler("say_maya", say_maya, filters=ADMIN_ONLY))
    app.add_handler(CommandHandler("say_special", say_special, filters=ADMIN_ONLY))
    app.add_handler(MessageHandler(ADMIN_TEXT_ONLY, on_admin_text))

    app.add_handler(CommandHandler("set_interval", set_interval, filters=ADMIN_ONLY))
    app.add_handler(CommandHandler("set_rate_gap", set_rate_gap, filters=ADMIN_ONLY))
    app.add_handler(CommandHandler("set_donation_range", set_donation_range, filters=ADMIN_ONLY))

    app.add_handler(CommandHandler("reload_configs", reload_configs, filters=ADMIN_ONLY))
    app.add_handler(CommandHandler("top", post_top, filters=ADMIN_ONLY))

    app.add_handler(CallbackQueryHandler(on_panel))

    schedule_top_refresh(app)
    schedule_drain(app)

    log.info("Starting Viral bot in silent mode")
    return app


if __name__ == "__main__":
    app = build_app()
    app.run_polling(close_loop=False)

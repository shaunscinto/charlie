# Model Finder — Display Filter + Color/Category Logic + MSRP Search & Display
# (v7a+accessories+fix2+wallovens+sizes-strict-v3 BRAND-LOCK + COUNTER-DEPTH + GE-EXPAND + TV-SIZES±5
#  + WASHER-MODES + DISHW-WASH FIX + SHOW-QUESTION-ABOVE-ANSWER + REFER-STYLES + FRENCH-DOOR
#  + FULL/STANDARD-DEPTH w/ SYNONYMS + FULL-DEPTH DEFAULT GROUPS + DOWNDRAFT ROUTING
#  + FOUR-DOOR (FOUR/CD4DR dual-depth) + ADA DISHWASHERS + CFM FILTER (HOODS/MICOTR SET)
#  + FREEZERS (UPRIGHT/CHEST strict group routing) + CAPACITY (cu ft next to model)
#  + DISPOSALS (DISPSR only) + DISHWASHER EXCLUDES DISPSR — ENFORCED ALWAYS)
#  + RANGES width (RANGE + PRORNG, Description-only)
#  + COOKTOPS (CTOP only; exclude RNGTOP) & RANGETOPS (RNGTOP only) width (Description-only)
#  + FIXES: capacity ±3 hard clamp using max cu ft; four-door excludes MIEL/MIELE; rangetop spelling variants
#  + RANGES control & fuel routing (rear/freestanding vs front/slide-in; gas/electric/induction/dual)
#  + Laundry Centers/towers (LCNTR), generic Laundry (WASH+DRY w/ accessory exclusions),
#    Compact Laundry (CMPCTW/COMPGS/COMPEL), Electrolux→ELUX brand alias
#  + BUGFIX: groups_override now strictly enforced even after category resolution
#  + BRAND-ONLY QUERIES: return all in-stock for the brand (with family expansions)
#  + BUILT-IN REFRIGERATION ROUTING (BIREFR + detailed groups)
#  + Built-in refrigeration width filter (Description/col H)
#  + Mattress Description-only display (only when Category=MATTS)
#  + Mattress name matching via Description (e.g., "dupont")
#  + Microwave routing — OTR→MICOTR, Countertop→CNTOP, Drawer→MWDRWR
#  + NEW: OTR low-profile / flush-mount filters (Description/col H)
#  + NEW: VENT (range hoods) — Description-only routing (ignore groups) + subtypes + width + brand/color
#  + UI UPDATE: Title/labels/placeholder + auto "Stock data last updated on: DD/MM" banner tied to Google Sheet

import os, io, re, json
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from email.utils import parsedate_to_datetime

import requests
import pandas as pd
import duckdb
import streamlit as st

# -------------------- Config --------------------
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
DATA_CSV_URL     = os.getenv("DATA_CSV_URL", "").strip()
ADMIN_PIN        = os.getenv("ADMIN_PIN", "standardtv2025")
BRAND_NAME       = os.getenv("BRAND_NAME", "Your Store")

# MSRP overrides (optional)
PRICE_COL_NAME   = os.getenv("PRICE_COL_NAME", "").strip() or None  # e.g., "MSRP"
PRICE_COL_INDEX  = os.getenv("PRICE_COL_INDEX", "").strip() or None # 1-based, e.g., "7"

# Files for caching data + source metadata
CACHE_FILE = "./_inventory_cache.parquet"
META_FILE  = "./_source_meta.json"

# -------------------- Page --------------------
st.set_page_config(page_title="Charlie - Your Personal Inventory Assistant (CLOSED BETA)", layout="wide")
st.title("Charlie - Your Personal Inventory Assistant (CLOSED BETA)")
st.caption("Inventory search query data sourced from STORIS.")

# -------------------- Utilities --------------------
def find_file(*names):
    for base in [Path("."), Path("/mnt/data"), Path(__file__).parent]:
        for n in names:
            p = base / n
            if p.exists():
                return str(p)
    return None

def _num_clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(r"[^\d,.\-]", "", regex=True)
         .str.replace(r"(?<=\d),(?=\d{3}\b)", "", regex=True)
         .str.replace(",", ".", regex=False, n=1),
        errors="coerce"
    )

def _safe_get_col_by_index(df: pd.DataFrame, one_based_index: int | None, default_one_based: int | None):
    if df is None or df.empty:
        return None, None
    idx_1 = None
    if one_based_index:
        try: idx_1 = int(one_based_index)
        except: idx_1 = None
    if idx_1 is None and default_one_based is not None:
        idx_1 = int(default_one_based)
    if idx_1 is None:
        return None, None
    idx_0 = idx_1 - 1
    if 0 <= idx_0 < df.shape[1]:
        return df.iloc[:, idx_0], df.columns[idx_0]
    return None, None

def tokens(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())

def _norm_letters(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())

def _read_meta() -> dict:
    try:
        if os.path.exists(META_FILE):
            with open(META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _write_meta(meta: dict):
    try:
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    except Exception:
        pass

def _la_dt_from_http(last_modified_hdr: str | None, fallback_utc: datetime | None = None) -> datetime:
    """
    Convert Last-Modified header (HTTP-date) to America/Los_Angeles datetime.
    If header missing/unparsable, use fallback_utc (assumed UTC) or now().
    """
    try:
        if last_modified_hdr:
            dt = parsedate_to_datetime(last_modified_hdr)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(ZoneInfo("America/Los_Angeles"))
    except Exception:
        pass
    if fallback_utc is None:
        fallback_utc = datetime.now(timezone.utc)
    return fallback_utc.astimezone(ZoneInfo("America/Los_Angeles"))

def _format_dd_mm(dt: datetime | None) -> str:
    if not dt:
        return "—"
    try:
        return dt.strftime("%d/%m")  # DD/MM
    except Exception:
        return "—"

# -------------------- Sheet fetch with change detection --------------------
def _head_sheet(url: str) -> tuple[str | None, str | None]:
    if not url:
        return None, None
    try:
        r = requests.head(url, timeout=10, allow_redirects=True)
        return r.headers.get("ETag"), r.headers.get("Last-Modified")
    except Exception:
        return None, None

def _fetch_sheet(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def _load_or_refresh_inventory() -> tuple[pd.DataFrame, datetime | None]:
    """
    Returns (df, last_updated_la_dt). Will refresh cache if ETag/Last-Modified changed.
    """
    if not DATA_CSV_URL:
        return pd.DataFrame(), None

    meta = _read_meta()
    etag_new, lm_new = _head_sheet(DATA_CSV_URL)

    etag_old = meta.get("etag")
    lm_old   = meta.get("last_modified")

    need_refresh = False
    if not os.path.exists(CACHE_FILE):
        need_refresh = True
    elif etag_new and etag_new != etag_old:
        need_refresh = True
    elif lm_new and lm_new != lm_old:
        need_refresh = True
    elif not (etag_old or lm_old):
        # No known source meta yet; initialize
        need_refresh = True

    last_updated_la = None

    if need_refresh:
        try:
            raw = _fetch_sheet(DATA_CSV_URL)
            df = normalize_columns(raw)
            if len(df):
                df.to_parquet(CACHE_FILE, index=False)
            # Prefer Last-Modified for display; fall back to now (UTC)
            now_utc = datetime.now(timezone.utc)
            last_updated_la = _la_dt_from_http(lm_new, now_utc)
            _write_meta({
                "etag": etag_new,
                "last_modified": lm_new,
                "last_fetch_utc": now_utc.isoformat(),
                "last_updated_display_la": last_updated_la.isoformat(),
            })
            return df, last_updated_la
        except Exception:
            # On failure, try to fall back to local cache if present
            pass

    # Fallback: load cached parquet and compute display date from stored meta or file mtime
    try:
        df = pd.read_parquet(CACHE_FILE)
    except Exception:
        df = pd.DataFrame()

    last_la_iso = _read_meta().get("last_updated_display_la")
    if last_la_iso:
        try:
            last_updated_la = datetime.fromisoformat(last_la_iso)
        except Exception:
            last_updated_la = None

    if last_updated_la is None and os.path.exists(CACHE_FILE):
        # Use file mtime (local tz) as a rough proxy
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE), tz=ZoneInfo("America/Los_Angeles"))
            last_updated_la = mtime
        except Exception:
            last_updated_la = None

    return df, last_updated_la

# -------------------- Mattress accessory/mattress code sets --------------------
_PROVIDED_CODES = {
    "TXBOX","TXADJB","TFOAM","THYBRD","TCOIL","TBOX","TADJBS",
    "QFOAM","QHYBRD","QCOIL","QBOX","QADJBS","BOX","SQBOX","PILL","BEDACC",
    "KFOAM","KHYBRD","KCOIL","KBOX","KADJBS","HYBRD","COIL",
    "FFOAM","FHYBRD","FCOIL","FBOX","FADJBS","FOAM",
    "CKFOAM","CKHYBR","CKCOIL","CKBOX","CKADJB","BEDFRM","ADJBSE"
}
_MATTRESS_GROUP_CODES = {
    "TFOAM","THYBRD","TCOIL",
    "QFOAM","QHYBRD","QCOIL",
    "KFOAM","KHYBRD","KCOIL",
    "HYBRD","COIL",
    "FFOAM","FHYBRD","FCOIL",
    "FOAM",
    "CKFOAM","CKHYBR","CKCOIL"
}
_EXPLICIT_ACCESSORY_CODES = _PROVIDED_CODES.difference(_MATTRESS_GROUP_CODES)

ACCESSORY_QUERY_TERMS = (
    "box spring","box springs","box","boxes",
    "foundation","foundations","found","fdn",
    "pillow","pillows","sheet","sheets",
    "frame","frames",
    "accessory","accessories","acc"
)
ACCESSORY_TEXT_KEYWORDS = [
    "ACC","ACCESSORY","ACCESSORIES","BOX","BOX SPRING","BOXSPRING",
    "HIGH BOX","LOW BOX","FOUNDATION","FOUND","FDN","PILLOW","PILLOWS",
    "SHEET","SHEETS","FRAME","FRAMES"
]

# -------------------- Wall ovens (BICOOK) --------------------
WALL_OVEN_GROUPS = {
    "SPEED","RNGTOP","STEAM",
    "MWCM27","OVNG27","OVNE27","DBEL27",
    "OVNE30","DBEL30","DBEL24","OVNE24","OVNG24","OVNG30",
    "MWCM30","OVNE36"
}
def filter_wall_groups_by_size(groups: list[str], size: str | None) -> list[str]:
    if not size:
        return groups
    size = str(size)
    return [g for g in groups if g.endswith(size)]

# -------------------- Depth helpers --------------------
COUNTER_DEPTH_GROUPS = {"CDSXS","CDFDIW","CDFDBM","CD4DR"}
def detect_counter_depth(q: str) -> bool:
    qn = q.lower()
    return (
        "counter depth" in qn or
        "counter-depth" in qn or
        "counterdepth" in qn or
        re.search(r"\bcounter\s*depth\b", qn) is not None
    )
FULL_DEPTH_SYNONYM_PAT = re.compile(
    r"""
    \b(
        full\s*depth
      | standard(\s*size)?\s*depth
      | standard[-\s]?depth
      | regular\s*depth
      | regular[-\s]?size\s*depth
    )\b
    """,
    re.I | re.X
)
def detect_full_or_standard_depth(q: str) -> bool:
    return FULL_DEPTH_SYNONYM_PAT.search(q or "") is not None

# -------------------- Televisions (ELECTR) --------------------
TV_GROUPS = {"4KTV40","4KTV50","4KTV60","4KTV70","4KTV80"}
def detect_tv(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in [" tv", " tvs", "television", "televisions", "4k tv", "4ktv"]) or ql.strip().startswith("tv") or ql.strip().startswith("tvs")
def detect_tv_size(q: str) -> int | None:
    qn = q.lower().replace("”", '"').replace("“", '"').replace("''", '"').replace("’", "'").replace("‘", "'")
    m = re.search(r'\b(\d{2,3})\s*(?:-|–)?\s*(?:in|inch|inches|")?\b', qn)
    if not m:
        return None
    try:
        val = int(m.group(1))
        if 20 <= val <= 120:
            return val
    except:
        pass
    return None
def tv_groups_within_5(size_int: int) -> list[str]:
    base_sizes = [40,50,60,70,80]
    near = [s for s in base_sizes if abs(s - size_int) <= 5]
    return [f"4KTV{s}" for s in near]
def tv_size_text_where(size_int: int) -> str:
    lo = max(0, size_int - 5); hi = size_int + 5
    pats = []
    for n in range(lo, hi + 1):
        s = str(n)
        pats.extend([f"%{s}\"%", f"%{s}''%", f"%{s} IN%", f"%{s}-INCH%", f"%{s}IN%", f"%{s} INCH%"])
    cols = ['UPPER("Description")', 'UPPER("Second_Description")']
    ors = []
    for col in cols:
        ors.append("(" + " OR ".join([f"{col} ILIKE '{p}'" for p in pats]) + ")")
    return "(" + " OR ".join(ors) + ")"

# -------------------- NEW: generic width parser + Description-only matcher --------------------
def detect_width_generic(q: str) -> int | None:
    qn = q.lower().replace("”", '"').replace("“", '"').replace("''", '"').replace("’", "'").replace("‘", "'")
    m = re.search(r'\b(\d{2,3})\s*(?:-|–)?\s*(?:in|inch|inches|")\b', qn)
    if not m:
        m = re.search(r'\b(\d{2,3})\b', qn)
    if not m: return None
    try:
        v = int(m.group(1))
        if 18 <= v <= 72:
            return v
    except: pass
    return None

def width_text_where_desc(size_int: int) -> str:
    s = str(size_int)
    pats = [f"%{s}\"%", f"%{s}''%", f"%{s} IN%", f"%{s}-INCH%", f"%{s}IN%", f"%{s} INCH%"]
    col = 'UPPER("Description")'
    return "(" + " OR ".join([f"{col} ILIKE '{p}'" for p in pats]) + ")"

# -------------------- NEW: RANGES detection + width (Description-only) --------------------
def detect_range(q: str) -> bool:
    ql = q.lower()
    if "hood" in ql and "range" in ql:
        return False
    return re.search(r'\branges?\b', ql) is not None
def detect_range_width(q: str) -> int | None:
    return detect_width_generic(q)
def range_size_text_where(size_int: int) -> str:
    return width_text_where_desc(size_int)

# -------- NEW: RANGE control & fuel intent --------
def detect_range_control(q: str) -> str | None:
    ql = q.lower()
    if re.search(r"\b(free[-\s]?standing|freestanding|rear[-\s]*control|back[-\s]*control)\b", ql):
        return "rear"
    if re.search(r"\b(slide[-\s]?in|front[-\s]*control)\b", ql):
        return "front"
    return None
def detect_range_fuel(q: str) -> str | None:
    ql = q.lower()
    if re.search(r"\b(dual[-\s]*fuel|duel[-\s]*fuel|dual)\b", ql): return "dual"
    if re.search(r"\binduction\b", ql): return "induction"
    if re.search(r"\b(electric|radiant)\b", ql): return "electric"
    if re.search(r"\bgas\b", ql): return "gas"
    return None
RANGE_GROUPS_REAR = {"RNGIRC","RNGGRC","RNGERC","RNGDF","24RNGG","24RNGE","20RNGG","20RNGE"}
RANGE_GROUPS_FRONT = {"RNGIFC","RNGGFC","RNGEFC","RNGDFC"}
RANGE_GROUPS_FUEL = {
    "gas":       {"RNGGRC","RNGGFC","RNGGDD","24RNGG","20RNGG"},
    "electric":  {"RNGERC","RNGEFC","RNGEDD","24RNGE","20RNGE"},
    "induction": {"RNGIRC","RNGIFC"},
    "dual":      {"RNGDF","RNGDFC","RNGDDD"},
}

# -------------------- NEW: COOKTOP + RANGETOP detection + width (Description-only) --------------------
def detect_cooktop(q: str) -> bool:
    return re.search(r'\bcooktops?\b', q.lower()) is not None
def detect_rangetop(q: str) -> bool:
    return re.search(r"\b(?:rangetop(?:'s)?|range[-\s]?top(?:'s)?|range[-\s]?tops(?:'s)?)\b", q.lower()) is not None
def ctop_rtop_width_from(q: str) -> int | None:
    return detect_width_generic(q)
def cooktop_size_text_where(size_int: int) -> str:
    return width_text_where_desc(size_int)

# -------------------- WASHERS (logic with dishwasher guard) --------------------
FRONT_LOAD_WASH_GROUPS = {"CMPCTW", "COMBO", "FLWM", "LCNTR"}
TOP_LOAD_WASH_GROUPS   = {"TLIWM", "TLAWM"}
AGITATOR_WASH_GROUPS   = {"TLAWM"}
IMPELLER_WASH_GROUPS   = {"TLIWM"}

def detect_dishwasher_intent(q: str) -> bool:
    ql = q.lower()
    return bool(re.search(r'\b(garbage\s+disposal|disposals?|disposers?)\b', ql)) or bool(re.search(r'\bdish(wash(?:er|ers)?)?\b', ql))

def detect_washer_style(q: str):
    ql = q.lower()
    if detect_dishwasher_intent(ql):
        return None, False
    washer_intent = (
        re.search(r'\bwasher(s)?\b', ql) is not None or
        re.search(r'\blaundry\b', ql) is not None
    )
    mode = None
    if washer_intent and (re.search(r"\bagitat(or|or\s+wash(?:er)?)\b", ql) or "agitator" in ql):
        mode = "agitator"
    elif washer_intent and ("impeller" in ql or "with an impeller" in ql):
        mode = "impeller"
    else:
        if re.search(r"\bfront[-\s]*load(ing|er|)?\b", ql) or "frontloader" in ql:
            mode = "front"; washer_intent = True
        elif re.search(r"\btop[-\s]*load(ing|er|)?\b", ql) or "toploader" in ql:
            mode = "top"; washer_intent = True
    return mode, washer_intent

# -------- NEW: Laundry-specific detectors (expanded) --------
def detect_laundry_center(q: str) -> bool:
    ql = q.lower()
    patterns = [
        r"\blaundry\s*centres?\b", r"\blaundry\s*centers?\b",
        r"\bwash\s*towers?\b", r"\bwash\-?tower\b", r"\bwash\s*tower\b", r"\bwashtower\b",
        r"\blaundry\s*towers?\b",
        r"\bcenter\s*laundry\b", r"\bcentre\s*laundry\b",
        r"\bwasher\s*dryer\s*(?:tower|center|centre)s?\b"
    ]
    return any(re.search(p, ql) for p in patterns)
def detect_compact_laundry(q: str) -> bool:
    ql = q.lower()
    return ("compact" in ql) and any(w in ql for w in ["washer","washers","dryer","dryers","laundry","stack"])
def detect_generic_laundry(q: str) -> bool:
    return re.search(r"\blaundry\b", q.lower()) is not None

# -------------------- REFER (FRIDGE) STYLE DETECTION --------------------
def detect_refer_style(q: str):
    ql = q.lower()
    refer_intent = any(s in ql for s in [
        "fridge","fridges","refrigerator","refrigerators","refer","refers","refr"
    ])
    style = None
    if re.search(r'\btop[-\s]*mount\b', ql) or re.search(r'\btop[-\s]*freezer\b', ql):
        style = "topmount"; refer_intent = True
    if re.search(r'\bside[-\s]*by[-\s]*side\b', ql) or re.search(r'\bsxs\b', ql) or re.search(r'\bsbs\b', ql):
        style = "sxs"; refer_intent = True
    if re.search(r'\bfrench[-\s]*door(s)?\b', ql) or re.search(r'\bfrenchdoor(s)?\b', ql):
        style = "french"; refer_intent = True
    if re.search(r'\b(4\s*door|4-door|four[-\s]*door|quad[-\s]*door|quaddoor|quad)\b', ql):
        style = "four"; refer_intent = True
    return style, refer_intent

# -------------------- NEW: BUILT-IN REFRIGERATION DETECTION --------------------
def detect_built_in_fridge(q: str):
    ql = q.lower()
    def has_any(words): return any(w in ql for w in words)
    def word(pat): return re.search(pat, ql) is not None

    fridge_words = ["fridge","fridges","refrigerator","refrigerators","refer","refers","refr"]
    built_in_words = ["built in","built-in","builtin","bi "]
    under_counter_words = ["under counter","undercounter","under-counter","under-counter","u/c"]

    if "wine" in ql and has_any(under_counter_words):
        return True, "REFER", ["WINEUN"]
    if ("wine" in ql) and (word(r"\bcolumn(s)?\b") or has_any(built_in_words)):
        return True, "REFER", ["WINE"]
    if (("beverage" in ql or "bev" in ql) and has_any(under_counter_words)) or \
       (has_any(under_counter_words) and ("beverage" in ql) or ("bev" in ql) or ("fridge" in ql) or ("refer" in ql)):
        return True, "REFER", ["BEVRG","REFUNC"]
    if ("beverage center" in ql or "beverage centres" in ql or "full size beverage" in ql or
        "tall beverage" in ql or "bev center" in ql or "bev centre" in ql):
        return True, "REFER", ["BEVFS"]

    bi_intent = (has_any(built_in_words) and (has_any(fridge_words) or "columns" in ql or word(r"\bcolumn(s)?\b"))) \
                or word(r"\bbi\s*(?:fridge|refer|refrigerator)s?\b")
    if not bi_intent:
        return False, None, None

    if word(r"\bcolumn(s)?\b") or "columns" in ql:
        if any(w in ql for w in ["freezer","freez","fzr"]):
            return True, "BIREFR", ["ALLBIF","FRECLM"]
        if any(w in ql for w in ["refer","fridge","refrigerator","fridges"]):
            return True, "BIREFR", ["ALLBIR","REFCLM"]
        return True, "BIREFR", ["ALLBIR","ALLBIF","FRECLM","REFCLM"]

    if any(word(p) for p in [r"\bsxs\b", r"\bsbs\b", r"\bside[-\s]*by[-\s]*side\b"]):
        return True, "BIREFR", ["BISXS"]

    if any(w in ql for w in ["refer drawer","refer drawers","fridge drawer","fridge drawers","refrigerator drawer","refrigerator drawers","drawers","drawer"]) and has_any(fridge_words):
        return True, "BIREFR", ["DRWR"]

    if re.search(r'\bfrench[-\s]*door(s)?\b', ql) or "frenchdoor" in ql:
        return True, "BIREFR", ["BIFDBM"]

    if any(word(p) for p in [r"\bbottom\s*mount\b", r"\bover[-\s]*under\b", r"\bover/?under\b", r"\bbottom\s*freezer\b"]):
        return True, "BIREFR", ["BIBOTT"]

    if has_any(built_in_words) and has_any(fridge_words):
        return True, "BIREFR", None

    return False, None, None

# -------------------- NEW: MICROWAVE DETECTION (OTR / Countertop / Drawer) --------------------
def _has_mw_word(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["microwave","microwaves","micro","mw"])

# Treat bare "otr" as OTR too (no microwave word required)
def detect_microwave_otr(q: str) -> bool:
    ql = q.lower()
    return (
        re.search(r"(?<![a-z0-9])otr(?![a-z0-9])", ql) is not None or
        re.search(r"\bover[-\s]*the[-\s]*range\b", ql) is not None or
        re.search(r"\bover\s*range\b", ql) is not None or
        (_has_mw_word(q) and "otr" in ql)
    )

def detect_microwave_countertop(q: str) -> bool:
    ql = q.lower()
    return (
        re.search(r"\bcounter[\s\-]*top\b", ql) is not None or "countertop" in ql
    ) and _has_mw_word(q)

def detect_microwave_drawer(q: str) -> bool:
    ql = q.lower()
    return (
        "microdrawer" in ql or "micro drawer" in ql or
        "microwave drawer" in ql or "drawer microwave" in ql or
        re.search(r"\bmw\s*drawer\b", ql) is not None
    )

# NEW: OTR refinement — detect "low profile" / "flush mount" intents
def detect_mw_low_profile_intent(q: str) -> bool:
    ql = q.lower()
    return bool(
        re.search(r"\blow[\s\-]*profile\b", ql) or
        "lowprofile" in ql or
        re.search(r"\bslim\b", ql)
    )

def detect_mw_flush_mount_intent(q: str) -> bool:
    ql = q.lower()
    return bool(
        re.search(r"\bflush[\s\-]*mount(ed)?\b", ql) or
        "flushmount" in ql or
        re.search(r"(?<!\w)flush(?!\w)", ql)
    )

# -------------------- NEW: DOWNDRAFT DETECTION --------------------
def detect_downdraft(q: str):
    ql = q.lower()
    has_dd = ("downdraft" in ql) or ("down draft" in ql) or ("pop up" in ql) or ("pop-up" in ql) or ("popup" in ql)
    if not has_dd:
        return False, None, None
    mentions_cooktop = any(w in ql for w in ["cooktop", "cook top", "ctop"])
    mentions_range   = "range" in ql
    mentions_oven    = "oven" in ql
    mentions_hood    = any(w in ql for w in ["hood", "vent", "ventilation", "extractor"])
    is_gas       = "gas" in ql
    is_induction = "induction" in ql
    is_electric  = ("electric" in ql) or ("radiant" in ql) or ("smoothtop" in ql) or ("smooth top" in ql)
    if mentions_hood or (has_dd and not (mentions_cooktop or mentions_range or mentions_oven)):
        return True, "VENT", None  # VENT path now description-only, so no group list
    if mentions_range or mentions_oven:
        return True, "RANGE", ["RNGDDD", "RNGGDD", "RNGEDD"]
    if mentions_cooktop or has_dd:
        if is_induction:
            return True, "CTOP", ["CTEI"]
        if is_gas:
            return True, "CTOP", ["CTEG"]
        if is_electric:
            return True, "CTOP", ["CTED"]
        return True, "CTOP", ["CTED", "CTEG", "CTEI"]
    return False, None, None

# -------------------- NEW: ADA DISHWASHER DETECTION --------------------
def detect_ada_dishwasher(q: str) -> bool:
    ql = q.lower()
    return ("ada" in ql) and detect_dishwasher_intent(ql)

# -------------------- NEW: CFM DETECTION & PARSING --------------------
CFM_GROUPS = {"MICOTR", "BLOW", "PHOOD", "DWNDFT", "ISLAND", "LINER", "WALLCH", "WALLUC"}
CFM_RANGE_PAT = re.compile(
    r"""
    (?:
        between\s*(\d{2,4})\s*(?:and|-|to)\s*(\d{2,4})\s*cfm
      | \b(?:under|below|less\s+than)\s*(\d{2,4})\s*cfm
      | \b(?:over|above|greater\s+than|at\s+least)\s*(\d{2,4})\s*cfm
      | \b(?:around|about|approx(?:imately)?)\s*(\d{2,4})\s*cfm
      | \b(\d{2,4})\s*cfm\b
    )
    """,
    re.I | re.X
)
def detect_cfm_intent(q: str) -> bool:
    return "cfm" in q.lower()
def parse_cfm_filter(q: str):
    m = CFM_RANGE_PAT.search(q)
    if not m: return None, None, None
    g = m.groups()
    def to_i(x):
        try: return int(x)
        except: return None
    if g[0] and g[1]:
        a,b = to_i(g[0]), to_i(g[1]); 
        if a and b: return (min(a,b), max(a,b), None)
    if g[2]: x = to_i(g[2]); return (None, x, None)
    if g[3]: x = to_i(g[3]); return (x, None, None)
    if g[4]:
        x = to_i(g[4])
        if x: return (max(0, int(round(x*0.9))), int(round(x*1.1)), "approx")
    if g[5]:
        x = to_i(g[5])
        if x: return (x, x, None)
    return None, None, None

CFM_EXTRACT_PAT = re.compile(r'(\d{2,4})\s*CFM', re.I)
def extract_cfm_from_text(text: str) -> int | None:
    if not text: return None
    m = CFM_EXTRACT_PAT.search(str(text))
    try:
        return int(m.group(1)) if m else None
    except:
        return None

# -------------------- NEW: CAPACITY (CU FT) EXTRACTION --------------------
CUFT_PAT = re.compile(
    r"""
    \b
    (\d{1,2}(?:\.\d{1,2})?)
    \s*
    (?:
        cu\.?\s*ft
      | cu\s*ft
      | cuft
      | cft
      | cubic\s*feet
      | ft\^?3
      | ft³
      | c\.?\s*f\.?
      | cf
    )
    \b
    """,
    re.I | re.X
)
def extract_cuft_max_from_text(text: str) -> float | None:
    if not text: return None
    vals = []
    for m in CUFT_PAT.finditer(str(text)):
        try:
            v = float(m.group(1))
            if 1.0 <= v <= 60.0:
                vals.append(v)
        except:
            pass
    if not vals:
        return None
    return max(vals)

COUNTER_DEPTH_TEXT_PAT = re.compile(
    r'(COUNTER[\s\-]*DEPTH|CNTR[\s\-]*DEPTH|CNTR[\s\-]*DPTH|CNTRDEPTH|CNTRDPTH)',
    re.I
)
def is_counter_depth_text(s: str) -> bool:
    return bool(COUNTER_DEPTH_TEXT_PAT.search(str(s or "")))

# -------------------- Load definition files --------------------
@st.cache_data(show_spinner=False)
def load_definitions():
    cat_map, grp_map, groups_by_cat = {}, {}, {}
    cat_path = find_file("category code definitions.csv")
    if cat_path:
        d = pd.read_csv(cat_path)
        d.columns = [c.strip().replace(" ", "_").lower() for c in d.columns]
        if "category" in d.columns and "category_desc" in d.columns:
            for _, r in d.iterrows():
                desc = str(r["category_desc"]).strip().lower()
                code = str(r["category"]).strip().upper()
                if desc and code:
                    cat_map[desc] = code
    grp_path = find_file("group code definitions.csv")
    if grp_path:
        d = pd.read_csv(grp_path)
        d.columns = [c.strip().replace(" ", "_").lower() for c in d.columns]
        if all(c in d.columns for c in ["group","group_desc","category"]):
            for _, r in d.iterrows():
                g = str(r["group"]).strip().upper()
                gd = str(r["group_desc"]).strip().lower()
                c = str(r["category"]).strip().upper()
                if gd and g:
                    grp_map[gd] = g
                if g and c:
                    groups_by_cat.setdefault(c, []).append(g)
    return cat_map, grp_map, groups_by_cat

CAT_MAP, GRP_MAP, GROUPS_BY_CAT = load_definitions()

# --------- Brand alias + expansions ----------
@st.cache_data(show_spinner=False)
def load_brand_aliases():
    alias_to_brand = {}
    path = find_file("brand code definitions.csv")
    if path:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        brand_col = None
        for c in df.columns:
            if c in {"brand","brand_name","brandcode","brand_code"}:
                brand_col = c; break
        if brand_col:
            alias_cols = [c for c in df.columns if c != brand_col]
            for _, r in df.iterrows():
                base = str(r[brand_col]).strip()
                if not base:
                    continue
                base_up = base.upper()
                alias_to_brand[base.lower()] = base_up
                for c in alias_cols:
                    val = str(r.get(c, "")).strip()
                    if not val: continue
                    for piece in re.split(r"[;/,|]+", val):
                        a = piece.strip()
                        if a:
                            alias_to_brand[a.lower()] = base_up

    alias_to_brand.update({
        "tempurpedic":"TEMP","tempur-pedic":"TEMP","tempur":"TEMP","temp":"TEMP",
        "stearns":"STFO","sterns":"STFO","stfo":"STFO","stearns & foster":"STFO","stearns and foster":"STFO",
        "sealy":"SEAL","smat":"SMAT","seal":"SEAL",
        "monogram":"MONO",
        "ge profile":"GPRO","profile":"GPRO",
        "café":"CAFE","cafe":"CAFE",
        "ge":"GE",
        "bosch":"BOSCH","thermador":"THERMADOR","kitchenaid":"KITCHENAID","jennair":"JENNAIR",
        "whirlpool":"WHIRLPOOL","maytag":"MAYTAG","amana":"AMANA","samsung":"SAMSUNG","lg":"LG",
        "fisher & paykel":"FISHERPAYKEL","fisher and paykel":"FISHERPAYKEL","fisherpaykel":"FISHERPAYKEL",
        "miele":"MIELE","cove":"COVE","wolf":"WOLF","sub-zero":"SUBZERO","subzero":"SUBZERO",
        "speed queen":"SPED","speedqueen":"SPED","s queen":"SPED",
        "electrolux":"ELUX","electro-lux":"ELUX","e-lux":"ELUX","elux":"ELUX"
    })

    PROVIDED = {
        "ZEPH":"ZEPHYR","WOLF":"WOLF","WINE":"WINEGARD","WHIR":"WHIRLPOOL CORPORATION",
        "WEBR":"WEBER GRILLS","VITA":"VITARA","VINT":"VINTEX ELECTROLUX","VERO":"VERONA",
        "VENT":"VENT-A-HOOD","USED":"USED APLIANCES","UNIQ":"UNIQUE GAS PRODUCTS LTD",
        "ULIN":"ULINE APPLIANCES","OROU":"TV","TRUE":"TRUE MANUFACTURING CO, LTD",
        "TRUI":"TRU OUTDOOR ISLANDS","THER":"THERMADOR","TEMP":"TEMPURPEDIC","SUMM":"SUMMIT APPLIANCES",
        "SUBZ":"SUBZERO","STFO":"STEARNS & FOSTER","SPED":"SPEED QUEEN","SONY":"SONY ELECTRONICS",
        "SONS":"SONOS","SONO":"SONOS","SMEG":"SMEG","SHAR":"SHARP","SEAL":"SEALY MATTRESS CO",
        "SMAT":"SEALY","SCOT":"SCOTSMAN","SAMG":"SAMSUNG-ELECTRONICS","ZSNG":"SAMSUNG ELECTRONICS",
        "SSNG":"SAMSUNG APPLIANCES","RTEQ":"RECTEQ","PURE":"PURECARE","PERL":"PERLICK",
        "MIEL":"MIELE APPLIANCES","MIDE":"MIDEA","MAYT":"MAYTAG","MARV":"MARVEL","LYNX":"LYNX",
        "LIEB":"LIEBHERR","LSKS":"LG SKS","LGEL":"LG ELECTRONICS","LGAP":"LG APPLIANCES",
        "COFU":"LA CORNUE CORNUFE","KOOL":"KOOLAIRE","KNIK":"KNICKERBOCKER BED CO","KLIP":"KLIPSCH AUDIO",
        "KITC":"KITCHENAID INC","JENS":"JENSEN","JENN":"JENN-AIR COMPANY","ILVE":"ILVE","HOTP":"HOTPOINT",
        "HOSH":"HOSHIZAKI","HEST":"HESTAN","HAIE":"HAIER","MONO":"GENERAL ELECTRIC MONOGRAM",
        "GENE":"GENERAL ELECTRIC COMPANY","GPRO":"GE PROFILE","GCAF":"GE CAFE","GALZ":"GALANZ",
        "GAGG":"GAGGENAU","FULG":"FULGOR","FPRO":"FRIGIDAIRE PROFESSIONAL","FGAL":"FRIGIDAIRE GALLERY",
        "FRIG":"FRIGIDAIRE","FORN":"FORNO","FONT":"FONTANA FORNI","FOLL":"FOLLETT",
        "FPKL":"FISHER & PAYKEL APPLIANCES","FAGR":"FAGOR","FABR":"FABER HOODS","EVOG":"EVO GRILLS",
        "ELIC":"ELICA","ELEM":"ELEMENT APPLIANCE COMPANY LLC","ELUX":"ELECTROLUX","DCSA":"DCS BY FISHER&PAYKEL",
        "DANB":"DANBY PRODUCTS INC","DACO":"DACOR","CROS":"CROSLEY","COYO":"COYOTE","COVE":"COVE",
        "COFE":"CORNUFE","KING":"BROIL KING","BROA":"BROAN","BREW":"BREW EXPRESS","BOSC":"BOSCH APPLIANCES",
        "BNCH":"BOSCH APPLIANCES","BLOM":"BLOMBERG","BLAZ":"BLAZE GRILLS","BEST":"BEST RANGE HOODS",
        "BERT":"BERTAZZONI","BEKO":"BEKO","BEDG":"BED GEAR","AZUR":"AZURE","AXIS":"AXIS","AVAN":"AVANTI APPLIANCES",
        "ASKO":"ASKO","AMRG":"AMERICAN RANGE","AMNA":"AMANA","ALFR":"ALFRESCO GRILLS"
    }
    for code, desc in PROVIDED.items():
        if desc:
            alias_to_brand[desc.lower()] = code
            fw = desc.split()[0].lower()
            if fw and fw not in alias_to_brand:
                alias_to_brand[fw] = code

    alias_to_brand.update({
        "samsung":"SAMSUNG",
        "lg electronics":"LG",
        "lg appliances":"LG",
        "sonos":"SONOS",
        "frigidaire":"FRIGIDAIRE",
        "sealy":"SEALY"
    })
    return alias_to_brand

BRAND_ALIASES = load_brand_aliases()
BRAND_EXPANSIONS = {
    "GE": {"GENE","GPRO"},
    "SAMSUNG": {"SAMG","ZSNG","SSNG"},
    "LG": {"LGEL","LGAP"},
    "SONOS": {"SONS","SONO"},
    "FRIGIDAIRE": {"FRIG","FGAL","FPRO"},
    "SEALY": {"SEAL","SMAT"},
}
GE_EQUIV = {"GENE", "GPRO"}

# -------------------- Relevant subcategories (Excel) --------------------
@st.cache_data(show_spinner=False)
def load_relevant_subcategories():
    xlsx = find_file("relevent subcategories.xlsx", "relevant subcategories.xlsx")
    if not xlsx:
        return {}, set(_EXPLICIT_ACCESSORY_CODES)
    groups_by_category, accessory_groups = {}, set()
    try:
        xl = pd.ExcelFile(xlsx)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df.columns = [re.sub(r"[^A-Za-z0-9_]", "", c.replace(" ", "_").lower()) for c in df.columns]
            if "category" not in df.columns or "group" not in df.columns:
                continue
            acc_cols = [c for c in df.columns if any(k in c for k in ["acc","accessory"])]
            for _, r in df.iterrows():
                c = str(r.get("category","")).upper().strip()
                g = str(r.get("group","")).upper().strip()
                if not c or not g:
                    continue
                groups_by_category.setdefault(c, set()).add(g)
                is_acc = g.endswith("ACC")
                for ac in acc_cols:
                    val = str(r.get(ac,"")).lower().strip()
                    if val in {"1","true","yes","y"}:
                        is_acc = True
                if is_acc:
                    accessory_groups.add(g)
        for k in list(groups_by_category.keys()):
            groups_by_category[k] = sorted(groups_by_category[k])
        accessory_groups |= set(_EXPLICIT_ACCESSORY_CODES)
        return groups_by_category, accessory_groups
    except Exception:
        return {}, set(_EXPLICIT_ACCESSORY_CODES)

GROUPS_BY_XLSX, ACCESSORY_GROUPS = load_relevant_subcategories()

# -------------------- Inventory normalization --------------------
def _pick_price_series(df: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    if PRICE_COL_NAME:
        for c in df.columns:
            if c.strip().lower() == PRICE_COL_NAME.strip().lower():
                return df[c], c
    s, nm = _safe_get_col_by_index(df, PRICE_COL_INDEX, None)
    if s is not None:
        return s, nm or f"INDEX_{PRICE_COL_INDEX}"
    s, nm = _safe_get_col_by_index(df, None, 7)
    if s is not None:
        return s, nm or "COLUMN_G"
    like_terms = ("msrp", "price", "list", "retail", "map", "tag", "ticket", "srp")
    for c in df.columns:
        if any(t in c.lower() for t in like_terms):
            return df[c], c
    return None, None

def _ensure_price_from_g_if_empty(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    used = ""
    if "Price" not in df.columns or df["Price"] is None or df["Price"].isna().all():
        s, nm = _safe_get_col_by_index(df, None, 7)
        if s is not None:
            df["Price"] = _num_clean(s)
            used = nm or "COLUMN_G"
    return df, used

def _promote_and_coalesce_price(df: pd.DataFrame) -> pd.DataFrame:
    s, nm = _pick_price_series(df)
    if s is not None:
        df["Price"] = _num_clean(s)
        df["Price_Source"] = nm
    else:
        df["Price"] = pd.NA
        df["Price_Source"] = None
    df, forced_nm = _ensure_price_from_g_if_empty(df)
    if forced_nm:
        df["Price_Source"] = f"{forced_nm} (forced G)"
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^A-Za-z0-9_]", "", c.replace(" ", "_")).strip() for c in df.columns]
    col_map = {
        "brand":"Brand","model":"Model","sku":"SKU",
        "category":"Category","group":"Group","qty":"Qty",
        "description":"Description",
        "second_description":"Second_Description","second_desc":"Second_Description",
        "2nd_desc":"Second_Description","2nd_description":"Second_Description",
        "2nd_desc_":"Second_Description",
        "dsp":"DSP","display":"DSP","on_display":"DSP","ondisplay":"DSP","showroom":"DSP"
    }
    for c in list(df.columns):
        lc = c.lower()
        if lc in col_map and col_map[lc] not in df.columns:
            df.rename(columns={c: col_map[lc]}, inplace=True)

    if "Qty" not in df.columns: df["Qty"] = 0
    def col_score(name: str) -> int:
        n = name.lower(); score = 0
        if "qty" in n or "quantity" in n: score += 3
        if "avail" in n: score += 2
        if "inv" in n: score += 2
        if "onhand" in n or "on_hand" in n: score += 2
        return score
    candidates_qty = sorted(df.columns, key=col_score, reverse=True)
    base_qty = pd.to_numeric(df.get("Qty", 0), errors="coerce").fillna(0)
    if float(base_qty.sum()) == 0:
        for c in candidates_qty:
            if col_score(c) == 0: continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any() and float(s.fillna(0).sum()) > 0:
                base_qty = s.fillna(0); break
    df["Qty"] = pd.to_numeric(base_qty, errors="coerce").fillna(0)

    for c in ["Category","Group","Second_Description","Brand","Description"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    if "Second_Description" not in df.columns:
        df["Second_Description"] = ""
    df["Second_Desc_NORM"] = df["Second_Description"].str.upper().str.replace(r"[^A-Z]", "", regex=True)

    if "DSP" in df.columns:
        def dsp_to_bool(x):
            s = str(x).strip().lower()
            return s in {"y","yes","true","1","on","display","floor","showroom"}
        df["DSP"] = df["DSP"].apply(dsp_to_bool)
    else:
        df["DSP"] = False

    df = _promote_and_coalesce_price(df)
    return df

# -------------------- Load Data (with auto-update) --------------------
def get_live_df_and_last_updated():
    df, last_dt = _load_or_refresh_inventory()
    if df.empty and DATA_CSV_URL:
        # Fallback one-shot load if parquet missing/corrupted
        try:
            df = normalize_columns(_fetch_sheet(DATA_CSV_URL))
            if len(df):
                df.to_parquet(CACHE_FILE, index=False)
            now_la = datetime.now(ZoneInfo("America/Los_Angeles"))
            meta = _read_meta()
            meta.update({
                "etag": meta.get("etag"),
                "last_modified": meta.get("last_modified"),
                "last_fetch_utc": datetime.now(timezone.utc).isoformat(),
                "last_updated_display_la": now_la.isoformat(),
            })
            _write_meta(meta)
            last_dt = now_la
        except Exception:
            pass
    return df, last_dt

# -------------------- Admin --------------------
def admin_gate():
    with st.expander("Admin (upload/refresh)"):
        pin = st.text_input("Enter admin PIN", type="password")
        c1, c2, _ = st.columns([1,1,1])
        ok = c1.button("Unlock"); force = c2.button("Force reload (clear cache)")
        if force:
            try: st.cache_data.clear()
            except: pass
            try: 
                if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
                if os.path.exists(META_FILE): os.remove(META_FILE)
            except: pass
            st.success("Cache cleared.")
        if ok:
            if pin == ADMIN_PIN:
                st.success("Admin unlocked"); return True
            st.error("Invalid PIN")
    return False

admin = admin_gate()
st.divider()

# -------------------- Load Data --------------------
df, _last_updated_la = get_live_df_and_last_updated()

# Banner: top-left "Stock data last updated on: DD/MM"
left, _ = st.columns([1,3])
with left:
    st.markdown(f"**Stock data last updated on:** {_format_dd_mm(_last_updated_la)}")

if df.empty:
    st.warning("No data loaded yet. Use Admin above.")
    st.stop()
else:
    con = duckdb.connect(); con.register("inventory", df)

# -------------------- Existing codes in data --------------------
EXISTING_CAT = set(str(x).upper().strip() for x in df.get("Category", pd.Series([], dtype="object")).dropna().unique())
EXISTING_GRP = set(str(x).upper().strip() for x in df.get("Group", pd.Series([], dtype="object")).dropna().unique())
EXISTING_BRANDS = set(str(x).upper().strip() for x in df.get("Brand", pd.Series([], dtype="object")).dropna().unique())
EXISTING_BRANDS_NORM = { _norm_letters(b): b for b in EXISTING_BRANDS }
EXISTING_MODELS = set(str(x).upper().strip() for x in df.get("Model", pd.Series([], dtype="object")).dropna().unique())
EXISTING_MODELS_NORM = { _norm_letters(m): m for m in EXISTING_MODELS }

def _quoted_chunks(q: str) -> list[str]:
    out = []
    for m in re.finditer(r'\"([^\"]{2,})\"|model\s+([A-Za-z0-9\-_/\.]{2,})', q, flags=re.I):
        s = (m.group(1) or m.group(2) or "").strip()
        if s: out.append(s)
    return out

def detect_model_search(q: str) -> tuple[list[str], list[str]]:
    q_up = (q or "").upper()
    candidates = _quoted_chunks(q)
    for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/\.]{1,}", q_up):
        t2 = t.strip(" ._-/")
        if len(t2) >= 3 and re.search(r"\d", t2):
            candidates.append(t2)
    seen = set(); cand = []
    for c in candidates:
        c2 = c.upper()
        if c2 not in seen:
            seen.add(c2); cand.append(c2)
    exact_models = []
    partial_tokens = []
    for c in cand:
        norm_c = _norm_letters(c)
        if norm_c in EXISTING_MODELS_NORM:
            exact_models.append(EXISTING_MODELS_NORM[norm_c]); continue
        if any(c in m for m in EXISTING_MODELS):
            partial_tokens.append(c)
    return sorted(set(exact_models)), sorted(set(partial_tokens))

# -------------------- Brand detection (STRICT brand lock + brand-only mode) --------------------
def _word_boundary_present(hay: str, needle: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])", hay, re.I) is not None
def _alias_in_query(ql: str, alias: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", ql, re.I) is not None

def detect_brand_candidates(q: str) -> list[str]:
    ql = q.lower()
    alias_hit = None
    for alias in sorted(BRAND_ALIASES.keys(), key=len, reverse=True):
        if alias and _alias_in_query(ql, alias):
            alias_hit = BRAND_ALIASES[alias].upper()
            break

    candidates = set()

    if alias_hit and alias_hit in BRAND_EXPANSIONS:
        fam = BRAND_EXPANSIONS[alias_hit]
        for code in fam:
            if code in EXISTING_BRANDS:
                candidates.add(code)

    if alias_hit == "GE":
        for g in GE_EQUIV:
            if g in EXISTING_BRANDS:
                candidates.add(g)

    if alias_hit and alias_hit in EXISTING_BRANDS:
        candidates.add(alias_hit)

    for b in sorted(EXISTING_BRANDS, key=len, reverse=True):
        if _word_boundary_present(q, b):
            candidates.add(b)

    def add_matches_for(token_up: str):
        norm_tok = _norm_letters(token_up)
        for norm_b, raw_b in EXISTING_BRANDS_NORM.items():
            if norm_tok == norm_b or norm_tok in norm_b or norm_b in norm_tok:
                candidates.add(raw_b)

    if alias_hit and alias_hit not in BRAND_EXPANSIONS and alias_hit not in EXISTING_BRANDS:
        add_matches_for(alias_hit)

    return sorted(candidates)

def is_brand_only_query(
    q: str,
    brand_candidates: list[str],
    *,
    color: str | None,
    price_min: float | None,
    price_max: float | None,
    model_exact_list: list[str],
    model_partial_tokens: list[str],
    has_any_other_intent: bool
) -> bool:
    if not brand_candidates:
        return False
    if color or price_min is not None or price_max is not None:
        return False
    if model_exact_list or model_partial_tokens:
        return False
    if has_any_other_intent:
        return False
    qtok = set(tokens(q))
    stop = {"brand", "brands", "stuff", "all", "everything", "items", "models", "inventory", "stock", "in", "show", "me"}
    brand_candidates = brand_candidates or []
    if qtok.issubset(stop | set(tokens(" ".join(brand_candidates)))):
        return True
    return True

# -------------------- Mattress understanding (now with name tokens) --------------------
MATTRESS_SIZE_PATTERNS = [
    ("CAL KING", r"\b(cal(?:ifornia)?\s*king|cal\s*king)\b"),
    ("KING",     r"\b(king)\b"),
    ("QUEEN",    r"\b(queen)\b"),
    ("FULL",     r"\b(full|double)\b"),
    ("TWIN XL",  r"\b(twin\s*xl|xl\s*twin)\b"),
    ("TWIN",     r"\b(twin)\b"),
]
MATTRESS_TYPE_PATTERNS = [
    ("HYBRID",      r"\b(hybrid)\b"),
    ("COIL",        r"\b(coil|innerspring|spring)\b"),
    ("FOAM",        r"\b(foam|memory|latex)\b"),
]
MATTRESS_TERMS = ("mattress", "mattresses", "matts", "matt")

def detect_mattress_filters(q: str):
    ql = q.lower()
    matt_intent = any(term in ql for term in MATTRESS_TERMS)
    size, mtype = None, None
    if matt_intent:
        for label, rx in MATTRESS_SIZE_PATTERNS:
            if re.search(rx, ql):
                size = label; break
        for label, rx in MATTRESS_TYPE_PATTERNS:
            if re.search(rx, ql):
                mtype = label; break
    return size, mtype, matt_intent

def extract_mattress_name_tokens(q: str) -> list[str]:
    ql = q.lower()
    stop = {"mattress","mattresses","matts","matt","bed","beds","set","sets","with","and","or","the"}
    for label, rx in MATTRESS_SIZE_PATTERNS + MATTRESS_TYPE_PATTERNS:
        ql = re.sub(rx, " ", ql)
    toks = [t for t in tokens(ql) if t not in stop and len(t) >= 3]
    return toks

def detect_accessory_intent(q: str) -> bool:
    ql = q.lower()
    return any(term in ql for term in ACCESSORY_QUERY_TERMS)

# -------------------- Wall oven detector + width --------------------
def detect_wall_oven(q: str) -> bool:
    ql = q.lower()
    return any(p in ql for p in [
        "wall oven","wall ovens","wall-oven","wall-ovens","walloven","wallovens",
        "built-in oven","built in oven","built-in ovens","built in ovens",
        "speed oven","speed ovens"
    ])
def detect_wall_oven_size(q: str) -> str | None:
    qn = q.lower().replace("”", '"').replace("“", '"').replace("''", '"').replace("’", "'").replace("‘", "'")
    m = re.search(r'\b(24|27|30|36)\s*(?:-|–)?\s*(?:in|inch|inches|")?\b', qn)
    if m:
        return m.group(1)
    return None

# -------------------- VENT (hoods) detectors --------------------
VENT_TRIG_PAT = re.compile(r"\b(hoods?|range\s*hoods?|vents?|ventilation|extractor|exhaust)\b", re.I)
def detect_vent_intent(q: str) -> bool:
    return VENT_TRIG_PAT.search(q or "") is not None

def vent_sub_keywords(q: str) -> list[str]:
    ql = (q or "").lower()
    keys = []
    if re.search(r"\bpro\b", ql) or "phood" in ql or "pro hood" in ql:
        keys += [" PRO ", "PRO-HOOD", "PRO HOOD", "PROHOOD", "PHOOD"]
    if "downdraft" in ql or "down draft" in ql or "ddraft" in ql or re.search(r"(?<!\w)dd(?!\w)", ql) or "pop-up" in ql or "pop up" in ql:
        keys += ["DOWNDRAFT", "DOWN DRAFT", "DDRAFT", " POP UP", "POP-UP"]
    if "island" in ql:
        keys += ["ISLAND"]
    if "insert" in ql or "liner" in ql or "panel ready" in ql or "custom hood" in ql or "custom" in ql:
        keys += ["INSERT", "LINER", "PANEL READY", "CUSTOM HOOD", "CUSTOM"]
    if "under cabinet" in ql or "under-cabinet" in ql or "undercabinet" in ql or "undcab" in ql or "uncab" in ql:
        keys += ["UNDER CABINET", "UNDER-CABINET", "UNDERCABINET", "UNDCAB", "UNCAB"]
    if "wall mount" in ql or "wall hood" in ql or re.search(r"(?<!\w)wall(?!\w)", ql):
        keys += ["WALL MOUNT", "WALL HOOD", "WALL"]
    # Dedup, preserve order-ish
    seen=set(); out=[]
    for k in keys:
        if k not in seen:
            seen.add(k); out.append(k)
    return out

# -------------------- Synonyms / definition matching --------------------
BASE_SYNONYMS = {
    "dishwasher":"DISHW","dishwashers":"DISHW",
    "dish":"DISHW","dishes":"DISHW",
    "fridge":"REFER","fridges":"REFER","refrigerator":"REFER","refrigerators":"REFER",
    "refer":"REFER","refers":"REFER","refr":"REFER",
    "range":"RANGE","ranges":"RANGE",
    "oven":"OVENS","ovens":"OVENS",
    "cooktop":"CTOP","cooktops":"CTOP",
    "washer":"WASH","washers":"WASH","laundry":"WASH",
    "dryer":"DRY","dryers":"DRY",
    # CHANGED: route hood/vent terms to VENT (Category) explicitly
    "hood":"VENT","hoods":"VENT","range hood":"VENT","range hoods":"VENT",
    "vent":"VENT","vents":"VENT","ventilation":"VENT","extractor":"VENT","exhaust":"VENT",
    "mattress":"MATTS","mattresses":"MATTS","matts":"MATTS","matt":"MATTS",
    "wall oven":"BICOOK","wall ovens":"BICOOK","wall-oven":"BICOOK","wall-ovens":"BICOOK","walloven":"BICOOK","wallovens":"BICOOK",
    "built-in oven":"BICOOK","built in oven":"BICOOK",
    "built-in ovens":"BICOOK","built in ovens":"BICOOK",
    "speed oven":"BICOOK","speed ovens":"BICOOK",
    "freezer":"FREEZ","freezers":"FREEZ",
}
def match_from_definitions(q: str):
    ql = q.lower()
    for phrase, code in BASE_SYNONYMS.items():
        if phrase in ql:
            return "Category", code
    for desc, code in CAT_MAP.items():
        if desc in ql:
            return "Category", code
    for desc, code in GRP_MAP.items():
        if desc in ql:
            return "Group", code
    qtok = set(tokens(ql))
    for desc, code in sorted(CAT_MAP.items(), key=lambda x: -len(x[0])):
        dtok = set(tokens(desc))
        if dtok and dtok.issubset(qtok):
            return "Category", code
    for desc, code in sorted(GRP_MAP.items(), key=lambda x: -len(x[0])):
        dtok = set(tokens(desc))
        if dtok and dtok.issubset(qtok):
            return "Group", code
    return None, None

def detect_category_or_group(q: str):
    ctype, code = match_from_definitions(q)
    if ctype and code:
        if ctype == "Category":
            if code not in EXISTING_CAT and code in GROUPS_BY_CAT:
                return ctype, code
            if code not in EXISTING_CAT:
                for cat in EXISTING_CAT:
                    if any(k in cat for k in [w.upper() for w in tokens(q)]):
                        return "Category", cat
        else:
            if code not in EXISTING_GRP:
                for grp in EXISTING_GRP:
                    if any(k in grp for k in [w.upper() for w in tokens(q)]):
                        return "Group", grp
        return ctype, code
    return None, None

# -------------------- Color / display / pricing --------------------
COLOR_PATTERNS = {
    "WHITE": re.compile(r"\b(white|wht|matte\s*white|bright\s*white|gloss\s*white)\b", re.I),
    "BLACK": re.compile(r"\b(black|blk|mblk|mbk|graphite\s*black|matte\s*black)\b", re.I),
    "STAINLESS": re.compile(r"\b(stainless(\s*steel)?|black\s*stainless)\b", re.I),
    "PANEL": re.compile(r"\b(panel(\s*ready)?|custom\s*panel)\b", re.I),
    "SLATE": re.compile(r"\b(slate)\b", re.I),
    "BISQUE": re.compile(r"\b(bisque|almond|beige)\b", re.I),
}
def detect_color(q: str):
    for name, rx in COLOR_PATTERNS.items():
        if rx.search(q): return name
    return None
def wants_matte(q: str) -> bool:
    return "matte" in q.lower()
def detect_display(q: str) -> bool:
    ql = q.lower()
    return any(p in ql for p in [
        "on display", "display model", "on the floor",
        "showroom", "floor model", "on showroom", "on the showroom"
    ])
def is_count_question(q: str) -> bool:
    ql = q.lower().strip()
    return re.search(r'\b(how\s+many|count|total)\b', ql) is not None

PRICE_RANGE_PAT = re.compile(
    r"""
    (?:
        between\s*\$?([\d,]+)\s*(?:and|-|to)\s*\$?([\d,]+)
      | \b(?:under|below|less\s+than)\s*\$?([\d,]+)
      | \b(?:over|above|greater\s+than|at\s+least)\s*\$?([\d,]+)
      | \b(?:around|about|approx(?:imately)?)\s*\$?([\d,]+)
      | (?:^|\s)\$?(\d[\d,]*)\s*-\s*\$?(\d[\d,]*)
      | \b(?:msrp|price|cost|priced?)\s*\$?([\d,]+)\b
    )
    """,
    re.I | re.X
)
def parse_price_filter(text: str):
    m = PRICE_RANGE_PAT.search(text)
    if not m:
        return None, None
    def num(x):
        if not x: return None
        try: return float(str(x).replace(",", ""))
        except: return None
    g = m.groups()
    if g[0] and g[1]:
        a, b = num(g[0]), num(g[1])
        if a is not None and b is not None: return (min(a,b), max(a,b))
    if g[2]: x = num(g[2]); return (None, x)
    if g[3]: x = num(g[3]); return (x, None)
    if g[4]: x = num(g[4]); return (x*0.90, x*1.10)
    if g[5] and g[6]:
        a, b = num(g[5]), num(g[6])
        if a is not None and b is not None: return (min(a,b), max(a,b))
    if g[7]:
        x = num(g[7]); return (x, x)
    return None, None

# -------------------- Category alias (REFR ↔ REFER) --------------------
def resolve_category_alias(df, code):
    existing = set(str(x).upper().strip() for x in df["Category"].unique())
    pairs = [("REFR", "REFER"), ("REFER", "REFR")]
    for a, b in pairs:
        if code == a and b in existing:
            return b
    return code

# -------------------- Group list by Category --------------------
def group_list_for_category(code):
    code = resolve_category_alias(df, code)
    valid = []
    if code in GROUPS_BY_XLSX:
        valid = [g for g in GROUPS_BY_XLSX[code] if g not in ACCESSORY_GROUPS and not g.endswith("ACC")]
    elif code in GROUPS_BY_CAT:
        valid = [g for g in GROUPS_BY_CAT[code] if not g.endswith("ACC")]
    if code == "BICOOK":
        valid = list(sorted(set((valid or [])) | set(WALL_OVEN_GROUPS)))
    if code == "ELECTR":
        valid = list(sorted(set((valid or [])) | set(TV_GROUPS)))
    return sorted(set(valid)), code

# -------------------- helpers: LIKE/ILIKE builders --------------------
def like_any(col: str, pats: list[str]) -> str:
    parts = [f"{col} ILIKE '%{p}%'" for p in pats]
    return "(" + " OR ".join(parts) + ")" if parts else "1=1"
def contains_any_in(cols: list[str], keywords: list[str]) -> str:
    conds = []
    for c in cols:
        conds.append(like_any(f'UPPER({c})', [k.upper() for k in keywords]))
    return "(" + " OR ".join(conds) + ")" if conds else "1=0"
def not_contains_any_in(cols: list[str], keywords: list[str]) -> str:
    pos = contains_any_in(cols, keywords)
    return f"NOT {pos}"

# -------------------- Color WHERE --------------------
def color_where(color: str, matte_only: bool):
    if not color: return None
    col = 'UPPER("Second_Description")'
    if color == "BLACK" and matte_only:
        return f"{col} ILIKE '%MATTE%BLACK%'"
    if color == "BLACK":
        return f"({col} ILIKE '%BLACK%' OR {col} ILIKE '% BLK %' OR {col} ILIKE '%BLK' OR {col} ILIKE 'BLK %' OR {col} ILIKE '%MBLK%' OR {col} ILIKE '%MBK%')"
    if color == "WHITE":
        return f"({col} ILIKE '%WHITE%' OR {col} ILIKE '%WHT%') AND {col} NOT ILIKE '%PANEL%'"
    if color == "STAINLESS":
        return f"({col} ILIKE '%STAINLESS%' OR {col} ILIKE '%BLACK STAINLESS%')"
    if color == "PANEL":
        return f"({col} ILIKE '%PANEL%' OR {col} ILIKE '%CUSTOM PANEL%')"
    if color == "SLATE":
        return f"{col} ILIKE '%SLATE%'"
    if color == "BISQUE":
        return f"({col} ILIKE '%BISQUE%' OR {col} ILIKE '%ALMOND%' OR {col} ILIKE '%BEIGE%')"
    return None

# -------------------- Mattress WHERE pieces --------------------
def mattress_where(size: str | None, mtype: str | None):
    conds = []
    def like_any_local(col, pats):
        return "(" + " OR ".join([f"{col} ILIKE '%{p}%'" for p in pats]) + ")"
    if size:
        size_map = {
            "TWIN": ["TWIN"],
            "TWIN XL": ["TWIN XL", "TWINXL", "TXL", "XLTWIN"],
            "FULL": ["FULL", "DOUBLE"],
            "QUEEN": ["QUEEN", "QN"],
            "KING": ["KING", "KG"],
            "CAL KING": ["CAL KING", "CALIFORNIA KING", "CA KING", "CK", "CAL-KING"],
        }
        pats = size_map.get(size, [size])
        conds.append("(" + " OR ".join([
            like_any_local('UPPER("Group")', [p.upper() for p in pats]),
            like_any_local('UPPER("Description")', [p.upper() for p in pats]),
            like_any_local('UPPER("Second_Description")', [p.upper() for p in pats]),
        ]) + ")")
    if mtype:
        type_map = {
            "COIL": ["COIL", "INNERSPRING", "SPRING"],
            "FOAM": ["FOAM", "MEMORY", "LATEX"],
            "HYBRID": ["HYBRID"],
        }
        pats = type_map.get(mtype, [mtype])
        conds.append("(" + " OR ".join([
            like_any_local('UPPER("Group")', [p.upper() for p in pats]),
            like_any_local('UPPER("Description")', [p.upper() for p in pats]),
            like_any_local('UPPER("Second_Description")', [p.upper() for p in pats]),
        ]) + ")")
    return conds

# -------------------- FREEZER style detection --------------------
def detect_freezer_style(q: str):
    ql = q.lower()
    freezer_intent = any(w in ql for w in ["freezer", "freezers"])
    style = None
    if freezer_intent and "upright" in ql:
        style = "upright"
    elif freezer_intent and "chest" in ql:
        style = "chest"
    return style, freezer_intent

# -------------------- NEW: GARBAGE DISPOSAL detection --------------------
def detect_disposal_intent(q: str) -> bool:
    ql = q.lower()
    return bool(re.search(r'\b(garbage\s+disposal|disposals?|disposers?)\b', ql))

# -------------------- WHERE builder --------------------
def build_where(code_type: str, code: str, color: str = None, matte_only: bool = False,
                on_display: bool = False, pmin=None, pmax=None, brand_list: list[str] | None = None,
                wall_only: bool = False, wall_size: str | None = None,
                matt_size: str | None = None, matt_type: str | None = None,
                matt_intent: bool = False, wants_accessory_only: bool = False, raw_query: str = "",
                counter_depth_only: bool = False,
                tv_only: bool = False, tv_size: int | None = None,
                washer_mode: str | None = None,
                refer_groups_override: list[str] | None = None,
                groups_override: list[str] | None = None,
                ada_dish_only: bool = False,
                require_cfm_in_desc: bool = False,
                exclude_groups: list[str] | None = None,
                model_exact_list: list[str] | None = None,
                model_partial_tokens: list[str] | None = None,
                range_only: bool = False,
                range_width: int | None = None,
                ctop_only: bool = False,
                ctop_width: int | None = None,
                rangetop_only: bool = False,
                rtop_width: int | None = None,
                brands_exclude: list[str] | None = None,
                range_control: str | None = None,
                range_fuel: str | None = None,
                laundry_both_categories: bool = False,
                brand_only_mode: bool = False,
                built_in_width: int | None = None,
                built_in_width_active: bool = False,
                matt_name_tokens: list[str] | None = None,
                microwave_desc_patterns: list[str] | None = None,
                # NEW: VENT description-only routing
                vent_mode: bool = False,
                vent_keywords: list[str] | None = None,
                vent_width: int | None = None):
    parts = ['COALESCE("Qty",0) > 0']

    if brand_only_mode and brand_list:
        vals = "', '".join(sorted(set(b.upper().strip() for b in brand_list)))
        parts.append(f'UPPER(TRIM("Brand")) IN (\'{vals}\')')
        if "Price" in df.columns:
            if pmin is not None: parts.append(f'COALESCE("Price", 0) >= {float(pmin):.2f}')
            if pmax is not None: parts.append(f'COALESCE("Price", 0) <= {float(pmax):.2f}')
        return " AND ".join(parts)

    if (model_exact_list and len(model_exact_list)) or (model_partial_tokens and len(model_partial_tokens)):
        if model_exact_list:
            in_list = "', '".join(x.upper().strip().replace("'", "''") for x in model_exact_list)
            parts.append(f'UPPER(TRIM("Model")) IN (\'{in_list}\')')
        if model_partial_tokens:
            for tok in model_partial_tokens:
                t = tok.upper().replace("'", "''")
                parts.append(f'UPPER("Model") LIKE \'%{t}%\'')
        if on_display and "DSP" in df.columns:
            parts.append('"DSP" = TRUE')
        if brand_list:
            vals = "', '".join(sorted(set(b.upper().strip() for b in brand_list)))
            parts.append(f'UPPER(TRIM("Brand")) IN (\'{vals}\')')
        if brands_exclude:
            ex = "', '".join(sorted(set(b.upper().strip() for b in brands_exclude)))
            parts.append(f'UPPER(TRIM("Brand")) NOT IN (\'{ex}\')')
        if "Price" in df.columns:
            if pmin is not None: parts.append(f'COALESCE("Price", 0) >= {float(pmin):.2f}')
            if pmax is not None: parts.append(f'COALESCE("Price", 0) <= {float(pmax):.2f}')
        return " AND ".join(parts)

    if rangetop_only:
        parts.append('UPPER(TRIM("Group")) = \'RNGTOP\'')
        if rtop_width is not None:
            parts.append(cooktop_size_text_where(rtop_width))

    if ctop_only:
        parts.append('UPPER(TRIM("Category")) = \'CTOP\'')
        parts.append('UPPER(TRIM("Group")) <> \'RNGTOP\'')
        parts.append('( "Group" IS NULL OR UPPER("Group") NOT LIKE \'%ACC\' )')
        if ctop_width is not None:
            parts.append(cooktop_size_text_where(ctop_width))

    if laundry_both_categories and not groups_override:
        parts.append("( UPPER(TRIM(\"Category\")) IN ('WASH','DRY') )")
        parts.append("( \"Group\" IS NULL OR UPPER(TRIM(\"Group\")) NOT IN ('WSHACC','PEDS','STACK','DRYACC') )")

    if code_type == "Category" and not (rangetop_only or ctop_only) and not laundry_both_categories:
        # --------- NEW: VENT description-only routing (ignore groups) ---------
        if vent_mode and code.upper() == "VENT":
            parts.append('UPPER(TRIM("Category")) = \'VENT\'')
            parts.append('( "Group" IS NULL OR UPPER("Group") NOT LIKE \'%ACC\' )')
            if vent_keywords:
                parts.append(contains_any_in(['"Description"','"Second_Description"'], vent_keywords))
            if vent_width is not None:
                parts.append(width_text_where_desc(vent_width))
        else:
            try:
                groups, resolved = group_list_for_category(code)
            except Exception:
                groups, resolved = [], code
                if groups_override:
                    in_list = "', '".join(groups_override)
                    parts.append(f'UPPER(TRIM("Group")) IN (\'{in_list}\')')
            else:
                if groups_override:
                    in_list = "', '".join(groups_override)
                    parts.append(f'UPPER(TRIM("Group")) IN (\'{in_list}\')')
                elif resolved == "RANGE" or range_only:
                    grps = None
                    if range_control or range_fuel:
                        cand = set(EXISTING_GRP)
                        control_set = None
                        fuel_set = None
                        if range_control == "rear":
                            control_set = RANGE_GROUPS_REAR
                        elif range_control == "front":
                            control_set = RANGE_GROUPS_FRONT
                        if range_fuel in RANGE_GROUPS_FUEL:
                            fuel_set = RANGE_GROUPS_FUEL[range_fuel]
                        if control_set and fuel_set:
                            grps = sorted((control_set & fuel_set) & cand)
                        elif control_set:
                            grps = sorted(control_set & cand)
                        elif fuel_set:
                            grps = sorted(fuel_set & cand)
                    if grps:
                        in_list = "', '".join(grps)
                        base = f'UPPER(TRIM("Group")) IN (\'{in_list}\')'
                        if range_width is not None:
                            parts.append(f'({base} AND {range_size_text_where(range_width)})')
                        else:
                            parts.append(base)
                    else:
                        cats = ["RANGE", "PRORNG"]
                        cat_in = "', '".join(sorted(set(cats)))
                        cat_clause = f'UPPER(TRIM("Category")) IN (\'{cat_in}\')'
                        base_clause = f'({cat_clause} AND ( "Group" IS NULL OR UPPER("Group") NOT LIKE \'%ACC\' ))'
                        if range_width is not None:
                            size_clause = range_size_text_where(range_width)
                            parts.append(f'({base_clause} AND {size_clause})')
                        else:
                            parts.append(base_clause)
                else:
                    cat_clause = None

                    if refer_groups_override and resolved in {"REFER", "REFR"}:
                        groups = refer_groups_override
                        in_list = "', '".join(groups)
                        if counter_depth_only:
                            fpkl_brand_clause = (
                                "(UPPER(TRIM(\"Category\")) IN ('REFER','REFR') AND "
                                "UPPER(TRIM(\"Brand\")) IN ('FPKL','FISHERPAYKEL'))"
                            )
                            parts.append(
                                f"(UPPER(TRIM(\"Group\")) IN ('{in_list}') OR {fpkl_brand_clause})"
                            )
                        else:
                            parts.append(f"UPPER(TRIM(\"Group\")) IN ('{in_list}')")
                    else:
                        strict_wall = (resolved == "BICOOK" and wall_size is not None)
                        if resolved == "BICOOK":
                            groups = filter_wall_groups_by_size(groups, wall_size)

                        if counter_depth_only and resolved in {"REFER", "REFR"}:
                            existing_grps = set(
                                str(x).upper().strip()
                                for x in df.get("Group", pd.Series([], dtype="object")).dropna().unique()
                            )
                            groups = sorted(set(COUNTER_DEPTH_GROUPS) & existing_grps) or sorted(COUNTER_DEPTH_GROUPS)
                            in_list = "', '".join(groups)
                            bott_cd_clause = (
                                "(UPPER(TRIM(\"Group\")) = 'BOTT' AND ("
                                "  UPPER(COALESCE(\"Description\", '')) ILIKE '%COUNTER%DEPTH%' OR "
                                "  UPPER(COALESCE(\"Description\", '')) ILIKE '%CNTR%DEPTH%' OR "
                                "  UPPER(COALESCE(\"Description\", '')) ILIKE '%CNTR%DPTH%'"
                                "))"
                            )
                            fpkl_brand_clause = (
                                "(UPPER(TRIM(\"Brand\")) IN ('FPKL','FISHERPAYKEL') AND "
                                " UPPER(TRIM(\"Category\")) IN ('REFER','REFR'))"
                            )
                            cat_clause = (
                                f"(UPPER(TRIM(\"Group\")) IN ('{in_list}') OR "
                                f"{bott_cd_clause} OR {fpkl_brand_clause})"
                            )

                        if tv_only and resolved == "ELECTR" and tv_size is not None:
                            near_groups = tv_groups_within_5(tv_size)
                            existing_grps = set(str(x).upper().strip() for x in df.get("Group", pd.Series([], dtype="object")).dropna().unique())
                            near_existing = sorted([g for g in near_groups if g in existing_grps])
                            size_text_clause = tv_size_text_where(tv_size)
                            subclauses = []
                            if near_existing:
                                in_list = "', '".join(near_existing)
                                subclauses.append(f'UPPER(TRIM("Group")) IN (\'{in_list}\')')
                            if size_text_clause:
                                subclauses.append(size_text_clause)
                            if subclauses:
                                cat_clause = f'(UPPER(TRIM("Category")) = \'{resolved}\' AND (' + " OR ".join(subclauses) + "))"
                            else:
                                cat_clause = f'(UPPER(TRIM("Category")) = \'{resolved}\' AND ( "Group" IS NULL OR UPPER("Group") NOT LIKE \'%ACC\' ))'
                            groups = near_existing or near_groups

                        if resolved == "WASH" and washer_mode is not None:
                            existing_grps = set(str(x).upper().strip() for x in df.get("Group", pd.Series([], dtype="object")).dropna().unique())
                            if washer_mode == "front":
                                wanted = FRONT_LOAD_WASH_GROUPS
                            elif washer_mode == "top":
                                wanted = TOP_LOAD_WASH_GROUPS
                            elif washer_mode == "agitator":
                                wanted = AGITATOR_WASH_GROUPS
                            elif washer_mode == "impeller":
                                wanted = IMPELLER_WASH_GROUPS
                            else:
                                wanted = set()
                            groups = sorted(set(wanted) & existing_grps) or sorted(wanted)

                        if cat_clause is not None:
                            parts.append(cat_clause)
                        else:
                            if groups:
                                in_list = "', '".join(groups)
                                if strict_wall or (counter_depth_only and resolved in {"REFER","REFR"}) or (resolved == "WASH" and washer_mode is not None):
                                    parts.append(f'UPPER(TRIM("Group")) IN (\'{in_list}\')')
                                else:
                                    parts.append(
                                        f'(UPPER(TRIM("Group")) IN (\'{in_list}\') OR '
                                        f'(UPPER(TRIM("Category")) = \'{resolved}\' AND ( "Group" IS NULL OR UPPER("Group") NOT LIKE \'%ACC\' )))'
                                    )
                            else:
                                if strict_wall or (resolved == "WASH" and washer_mode is not None):
                                    parts.append("1=0")
                                else:
                                    parts.append(f'UPPER(TRIM("Category")) = \'{resolve_category_alias(df, code)}\' AND ( "Group" IS NULL OR UPPER("Group") NOT LIKE \'%ACC\' )')

    elif code_type == "Group" and not (rangetop_only or ctop_only):
        parts.append(f'UPPER(TRIM("Group")) = \'{code}\'')

    elif code_type == "FallbackText" and not (rangetop_only or ctop_only):
        like_parts = []
        qtok = [t for t in tokens(code) if len(t) > 1]
        search_cols = ['Category','Group','Description','Second_Description']
        for col in search_cols:
            if col in df.columns:
                sub = " OR ".join([f'LOWER("{col}") LIKE \'%{t}%\'' for t in qtok])
                if sub:
                    like_parts.append(f'({sub})')
        if like_parts:
            parts.append("(" + " OR ".join(like_parts) + ")")
        parts.append('( "Group" IS NULL OR UPPER("Group") NOT LIKE \'%ACC\' )')

    if wall_only:
        parts.append("(TRUE)")

    # Mattress filters
    if matt_intent:
        for c in mattress_where(matt_size, matt_type):
            parts.append(c)
        if matt_name_tokens:
            for t in matt_name_tokens:
                t_up = t.upper().replace("'", "''")
                parts.append(f'UPPER("Description") ILIKE \'%{t_up}%\'')

    if wants_accessory_only:
        acc_in = "', '".join(sorted(ACCESSORY_GROUPS)) if ACCESSORY_GROUPS else ""
        group_in_clause = f'UPPER(TRIM("Group")) IN (\'{acc_in}\')' if acc_in else "1=0"
        parts.append("(" + " OR ".join([
            group_in_clause,
            contains_any_in(['"Group"','"Description"','"Second_Description"'], ACCESSORY_TEXT_KEYWORDS)
        ]) + ")")
        if _MATTRESS_GROUP_CODES:
            m_in = "', '".join(sorted(_MATTRESS_GROUP_CODES))
            parts.append(f'UPPER(TRIM("Group")) NOT IN (\'{m_in}\')')
    elif matt_intent:
        if ACCESSORY_GROUPS:
            acc_in = "', '".join(sorted(ACCESSORY_GROUPS))
            parts.append(f'UPPER(TRIM("Group")) NOT IN (\'{acc_in}\')')
        parts.append(not_contains_any_in(['"Group"','"Description"','"Second_Description"'], ACCESSORY_TEXT_KEYWORDS))

    if color:
        cc = color_where(color, matte_only)
        parts.append(cc if cc else "1=0")

    if on_display and "DSP" in df.columns:
        parts.append('"DSP" = TRUE')

    if brand_list:
        vals = "', '".join(sorted(set(b.upper().strip() for b in brand_list)))
        parts.append(f'UPPER(TRIM("Brand")) IN (\'{vals}\')')

    if brands_exclude:
        ex = "', '".join(sorted(set(b.upper().strip() for b in brands_exclude)))
        parts.append(f'UPPER(TRIM("Brand")) NOT IN (\'{ex}\')')

    if "Price" in df.columns:
        if pmin is not None:
            parts.append(f'COALESCE("Price", 0) >= {float(pmin):.2f}')
        if pmax is not None:
            parts.append(f'COALESCE("Price", 0) <= {float(pmax):.2f}')

    if ada_dish_only:
        parts.append('UPPER(COALESCE("Description", \'\')) ILIKE \'%ADA%\'')

    if require_cfm_in_desc:
        parts.append('UPPER(COALESCE("Description", \'\')) ILIKE \'%CFM%\'')

    if exclude_groups:
        ex_list = "', '".join(sorted(set(g.upper().strip() for g in exclude_groups)))
        parts.append(f'UPPER(TRIM("Group")) NOT IN (\'{ex_list}\')')

    if built_in_width_active and (built_in_width is not None):
        parts.append(width_text_where_desc(built_in_width))

    # NEW: add extra Description filters for OTR refinements
    if microwave_desc_patterns:
        desc_col = 'UPPER(COALESCE("Description", ""))'
        ors = " OR ".join([f"{desc_col} ILIKE '%{p.upper()}%'" for p in microwave_desc_patterns])
        parts.append("(" + ors + ")")

    return " AND ".join(parts)

# -------------------- Answer --------------------
def fmt_price(p):
    try:
        if pd.isna(p): return None
        return f"${int(round(float(p))):,}"
    except Exception:
        return None

def fmt_cuft(val: float | None) -> str | None:
    if val is None:
        return None
    try:
        if abs(val - round(val)) < 1e-6:
            return f"{int(round(val))} cu ft"
        return f"{val:.1f} cu ft"
    except:
        return None

def compute_answer(q: str, code_type: str, code: str, color: str, brand_candidates: list[str]):
    matte_only = wants_matte(q) and (color == "BLACK")
    on_display = detect_display(q)
    pmin, pmax = parse_price_filter(q)
    model_exact_list, model_partial_tokens = detect_model_search(q)
    model_search_active = bool(model_exact_list or model_partial_tokens)

    # Mattress intent + tokens
    matt_size, matt_type, matt_intent = detect_mattress_filters(q)
    matt_name_tokens = extract_mattress_name_tokens(q) if matt_intent else []
    accessory_intent = detect_accessory_intent(q)

    if accessory_intent and not matt_intent:
        code_type, code = "Category", "MATTS"
    if matt_intent:
        code_type, code = "Category", "MATTS"

    wall_only = detect_wall_oven(q)
    wall_size = detect_wall_oven_size(q) if wall_only else None
    if wall_only:
        code_type, code = "Category", "BICOOK"

    # NEW: Microwave routing + OTR refinements (run before VENT)
    groups_override = None
    microwave_desc_patterns = None
    if detect_microwave_otr(q):
        code_type, code = "Category", "OVENS"
        groups_override = ["MICOTR"]
        low_profile = detect_mw_low_profile_intent(q)
        flush_mount = detect_mw_flush_mount_intent(q)
        if low_profile and flush_mount:
            microwave_desc_patterns = ["LOW PROFILE", "LOW-PROFILE", "LOWPROFILE", "SLIM", "FLUSH", "FLUSH MOUNT", "FLUSH-MOUNT", "FLUSHMOUNT"]
        elif low_profile:
            microwave_desc_patterns = ["LOW PROFILE", "LOW-PROFILE", "LOWPROFILE", "SLIM"]
        elif flush_mount:
            microwave_desc_patterns = ["FLUSH", "FLUSH MOUNT", "FLUSH-MOUNT", "FLUSHMOUNT"]
    elif detect_microwave_countertop(q):
        code_type, code = "Category", "OVENS"
        groups_override = ["CNTOP"]
    elif detect_microwave_drawer(q):
        code_type, code = "Category", "OVENS"
        groups_override = ["MWDRWR"]

    # -------- NEW: VENT intent (description-only, ignore groups) --------
    vent_mode = False
    vent_kw = None
    vent_width = None
    if detect_vent_intent(q):
        code_type, code = "Category", "VENT"
        vent_mode = True
        vent_kw = vent_sub_keywords(q)  # may be empty/None
        vent_width = detect_width_generic(q)

    tv_only = detect_tv(q)
    tv_size = detect_tv_size(q) if tv_only else None
    if tv_only:
        code_type, code = "Category", "ELECTR"

    range_only = detect_range(q)
    range_width = detect_range_width(q) if range_only else None
    range_control = detect_range_control(q) if range_only else None
    range_fuel = detect_range_fuel(q) if range_only else None
    if range_only:
        code_type, code = "Category", "RANGE"

    cooktop_intent = detect_cooktop(q)
    rangetop_intent = detect_rangetop(q)
    ctop_only = False
    rangetop_only = False
    ctop_width = None
    rtop_width = None
    if rangetop_intent:
        rangetop_only = True
        rtop_width = ctop_rtop_width_from(q)
    elif cooktop_intent:
        ctop_only = True
        ctop_width = ctop_rtop_width_from(q)
        code_type, code = "Category", "CTOP"

    dishwasher_intent = detect_dishwasher_intent(q)
    washer_mode, washer_intent = detect_washer_style(q)

    laundry_center = detect_laundry_center(q)
    compact_laundry = detect_compact_laundry(q)
    generic_laundry = detect_generic_laundry(q)

    refer_groups_override = None
    exclude_groups = None
    laundry_both_categories = False

    if laundry_center:
        code_type, code = "Category", "WASH"
        groups_override = ["LCNTR"]

    elif compact_laundry:
        code_type, code = "Category", "WASH"
        groups_override = ["CMPCTW", "COMPGS", "COMPEL"]

    elif generic_laundry and not (range_only or ctop_only or rangetop_only or wall_only or tv_only):
        code_type, code = "Category", "WASH"
        laundry_both_categories = True

    if washer_intent and not dishwasher_intent and not (range_only or ctop_only or rangetop_only or wall_only or tv_only or laundry_center or compact_laundry or laundry_both_categories):
        code_type, code = "Category", "WASH"

    freezer_style, freezer_intent = detect_freezer_style(q)
    if freezer_intent:
        code_type, code = "Category", "FREEZ"
        if freezer_style == "upright":
            groups_override = ["MANUPR", "FFUPR"]
        elif freezer_style == "chest":
            groups_override = ["CHEST"]

    disposal_intent = detect_disposal_intent(q)
    if disposal_intent:
        code_type, code = "Category", "DISHW"
        groups_override = ["DISPSR"]
        exclude_groups = None

    ada_dish_only = detect_ada_dishwasher(q)
    if ada_dish_only:
        code_type, code = "Category", "DISHW"
        exclude_groups = ["DISPSR"]

    counter_depth = detect_counter_depth(q)
    full_or_standard_depth = detect_full_or_standard_depth(q)

    # ---------- BUILT-IN refrigeration routing ----------
    bi_hit, bi_cat, bi_groups = detect_built_in_fridge(q)
    built_in_width = detect_width_generic(q)
    built_in_width_active = False

    if bi_hit:
        code_type, code = "Category", (bi_cat or "BIREFR")
        if bi_groups:
            groups_override = bi_groups
        refer_groups_override = None
        built_in_width_active = True

    refer_style, refer_intent = detect_refer_style(q)
    if (refer_intent or (not code_type and not code)) and not (range_only or ctop_only or rangetop_only or bi_hit or vent_mode):
        code_type, code = "Category", "REFER"

    if not bi_hit:
        if refer_style == "topmount":
            refer_groups_override = ["TOPMNT"]
        elif refer_style == "sxs":
            if counter_depth:
                refer_groups_override = ["CDSXS"]
            elif full_or_standard_depth:
                refer_groups_override = ["SXS"]
            else:
                refer_groups_override = ["SXS", "CDSXS"]
        elif refer_style == "french":
            if counter_depth:
                refer_groups_override = ["CDFDIW", "CDFDBM", "CD4DR"]
            elif full_or_standard_depth:
                refer_groups_override = ["FDBM", "FDIW", "FOUR"]
            else:
                refer_groups_override = ["FDBM", "FDIW", "CDFDIW", "CDFDBM", "CD4DR", "FOUR"]
        elif refer_style == "four":
            if counter_depth:
                refer_groups_override = ["CD4DR"]
            elif full_or_standard_depth:
                refer_groups_override = ["FOUR"]
            else:
                refer_groups_override = ["FOUR", "CD4DR"]
        if (refer_style is None) and full_or_standard_depth and not counter_depth:
            refer_groups_override = ["SXS", "FDBM", "FDIW", "FOUR", "BOTT"]

    if dishwasher_intent and not disposal_intent:
        if not code_type or code_type == "FallbackText":
            code_type, code = "Category", "DISHW"
        exclude_groups = (exclude_groups or []) + ["DISPSR"]

    brands_exclude = []
    if refer_style == "four":
        brands_exclude.extend(["MIEL", "MIELE"])

    dd_intent, dd_cat, dd_groups = detect_downdraft(q)
    if dd_intent and dd_cat:
        # If the user meant a HOOD downdraft, we keep VENT description-only (no group filter).
        if dd_cat == "VENT":
            code_type, code = "Category", "VENT"
            vent_mode = True
            # Ensure downdraft terms included even if user only said "downdraft"
            add_dd = ["DOWNDRAFT", "DOWN DRAFT", "DDRAFT", " POP UP", "POP-UP"]
            vent_kw = (vent_kw or []) + add_dd
        else:
            code_type, code = "Category", dd_cat
            groups_override = dd_groups
            exclude_groups = None
            range_control = None
            range_fuel = None

    cfm_intent = detect_cfm_intent(q)
    cfm_min, cfm_max, cfm_approx = parse_cfm_filter(q)
    require_cfm_in_desc = False
    if cfm_intent:
        # For VENT searches we stay description-only (no groups). For general CFM without VENT words,
        # we still bias to ventilation-y groups, but description must contain CFM.
        require_cfm_in_desc = True
        if not vent_mode and not (dd_intent and dd_cat == "VENT"):
            existing = set(EXISTING_GRP)
            grps = sorted([g for g in CFM_GROUPS if g in existing]) or sorted(CFM_GROUPS)
            groups_override = grps
            if not refer_intent and not washer_intent and not dishwasher_intent and not tv_only and not wall_only and not matt_intent and not range_only and not ctop_only and not rangetop_only:
                code_type, code = "Category", "VENT"

    brand_only_mode = False
    has_any_other_intent = any([
        matt_intent, accessory_intent, wall_only, tv_only, range_only, cooktop_intent, rangetop_intent,
        dishwasher_intent, washer_intent, laundry_center, compact_laundry, generic_laundry,
        freezer_intent, disposal_intent, ada_dish_only, dd_intent, cfm_intent, refer_intent, bi_hit,
        detect_microwave_otr(q), detect_microwave_countertop(q), detect_microwave_drawer(q), vent_mode
    ])

    brand_only_mode = is_brand_only_query(
        q,
        brand_candidates,
        color=color,
        price_min=pmin,
        price_max=pmax,
        model_exact_list=model_exact_list,
        model_partial_tokens=model_partial_tokens,
        has_any_other_intent=has_any_other_intent
    )

    if not code_type or not code:
        code_type = "FallbackText"
        code = q

    if model_search_active:
        code_type, code = "FallbackText", ""
        refer_groups_override = None
        groups_override = None
        exclude_groups = None
        color = None

    where = build_where(
        code_type, code, color, matte_only, on_display, pmin, pmax, brand_candidates,
        wall_only=wall_only, wall_size=wall_size,
        matt_size=matt_size, matt_type=matt_type,
        matt_intent=matt_intent, wants_accessory_only=accessory_intent, raw_query=q,
        counter_depth_only=counter_depth,
        tv_only=tv_only, tv_size=tv_size,
        washer_mode=washer_mode,
        refer_groups_override=refer_groups_override,
        groups_override=groups_override,
        ada_dish_only=ada_dish_only,
        require_cfm_in_desc=require_cfm_in_desc,
        exclude_groups=exclude_groups,
        model_exact_list=(model_exact_list if model_search_active else None),
        model_partial_tokens=(model_partial_tokens if model_search_active else None),
        range_only=range_only,
        range_width=range_width,
        ctop_only=ctop_only,
        ctop_width=ctop_width,
        rangetop_only=rangetop_only,
        rtop_width=rtop_width,
        brands_exclude=brands_exclude,
        range_control=range_control,
        range_fuel=range_fuel,
        laundry_both_categories=laundry_both_categories,
        brand_only_mode=brand_only_mode,
        built_in_width=built_in_width,
        built_in_width_active=built_in_width_active,
        matt_name_tokens=matt_name_tokens,
        microwave_desc_patterns=microwave_desc_patterns,
        vent_mode=vent_mode,
        vent_keywords=vent_kw,
        vent_width=vent_width
    )

    base_cols = ['Brand','Model','Qty','Second_Description','Category','Group']
    select_parts = []
    for c in base_cols:
        if c in df.columns:
            select_parts.append('"Group"' if c == "Group" else c)
        else:
            fallback = {
                'Brand':'"" AS Brand','Model':'"" AS Model','Qty':'0 AS Qty',
                'Second_Description':'"" AS Second_Description',
                'Category':'"" AS Category','Group':'NULL AS "Group"'
            }[c]
            select_parts.append(fallback)

    if "DSP" in df.columns:
        select_parts.insert(3, "DSP")

    if "Price" in df.columns:
        select_parts.insert(4, "Price")
    else:
        select_parts.insert(4, "NULL AS Price")

    if "Description" not in df.columns:
        select_parts.append("'' AS Description")
    else:
        select_parts.append("Description")

    select_clause = ", ".join(select_parts)

    res = con.execute(f'''
        SELECT {select_clause}
        FROM inventory
        WHERE {where}
        ORDER BY Qty DESC, Brand, Model
        LIMIT 500
    ''').df()

    cfm_intent = detect_cfm_intent(q)
    if cfm_intent and not res.empty:
        res = res.copy()
        res["CFM_VAL"] = res["Description"].apply(extract_cfm_from_text)
        res = res[res["CFM_VAL"].notna()]
        cfm_min, cfm_max, _ = parse_cfm_filter(q)
        if (cfm_min is not None) or (cfm_max is not None):
            if cfm_min is not None:
                res = res[res["CFM_VAL"] >= cfm_min]
            if cfm_max is not None:
                res = res[res["CFM_VAL"] <= cfm_max]

    counter_depth = detect_counter_depth(q)
    full_or_standard_depth = detect_full_or_standard_depth(q)
    if (not counter_depth) and full_or_standard_depth and not res.empty:
        res = res.copy()
        mask_cd_bott = (
            (res["Group"].astype(str).str.upper() == "BOTT") &
            res["Description"].astype(str).str.contains(
                r"COUNTER[\s\-]*DEPTH|CNTR[\s\-]*DEPTH|CNTR[\s\-]*DPTH",
                case=False, regex=True
            )
        )
        mask_fpkl_fridge = (
            res["Category"].astype(str).str.upper().isin(["REFER","REFR"]) &
            res["Brand"].astype(str).str.upper().isin(["FPKL","FISHERPAYKEL"])
        )
        res = res[~(mask_cd_bott | mask_fpkl_fridge)]

    cuft_query = extract_cuft_max_from_text(q)
    if (cuft_query is not None) and not res.empty:
        res = res.copy()
        res["CUFT_VAL"] = res["Description"].apply(extract_cuft_max_from_text)
        res = res[res["CUFT_VAL"].notna()]
        lo, hi = cuft_query - 3.0, cuft_query + 3.0
        res = res[(res["CUFT_VAL"] >= lo) & (res["CUFT_VAL"] <= hi)]

    if res.empty:
        quals = []
        if brand_candidates: quals.append("/".join(sorted(set(brand_candidates))))
        if color: quals.append("Matte Black" if matte_only else color.title())
        if on_display: quals.append("On Display")
        if counter_depth: quals.append("Counter-Depth")
        if full_or_standard_depth: quals.append("Full/Standard Depth")
        if tv_only and tv_size: quals.append(f'{tv_size}" TV ±5"')
        if range_only and range_width: quals.append(f'{range_width}" Range')
        if ctop_only and ctop_width: quals.append(f'{ctop_width}" Cooktop')
        if rangetop_only and rtop_width: quals.append(f'{rtop_width}" Rangetop')
        if washer_intent and washer_mode:
            mapping = {"front":"Front-Load","top":"Top-Load","agitator":"Agitator","impeller":"Impeller"}
            quals.append(f'Washer: {mapping.get(washer_mode, washer_mode)}')
        if ada_dish_only: quals.append("ADA Dishwasher")
        if disposal_intent: quals.append("Garbage Disposals (DISPSR)")
        if dd_intent: quals.append("Downdraft")
        if detect_microwave_otr(q): quals.append("OTR Microwave (MICOTR)")
        if detect_microwave_countertop(q): quals.append("Countertop Microwave (CNTOP)")
        if detect_microwave_drawer(q): quals.append("Microwave Drawer (MWDRWR)")
        if microwave_desc_patterns: quals.append("Description filter applied")
        if range_only:
            if range_control: quals.append(f'{range_control.title()} Control')
            if range_fuel: quals.append(f'{range_fuel.title()}')
        if laundry_both_categories: quals.append("Laundry (WASH + DRY, no accessories)")
        if compact_laundry: quals.append("Compact Laundry")
        if laundry_center: quals.append("Laundry Center/Tower")
        if vent_mode and vent_width: quals.append(f'{vent_width}" Hood')
        if pmin is not None or pmax is not None:
            if pmin is not None and pmax is not None: quals.append(f"{fmt_price(pmin)}–{fmt_price(pmax)}")
            elif pmin is not None: quals.append(f"≥ {fmt_price(pmin)}")
            elif pmax is not None: quals.append(f"≤ {fmt_price(pmax)}")
        suffix = (" (" + ", ".join([q for q in quals if q]) + ")") if quals else ""
        return f"No in-stock units found{suffix}."

    # -------- OUTPUT RENDERING --------
    show_mattress_description_only = (res["Category"].astype(str).str.upper() == "MATTS").any()

    lines = []
    for _, r in res.iterrows():
        cat_up = str(r.get("Category","")).upper()
        is_matt = (cat_up == "MATTS")
        if is_matt:
            primary_text = str(r.get("Description","")).title()
        else:
            primary_text = str(r.get("Second_Description","")).title()

        price_str = fmt_price(r.get("Price"))
        price_seg = f" — MSRP: {price_str}" if price_str else " — MSRP: N/A"
        dsp_tag = " — On Display" if ("DSP" in res.columns and bool(r.get("DSP"))) else ""
        qty_val = int(pd.to_numeric(r.get("Qty", 0), errors="coerce") or 0)

        cuft_seg = ""
        if not is_matt:
            cuft_val = extract_cuft_max_from_text(r.get("Description",""))
            if cuft_val is not None:
                cuft_seg = f" ({fmt_cuft(cuft_val)})"

        line = f"{r.get('Brand','')} {r.get('Model','')}{cuft_seg} — {primary_text}{price_seg} — Qty {qty_val}{dsp_tag}"

        cfm_val = extract_cfm_from_text(r.get("Description",""))
        if cfm_val and not is_matt:
            line += f" — {cfm_val} CFM"

        lines.append(line)
    return "\n".join(lines)

# -------------------- Q&A State --------------------
if "qa_history" not in st.session_state: st.session_state.qa_history = []
if "last_q" not in st.session_state: st.session_state.last_q = None

# -------------------- UI --------------------
st.subheader("Search Results")
q = st.chat_input('Ask about stock, in your own words.')

if q and len(df) and q != st.session_state.last_q:
    st.session_state.last_q = q
    brand_candidates = detect_brand_candidates(q)
    code_type, code = detect_category_or_group(q)
    color = detect_color(q)
    ans = compute_answer(q, code_type, code, color, brand_candidates)

    st.markdown(f"**Question:** {q}")
    st.text(ans)

    st.session_state.qa_history.append({"q": q, "a": ans})

# -------------------- History --------------------
with st.expander("Q&A History", expanded=False):
    if st.session_state.qa_history:
        for i, item in enumerate(reversed(st.session_state.qa_history), start=1):
            st.markdown(f"**Q{i}:** {item['q']}")
            st.text(f"A{i}: {item['a']}")
            st.markdown("---")
    else:
        st.caption("No history yet — ask a question above.")

# -------------------- Admin diagnostics (hidden unless unlocked) --------------------
if admin:
    with st.expander("Price Diagnostics (Admin)"):
        st.write("Detected Price column summary:")
        if "Price_Source" in df.columns:
            st.write(f"**Price_Source:** {df['Price_Source'].dropna().unique()[:3]}")
        st.write("**Sample Prices (first 5 non-null):**")
        st.write(df["Price"].dropna().head(5).to_list())
        st.caption("If needed, set env vars: PRICE_COL_NAME='MSRP' or PRICE_COL_INDEX='7', then Admin → Force reload.")
import io
import re
import pandas as pd

# WhatsApp Android: 12/03/2024, 10:15 am - Name: message
# WhatsApp iOS:     [12/03/24, 10:15:32 AM] Name: message
_TS = (
    r"\d{1,2}/\d{1,2}/\d{2,4},\s*"
    r"\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:\s*[APap][Mm])?"
)
_LINE_START = re.compile(rf"^\[?(?P<ts>{_TS})\]?(?:\s*-\s*|\s+)(?P<rest>.*)$")
_USER_SPLIT = re.compile(r"([\w\W]+?):\s")
_URL_RE = re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+", re.IGNORECASE)

_DATE_FORMATS = (
    "%d/%m/%Y, %I:%M %p",
    "%d/%m/%y, %I:%M %p",
    "%d/%m/%Y, %I:%M:%S %p",
    "%d/%m/%y, %I:%M:%S %p",
    "%d/%m/%Y, %H:%M",
    "%d/%m/%y, %H:%M",
    "%d/%m/%Y, %H:%M:%S",
    "%d/%m/%y, %H:%M:%S",
    "%m/%d/%Y, %I:%M %p",
    "%m/%d/%y, %I:%M %p",
    "%m/%d/%Y, %I:%M:%S %p",
    "%m/%d/%y, %I:%M:%S %p",
    "%m/%d/%Y, %H:%M",
    "%m/%d/%y, %H:%M",
    "%m/%d/%Y, %H:%M:%S",
    "%m/%d/%y, %H:%M:%S",
)

_DELETED_RE = re.compile(
    r"(this message was deleted|you deleted this message)",
    re.IGNORECASE,
)
_MEDIA_ONLY_RE = re.compile(
    r"(<media omitted>|image omitted|video omitted|audio omitted|"
    r"sticker omitted|gif omitted|document omitted|contact card omitted)",
    re.IGNORECASE,
)

BATCH_ROWS = 25_000
ZIP_MAGIC = b"PK\x03\x04"


def _clean_timestamps(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\u202f", " ", regex=False)
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"[\[\]]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _parse_dates(series: pd.Series) -> pd.Series:
    cleaned = _clean_timestamps(series)
    parsed = pd.Series(pd.NaT, index=cleaned.index, dtype="datetime64[ns]")

    for fmt in _DATE_FORMATS:
        mask = parsed.isna()
        if not mask.any():
            break
        parsed.loc[mask] = pd.to_datetime(cleaned[mask], format=fmt, errors="coerce")

    still_invalid = parsed.isna()
    if still_invalid.any():
        leftover = cleaned[still_invalid]
        try:
            parsed.loc[still_invalid] = pd.to_datetime(
                leftover, dayfirst=True, format="mixed", errors="coerce"
            )
        except (TypeError, ValueError):
            parsed.loc[still_invalid] = pd.to_datetime(
                leftover, dayfirst=True, errors="coerce"
            )

    return parsed


def _split_user_message(raw: str):
    entry = _USER_SPLIT.split(raw, maxsplit=1)
    if len(entry) >= 3:
        return entry[1].strip(), entry[2]
    return "group_notification", entry[0]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_media"] = df["message"].str.match(_MEDIA_ONLY_RE, na=False)
    df["is_deleted"] = df["message"].str.match(_DELETED_RE, na=False)
    df["is_system"] = df["user"].eq("group_notification")
    text_mask = ~(df["is_media"] | df["is_deleted"] | df["is_system"])
    df["word_count"] = 0
    df.loc[text_mask, "word_count"] = (
        df.loc[text_mask, "message"].str.split().str.len().fillna(0).astype("int32")
    )
    df["char_count"] = df["message"].str.len().fillna(0).astype("int32")
    df["link_count"] = df["message"].str.count(_URL_RE).fillna(0).astype("int16")
    df["emoji_count"] = df["message"].map(_count_emojis).astype("int16")

    df["year"] = df["date"].dt.year.astype("int16")
    df["month_num"] = df["date"].dt.month.astype("int8")
    df["month"] = df["date"].dt.month_name().astype("category")
    df["day"] = df["date"].dt.day.astype("int8")
    df["only_date"] = df["date"].dt.date
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour.astype("int8")
    df["minute"] = df["date"].dt.minute.astype("int8")
    df["period"] = df["hour"].map(lambda h: "23-00" if int(h) == 23 else f"{int(h)}-{int(h) + 1}").astype("category")
    df["day_name"] = pd.Categorical(
        df["day_name"],
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )
    df["user"] = df["user"].astype("category")
    return df


def _count_emojis(text):
    if not isinstance(text, str) or not text:
        return 0
    try:
        import emoji

        return int(emoji.emoji_count(text))
    except Exception:
        return 0


def _batch_to_frame(rows):
    if not rows:
        return None
    users = []
    bodies = []
    dates = []
    for ts, rest in rows:
        user, body = _split_user_message(rest.lstrip(" -"))
        users.append(user)
        bodies.append(body.replace("\r", "").strip())
        dates.append(ts)

    df = pd.DataFrame({"user": users, "message": bodies, "message_date": dates})
    df["date"] = _parse_dates(df["message_date"])
    df.drop(columns=["message_date"], inplace=True)
    df = df.dropna(subset=["date"])
    if df.empty:
        return None
    return _features(df)


def preprocessor_from_lines(lines, batch_size=BATCH_ROWS):
    """Stream a WhatsApp export line-by-line so the full file is not regex-split in RAM."""
    frames = []
    batch = []
    current_ts = None
    current_parts = []

    def flush_message():
        if current_ts is None:
            return
        batch.append((current_ts, "\n".join(current_parts)))

    def flush_batch():
        frame = _batch_to_frame(batch)
        batch.clear()
        if frame is not None:
            frames.append(frame)

    for raw in lines:
        line = raw[:-1] if raw.endswith("\n") else raw
        if line.endswith("\r"):
            line = line[:-1]
        match = _LINE_START.match(line)
        if match:
            flush_message()
            if len(batch) >= batch_size:
                flush_batch()
            current_ts = match.group("ts")
            current_parts = [match.group("rest")]
        elif current_ts is not None:
            current_parts.append(line)

    flush_message()
    if batch:
        flush_batch()

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)


def preprocessor_from_file(file_obj, encoding="utf-8"):
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
        peek = file_obj.read(4)
        file_obj.seek(0)
        if peek.startswith(ZIP_MAGIC):
            raise ValueError("zip")

    buffer = file_obj
    if isinstance(file_obj, (bytes, bytearray)):
        buffer = io.BytesIO(file_obj)

    text = io.TextIOWrapper(buffer, encoding=encoding, errors="replace")
    try:
        return preprocessor_from_lines(text)
    finally:
        try:
            text.detach()
        except Exception:
            pass


def preprocessor(data):
    if isinstance(data, (bytes, bytearray)):
        return preprocessor_from_file(io.BytesIO(data))
    return preprocessor_from_lines(io.StringIO(data))

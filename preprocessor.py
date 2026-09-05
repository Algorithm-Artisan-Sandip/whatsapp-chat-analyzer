import re
import pandas as pd


# WhatsApp Android: 12/03/2024, 10:15 am - Name: message
# WhatsApp iOS:     [12/03/24, 10:15:32 AM] Name: message
# Year may be 2 or 4 digits; time may be 12h or 24h, with optional seconds.
_TIMESTAMP = (
    r"\[?"
    r"\d{1,2}/\d{1,2}/\d{2,4},\s*"
    r"\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:\s*[APap][Mm])?"
    r"\]?"
)
_SPLIT_PATTERN = re.compile(_TIMESTAMP)

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
    """Parse WhatsApp timestamps without deprecated pandas kwargs."""
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


def preprocessor(data):
    messages = _SPLIT_PATTERN.split(data)[1:]
    dates = _SPLIT_PATTERN.findall(data)

    df = pd.DataFrame({"user_message": messages, "message_date": dates})
    df["user_message"] = df["user_message"].str.lstrip(" -")

    df["date"] = _parse_dates(df["message_date"])
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    users = []
    message_bodies = []
    for message in df["user_message"]:
        entry = re.split(r"([\w\W]+?):\s", message, maxsplit=1)
        if entry[1:]:
            users.append(entry[1].strip())
            message_bodies.append(entry[2])
        else:
            users.append("group_notification")
            message_bodies.append(entry[0])

    df["user"] = users
    df["message"] = (
        pd.Series(message_bodies, index=df.index)
        .astype(str)
        .str.replace("\r", "", regex=False)
        .str.strip()
    )
    df.drop(columns=["user_message"], inplace=True)

    df["is_media"] = df["message"].str.match(_MEDIA_ONLY_RE, na=False)
    df["is_deleted"] = df["message"].str.match(_DELETED_RE, na=False)
    df["is_system"] = df["user"].eq("group_notification")
    text_mask = ~(df["is_media"] | df["is_deleted"] | df["is_system"])
    df["word_count"] = 0
    df.loc[text_mask, "word_count"] = (
        df.loc[text_mask, "message"].str.split().str.len().fillna(0).astype(int)
    )
    df["char_count"] = df["message"].str.len().fillna(0).astype(int)

    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["only_date"] = df["date"].dt.date
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute

    def _period(hour):
        if pd.isna(hour):
            return None
        hour = int(hour)
        if hour == 23:
            return "23-00"
        return f"{hour}-{hour + 1}"

    df["period"] = df["hour"].map(_period)

    df["day_name"] = pd.Categorical(
        df["day_name"],
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )

    return df.sort_values("date").reset_index(drop=True)

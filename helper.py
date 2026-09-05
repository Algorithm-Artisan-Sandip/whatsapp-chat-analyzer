import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_EXTRACTOR = URLExtract()
_SENTIMENT = SentimentIntensityAnalyzer()

_STOP_WORDS_CACHE = None


def _stop_words():
    global _STOP_WORDS_CACHE
    if _STOP_WORDS_CACHE is None:
        with open("stop_words.txt", "r", encoding="utf-8") as f:
            _STOP_WORDS_CACHE = set(f.read().split())
    return _STOP_WORDS_CACHE


def filter_user(df, selected_user):
    if selected_user != "Overall":
        return df[df["user"] == selected_user].copy()
    return df.copy()


def human_messages(df):
    return df[df["user"] != "group_notification"].copy()


def fetch_stats(selected_user, df):
    view = filter_user(df, selected_user)
    total_messages = view.shape[0]
    words = int(view["word_count"].sum()) if "word_count" in view else 0
    media_messages = int(view["is_media"].sum()) if "is_media" in view else 0
    links = []
    for message in view["message"]:
        if isinstance(message, str):
            links.extend(_EXTRACTOR.find_urls(message))
    return total_messages, words, media_messages, len(links)


def overview_metrics(selected_user, df):
    view = filter_user(df, selected_user)
    people = human_messages(df if selected_user == "Overall" else view)
    if view.empty:
        return {
            "messages": 0,
            "words": 0,
            "media": 0,
            "links": 0,
            "deleted": 0,
            "members": 0,
            "days_active": 0,
            "span_days": 0,
            "avg_per_day": 0.0,
            "avg_words": 0.0,
            "first": None,
            "last": None,
        }

    _, words, media, links = fetch_stats(selected_user, df)
    span_days = max((view["date"].max() - view["date"].min()).days + 1, 1)
    text_rows = view[~view["is_media"] & ~view["is_deleted"] & ~view["is_system"]]
    return {
        "messages": int(view.shape[0]),
        "words": words,
        "media": media,
        "links": links,
        "deleted": int(view["is_deleted"].sum()) if "is_deleted" in view else 0,
        "members": people["user"].nunique(),
        "days_active": int(view["only_date"].nunique()),
        "span_days": int(span_days),
        "avg_per_day": round(view.shape[0] / span_days, 2),
        "avg_words": round(float(text_rows["word_count"].mean()) if not text_rows.empty else 0.0, 2),
        "first": view["date"].min(),
        "last": view["date"].max(),
    }


def most_busy_users(df):
    people = human_messages(df)
    counts = people["user"].value_counts()
    x = counts.head(10)
    table = (
        counts.rename("messages")
        .to_frame()
        .assign(percent=lambda d: (d["messages"] / d["messages"].sum() * 100).round(2))
        .reset_index()
        .rename(columns={"user": "name"})
    )
    return x, table


def user_leaderboard(df):
    people = human_messages(df)
    if people.empty:
        return pd.DataFrame(columns=["user", "messages", "words", "media", "emojis", "avg_words", "percent"])

    rows = []
    total = people.shape[0]
    for user, group in people.groupby("user"):
        emojis = []
        for message in group["message"]:
            if isinstance(message, str):
                emojis.extend(item["emoji"] for item in emoji.emoji_list(message))
        text = group[~group["is_media"] & ~group["is_deleted"]]
        rows.append(
            {
                "user": user,
                "messages": int(group.shape[0]),
                "words": int(group["word_count"].sum()),
                "media": int(group["is_media"].sum()),
                "emojis": len(emojis),
                "avg_words": round(float(text["word_count"].mean()) if not text.empty else 0.0, 2),
                "percent": round(group.shape[0] / total * 100, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("messages", ascending=False).reset_index(drop=True)


def create_wordcloud(selected_user, df):
    view = filter_user(df, selected_user)
    stop_words = _stop_words()
    temp = view[(view["user"] != "group_notification") & ~view["is_media"] & ~view["is_deleted"]].copy()
    if temp.empty:
        return None

    def remove_stop_words(message):
        return " ".join(word for word in str(message).lower().split() if word not in stop_words)

    temp["clean"] = temp["message"].apply(remove_stop_words)
    text = temp["clean"].str.cat(sep=" ").strip()
    if not text:
        return None
    wc = WordCloud(width=800, height=400, min_font_size=10, background_color="white", colormap="viridis")
    return wc.generate(text)


def most_common_words(selected_user, df):
    view = filter_user(df, selected_user)
    stop_words = _stop_words()
    temp = view[(view["user"] != "group_notification") & ~view["is_media"] & ~view["is_deleted"]]
    words = []
    for message in temp["message"]:
        for word in str(message).lower().split():
            if word not in stop_words and word.isalpha():
                words.append(word)
    return pd.DataFrame(Counter(words).most_common(25), columns=["word", "count"])


def emoji_helper(selected_user, df):
    view = filter_user(df, selected_user)
    emojis = []
    for message in view["message"]:
        if not isinstance(message, str):
            continue
        emojis.extend(item["emoji"] for item in emoji.emoji_list(message))
    return pd.DataFrame(Counter(emojis).most_common(), columns=["emoji", "count"])


def monthly_timeline(selected_user, df):
    view = filter_user(df, selected_user)
    timeline = view.groupby(["year", "month_num", "month"], observed=False).count()["message"].reset_index()
    timeline["time"] = timeline["month"] + " " + timeline["year"].astype(str)
    return timeline.sort_values(["year", "month_num"])


def daily_timeline(selected_user, df):
    view = filter_user(df, selected_user)
    daily = view.groupby("only_date").count()["message"].reset_index()
    daily = daily.rename(columns={"only_date": "date", "message": "message"})
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date")


def week_activity_map(selected_user, df):
    view = filter_user(df, selected_user)
    return view["day_name"].value_counts()


def month_activity_map(selected_user, df):
    view = filter_user(df, selected_user)
    return view["month"].value_counts()


def activity_heatmap(selected_user, df):
    view = filter_user(df, selected_user)
    if view.empty:
        return pd.DataFrame()
    heatmap = view.pivot_table(index="day_name", columns="period", values="message", aggfunc="count").fillna(0)
    return heatmap


def hourly_activity(selected_user, df):
    view = filter_user(df, selected_user)
    hours = (
        view.groupby("hour")
        .size()
        .reindex(range(24), fill_value=0)
        .rename("messages")
        .reset_index()
    )
    return hours


def chronotype(selected_user, df):
    view = filter_user(df, selected_user)
    if view.empty:
        return pd.DataFrame(columns=["bucket", "messages", "percent"])
    buckets = pd.cut(
        view["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night (12-6am)", "Morning (6-12)", "Afternoon (12-6pm)", "Evening (6-12am)"],
    )
    counts = buckets.value_counts().rename("messages")
    table = counts.to_frame()
    table["percent"] = (table["messages"] / table["messages"].sum() * 100).round(1)
    return table.reset_index().rename(columns={"index": "bucket", "hour": "bucket"})


def activity_streaks(selected_user, df):
    view = filter_user(df, selected_user)
    dates = sorted(set(view["only_date"]))
    if not dates:
        return {"current": 0, "longest": 0, "active_days": 0}
    longest = current = 1
    streak = 1
    for prev, nxt in zip(dates, dates[1:]):
        if (nxt - prev).days == 1:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 1
    last = dates[-1]
    # current streak relative to last active day in the chat, not "today"
    current = 1
    for prev, nxt in zip(reversed(dates[:-1]), reversed(dates[1:])):
        if (nxt - prev).days == 1:
            current += 1
        else:
            break
    return {"current": current, "longest": max(longest, current), "active_days": len(dates), "last_day": last}


def longest_messages(selected_user, df, n=8):
    view = filter_user(df, selected_user)
    text = view[(view["user"] != "group_notification") & ~view["is_media"] & ~view["is_deleted"]].copy()
    text = text.sort_values("word_count", ascending=False).head(n)
    preview = text["message"].str.replace("\n", " ", regex=False).str.slice(0, 180)
    return pd.DataFrame(
        {
            "date": text["date"].dt.strftime("%Y-%m-%d %H:%M"),
            "user": text["user"],
            "words": text["word_count"],
            "preview": preview,
        }
    )


def response_times(df):
    people = human_messages(df).sort_values("date")
    if people.shape[0] < 2:
        return pd.DataFrame(columns=["user", "replies", "median_minutes", "mean_minutes"])
    people["prev_user"] = people["user"].shift()
    people["prev_date"] = people["date"].shift()
    people["delta"] = people["date"] - people["prev_date"]
    replies = people[
        people["prev_user"].notna()
        & (people["user"] != people["prev_user"])
        & (people["delta"] >= pd.Timedelta(seconds=5))
        & (people["delta"] <= pd.Timedelta(hours=12))
    ].copy()
    if replies.empty:
        return pd.DataFrame(columns=["user", "replies", "median_minutes", "mean_minutes"])
    replies["minutes"] = replies["delta"].dt.total_seconds() / 60.0
    summary = (
        replies.groupby("user")["minutes"]
        .agg(replies="count", median_minutes="median", mean_minutes="mean")
        .reset_index()
        .round(1)
        .sort_values("median_minutes")
    )
    return summary


def conversation_starters(df, gap_hours=6):
    people = human_messages(df).sort_values("date")
    if people.empty:
        return pd.DataFrame(columns=["user", "conversations_started", "percent"])
    people["prev_date"] = people["date"].shift()
    people["gap"] = people["date"] - people["prev_date"]
    gap = pd.Timedelta(hours=gap_hours)
    starters = people[people["gap"].isna() | (people["gap"] > gap)]
    counts = starters["user"].value_counts().rename("conversations_started")
    table = counts.to_frame()
    table["percent"] = (table["conversations_started"] / table["conversations_started"].sum() * 100).round(2)
    return table.reset_index().rename(columns={"index": "user", "user": "user"})


def sentiment_table(selected_user, df):
    view = filter_user(df, selected_user)
    text = view[(view["user"] != "group_notification") & ~view["is_media"] & ~view["is_deleted"]].copy()
    if text.empty:
        return text
    scores = text["message"].map(lambda m: _SENTIMENT.polarity_scores(str(m)))
    text["compound"] = scores.map(lambda s: s["compound"])
    text["positive"] = scores.map(lambda s: s["pos"])
    text["neutral"] = scores.map(lambda s: s["neu"])
    text["negative"] = scores.map(lambda s: s["neg"])
    text["label"] = pd.cut(
        text["compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["Negative", "Neutral", "Positive"],
    )
    return text


def sentiment_summary(selected_user, df):
    scored = sentiment_table(selected_user, df)
    if scored.empty or "label" not in scored:
        return {
            "avg_compound": 0.0,
            "positive_pct": 0.0,
            "neutral_pct": 0.0,
            "negative_pct": 0.0,
            "by_day": pd.DataFrame(columns=["date", "compound"]),
            "by_user": pd.DataFrame(columns=["user", "compound", "messages"]),
            "scored": scored,
        }
    label_share = scored["label"].value_counts(normalize=True)
    by_day = (
        scored.assign(day=scored["date"].dt.date)
        .groupby("day")["compound"]
        .mean()
        .reset_index()
        .rename(columns={"day": "date", "compound": "compound"})
    )
    by_day["date"] = pd.to_datetime(by_day["date"])
    by_user = (
        scored.groupby("user")
        .agg(compound=("compound", "mean"), messages=("message", "count"))
        .reset_index()
        .round(3)
        .sort_values("compound", ascending=False)
    )
    return {
        "avg_compound": round(float(scored["compound"].mean()), 3),
        "positive_pct": round(float(label_share.get("Positive", 0) * 100), 1),
        "neutral_pct": round(float(label_share.get("Neutral", 0) * 100), 1),
        "negative_pct": round(float(label_share.get("Negative", 0) * 100), 1),
        "by_day": by_day,
        "by_user": by_user,
        "scored": scored,
    }


def search_messages(df, query, selected_user="Overall"):
    view = filter_user(df, selected_user)
    view = view[~view["is_system"]]
    if not query or not str(query).strip():
        return view[["date", "user", "message"]].head(50)
    q = str(query).strip()
    matches = view[view["message"].str.contains(q, case=False, na=False, regex=False)]
    return matches[["date", "user", "message"]]


def export_frame(df):
    out = df.copy()
    cols = [
        "date",
        "user",
        "message",
        "year",
        "month",
        "day_name",
        "hour",
        "word_count",
        "is_media",
        "is_deleted",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols]

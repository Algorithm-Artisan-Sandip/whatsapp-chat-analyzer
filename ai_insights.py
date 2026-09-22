"""Local NLP / 'AI' layer. No API keys: TF-IDF, NMF, VADER, and rule-based retrieval."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import helper

_SENTIMENT = SentimentIntensityAnalyzer()
_TFIDF_CAP = 12_000
_QUESTION_RE = re.compile(
    r"^\s*(who|what|when|where|why|how|which|can|could|would|is|are|do|did|does)\b",
    re.I,
)
_TENSION_WORDS = {
    "hate", "stupid", "idiot", "shut up", "angry", "worst", "useless",
    "disgusting", "kill", "dumb", "trash", "pathetic", "nonsense",
}


def _text_frame(df, selected_user="Overall"):
    view = helper.filter_user(df, selected_user)
    text = view[
        (view["user"] != "group_notification")
        & ~view["is_media"]
        & ~view["is_deleted"]
        & view["message"].map(lambda m: isinstance(m, str) and len(m.strip()) > 2)
    ]
    if text.empty:
        return text
    if len(text) > _TFIDF_CAP:
        text = text.sample(n=_TFIDF_CAP, random_state=42)
    return text.copy().reset_index(drop=True)


def _vectorize(messages):
    token = re.compile(r"^[a-zA-Z]{3,}$")
    stops = sorted({w.lower() for w in helper._stop_words() if token.match(str(w))})
    n = len(messages)
    vec = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        min_df=1 if n < 80 else 2,
        max_df=0.9 if n > 20 else 1.0,
        stop_words=stops or "english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    matrix = vec.fit_transform(messages.astype(str))
    return vec, matrix


def build_index(df, selected_user="Overall"):
    text = _text_frame(df, selected_user)
    if text.empty or text["message"].str.len().sum() < 20:
        return None
    try:
        vec, matrix = _vectorize(text["message"])
    except ValueError:
        return None
    if matrix.shape[1] == 0:
        return None
    return {"frame": text, "vectorizer": vec, "matrix": matrix}


def topic_model(index, n_topics=6):
    if not index:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    frame = index["frame"]
    matrix = index["matrix"]
    vec = index["vectorizer"]
    n_topics = int(max(2, min(n_topics, matrix.shape[0] // 4 or 2, matrix.shape[1], 8)))
    if matrix.shape[0] < n_topics or matrix.shape[1] < n_topics:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    model = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=400)
    w = model.fit_transform(matrix)
    h = model.components_
    terms = np.array(vec.get_feature_names_out())
    rows = []
    for i, comp in enumerate(h):
        top = terms[comp.argsort()[::-1][:8]]
        rows.append({"topic": f"Topic {i + 1}", "label": ", ".join(top[:4]), "top_words": ", ".join(top)})
    topics = pd.DataFrame(rows)
    assigned = w.argmax(axis=1)
    strength = w.max(axis=1)
    labeled = frame.copy()
    labeled["topic"] = [topics.iloc[i]["topic"] for i in assigned]
    labeled["topic_label"] = [topics.iloc[i]["label"] for i in assigned]
    labeled["topic_strength"] = strength
    timeline = (
        labeled.assign(day=labeled["date"].dt.date)
        .groupby(["day", "topic"], observed=False)
        .size()
        .reset_index(name="messages")
    )
    timeline["day"] = pd.to_datetime(timeline["day"])
    examples = (
        labeled.sort_values("topic_strength", ascending=False)
        .groupby("topic", observed=False)
        .head(3)[["topic", "topic_label", "date", "user", "message", "topic_strength"]]
    )
    return topics, timeline, examples


def semantic_search(index, query, k=8):
    if not index or not str(query).strip():
        return pd.DataFrame(columns=["date", "user", "message", "similarity"])
    vec, matrix, frame = index["vectorizer"], index["matrix"], index["frame"]
    q = vec.transform([str(query).strip()])
    scores = cosine_similarity(q, matrix).ravel()
    top = np.argsort(scores)[::-1][:k]
    hits = frame.iloc[top][["date", "user", "message"]].copy()
    hits["similarity"] = scores[top].round(3)
    return hits.loc[hits["similarity"] > 0].reset_index(drop=True)


def classify_intent(message: str) -> str:
    if not isinstance(message, str) or not message.strip():
        return "Other"
    raw = message.strip()
    low = raw.lower()
    if raw.lower().startswith("<media") or "omitted" in low:
        return "Media"
    if "?" in raw or _QUESTION_RE.search(raw):
        return "Question"
    if any(w in low for w in ("thanks", "thank you", "thx", "tysm")):
        return "Thanks"
    if any(w in low for w in ("sorry", "my bad", "apologies")):
        return "Apology"
    if any(tok in low for tok in ("lol", "lmao", "haha", "rofl")) or any(e in raw for e in "😂😅🤣😊😄"):
        return "Humor"
    if any(w in low for w in ("let's", "lets ", "we should", "plan", "ship", "deploy", "meeting")):
        return "Plan"
    if any(w in low for w in _TENSION_WORDS):
        return "Tension"
    return "Statement"


def intent_table(df, selected_user="Overall"):
    view = helper.filter_user(df, selected_user)
    human = view[(view["user"] != "group_notification") & ~view["is_media"]]
    if human.empty:
        return pd.DataFrame(columns=["intent", "messages"]), pd.DataFrame()
    sample, _ = helper.sample_text_rows(human, cap=20_000)
    labeled = sample.copy()
    labeled["intent"] = labeled["message"].map(classify_intent)
    counts = labeled["intent"].value_counts().rename("messages").reset_index().rename(columns={"index": "intent", "intent": "intent"})
    by_user = (
        labeled.groupby(["user", "intent"], observed=True)
        .size()
        .reset_index(name="messages")
    )
    return counts, by_user


def tension_timeline(df, selected_user="Overall"):
    view = helper.filter_user(df, selected_user)
    human = view[(view["user"] != "group_notification") & ~view["is_media"] & ~view["is_deleted"]]
    if human.empty:
        return pd.DataFrame(columns=["date", "tension"]), pd.DataFrame()
    sample, _ = helper.sample_text_rows(human, cap=20_000)
    rows = []
    for _, row in sample.iterrows():
        msg = str(row["message"])
        letters = [c for c in msg if c.isalpha()]
        caps = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0
        compound = _SENTIMENT.polarity_scores(msg)["compound"]
        lex = sum(1 for w in _TENSION_WORDS if w in msg.lower())
        score = max(0.0, -compound) + (0.4 if caps > 0.65 and len(msg) > 8 else 0) + 0.25 * lex
        rows.append(score)
    out = sample.copy()
    out["tension"] = rows
    daily = (
        out.assign(day=out["date"].dt.date)
        .groupby("day")["tension"]
        .mean()
        .reset_index()
        .rename(columns={"day": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    spikes = out.sort_values("tension", ascending=False).head(8)[["date", "user", "message", "tension"]]
    return daily, spikes


def activity_anomalies(df, selected_user="Overall"):
    daily = helper.daily_timeline(selected_user, df)
    if daily.empty or len(daily) < 5:
        return daily.assign(z_score=np.nan, flag="")
    mu = daily["message"].mean()
    sd = daily["message"].std(ddof=0) or 1.0
    daily = daily.copy()
    daily["z_score"] = ((daily["message"] - mu) / sd).round(2)
    daily["flag"] = np.where(daily["z_score"] >= 2, "Burst", np.where(daily["z_score"] <= -2, "Unusually quiet", "Normal"))
    return daily


def language_mix(df, selected_user="Overall"):
    view = helper.filter_user(df, selected_user)
    human = view[(view["user"] != "group_notification") & ~view["is_media"]]
    if human.empty:
        return pd.DataFrame(columns=["script", "messages"])
    sample, _ = helper.sample_text_rows(human, cap=8_000)

    def script_of(text):
        t = str(text)
        if re.search(r"[\u0900-\u097F]", t):
            return "Devanagari (Hindi/Marathi)"
        if re.search(r"[\u0980-\u09FF]", t):
            return "Bangla"
        if re.search(r"[\u0A80-\u0AFF]", t):
            return "Gujarati"
        if re.search(r"[\u0B80-\u0BFF]", t):
            return "Tamil"
        if re.search(r"[\u0C00-\u0C7F]", t):
            return "Telugu"
        if re.search(r"[\u0C80-\u0CFF]", t):
            return "Kannada"
        if re.search(r"[\u0D00-\u0D7F]", t):
            return "Malayalam"
        if re.search(r"[\u0600-\u06FF]", t):
            return "Arabic script"
        if re.search(r"[A-Za-z]", t):
            return "Latin (English-like)"
        return "Other / emoji"

    counts = sample["message"].map(script_of).value_counts().rename("messages")
    return counts.reset_index().rename(columns={"index": "script", "message": "script"})


def member_similarity(df):
    board = helper.user_leaderboard(df)
    chrono_cache = {}
    if board.empty or board.shape[0] < 2:
        return board, pd.DataFrame()
    replies = helper.response_times(df)
    reply_map = dict(zip(replies["user"], replies["median_minutes"])) if not replies.empty else {}
    scored = helper.sentiment_table("Overall", df)
    sent_map = {}
    if not scored.empty and "compound" in scored:
        sent_map = scored.groupby("user")["compound"].mean().to_dict()

    feats = []
    for _, row in board.iterrows():
        user = row["user"]
        if user not in chrono_cache:
            ch = helper.chronotype(user, df)
            night = float(ch.loc[ch["bucket"].astype(str).str.startswith("Night"), "percent"].sum()) if not ch.empty else 0
            chrono_cache[user] = night
        feats.append(
            [
                row["messages"],
                row["avg_words"],
                row["media"],
                row["emojis"],
                reply_map.get(user, 30),
                sent_map.get(user, 0),
                chrono_cache[user],
            ]
        )
    arr = np.array(feats, dtype=float)
    arr = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-9)
    sim = cosine_similarity(arr)
    names = board["user"].astype(str).tolist()
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append({"member_a": names[i], "member_b": names[j], "style_similarity": round(float(sim[i, j]), 3)})
    pair_df = pd.DataFrame(pairs).sort_values("style_similarity", ascending=False)
    return board, pair_df


def extractive_recap(df, selected_user="Overall"):
    stats = helper.overview_metrics(selected_user, df)
    streaks = helper.activity_streaks(selected_user, df)
    daily = helper.daily_timeline(selected_user, df)
    starters = helper.conversation_starters(df)
    replies = helper.response_times(df)
    mood = helper.sentiment_summary(selected_user, df)
    anomalies = activity_anomalies(df, selected_user)

    busiest = None
    if not daily.empty:
        top = daily.sort_values("message", ascending=False).iloc[0]
        busiest = f"{pd.to_datetime(top['date']).strftime('%d %b %Y')} ({int(top['message'])} messages)"

    starter = None
    if selected_user == "Overall" and not starters.empty:
        starter = f"{starters.iloc[0]['user']} ({int(starters.iloc[0]['conversations_started'])} sessions)"
    fastest = None
    if selected_user == "Overall" and not replies.empty:
        fastest = f"{replies.iloc[0]['user']} (median {replies.iloc[0]['median_minutes']} min)"

    bursts = anomalies.loc[anomalies.get("flag", "") == "Burst"] if "flag" in anomalies else pd.DataFrame()
    quiet = anomalies.loc[anomalies.get("flag", "") == "Unusually quiet"] if "flag" in anomalies else pd.DataFrame()

    bullets = [
        f"Chat spans {stats['span_days']} days with {stats['messages']:,} messages and {stats['members']} member(s).",
        f"Longest activity streak is {streaks['longest']} day(s); average volume is {stats['avg_per_day']} messages/day.",
        f"Busiest day: {busiest}." if busiest else "Not enough daily volume to name a peak day.",
        f"Mood (VADER): {mood['avg_compound']} · {mood['positive_pct']}% positive / {mood['negative_pct']}% negative.",
        f"Most sessions started by {starter}." if starter else "Conversation-starter stats need multiple members.",
        f"Fastest typical reply: {fastest}." if fastest else "Reply-speed needs back-and-forth turns.",
        f"Activity bursts on {len(bursts)} day(s); unusually quiet on {len(quiet)} day(s)."
        if "flag" in anomalies
        else "Need more days to flag anomalies.",
    ]
    index = build_index(df, selected_user)
    quotes = pd.DataFrame()
    if index:
        frame = index["frame"]
        centrality = np.asarray(index["matrix"].sum(axis=1)).ravel()
        frame = frame.assign(rank=centrality)
        quotes = frame.sort_values("rank", ascending=False).head(5)[["date", "user", "message"]]
    return {"bullets": bullets, "quotes": quotes, "anomalies": anomalies}


def answer_question(df, question, selected_user="Overall"):
    q = str(question).strip()
    if not q:
        return "", pd.DataFrame()
    low = q.lower()
    stats = helper.overview_metrics(selected_user, df)
    daily = helper.daily_timeline(selected_user, df)
    board = helper.user_leaderboard(df)
    mood = helper.sentiment_summary(selected_user, df)
    starters = helper.conversation_starters(df)
    replies = helper.response_times(df)

    def cite(frame):
        if frame is None or frame.empty:
            return pd.DataFrame()
        out = frame.copy()
        if "date" in out:
            out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d %H:%M")
        return out.head(8)

    if any(k in low for k in ("who talk", "most message", "busiest user", "most active person", "leaderboard")):
        if board.empty:
            return "No members found.", pd.DataFrame()
        top = board.iloc[0]
        return (
            f"{top['user']} sent the most messages ({int(top['messages'])}, {top['percent']}% of the chat).",
            cite(board[["user", "messages", "percent", "words"]]),
        )
    if any(k in low for k in ("busiest day", "most active day", "when were we active", "peak day")):
        if daily.empty:
            return "Not enough daily data.", pd.DataFrame()
        top = daily.sort_values("message", ascending=False).iloc[0]
        return (
            f"The busiest day is {pd.to_datetime(top['date']).strftime('%d %b %Y')} with {int(top['message'])} messages.",
            cite(daily.sort_values("message", ascending=False).head(5)),
        )
    if any(k in low for k in ("sentiment", "mood", "positive", "negative", "happy", "sad")):
        return (
            f"Average VADER compound is {mood['avg_compound']} "
            f"({mood['positive_pct']}% positive, {mood['neutral_pct']}% neutral, {mood['negative_pct']}% negative).",
            cite(mood["by_user"] if selected_user == "Overall" else pd.DataFrame()),
        )
    if any(k in low for k in ("who start", "initiator", "starts conversation")):
        if starters.empty:
            return "Not enough gaps to detect conversation starters.", pd.DataFrame()
        top = starters.iloc[0]
        return (
            f"{top['user']} starts the most sessions after a quiet gap ({int(top['conversations_started'])}).",
            cite(starters),
        )
    if any(k in low for k in ("reply", "respond", "fastest", "slowest")):
        if replies.empty:
            return "Not enough back-and-forth to estimate reply speed.", pd.DataFrame()
        top = replies.iloc[0]
        return (
            f"{top['user']} has the fastest median reply ({top['median_minutes']} minutes among in-thread turns).",
            cite(replies),
        )
    if any(k in low for k in ("how many", "total message", "count")):
        return (
            f"This view has {stats['messages']:,} messages, {stats['words']:,} words, "
            f"{stats['media']} media items, across {stats['span_days']} days.",
            pd.DataFrame(),
        )

    index = build_index(df, selected_user)
    hits = semantic_search(index, q, k=8)
    if hits.empty:
        return "No locally similar messages found. Try different keywords.", pd.DataFrame()
    first = hits.iloc[0]
    answer = (
        f"Closest match (similarity {first['similarity']}): {first['user']} on "
        f"{pd.to_datetime(first['date']).strftime('%d %b %Y')} — “{str(first['message'])[:220]}”"
    )
    return answer, cite(hits)

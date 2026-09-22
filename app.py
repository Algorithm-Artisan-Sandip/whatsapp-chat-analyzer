import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import preprocessor, helper, ai_insights

st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20),
    font=dict(color="#FAFAFA"),
)
MAX_UPLOAD_MB = 400
ACCENT = "#25D366"

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { position: relative; }
    .sidebar-footer {
        position: fixed; bottom: 0; left: 0; width: inherit;
        background-color: inherit; padding: 1rem;
        border-top: 1px solid rgba(250, 250, 250, 0.15); z-index: 999;
    }
    [data-testid="stSidebar"] > div:first-child { padding-bottom: 90px; }
    .hero-card {
        background: linear-gradient(135deg, #128C7E 0%, #075E54 60%, #0E1117 100%);
        border: 1px solid rgba(37, 211, 102, 0.35);
        border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    }
    .hero-card h1 { margin: 0 0 0.35rem 0; font-size: 2rem; }
    .hero-card p { margin: 0; color: #d7e6df; }
    div[data-testid="stMetric"] {
        background: #16202b; border: 1px solid rgba(37, 211, 102, 0.18);
        border-radius: 12px; padding: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def load_chat(raw_text):
    df = preprocessor.preprocessor(raw_text)
    if df.empty:
        return None
    return df


def validate_upload(uploaded_file):
    name = (uploaded_file.name or "").lower()
    size = int(getattr(uploaded_file, "size", 0) or 0)
    size_mb = size / (1024 * 1024)
    if name.endswith((".zip", ".opus", ".jpg", ".jpeg", ".png", ".mp4", ".pdf", ".webp")):
        return "Upload the WhatsApp .txt export (Without media), not a zip or media file."
    if size_mb > MAX_UPLOAD_MB:
        return (
            f"This file is {size_mb:.1f} MB. The app accepts up to {MAX_UPLOAD_MB} MB. "
            "Export the chat Without media, or run locally with a larger limit."
        )
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
        magic = uploaded_file.read(4)
        uploaded_file.seek(0)
        if magic.startswith(preprocessor.ZIP_MAGIC):
            return "This looks like a zip/media export. Upload the .txt file from WhatsApp instead."
    return None


def parse_uploaded_file(uploaded_file):
    sig = (uploaded_file.name, int(getattr(uploaded_file, "size", 0) or 0))
    if st.session_state.get("chat_sig") != sig:
        size_mb = (sig[1] / (1024 * 1024)) if sig[1] else 0
        with st.spinner(f"Parsing {uploaded_file.name} ({size_mb:.1f} MB). Large chats are read in chunks..."):
            try:
                parsed = preprocessor.preprocessor_from_file(uploaded_file)
            except ValueError as exc:
                if str(exc) == "zip":
                    return None, "This looks like a zip/media export. Upload the .txt file instead."
                raise
        st.session_state.chat_sig = sig
        st.session_state.chat_df = parsed
    return st.session_state.chat_df, None


@st.cache_data(show_spinner=False)
def parse_sample():
    with open("sample_chat.txt", "r", encoding="utf-8") as f:
        return load_chat(f.read())


def render_overview(view_user, df):
    stats = helper.overview_metrics(view_user, df)
    streaks = helper.activity_streaks(view_user, df)
    st.markdown(
        f"""
        <div class="hero-card">
            <h1>Chat intelligence dashboard</h1>
            <p>
                {stats['first'].strftime('%d %b %Y') if stats['first'] is not None else '—'}
                → {stats['last'].strftime('%d %b %Y') if stats['last'] is not None else '—'}
                · {stats['span_days']} day span · {stats['members']} member(s)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Messages", f"{stats['messages']:,}")
    c2.metric("Words", f"{stats['words']:,}")
    c3.metric("Media", f"{stats['media']:,}")
    c4.metric("Links", f"{stats['links']:,}")
    c5.metric("Avg / day", stats["avg_per_day"])
    c6.metric("Longest streak", f"{streaks['longest']} days")

    c7, c8, c9, c10 = st.columns(4)
    c7.metric("Active days", streaks["active_days"])
    c8.metric("Deleted", stats["deleted"])
    c9.metric("Avg words / msg", stats["avg_words"])
    c10.metric("Members", stats["members"])

    left, right = st.columns(2)
    with left:
        daily = helper.daily_timeline(view_user, df)
        if daily.empty:
            st.info("Not enough data for a daily timeline.")
        else:
            fig = px.area(daily, x="date", y="message", title="Messages per day")
            fig.update_traces(line_color=ACCENT, fillcolor="rgba(37,211,102,0.25)")
            st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        monthly = helper.monthly_timeline(view_user, df)
        if monthly.empty:
            st.info("Not enough data for a monthly timeline.")
        else:
            fig = px.bar(monthly, x="time", y="message", title="Messages per month")
            fig.update_traces(marker_color=ACCENT)
            st.plotly_chart(style_fig(fig), use_container_width=True)


def render_activity(view_user, df):
    col1, col2 = st.columns(2)
    with col1:
        busy_day = helper.week_activity_map(view_user, df)
        if busy_day.empty:
            st.info("No weekday activity to plot.")
        else:
            day_df = busy_day.rename("messages").reset_index().rename(columns={"index": "day", "day_name": "day"})
            fig = px.bar(day_df, x="day", y="messages", title="Busiest weekdays")
            fig.update_traces(marker_color="#9B59B6")
            st.plotly_chart(style_fig(fig), use_container_width=True)
    with col2:
        busy_month = helper.month_activity_map(view_user, df)
        if busy_month.empty:
            st.info("No monthly activity to plot.")
        else:
            month_df = busy_month.rename("messages").reset_index().rename(columns={"index": "month", "month": "month"})
            fig = px.bar(month_df, x="month", y="messages", title="Busiest months")
            fig.update_traces(marker_color="#EEC05A")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    hours = helper.hourly_activity(view_user, df)
    fig = px.bar(hours, x="hour", y="messages", title="Messages by hour of day")
    fig.update_traces(marker_color="#34B7F1")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(style_fig(fig), use_container_width=True)

    chrono = helper.chronotype(view_user, df)
    if not chrono.empty:
        fig = px.pie(chrono, names="bucket", values="messages", title="When this chat is actually alive", hole=0.45)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    heatmap = helper.activity_heatmap(view_user, df)
    st.subheader("Weekly activity heatmap")
    if heatmap.empty or heatmap.values.sum() == 0:
        st.info("Not enough activity to build a heatmap.")
    else:
        fig = px.imshow(
            heatmap,
            aspect="auto",
            color_continuous_scale="YlOrBr",
            title="Day vs hour-range intensity",
        )
        fig.update_layout(xaxis_title="Hour range", yaxis_title="Day")
        st.plotly_chart(style_fig(fig), use_container_width=True)


def render_members(view_user, df):
    board = helper.user_leaderboard(df)
    if board.empty:
        st.info("No members found.")
        return
    if view_user != "Overall":
        board = board[board["user"] == view_user]
    st.subheader("Member leaderboard")
    st.dataframe(board, use_container_width=True, hide_index=True)

    fig = px.bar(board.head(12), x="user", y="messages", title="Who talks the most")
    fig.update_traces(marker_color=ACCENT)
    st.plotly_chart(style_fig(fig), use_container_width=True)

    if board.shape[0] >= 2:
        radar = board.head(5).copy()
        metrics = ["messages", "words", "media", "emojis", "avg_words"]
        for col in metrics:
            max_v = radar[col].max() or 1
            radar[col + "_n"] = radar[col] / max_v
        melted = radar.melt(id_vars="user", value_vars=[c + "_n" for c in metrics], var_name="metric", value_name="score")
        melted["metric"] = melted["metric"].str.replace("_n", "", regex=False)
        fig = px.line_polar(melted, r="score", theta="metric", color="user", line_close=True, title="Top members compared")
        fig.update_traces(fill="toself")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    _, pairs = ai_insights.member_similarity(df)
    if view_user == "Overall" and pairs is not None and not pairs.empty:
        st.subheader("Behavioral style similarity")
        st.caption("Cosine similarity on volume, length, media, emojis, reply speed, mood, and night-owl share — not personality.")
        st.dataframe(pairs.head(10), use_container_width=True, hide_index=True)


def render_content(view_user, df):
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Word cloud")
        image = helper.create_wordcloud(view_user, df)
        if image is None:
            st.info("Not enough text to build a word cloud.")
        else:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.imshow(image, interpolation="bilinear")
            ax.axis("off")
            fig.patch.set_facecolor("#0E1117")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    with col2:
        common = helper.most_common_words(view_user, df)
        st.subheader("Most used words")
        if common.empty:
            st.info("No frequent words found.")
        else:
            fig = px.bar(common.sort_values("count"), x="count", y="word", orientation="h", title="")
            fig.update_traces(marker_color="#EEC05A")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    emoji_df = helper.emoji_helper(view_user, df)
    st.subheader("Emoji analysis")
    if emoji_df.empty:
        st.info("No emojis found for this selection.")
    else:
        left, right = st.columns(2)
        with left:
            st.dataframe(emoji_df.head(20), use_container_width=True, hide_index=True)
        with right:
            top = emoji_df.head(10)
            fig = px.pie(top, names="emoji", values="count", title="Top emojis", hole=0.35)
            st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Longest messages")
    longest = helper.longest_messages(view_user, df)
    if longest.empty:
        st.info("No text messages to rank.")
    else:
        st.dataframe(longest, use_container_width=True, hide_index=True)


def render_sentiment(view_user, df):
    summary = helper.sentiment_summary(view_user, df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg sentiment", summary["avg_compound"], help="VADER compound score from -1 (negative) to +1 (positive)")
    c2.metric("Positive", f"{summary['positive_pct']}%")
    c3.metric("Neutral", f"{summary['neutral_pct']}%")
    c4.metric("Negative", f"{summary['negative_pct']}%")
    if len(helper.filter_user(df, view_user)) > helper.NLP_SAMPLE_ROWS:
        st.caption(f"Large chat: sentiment is estimated from a random sample of {helper.NLP_SAMPLE_ROWS:,} messages.")

    if summary["by_day"].empty:
        st.info("Not enough text for sentiment analysis.")
        return

    fig = px.line(summary["by_day"], x="date", y="compound", title="Mood over time")
    fig.update_traces(line_color=ACCENT)
    fig.add_hline(y=0, line_dash="dot", line_color="#888")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    if not summary["by_user"].empty and view_user == "Overall":
        fig = px.bar(summary["by_user"], x="user", y="compound", title="Average sentiment by member")
        fig.update_traces(marker_color="#34B7F1")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    scored = summary["scored"]
    share = scored["label"].value_counts().rename("messages").reset_index().rename(columns={"index": "label", "label": "label"})
    fig = px.pie(share, names="label", values="messages", title="Message tone mix", hole=0.45)
    st.plotly_chart(style_fig(fig), use_container_width=True)


def render_insights(view_user, df):
    st.subheader("Who starts conversations")
    st.caption("A new conversation is counted after a 6-hour gap.")
    starters = helper.conversation_starters(df)
    if view_user != "Overall" and not starters.empty:
        starters = starters[starters["user"] == view_user]
    if starters.empty:
        st.info("Not enough turns to detect conversation starters.")
    else:
        fig = px.bar(starters, x="user", y="conversations_started", title="")
        fig.update_traces(marker_color="#9B59B6")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.dataframe(starters, use_container_width=True, hide_index=True)

    st.subheader("Reply speed")
    st.caption("Median time to reply after someone else, ignoring gaps longer than 12 hours.")
    replies = helper.response_times(df)
    if view_user != "Overall" and not replies.empty:
        replies = replies[replies["user"] == view_user]
    if replies.empty:
        st.info("Not enough back-and-forth messages to estimate reply speed.")
    else:
        fig = px.bar(replies, x="user", y="median_minutes", title="Median reply time (minutes)")
        fig.update_traces(marker_color="#128C7E")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.dataframe(replies, use_container_width=True, hide_index=True)

    streaks = helper.activity_streaks(view_user, df)
    st.subheader("Activity streaks")
    s1, s2, s3 = st.columns(3)
    s1.metric("Longest consecutive days", streaks["longest"])
    s2.metric("Streak at end of chat", streaks["current"])
    s3.metric("Total active days", streaks["active_days"])


def render_explorer(view_user, df):
    st.subheader("Search messages")
    query = st.text_input("Keyword", placeholder="e.g. heatmap, meeting, 😂")
    matches = helper.search_messages(df, query, view_user)
    st.caption(f"{len(matches):,} matching row(s)" if query else "Showing a recent slice until you search.")
    show = matches.copy()
    if not show.empty:
        show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(show, use_container_width=True, hide_index=True, height=420)

    st.subheader("Export")
    export_df = helper.filter_user(helper.export_frame(df), view_user)
    if len(export_df) > helper.EXPORT_MAX_ROWS:
        st.caption(f"CSV export is capped at {helper.EXPORT_MAX_ROWS:,} rows so the browser does not run out of memory.")
        export_df = export_df.head(helper.EXPORT_MAX_ROWS)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered chat as CSV",
        data=csv,
        file_name="whatsapp_chat_analysis.csv",
        mime="text/csv",
    )


def render_ai_recap(view_user, df):
    st.caption("Local extractive AI — TF-IDF retrieval + VADER + rules. No API key, nothing leaves this session.")
    recap = ai_insights.extractive_recap(df, view_user)
    st.subheader("Auto recap")
    for bullet in recap["bullets"]:
        st.markdown(f"- {bullet}")
    if recap["quotes"] is not None and not recap["quotes"].empty:
        st.subheader("High-signal messages")
        st.caption("Ranked by TF-IDF mass (distinctive wording), not by likes.")
        show = recap["quotes"].copy()
        show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("Ask the chat")
    st.caption("Fact questions (who talks most, mood, reply speed) use metrics. Other questions retrieve similar messages.")
    question = st.text_input("Question", placeholder="e.g. Who starts conversations? What did we say about heatmap?")
    if question.strip():
        answer, cites = ai_insights.answer_question(df, question, view_user)
        st.success(answer)
        if cites is not None and not cites.empty:
            st.dataframe(cites, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Tension over time")
        daily_t, spikes = ai_insights.tension_timeline(df, view_user)
        if daily_t.empty:
            st.info("Not enough text to estimate tension.")
        else:
            fig = px.area(daily_t, x="date", y="tension", title="")
            fig.update_traces(line_color="#ff6b6b", fillcolor="rgba(255,107,107,0.25)")
            st.plotly_chart(style_fig(fig), use_container_width=True)
            if not spikes.empty:
                spikes = spikes.copy()
                spikes["date"] = pd.to_datetime(spikes["date"]).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(spikes, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Activity anomalies")
        st.caption("Days with |z-score| ≥ 2 vs the chat’s own daily average.")
        anomalies = recap["anomalies"]
        if anomalies.empty or "z_score" not in anomalies:
            st.info("Need more days to flag outliers.")
        else:
            fig = px.scatter(anomalies, x="date", y="message", color="flag", title="")
            st.plotly_chart(style_fig(fig), use_container_width=True)
            flagged = anomalies[anomalies["flag"] != "Normal"]
            if flagged.empty:
                st.info("No strong bursts or quiet days in this range.")
            else:
                st.dataframe(flagged, use_container_width=True, hide_index=True)

    st.subheader("Writing-script mix")
    st.caption("Lightweight script detection (Latin, Devanagari, Bangla, …). VADER is English-oriented.")
    langs = ai_insights.language_mix(df, view_user)
    if langs.empty:
        st.info("No text for language mix.")
    else:
        fig = px.pie(langs, names="script", values="messages", hole=0.4)
        st.plotly_chart(style_fig(fig), use_container_width=True)


def render_topics(view_user, df):
    st.caption("Unsupervised NMF topics and TF-IDF semantic search. Runs on-device, no embeddings API.")
    index = ai_insights.build_index(df, view_user)
    if not index:
        st.info("Need more alphabetic text (after stopwords) to fit topics.")
        return

    topics, timeline, examples = ai_insights.topic_model(index)
    if topics.empty:
        st.info("Corpus is too small for NMF. Try Overall or a wider date range.")
    else:
        st.subheader("Discovered topics")
        st.dataframe(topics, use_container_width=True, hide_index=True)
        if not timeline.empty:
            fig = px.area(timeline, x="day", y="messages", color="topic", title="Topic volume over time")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        if not examples.empty:
            st.subheader("Example messages")
            ex = examples.copy()
            ex["date"] = pd.to_datetime(ex["date"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(ex, use_container_width=True, hide_index=True)

    st.subheader("Semantic search")
    query = st.text_input("Find similar meaning", placeholder="e.g. deadline, shipping the app, late night work")
    if query.strip():
        hits = ai_insights.semantic_search(index, query, k=10)
        if hits.empty:
            st.info("No similar messages. Try a more specific phrase.")
        else:
            hits = hits.copy()
            hits["date"] = pd.to_datetime(hits["date"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(hits, use_container_width=True, hide_index=True)

    st.subheader("Dialogue acts")
    intents, by_user = ai_insights.intent_table(df, view_user)
    if intents.empty:
        st.info("No messages to classify.")
        return
    fig = px.pie(intents, names="intent", values="messages", hole=0.4, title="Intent mix")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    if view_user == "Overall" and not by_user.empty:
        fig = px.bar(by_user, x="user", y="messages", color="intent", title="Intents by member")
        st.plotly_chart(style_fig(fig), use_container_width=True)


with st.sidebar:
    st.title("💬 Chat Analyzer")
    st.caption("Turn a WhatsApp export into interview-ready insights.")
    source = st.radio("Data source", ["Upload export", "Try sample chat"], index=0)
    uploaded_file = None
    if source == "Upload export":
        uploaded_file = st.file_uploader(
            "WhatsApp .txt export",
            type=["txt"],
            help=f"Up to {MAX_UPLOAD_MB} MB. Use Export chat → Without media. Zip/media backups are rejected.",
        )
        st.markdown(
            f"<small>WhatsApp → Chat → Export chat → Without media · max {MAX_UPLOAD_MB} MB</small>",
            unsafe_allow_html=True,
        )

df = None
upload_error = None
if source == "Try sample chat":
    df = parse_sample()
elif uploaded_file is not None:
    upload_error = validate_upload(uploaded_file)
    if upload_error is None:
        df, upload_error = parse_uploaded_file(uploaded_file)
        if df is not None and df.empty:
            df = None

if upload_error:
    st.error(upload_error)
elif df is None and source == "Upload export" and uploaded_file is None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>WhatsApp Chat Analyzer</h1>
            <p>
                Upload a chat export or load the built-in sample to explore activity, VADER sentiment,
                local TF-IDF topics, semantic search, and extractive Q&amp;A — no API key required.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    a.markdown("**1. Parse**\n\nAndroid and iOS timestamps, media, deleted messages.")
    b.markdown("**2. Measure**\n\nVolume, streaks, emoji, words, links, and mood.")
    c.markdown("**3. Explain**\n\nWho starts chats, who replies fastest, when the group is awake.")
    st.info("Use the sidebar: upload a `.txt` export or click **Try sample chat**.")
elif df is None:
    st.error("Could not parse any messages. Export the chat as a .txt file (without media) and try again.")
else:
    user_list = sorted(u for u in df["user"].unique() if u != "group_notification")
    with st.sidebar:
        selected_user = st.selectbox("Analyze", ["Overall"] + user_list)
        min_day = df["only_date"].min()
        max_day = df["only_date"].max()
        picked = st.date_input("Date range", value=(min_day, max_day), min_value=min_day, max_value=max_day)
        if isinstance(picked, tuple) and len(picked) == 2:
            start_day, end_day = picked
            df = df.loc[(df["only_date"] >= start_day) & (df["only_date"] <= end_day)]

    if df.empty:
        st.warning("No messages in that date range.")
    else:
        if len(df) > helper.NLP_SAMPLE_ROWS:
            st.caption(
                f"Parsed {len(df):,} messages. Word cloud, emojis, and sentiment use a "
                f"{helper.NLP_SAMPLE_ROWS:,}-message sample so Streamlit Cloud stays within RAM."
            )
        tabs = st.tabs(
            ["Overview", "Activity", "Members", "Content", "Sentiment", "Insights", "AI Recap", "Topics", "Explorer"]
        )
        with tabs[0]:
            render_overview(selected_user, df)
        with tabs[1]:
            render_activity(selected_user, df)
        with tabs[2]:
            render_members(selected_user, df)
        with tabs[3]:
            render_content(selected_user, df)
        with tabs[4]:
            render_sentiment(selected_user, df)
        with tabs[5]:
            render_insights(selected_user, df)
        with tabs[6]:
            render_ai_recap(selected_user, df)
        with tabs[7]:
            render_topics(selected_user, df)
        with tabs[8]:
            render_explorer(selected_user, df)

st.sidebar.markdown(
    """
    <div class='sidebar-footer'>
        <div style='text-align: center; font-size: 14px;'>
            Made with ❤️ by
            <b><a href='https://www.linkedin.com/in/pramaniksandip/' target='_blank' style='text-decoration: none; color: #25D366; font-weight: bold;'>
                Sandip
            </a></b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

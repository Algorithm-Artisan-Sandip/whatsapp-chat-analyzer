# WhatsApp Chat Analyzer

A Streamlit dashboard that turns a WhatsApp `.txt` export into product-style analytics: volume, member behavior, language, mood, and conversation dynamics.

Built for a portfolio: it is not only charts, it is a small end-to-end data product (ingest → clean → analyze → interactive UI → export).

## Features

- **Robust ingest** for Android and iOS exports (12/24-hour time, 2/4-digit years, Unicode spaces, media and deleted messages)
- **Overview KPIs**: messages, words, media, links, members, messages/day, deleted messages, activity streaks
- **Activity**: daily/monthly trends, weekday and hour patterns, chronotype mix, day×hour heatmap
- **Members**: leaderboard (messages, words, media, emojis, average length) and a comparison radar
- **Content**: word cloud, frequent words, emoji mix, longest messages
- **Sentiment**: VADER scores overall, by day, and by member
- **Insights**: who starts conversations after a quiet gap, median reply time
- **Explorer**: keyword search and CSV download
- **Sample chat** so anyone can demo the app without uploading a personal backup

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the sidebar, choose **Try sample chat**, or export a real chat:

WhatsApp → Chat → **Export chat** → **Without media** → upload the `.txt` file.

## Tech stack

Python, Pandas, Streamlit, Plotly, Matplotlib, WordCloud, VADER sentiment, URLExtract, emoji

## Resume bullets you can use

- Built an end-to-end WhatsApp analytics web app: custom parser, feature engineering, interactive Plotly dashboard, and CSV export
- Implemented conversation-level metrics (reply latency, conversation starters, streaks) on top of standard EDA charts
- Added VADER sentiment trending and empty-state guards so live demos do not crash on sparse chats

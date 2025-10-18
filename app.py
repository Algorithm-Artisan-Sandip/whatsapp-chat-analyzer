import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import preprocessor, helper

# Custom CSS to fix the sidebar footer at the bottom
st.markdown("""
    <style>
    /* Target the sidebar */
    [data-testid="stSidebar"] {
        position: relative;
    }
    
    /* Create a fixed footer container */
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: inherit;
        background-color: inherit;
        padding: 1rem;
        border-top: 1px solid rgba(250, 250, 250, 0.2);
        z-index: 999;
    }
    
    /* Add padding to sidebar content to prevent overlap with fixed footer */
    [data-testid="stSidebar"] > div:first-child {
        padding-bottom: 80px;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocessor(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis for", user_list)

    if st.sidebar.button("Show Analysis"):
        # stats area
        num_messages, words, media_messages, links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(media_messages)
        with col4:
            st.header("Links Shared")
            st.title(links)


        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timeline['time'], timeline['message'], color='green')

        plt.xticks(rotation=45, ha='right', fontsize=8)

        if len(timeline) > 20:
            every_nth = len(timeline) // 20
            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)

        st.pyplot(fig)


        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_timeline['date'], daily_timeline['message'], color='orange')

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))

        plt.xticks(rotation=45, ha='right', fontsize=8)

        fig.tight_layout()
        st.pyplot(fig)


        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(user_heatmap, cmap='YlOrBr', ax=ax)

        ax.set_xlabel("Time of Day (Hour Range)", fontsize=12)
        ax.set_ylabel("Day of the Week", fontsize=12)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

        st.pyplot(fig)


        # finding the busiest users in the group
        if selected_user == 'Overall':
            st.title("Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)


        # wordcloud
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc, interpolation='bilinear')
        st.title("Word Cloud")
        st.pyplot(fig)


        # most common words
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        st.title("Most Common Words")
        ax.barh(most_common_df['word'], most_common_df['count'], color='orange')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)

        with col2:
            top_emojis = emoji_df.head(10)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                top_emojis['count'],
                labels=top_emojis['emoji'],
                autopct="%0.2f%%",
                startangle=90,
                textprops={'fontsize': 14}
            )
            ax.axis('equal')
            st.pyplot(fig)


# ----- FIXED SIDEBAR FOOTER -----
st.sidebar.markdown(
    """
    <div class='sidebar-footer'>
        <div style='text-align: center; font-size: 14px;'>
            Made with ❤️ by 
            <b><a href='https://www.linkedin.com/in/sandip-pramanik' target='_blank' style='text-decoration: none; color: #ff4b4b; font-weight: bold;'>
                Sandip
            </a></b>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
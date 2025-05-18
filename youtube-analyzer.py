import streamlit as st
from googleapiclient.discovery import build
from datetime import datetime, timezone
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(
    page_title="YouTube Analyzer Advanced",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.google.com',
        'About': "YouTube Analyzer dengan fitur lebih lengkap dari VidIQ dan TubeBuddy"
    }
)

# --- Session State ---
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = []
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'keyword' not in st.session_state:
    st.session_state.keyword = ''

# --- Sidebar: API Key Management ---
st.sidebar.title("⚙️ Settings")
with st.sidebar.expander("API Keys", expanded=True):
    new_key = st.text_input("Tambah API Key baru", type="password")
    if st.button("➕ Tambah"):
        if len(st.session_state.api_keys) >= 5:
            st.warning("Maksimal 5 API Key")
        elif not new_key.strip():
            st.warning("API Key tidak boleh kosong")
        else:
            st.session_state.api_keys.append(new_key.strip())
            st.success("API Key berhasil ditambahkan")
    for idx, key in enumerate(st.session_state.api_keys):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text(f"{idx+1}. ************{key[-5:]}")
        with col2:
            if st.button("🗑️", key=f"del_{idx}"):
                st.session_state.api_keys.pop(idx)
                st.experimental_rerun()
    if not st.session_state.api_keys:
        st.info("Belum ada API Key tersimpan.")

# --- Pilih API Key ---
selected_key = None
if st.session_state.api_keys:
    selected_idx = st.sidebar.selectbox(
        "Pilih API Key",
        options=range(len(st.session_state.api_keys)),
        format_func=lambda i: f"API Key {i+1} (****{st.session_state.api_keys[i][-5:]})"
    )
    selected_key = st.session_state.api_keys[selected_idx]

# --- Fungsi untuk mengambil data dari YouTube API ---
@st.cache_data(ttl=1800)
def get_youtube_videos(api_key, keyword, max_results=30, order="relevance"):
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        search_response = youtube.search().list(
            q=keyword,
            part="id,snippet",
            maxResults=max_results,
            order=order,
            type="video"
        ).execute()
        video_ids = [item['id']['videoId'] for item in search_response['items']]
        videos_response = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids)
        ).execute()
        videos = []
        for item in videos_response['items']:
            published = item['snippet']['publishedAt']
            published_dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            days_old = max((datetime.now(timezone.utc) - published_dt).days, 1)
            views = int(item['statistics'].get('viewCount', 0))
            videos.append({
                "video_id": item['id'],
                "title": item['snippet']['title'],
                "channel": item['snippet']['channelTitle'],
                "published": published_dt.strftime("%Y-%m-%d"),
                "views": views,
                "likes": int(item['statistics'].get('likeCount', 0)),
                "comments": int(item['statistics'].get('commentCount', 0)),
                "views_per_day": views / days_old,
                "days_old": days_old
            })
        return videos
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []

# --- Main App UI ---
st.title("🔎 YouTube Analyzer Advanced")

with st.form("search_form"):
    keyword = st.text_input("Masukkan Judul / Keyword YouTube", value=st.session_state.keyword)
    filter_order = st.radio("Filter Video:", ["Relevance", "Newest", "Popular"], horizontal=True)
    submitted = st.form_submit_button("Cari Video")

if submitted:
    if not selected_key:
        st.error("Pilih API Key dulu di sidebar.")
    elif not keyword.strip():
        st.warning("Masukkan keyword dulu.")
    else:
        st.session_state.keyword = keyword.strip()
        order_map = {"Relevance": "relevance", "Newest": "date", "Popular": "viewCount"}
        with st.spinner("Mengambil data..."):
            videos = get_youtube_videos(selected_key, st.session_state.keyword, order=order_map[filter_order])
            if videos:
                df = pd.DataFrame(videos)
                st.success(f"Ditemukan {len(df)} video untuk '{st.session_state.keyword}'")
                # Show summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Rata-rata Views", f"{int(df['views'].mean()):,}")
                col2.metric("Rata-rata Views/Hari", f"{int(df['views_per_day'].mean()):,}")
                col3.metric("Video Terlama (hari)", f"{df['days_old'].max()}")

                # Dataframe
                st.dataframe(df[['title', 'channel', 'published', 'views', 'likes', 'comments', 'views_per_day']])

                # Sentiment analysis of titles
                df['sentiment'] = df['title'].apply(lambda t: TextBlob(t).sentiment.polarity)
                st.subheader("Analisis Sentimen Judul Video")
                fig, ax = plt.subplots()
                ax.hist(df['sentiment'], bins=20, color='skyblue')
                ax.set_xlabel("Sentimen")
                ax.set_ylabel("Jumlah Video")
                st.pyplot(fig)

                # WordCloud
                st.subheader("Word Cloud Judul Video")
                all_titles = " ".join(df['title'])
                wc = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)

                # Export data to CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Data CSV",
                    data=csv,
                    file_name=f"youtube_{keyword}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Tidak ditemukan data video.")

else:
    st.info("Masukkan keyword dan pilih API Key di sidebar, lalu klik Cari Video.")

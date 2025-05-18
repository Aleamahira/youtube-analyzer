import streamlit as st
from googleapiclient.discovery import build
from datetime import datetime, timezone
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import openai

# --- CONFIG ---
st.set_page_config(page_title="YouTube Analyzer + AI Features", layout="wide")

# --- SESSION STATE ---
if "api_keys" not in st.session_state:
    st.session_state.api_keys = []
if "selected_key_idx" not in st.session_state:
    st.session_state.selected_key_idx = 0

# --- SIDEBAR: API Keys and OpenAI key ---
st.sidebar.title("Settings & API Keys")
# Input Youtube API Keys
new_key = st.sidebar.text_input("Tambah YouTube API Key (max 5)", type="password")
if st.sidebar.button("Tambah API Key"):
    if len(st.session_state.api_keys) >= 5:
        st.sidebar.warning("Maksimal 5 API Key.")
    elif new_key.strip():
        st.session_state.api_keys.append(new_key.strip())
        st.sidebar.success("API Key ditambahkan.")
# Pilih API Key
if st.session_state.api_keys:
    st.session_state.selected_key_idx = st.sidebar.selectbox(
        "Pilih API Key",
        options=range(len(st.session_state.api_keys)),
        format_func=lambda i: f"Key {i+1} (****{st.session_state.api_keys[i][-5:]})",
        index=st.session_state.selected_key_idx,
    )
youtube_api_key = st.session_state.api_keys[st.session_state.selected_key_idx] if st.session_state.api_keys else None

# Input OpenAI API Key for AI features
openai_api_key = st.sidebar.text_input("OpenAI API Key (untuk AI rekomendasi judul)", type="password")

# --- YouTube API Helper ---
def get_youtube_client():
    if youtube_api_key:
        return build("youtube", "v3", developerKey=youtube_api_key)
    else:
        return None

def fetch_videos(youtube, query=None, channel_id=None, max_results=20, order="relevance"):
    try:
        if channel_id:
            # Get videos from channel
            search_response = youtube.search().list(
                channelId=channel_id,
                part="id,snippet",
                maxResults=max_results,
                order="date"
            ).execute()
        else:
            search_response = youtube.search().list(
                q=query,
                part="id,snippet",
                maxResults=max_results,
                order=order
            ).execute()
        video_ids = [item['id']['videoId'] for item in search_response['items'] if item['id']['kind']=='youtube#video']
        if not video_ids:
            return []
        videos_response = youtube.videos().list(
            id=",".join(video_ids),
            part="snippet,statistics"
        ).execute()
        videos = []
        for item in videos_response["items"]:
            published = item['snippet']['publishedAt']
            published_dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            days_old = (datetime.now(timezone.utc) - published_dt).days + 1
            views = int(item['statistics'].get('viewCount', 0))
            likes = int(item['statistics'].get('likeCount', 0))
            comments = int(item['statistics'].get('commentCount', 0))
            videos.append({
                "video_id": item['id'],
                "title": item['snippet']['title'],
                "channel_title": item['snippet']['channelTitle'],
                "published": published_dt.strftime("%Y-%m-%d"),
                "days_old": days_old,
                "views": views,
                "likes": likes,
                "comments": comments,
                "views_per_day": views / days_old if days_old > 0 else views,
                "thumbnail": item['snippet']['thumbnails']['medium']['url']
            })
        return videos
    except Exception as e:
        st.error(f"Error fetching videos: {e}")
        return []

# --- AI Title Generator ---
def generate_ai_titles(prompt, openai_key, n=5):
    import openai
    openai.api_key = openai_key
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Buat {n} judul video YouTube menarik dan SEO friendly berdasarkan keyword: {prompt}",
            max_tokens=150,
            n=n,
            stop=None,
            temperature=0.7,
        )
        titles = [choice['text'].strip() for choice in response['choices']]
        return titles
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return []

# --- Prediksi sederhana berdasarkan views_per_day rata-rata ---
def predict_performance(df):
    if df.empty:
        return "Data tidak cukup untuk prediksi."
    avg_vpd = df["views_per_day"].mean()
    if avg_vpd > 1000:
        return "Performa video diperkirakan sangat baik (views/hari tinggi)."
    elif avg_vpd > 100:
        return "Performa video diperkirakan baik."
    else:
        return "Performa video diperkirakan rendah."

# --- MAIN UI ---
st.title("YouTube Analyzer dengan Fitur AI dan Kompetitor")

if not youtube_api_key:
    st.warning("Masukkan YouTube API Key di sidebar untuk menggunakan aplikasi ini.")
    st.stop()

youtube = get_youtube_client()
if not youtube:
    st.error("Gagal membuat klien YouTube API.")
    st.stop()

with st.form("search_form"):
    query = st.text_input("Masukkan Keyword pencarian YouTube")
    channel_id = st.text_input("Atau masukkan Channel ID kompetitor (opsional)")
    submitted = st.form_submit_button("Cari")

if submitted:
    videos_main = []
    if query:
        videos_main = fetch_videos(youtube, query=query, max_results=30)
    elif channel_id:
        videos_main = fetch_videos(youtube, channel_id=channel_id, max_results=30)
    else:
        st.warning("Masukkan keyword pencarian atau Channel ID")
        st.stop()

    if not videos_main:
        st.info("Tidak ada video ditemukan.")
    else:
        df = pd.DataFrame(videos_main)
        st.subheader("Hasil Pencarian Video")
        # Tabel data dengan link ke YouTube
        def make_link(row):
            url = f"https://www.youtube.com/watch?v={row['video_id']}"
            return f"[{row['title'][:60]}...]({url})"
        df["video_link"] = df.apply(make_link, axis=1)
        st.dataframe(df[["video_link", "channel_title", "views", "likes", "comments", "published"]], height=400)

        # Thumbnail grid dengan link
        st.subheader("Thumbnail Preview")
        cols = st.columns(5)
        for i, video in enumerate(videos_main[:20]):
            with cols[i % 5]:
                st.markdown(f"[![Thumbnail]( {video['thumbnail']} )](https://www.youtube.com/watch?v={video['video_id']})")

        # Analisis Sentimen judul
        st.subheader("Analisis Sentimen Judul")
        df["sentiment"] = df["title"].apply(lambda t: TextBlob(t).sentiment.polarity)
        fig, ax = plt.subplots()
        ax.hist(df["sentiment"], bins=15, color="skyblue")
        ax.set_xlabel("Sentimen")
        ax.set_ylabel("Jumlah Video")
        st.pyplot(fig)

        # Wordcloud judul
        st.subheader("Wordcloud Judul Video")
        text_all = " ".join(df["title"])
        wc = WordCloud(width=800, height=400, background_color="white").generate(text_all)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # Prediksi performa video
        st.subheader("Prediksi Performa Video (Sederhana)")
        pred = predict_performance(df)
        st.info(pred)

        # AI Judul Generator
        if openai_api_key:
            st.subheader("Rekomendasi Judul AI")
            ai_titles = generate_ai_titles(query if query else channel_id, openai_api_key, n=5)
            for i, title in enumerate(ai_titles, 1):
                st.markdown(f"{i}. **{title}**")
        else:
            st.info("Masukkan OpenAI API Key di sidebar untuk rekomendasi judul AI.")

        # Analisis channel kompetitor jika ada channel_id
        if channel_id:
            st.subheader("Video Kompetitor (Channel ID diberikan)")
            videos_comp = fetch_videos(youtube, channel_id=channel_id, max_results=15)
            if videos_comp:
                df_comp = pd.DataFrame(videos_comp)
                df_comp["video_link"] = df_comp.apply(make_link, axis=1)
                st.dataframe(df_comp[["video_link", "views", "likes", "comments", "published"]])
                # Thumbnail kompetitor
                st.subheader("Thumbnail Kompetitor")
                cols2 = st.columns(5)
                for i, video in enumerate(videos_comp[:20]):
                    with cols2[i % 5]:
                        st.markdown(f"[![Thumbnail]( {video['thumbnail']} )](https://www.youtube.com/watch?v={video['video_id']})")
            else:
                st.info("Tidak ditemukan video kompetitor.")

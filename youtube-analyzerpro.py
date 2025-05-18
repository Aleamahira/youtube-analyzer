import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import openai
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# ------------------- SIDEBAR ---------------------
st.sidebar.title("🔐 API Configuration")
yt_api_key = st.sidebar.text_input("🔑 YouTube API Key", type="password")
openai_api_key = st.sidebar.text_input("🧠 OpenAI API Key", type="password")
openai.api_key = openai_api_key

# ------------------- FUNCTION ---------------------
@st.cache_data(show_spinner=False)
def get_channel_videos(channel_id, max_results=30):
    youtube = build('youtube', 'v3', developerKey=yt_api_key)
    req = youtube.search().list(part='snippet', channelId=channel_id, maxResults=max_results, order='date')
    res = req.execute()
    videos = []
    for item in res['items']:
        if item['id']['kind'] == 'youtube#video':
            videos.append({
                'title': item['snippet']['title'],
                'videoId': item['id']['videoId'],
                'publishedAt': item['snippet']['publishedAt'],
                'thumbnail': item['snippet']['thumbnails']['medium']['url']
            })
    return videos

def search_videos_by_keyword(keyword, max_results=30):
    youtube = build('youtube', 'v3', developerKey=yt_api_key)
    req = youtube.search().list(part='snippet', q=keyword, maxResults=max_results, order='relevance')
    res = req.execute()
    videos = []
    for item in res['items']:
        if item['id']['kind'] == 'youtube#video':
            videos.append({
                'title': item['snippet']['title'],
                'videoId': item['id']['videoId'],
                'channelTitle': item['snippet']['channelTitle'],
                'publishedAt': item['snippet']['publishedAt'],
                'thumbnail': item['snippet']['thumbnails']['medium']['url']
            })
    return videos

def generate_wordcloud(titles):
    text = " ".join(titles)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def recommend_with_openai(prompt):
    if not openai_api_key:
        return "Masukkan API Key OpenAI dulu."
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ------------------- UI ---------------------
st.title("📊 YouTube Analyzer Pro")

menu = st.selectbox("🔍 Pilih Mode Analisa", ["🔑 Berdasarkan Keyword", "📺 Berdasarkan Channel ID"])

if menu == "📺 Berdasarkan Channel ID":
    channel_id = st.text_input("Masukkan Channel ID YouTube:")
    if channel_id and yt_api_key:
        videos = get_channel_videos(channel_id)
        if videos:
            df = pd.DataFrame(videos)
            st.subheader("📹 Video Terbaru")
            for _, row in df.iterrows():
                st.image(row['thumbnail'], width=300)
                st.markdown(f"**{row['title']}**")
                st.markdown(f"[Tonton di YouTube](https://youtube.com/watch?v={row['videoId']})")

            st.subheader("☁️ WordCloud Judul Video")
            wc = generate_wordcloud(df['title'].tolist())
            st.image(wc.to_array())

            st.subheader("🧠 Rekomendasi Judul AI")
            title_text = "\n".join(df['title'].tolist())
            prompt = f"Berdasarkan judul-judul berikut:\n{title_text}\nBuatlah 5 rekomendasi judul baru yang menarik dan SEO friendly:"
            ai_titles = recommend_with_openai(prompt)
            st.write(ai_titles)

            st.subheader("🏷️ Rekomendasi Tag & Strategi")
            tag_prompt = f"Berdasarkan channel dengan judul-judul:\n{title_text}\nBerikan rekomendasi tag relevan dan strategi konten yang sedang tren:"
            tag_response = recommend_with_openai(tag_prompt)
            st.write(tag_response)

elif menu == "🔑 Berdasarkan Keyword":
    keyword = st.text_input("Masukkan Keyword Pencarian:")
    if keyword and yt_api_key:
        results = search_videos_by_keyword(keyword)
        if results:
            df = pd.DataFrame(results)
            st.subheader("🔥 Video Populer Berdasarkan Keyword")
            for _, row in df.iterrows():
                st.image(row['thumbnail'], width=300)
                st.markdown(f"**{row['title']}** by {row['channelTitle']}")
                st.markdown(f"[Tonton di YouTube](https://youtube.com/watch?v={row['videoId']})")

            st.subheader("☁️ WordCloud Judul Video")
            wc = generate_wordcloud(df['title'].tolist())
            st.image(wc.to_array())

            st.subheader("🧠 AI Rekomendasi Judul")
            keyword_titles = "\n".join(df['title'].tolist())
            prompt = f"Berikut daftar judul video berdasarkan keyword '{keyword}':\n{keyword_titles}\nBuat 5 judul baru yang viral dan cocok untuk YouTube:"
            ai_titles = recommend_with_openai(prompt)
            st.write(ai_titles)

            st.subheader("📈 Strategi & Ide Konten")
            prompt2 = f"Apa saja ide konten, tag, dan strategi untuk keyword '{keyword}' berdasarkan data ini:\n{keyword_titles}"
            tag_response = recommend_with_openai(prompt2)
            st.write(tag_response)

# ------------------ FOOTER -------------------
st.markdown("---")
st.markdown("🚀 Dibuat oleh YouTube Analyzer Pro · Powered by OpenAI & YouTube Data API")

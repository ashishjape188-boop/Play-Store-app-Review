import pandas as pd
import numpy as np
import re
import emoji

from google_play_scraper import reviews, search
from textblob import TextBlob

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import gradio as gr

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# 🔍 Get App ID from Name
def get_app_id(app_input):

    if "." in app_input:
        return app_input

    result = search(
        app_input,
        lang="en",
        country="in",
        n_hits=5
    )

    for app in result:

        if app.get("appId"):

            print("Selected App:", app["title"])
            print("App ID:", app["appId"])

            return app["appId"]

    raise ValueError("App not found.")


# ⭐ Extract Star Value
def extract_star_value(star_label):

    if star_label == "All Stars":
        return "All"

    if "1" in star_label:
        return 1
    elif "2" in star_label:
        return 2
    elif "3" in star_label:
        return 3
    elif "4" in star_label:
        return 4
    elif "5" in star_label:
        return 5


# 🧹 Clean Text
def clean_text(text):

    text = str(text).lower()

    text = emoji.replace_emoji(text, replace="")

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^\w\s]", "", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


# 🚀 Main Function
def analyze_app(app_input, star_label, review_count):

    try:

        app_id = get_app_id(app_input)

        star_filter = extract_star_value(star_label)

        review_count = int(review_count)

        result, _ = reviews(
            app_id,
            lang="en",
            country="in",
            count=review_count
        )

        df = pd.DataFrame(result)

        df = df[["content", "score"]]

        # ⭐ Apply Star Filter
        if star_filter != "All":

            df = df[df["score"] == star_filter]

        if len(df) == 0:
            raise ValueError("No reviews found.")

        # Clean
        df["cleaned"] = df["content"].apply(clean_text)

        # Sentiment
        df["polarity"] = df["cleaned"].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )

        df["Sentiment"] = np.where(
            df["polarity"] >= 0,
            "Positive",
            "Negative"
        )

        sentiment_counts = df["Sentiment"].value_counts()

        # 🍩 Donut Chart
        plt.figure(figsize=(5,5))

        colors = ["#1f77b4", "#ff7f0e"]

        wedges, texts, autotexts = plt.pie(
            sentiment_counts,
            autopct="%0.0f%%",
            startangle=90,
            colors=colors,
            radius=0.78,
            wedgeprops=dict(width=0.26),
            pctdistance=0.72
        )

        for t in texts:
            t.set_visible(False)

        for autotext in autotexts:
            autotext.set_color("black")
            autotext.set_fontsize(12)

        total_reviews = len(df)

        plt.text(
            0, 0,
            f"{total_reviews}\nReviews",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold"
        )

        plt.legend(
            wedges,
            sentiment_counts.index,
            title="Sentiment",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=2
        )

        plt.axis("equal")

        pie_path = "sentiment_donut.png"

        plt.savefig(
            pie_path,
            transparent=True,
            bbox_inches="tight",
            pad_inches=1.2
        )

        plt.close()

        # ⭐ Split Positive & Negative Text

        positive_text = " ".join(
            df[df["Sentiment"] == "Positive"]["cleaned"]
        )

        negative_text = " ".join(
            df[df["Sentiment"] == "Negative"]["cleaned"]
        )

        # ☁️ Positive Word Cloud

        positive_wc = WordCloud(
            background_color=None,
            mode="RGBA",
            width=600,
            height=300,
            colormap="Greens"
        ).generate(positive_text)

        positive_path = "positive_wc.png"

        positive_wc.to_file(positive_path)

        # ☁️ Negative Word Cloud

        negative_wc = WordCloud(
            background_color=None,
            mode="RGBA",
            width=600,
            height=300,
            colormap="Reds"
        ).generate(negative_text)

        negative_path = "negative_wc.png"

        negative_wc.to_file(negative_path)

        return pie_path, positive_path, negative_path

    except Exception as e:

        print("ERROR:", e)

        return None, None, None


# 🎯 UI Layout
with gr.Blocks() as interface:

    gr.Markdown(
        "# 📱 Play Store Review Sentiment Analyzer"
    )

    gr.Markdown(
        "Enter App Name (PhonePe) or App ID (com.phonepe.app)"
    )

    app_input = gr.Textbox(
        label="Enter App Name"
    )

    with gr.Row():

        star_input = gr.Radio(
            choices=[
                "All Stars",
                "☆ 1 Star",
                "☆☆ 2 Stars",
                "☆☆☆ 3 Stars",
                "☆☆☆☆ 4 Stars",
                "☆☆☆☆☆ 5 Stars"
            ],
            label="Select Star Rating",
            value="All Stars"
        )

        review_input = gr.Dropdown(
            choices=["500","1000","1500","2000"],
            label="Number of Reviews",
            value="500"
        )

    submit_btn = gr.Button("Submit")

    # Donut Row
    with gr.Row():

        pie_output = gr.Image(
            label="Sentiment Donut Chart"
        )

    # Word Clouds Row
    with gr.Row():

        positive_output = gr.Image(
            label="Positive Word Cloud"
        )

        negative_output = gr.Image(
            label="Negative Word Cloud"
        )

    submit_btn.click(
        analyze_app,
        inputs=[
            app_input,
            star_input,
            review_input
        ],
        outputs=[
            pie_output,
            positive_output,
            negative_output
        ]
    )


interface.launch()
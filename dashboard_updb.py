#!/usr/bin/env python
"""
UPDB Comprehensive Battlespace-Aware Dashboard
================================================

This dashboard is built for your UPDB dataset. It requires a CSV file named "updb.csv"
with the following columns:
    city, country, Month, Day, Year, Time, AMPM, date_detail, description, district,
    geoname_id, id, latitude, location, longitude, other, source, source_id, ts, water, word_count

The script performs the following:
  • Creates a new datetime column ("Datetime") from Year, Month, Day, Time, and AMPM.
  • Converts latitude and longitude to numeric.
  • Ensures the description column is a string.
  • Uses existing or computed columns such as comment_length and extracted_keywords.
  • Adds new computed columns:
         - comment_length: word count of "description"
         - extracted_keywords: top 3 keywords from "description" (using RAKE)
  • Provides multiple interactive tabs:
       - Data Overview: Shows raw data, column types, and summary statistics.
       - Visualizations: Interactive scatter plots, histograms, and a geospatial map.
       - Correlation Analysis: Heatmap of numeric correlations.
       - Advanced Analysis: Clustering on selected numeric columns, word cloud, and 3D scatter plot.
       - Search Descriptions: Search the "description" text by keywords (with highlighted matches).
       - NLP Insights: Generate summary and sentiment analysis for sample descriptions.
       - Battlespace Awareness: Filters events for threat keywords (e.g., "attack", "bomb", etc.),
         and displays a table, timeline, and geospatial map for situational awareness.

Instructions:
  1. Install required packages.
  2. Place "updb.csv" in the same folder as this script.
  3. Run the script: python dashboard_updb.py
  4. Access the app at http://127.0.0.1:8051/

This script is designed to run in any Python environment, including a Conda VM.
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from wordcloud import WordCloud
import base64
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import folium
from folium.plugins import MarkerCluster
import os
from rake_nltk import Rake
import nltk
from transformers import pipeline

# Download required NLTK resources for RAKE
nltk.download('stopwords')
nltk.download('punkt')

# Initialize RAKE for keyword extraction
rake_extractor = Rake()

# Initialize NLP pipelines for insights (optional but recommended in production to specify model versions)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", revision="main")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="main")

# -----------------------------
# Data Loading & Preparation
# -----------------------------
df = pd.read_csv("updb.csv", low_memory=False)

# Convert latitude and longitude columns to numeric
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# Create a new datetime column ("Datetime") using Year, Month, Day, Time, and AMPM.
df["Datetime"] = pd.to_datetime(
    df["Year"].astype(str) + "-" +
    df["Month"].astype(str).str.zfill(2) + "-" +
    df["Day"].astype(str).str.zfill(2) + " " +
    df["Time"].astype(str) + " " +
    df["AMPM"].astype(str),
    format='%m/%d/%Y', errors='coerce'
)
# Replace out-of-bound dates with NaT (e.g. dates before 1677-09-21)
df.loc[df["Datetime"] < pd.Timestamp("1677-09-21"), "Datetime"] = pd.NaT

# Ensure "description" is a string.
df["description"] = df["description"].astype(str)

# -----------------------------
# Compute Additional Columns
# -----------------------------
# Compute comment length as word count.
df["comment_length"] = df["description"].apply(lambda x: len(x.split()))
# Extract top 3 keywords from description using RAKE.
def extract_keywords(text):
    rake_extractor.extract_keywords_from_text(text)
    keywords = rake_extractor.get_ranked_phrases()
    return ", ".join(keywords[:3]) if keywords else ""
df["extracted_keywords"] = df["description"].apply(extract_keywords)

# -----------------------------
# Data Overview Objects
# -----------------------------
dtypes_df = pd.DataFrame({"Column": df.columns, "DataType": df.dtypes.astype(str)})
numeric_df = df.select_dtypes(include=["number"])
numeric_summary = numeric_df.describe().reset_index() if not numeric_df.empty else pd.DataFrame()
categorical_df = df.select_dtypes(include=["object"])
categorical_summary = categorical_df.describe().reset_index() if not categorical_df.empty else pd.DataFrame()

numeric_columns = numeric_df.columns.tolist()
categorical_columns = categorical_df.columns.tolist()

# -----------------------------
# Helper Functions
# -----------------------------
def generate_wordcloud(text):
    if not text.strip():
        return ""
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"

def perform_clustering(texts, n_clusters=5):
    if len(texts) < n_clusters:
        n_clusters = len(texts)
    if n_clusters == 0:
        return [], None
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X)
    return labels, km

def generate_folium_map(dataframe):
    if "latitude" in dataframe.columns and "longitude" in dataframe.columns:
        lat_center = dataframe["latitude"].mean()
        lon_center = dataframe["longitude"].mean()
    else:
        lat_center, lon_center = 0, 0
    m = folium.Map(location=[lat_center, lon_center], zoom_start=2)
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in dataframe.iterrows():
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.notnull(lat) and pd.notnull(lon):
            popup_text = (
                f"<b>Description:</b> {row.get('description', 'N/A')}<br>"
                f"<b>City:</b> {row.get('city', 'N/A')}<br>"
                f"<b>Country:</b> {row.get('country', 'N/A')}"
            )
            folium.Marker(location=[lat, lon], popup=popup_text).add_to(marker_cluster)
    return m

def highlight_keywords(text, keywords):
    for kw in keywords:
        if kw:
            text = re.sub(f"({re.escape(kw)})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    return text

def get_nlp_insights():
    sample_text = " ".join(df["description"].head(50).tolist())
    try:
        summary = summarizer(sample_text, max_length=200, min_length=50, do_sample=False)
        summary_text = summary[0]["summary_text"]
    except Exception as e:
        summary_text = f"Error during summarization: {e}"
    try:
        sentiment = sentiment_analyzer(sample_text[:512])
        sentiment_result = sentiment[0]["label"]
    except Exception as e:
        sentiment_result = f"Error during sentiment analysis: {e}"
    return summary_text, sentiment_result

# -----------------------------
# Battlespace Awareness Helper
# -----------------------------
def get_battlespace_events(dataframe):
    threat_keywords = ["attack", "bomb", "explosion", "assault", "conflict", "military", "strike", "hostile", "alert"]
    pattern = "|".join(re.escape(kw) for kw in threat_keywords)
    mask = dataframe["description"].str.contains(pattern, case=False, na=False)
    return dataframe[mask]

# -----------------------------
# Build the Dash App with Tabs
# -----------------------------
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
app.title = "UPDB Comprehensive Dashboard"

app.layout = html.Div([
    html.H1("UPDB Comprehensive Dashboard", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Data Overview", value="overview"),
        dcc.Tab(label="Visualizations", value="visualizations"),
        dcc.Tab(label="Correlation Analysis", value="correlation"),
        dcc.Tab(label="Advanced Analysis", value="advanced"),
        dcc.Tab(label="Search Descriptions", value="search"),
        dcc.Tab(label="NLP Insights", value="nlp"),
        dcc.Tab(label="Battlespace Awareness", value="battlespace")
    ]),
    html.Div(id="tab-content")
], style={"fontFamily": "Arial, sans-serif", "padding": "20px"})

# -----------------------------
# Callback: Render Tab Content
# -----------------------------
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_tab_content(tab):
    if tab == "overview":
        return html.Div([
            html.H2("Raw Data"),
            dash_table.DataTable(
                id="raw-data-table",
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict("records"),
                page_size=10,
                style_table={"overflowX": "auto"}
            ),
            html.Br(),
            html.H2("Column Data Types"),
            dash_table.DataTable(
                id="dtypes-table",
                columns=[{"name": "Column", "id": "Column"}, {"name": "DataType", "id": "DataType"}],
                data=dtypes_df.to_dict("records"),
                page_size=20,
                style_table={"overflowX": "auto"}
            ),
            html.Br(),
            html.H2("Numeric Summary"),
            dash_table.DataTable(
                id="numeric-summary-table",
                columns=[{"name": col, "id": col} for col in numeric_summary.columns],
                data=numeric_summary.to_dict("records") if not numeric_summary.empty else [],
                page_size=10,
                style_table={"overflowX": "auto"}
            ),
            html.Br(),
            html.H2("Categorical Summary"),
            dash_table.DataTable(
                id="categorical-summary-table",
                columns=[{"name": col, "id": col} for col in categorical_summary.columns],
                data=categorical_summary.to_dict("records") if not categorical_summary.empty else [],
                page_size=10,
                style_table={"overflowX": "auto"}
            )
        ], style={"padding": "20px"})
    elif tab == "visualizations":
        return html.Div([
            html.H2("Interactive Visualizations"),
            html.Div([
                html.Label("Select X-axis (Numeric):"),
                dcc.Dropdown(
                    id="x-axis-dropdown",
                    options=[{"label": col, "value": col} for col in numeric_columns],
                    value=numeric_columns[0] if numeric_columns else None
                )
            ], style={"width": "45%", "display": "inline-block", "padding": "10px"}),
            html.Div([
                html.Label("Select Y-axis (Numeric):"),
                dcc.Dropdown(
                    id="y-axis-dropdown",
                    options=[{"label": col, "value": col} for col in numeric_columns],
                    value=numeric_columns[1] if len(numeric_columns) > 1 else (numeric_columns[0] if numeric_columns else None)
                )
            ], style={"width": "45%", "display": "inline-block", "padding": "10px"}),
            html.Div([
                html.Label("Select Color (Categorical):"),
                dcc.Dropdown(
                    id="color-dropdown",
                    options=[{"label": col, "value": col} for col in categorical_columns],
                    value=categorical_columns[0] if categorical_columns else None
                )
            ], style={"width": "45%", "display": "inline-block", "padding": "10px"}),
            dcc.Graph(id="scatter-plot"),
            html.Hr(),
            html.Div([
                html.Label("Select a Numeric Column for Histogram:"),
                dcc.Dropdown(
                    id="histogram-dropdown",
                    options=[{"label": col, "value": col} for col in numeric_columns],
                    value=numeric_columns[0] if numeric_columns else None
                )
            ], style={"width": "45%", "display": "inline-block", "padding": "10px"}),
            dcc.Graph(id="histogram-graph"),
            html.Hr(),
            html.Div([
                html.H2("Geospatial Map"),
                dcc.Graph(id="map-plot")
            ])
        ], style={"padding": "20px"})
    elif tab == "correlation":
        if numeric_columns:
            corr = df[numeric_columns].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        else:
            fig = go.Figure()
        return html.Div([
            html.H2("Correlation Analysis"),
            dcc.Graph(figure=fig)
        ], style={"padding": "20px"})
    elif tab == "advanced":
        return html.Div([
            html.H2("Advanced Analysis"),
            html.Div([
                html.Label("Generate Word Cloud from Description:"),
                html.Button("Generate Word Cloud", id="wc-button", n_clicks=0),
                html.Br(),
                html.Img(id="wc-image", style={"width": "100%", "height": "auto", "border": "1px solid #ddd", "borderRadius": "4px"})
            ], style={"padding": "10px", "marginBottom": "30px"}),
            html.Div([
                html.Label("Select Numeric Columns for Clustering:"),
                dcc.Dropdown(
                    id="clustering-columns",
                    options=[{"label": col, "value": col} for col in numeric_columns],
                    value=numeric_columns[:2] if len(numeric_columns) >= 2 else numeric_columns,
                    multi=True
                )
            ], style={"width": "60%", "padding": "10px", "margin": "auto"}),
            html.Div([
                html.Label("Select Number of Clusters:"),
                dcc.Slider(
                    id="num-clusters",
                    min=2,
                    max=10,
                    step=1,
                    value=3,
                    marks={i: str(i) for i in range(2, 11)}
                )
            ], style={"width": "60%", "padding": "10px", "margin": "auto"}),
            dcc.Graph(id="clustering-graph"),
            html.Hr(),
            html.Div([
                html.H2("3D Scatter Plot of Events"),
                dcc.Graph(id="scatter-3d-graph")
            ])
        ], style={"padding": "20px"})
    elif tab == "search":
        return html.Div([
            html.H2("Search Descriptions"),
            dcc.Input(
                id="search-input",
                type="text",
                placeholder="Enter keywords to search...",
                style={"width": "80%", "padding": "10px", "fontSize": "16px"},
                value=""
            ),
            html.Button("Search", id="search-button", n_clicks=0,
                        style={"padding": "10px 20px", "fontSize": "16px", "cursor": "pointer", "marginLeft": "10px"}),
            html.Br(), html.Br(),
            html.Div(id="search-results")
        ], style={"padding": "20px"})
    elif tab == "nlp":
        return html.Div([
            html.H2("NLP Insights"),
            html.Button("Generate NLP Insights", id="nlp-button", n_clicks=0,
                        style={"padding": "10px 20px", "fontSize": "16px", "cursor": "pointer"}),
            html.Br(), html.Br(),
            html.Div(id="nlp-output", style={"padding": "20px", "fontSize": "16px"})
        ], style={"padding": "20px"})
    elif tab == "battlespace":
        threat_keywords = ["attack", "bomb", "explosion", "assault", "conflict", "military", "strike", "hostile", "alert"]
        pattern = "|".join(re.escape(kw) for kw in threat_keywords)
        battlespace_df = df[df["description"].str.contains(pattern, case=False, na=False)]
        timeline_fig = go.Figure()
        if not battlespace_df.empty:
            timeline_fig.add_trace(go.Scatter(
                x=battlespace_df["Datetime"],
                y=[1] * len(battlespace_df),
                mode="markers+text",
                text=battlespace_df["city"],
                marker=dict(size=10, color="red"),
                textposition="top center",
                name="Event"
            ))
            timeline_fig.update_layout(title="Timeline of Battlespace Events", xaxis_title="Datetime", yaxis=dict(visible=False))
        else:
            timeline_fig = go.Figure()
        if "latitude" in battlespace_df.columns and "longitude" in battlespace_df.columns and not battlespace_df.empty:
            geo_fig = px.scatter_mapbox(battlespace_df, lat="latitude", lon="longitude",
                                        hover_name="location", color="country",
                                        zoom=1, height=600, title="Battlespace Geospatial Map")
            geo_fig.update_layout(mapbox_style="open-street-map", margin={"r":0, "t":40, "l":0, "b":0})
        else:
            geo_fig = go.Figure()
        table = dash_table.DataTable(
            id="battlespace-table",
            columns=[{"name": col, "id": col} for col in battlespace_df.columns],
            data=battlespace_df.to_dict("records"),
            page_size=10,
            style_table={"overflowX": "auto"}
        )
        return html.Div([
            html.H2("Battlespace Awareness"),
            html.H3("Events Matching Threat Keywords"),
            table,
            html.Br(),
            html.H3("Timeline of Events"),
            dcc.Graph(figure=timeline_fig),
            html.Br(),
            html.H3("Geospatial Map of Events"),
            dcc.Graph(figure=geo_fig)
        ], style={"padding": "20px"})
    else:
        return html.Div("Tab not found", style={"padding": "20px"})

# -----------------------------
# Callbacks for Visualizations
# -----------------------------
@app.callback(
    Output("scatter-plot", "figure"),
    [Input("x-axis-dropdown", "value"),
     Input("y-axis-dropdown", "value"),
     Input("color-dropdown", "value")]
)
def update_scatter_plot(x_axis, y_axis, color_col):
    if x_axis and y_axis:
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, title="Scatter Plot")
        return fig
    return go.Figure()

@app.callback(
    Output("histogram-graph", "figure"),
    Input("histogram-dropdown", "value")
)
def update_histogram(numeric_col):
    if numeric_col:
        fig = px.histogram(df, x=numeric_col, title=f"Histogram of {numeric_col}")
        return fig
    return go.Figure()

@app.callback(
    Output("map-plot", "figure"),
    Input("tabs", "value")
)
def update_map(tab):
    if tab == "visualizations":
        if "latitude" in df.columns and "longitude" in df.columns:
            fig = px.scatter_mapbox(df, lat="latitude", lon="longitude",
                                    hover_name="Location", color="Country",
                                    zoom=1, height=600, title="Geospatial Map")
            fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
            return fig
    return go.Figure()

# -----------------------------
# Callbacks for Advanced Analysis (Word Cloud, Clustering, 3D Scatter)
# -----------------------------
@app.callback(
    Output("wc-image", "src"),
    Input("wc-button", "n_clicks")
)
def update_wordcloud(n_clicks):
    combined_text = " ".join(df["description"].tolist())
    return generate_wordcloud(combined_text)

@app.callback(
    Output("clustering-graph", "figure"),
    [Input("clustering-columns", "value"),
     Input("num-clusters", "value")]
)
def update_clustering(selected_columns, num_clusters):
    if not selected_columns or len(selected_columns) < 1:
        return go.Figure()
    data = df[selected_columns].dropna()
    if data.empty:
        return go.Figure()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    data["Cluster"] = clusters.astype(str)
    if len(selected_columns) >= 2:
        fig = px.scatter(data, x=selected_columns[0], y=selected_columns[1],
                         color="Cluster", title="Clustering Analysis")
    else:
        fig = px.histogram(data, x=selected_columns[0], color="Cluster", title="Clustering Analysis")
    return fig

@app.callback(
    Output("scatter-3d-graph", "figure"),
    Input("tabs", "value")
)
def update_3d_scatter(tab):
    if tab == "advanced" and not df.empty:
        sample_df = df.copy()
        sample_df["Datetime"] = pd.to_datetime(sample_df["Datetime"], errors="coerce")
        sample_size = min(55000, len(sample_df))
        if sample_size > 0:
            sample_df = sample_df.sample(sample_size)
            fig = px.scatter_3d(
                sample_df,
                x="Datetime",
                y="longitude",
                z="latitude",
                color="Country" if "Country" in sample_df.columns else None,
                hover_data=["description"],
                opacity=0.7,
                title="3D Scatter Plot of Events"
            )
            return fig
    return go.Figure()

# -----------------------------
# Callback for Search Descriptions Tab
# -----------------------------
@app.callback(
    Output("search-results", "children"),
    Input("search-button", "n_clicks"),
    State("search-input", "value")
)
def update_search_results(n_clicks, search_query):
    if not search_query or n_clicks is None or n_clicks == 0:
        return html.Div("Enter keywords and click Search to see results.", style={"fontSize": "16px"})
    keywords = [kw.strip() for kw in search_query.split(",") if kw.strip()]
    if not keywords:
        return html.Div("No valid keywords provided.", style={"fontSize": "16px"})
    pattern = "|".join(re.escape(kw) for kw in keywords)
    mask = df["description"].str.contains(pattern, case=False, na=False)
    results = df[mask]
    if results.empty:
        return html.Div("No matching descriptions found.", style={"fontSize": "16px"})
    children = []
    for _, row in results.iterrows():
        highlighted = highlight_keywords(row["description"], keywords)
        children.append(dcc.Markdown(highlighted, dangerously_allow_html=True,
                                     style={"marginBottom": "20px", "padding": "10px", "border": "1px solid #ddd"}))
    return children

# -----------------------------
# Callback for NLP Insights Tab
# -----------------------------
@app.callback(
    Output("nlp-output", "children"),
    Input("nlp-button", "n_clicks")
)
def update_nlp_insights(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return "Click the button to generate NLP insights."
    sample_text = " ".join(df["description"].head(50).tolist())
    try:
        summary = summarizer(sample_text, max_length=200, min_length=50, do_sample=False)
        summary_text = summary[0]["summary_text"]
    except Exception as e:
        summary_text = f"Error during summarization: {e}"
    try:
        sentiment = sentiment_analyzer(sample_text[:512])
        sentiment_result = sentiment[0]["label"]
    except Exception as e:
        sentiment_result = f"Error during sentiment analysis: {e}"
    return html.Div([
        html.H3("Summary:"),
        html.P(summary_text),
        html.H3("Overall Sentiment:"),
        html.P(sentiment_result)
    ], style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "4px"})

# -----------------------------
# Run the App
# -----------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=8051)

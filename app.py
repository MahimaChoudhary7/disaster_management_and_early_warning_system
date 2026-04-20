import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu


MODEL_PATH = "model/model.pkl"
REGIONS = ["North", "South", "East", "West"]
ALERT_STYLES = {
    "Flood": {
        "badge": "High Water Risk",
        "accent": "#2f80ed",
        "message": "Flood-like conditions detected. Review drainage, shelter, and response routes.",
    },
    "Cyclone": {
        "badge": "Storm Escalation",
        "accent": "#f2994a",
        "message": "Cyclone-like weather pattern detected. Prepare communication and evacuation plans.",
    },
    "Heatwave": {
        "badge": "Extreme Heat",
        "accent": "#eb5757",
        "message": "Heat stress conditions detected. Prioritize cooling centers and medical readiness.",
    },
    "No Disaster": {
        "badge": "Stable Outlook",
        "accent": "#27ae60",
        "message": "No immediate disaster signal detected for the current environmental inputs.",
    },
}


st.set_page_config(
    page_title="Disaster Prediction Dashboard",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --bg-top: #f4efe6;
        --bg-mid: #eef5f2;
        --card: rgba(255, 255, 255, 0.78);
        --card-strong: rgba(255, 255, 255, 0.92);
        --border: rgba(47, 79, 79, 0.14);
        --ink: #18342f;
        --muted: #5d756e;
        --accent: #0f766e;
        --accent-soft: #d9efe9;
        --warm: #f2994a;
        --danger: #c34a36;
        --ok: #2d8f60;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 200, 124, 0.4), transparent 28%),
            radial-gradient(circle at top right, rgba(255, 220, 150, 0.3), transparent 32%),
            linear-gradient(135deg, rgba(255, 228, 196, 1) 0%, rgba(255, 240, 220, 1) 55%, rgba(255, 250, 240, 1) 100%);
        color: var(--ink);
        font-family: "Trebuchet MS", "Gill Sans", sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #163c3b 0%, #1f5a57 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #f7fbfa !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero-panel {
        background: linear-gradient(135deg, rgba(16, 47, 45, 0.96), rgba(15, 118, 110, 0.86));
        border-radius: 24px;
        padding: 2rem 2rem 1.5rem 2rem;
        color: #f7fbfa;
        box-shadow: 0 24px 70px rgba(19, 53, 49, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.25rem;
    }

    .hero-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2.3rem;
        line-height: 1.1;
        margin-bottom: 0.4rem;
        letter-spacing: 0.01em;
    }

    .hero-copy {
        font-size: 1.02rem;
        max-width: 760px;
        color: rgba(247, 251, 250, 0.88);
        margin-bottom: 0.6rem;
    }

    .chip-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }

    .chip {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        padding: 0.45rem 0.9rem;
        font-size: 0.9rem;
    }

    .glass-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        box-shadow: 0 18px 40px rgba(21, 43, 39, 0.08);
        backdrop-filter: blur(6px);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: var(--card-strong);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 18px 40px rgba(21, 43, 39, 0.08);
        min-height: 118px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: var(--ink);
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
        font-family: Georgia, "Times New Roman", serif;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: 0.35rem;
    }

    .section-title {
        font-family: Georgia, "Times New Roman", serif;
        color: var(--ink);
        font-size: 1.45rem;
        margin-bottom: 0.15rem;
    }

    .section-copy {
        color: var(--muted);
        margin-bottom: 0.8rem;
        font-size: 0.98rem;
    }

    .alert-banner {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 14px 34px rgba(21, 43, 39, 0.12);
    }

    .glass-card {
        color: #000000 !important;
    }

    .glass-card * {
        color: #000000 !important;
    }

    .glass-card div[data-testid="stWidgetLabel"] {
        color: #000000 !important;
    }
    
    .glass-card div[data-testid="stWidgetLabel"] label,
    .glass-card div[data-testid="stWidgetLabel"] p,
    .glass-card div[data-testid="stWidgetLabel"] span,
    .glass-card div[data-testid="stWidgetLabel"] * {
        color: #000000 !important;
    }
    
    .glass-card label {
        color: #000000 !important;
    }
    
    .glass-card p {
        color: #000000 !important;
    }

    .glass-card div[data-testid="stSlider"] [data-baseweb="slider"] {
        margin-top: 0.35rem;
    }

    .glass-card div[data-testid="stSlider"] [data-baseweb="slider"] * {
        color: #ff5a52 !important;
    }

    .glass-card div[data-testid="stSlider"] div[role="slider"] {
        background-color: #ff5a52 !important;
        box-shadow: 0 0 0 4px rgba(255, 90, 82, 0.16);
    }

    .glass-card button[kind="secondary"] {
        color: #f6fbfa !important;
        background: linear-gradient(135deg, #163c3b, #0f766e) !important;
        border: 0 !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.84);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def render_hero(title: str, copy: str, chips: list[str]) -> None:
    chip_markup = "".join(f"<span class='chip'>{chip}</span>" for chip in chips)
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-title">{title}</div>
            <div class="hero-copy">{copy}</div>
            <div class="chip-row">{chip_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


artifact = load_artifact()
model = artifact["model"] if artifact else None
metrics = artifact["metrics"] if artifact else {}
classification_report_df = (
    artifact["classification_report"] if artifact else pd.DataFrame()
)
confusion_matrix_df = artifact["confusion_matrix"] if artifact else pd.DataFrame()


with st.sidebar:
    st.markdown("## Disaster Prediction Dashboard")

    selected = option_menu(
        "Navigation",
        ["Home", "Predict Disaster", "Model Info", "Weather Summary"],
        icons=["house", "activity", "bar-chart", "cloud"],
        menu_icon="grid",
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#f8d98b", "font-size": "16px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "4px 0px",
                "padding": "10px 14px",
                "border-radius": "12px",
            },
            "nav-link-selected": {
                "background-color": "rgba(255,255,255,0.14)",
                "font-weight": "700",
            },
        },
    )

    st.markdown("---")
    if artifact:
        st.success("Trained model loaded")
        st.caption(f"Dataset size: {artifact['dataset_size']} samples")
    else:
        st.error("Model file not found")
        st.caption("Run `python train_model.py` before opening the dashboard.")


if selected == "Home":
    render_hero(
        "Disaster Early Warning Dashboard",
        "A cleaner command center for classifying weather-driven risk. Use the dashboard to test conditions, inspect model behavior, and communicate performance in a presentation-ready layout.",
        ["Random Forest Pipeline", "Live Prediction View", "Metrics + Classification Report"],
    )

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        render_section_intro(
            "What this project does",
            "The system estimates the most likely disaster type from a small set of environmental measurements and a region label.",
        )
        st.markdown(
            """
            - Temperature, humidity, rainfall, and wind speed are processed with a trained classification pipeline.
            - The app predicts one of four classes: Flood, Cyclone, Heatwave, or No Disaster.
            - The performance page exposes the model metrics, class-wise report, and confusion matrix for review.
            """,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        render_section_intro(
            "Project snapshot",
            "A quick overview of the model artifact that is currently loaded in the app.",
        )
        if artifact:
            snapshot_cols = st.columns(2)
            with snapshot_cols[0]:
                render_metric_card("Accuracy", f"{metrics['accuracy'] * 100:.1f}%", "Current held-out test score")
            with snapshot_cols[1]:
                render_metric_card("Classes", str(len(artifact["labels"])), "Target categories")
        else:
            st.info("Train the model first to populate the live dashboard snapshot.")
        st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Predict Disaster":
    render_hero(
        "Interactive Risk Prediction",
        "Tune the environmental inputs below to generate a model prediction and compare class probabilities in a more visual format.",
        ["Fast Scenario Testing", "Probability Ranking", "Preparedness Messaging"],
    )

    if model is None:
        st.error("Model not loaded. Train the model first.")
    else:
        controls_col, result_col = st.columns([1, 1.15], gap="large")

        with controls_col:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            render_section_intro(
                "Scenario inputs",
                "Adjust the values to simulate a local weather situation.",
            )

            temperature = st.slider("Temperature (deg C)", 0.0, 60.0, 30.0, 0.5)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, 1.0)
            rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 100.0, 5.0)
            wind_speed = st.slider("Wind Speed (km/h)", 0.0, 200.0, 40.0, 1.0)
            region = st.select_slider("Region", options=REGIONS, value="North")

            predict_clicked = st.button("Run Prediction", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with result_col:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            render_section_intro(
                "Model response",
                "The prediction panel summarizes the top class and the model confidence distribution.",
            )

            input_df = pd.DataFrame(
                {
                    "temperature": [temperature],
                    "humidity": [humidity],
                    "rainfall": [rainfall],
                    "wind_speed": [wind_speed],
                    "region": [region],
                }
            )

            if predict_clicked:
                try:
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]
                    probability_df = pd.DataFrame(
                        {
                            "Disaster Type": model.classes_,
                            "Probability (%)": probabilities * 100,
                        }
                    ).sort_values("Probability (%)", ascending=False)

                    alert_style = ALERT_STYLES[prediction]
                    st.markdown(
                        f"""
                        <div class="alert-banner" style="background: linear-gradient(135deg, {alert_style['accent']}, #163c3b);">
                            <div style="font-size: 0.82rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.88;">{alert_style['badge']}</div>
                            <div style="font-size: 1.9rem; font-family: Georgia, 'Times New Roman', serif; margin: 0.2rem 0 0.35rem 0;">{prediction}</div>
                            <div style="font-size: 1rem; opacity: 0.92;">{alert_style['message']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    top_prob = probability_df.iloc[0]["Probability (%)"]
                    metric_cols = st.columns(2)
                    with metric_cols[0]:
                        render_metric_card("Top Confidence", f"{top_prob:.2f}%", "Highest predicted class probability")
                    with metric_cols[1]:
                        render_metric_card("Selected Region", region, "Geographic context used for this run")

                    fig, ax = plt.subplots(figsize=(7.5, 4))
                    sns.barplot(
                        data=probability_df,
                        x="Probability (%)",
                        y="Disaster Type",
                        palette=["#0f766e", "#4ea699", "#d0b49f", "#f2994a"],
                        ax=ax,
                    )
                    ax.set_xlabel("Probability (%)")
                    ax.set_ylabel("")
                    ax.set_title("Class probability ranking", loc="left", fontsize=13)
                    ax.set_xlim(0, 100)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.set_facecolor((1, 1, 1, 0))
                    fig.patch.set_alpha(0)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.dataframe(
                        probability_df.style.format({"Probability (%)": "{:.2f}"}),
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
            else:
                st.info("Set the inputs and click Run Prediction to see the dashboard response.")

            st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Model Info":
    render_hero(
        "Model Performance Review",
        "A polished performance page with headline metrics, class-level detail, and the confusion matrix in one place.",
        ["Held-out Evaluation", "Per-class Report", "Operational Readiness"],
    )

    if artifact is None:
        st.error("Model not found. Run `python train_model.py` first.")
    else:
        overview_cols = st.columns(4, gap="medium")
        with overview_cols[0]:
            render_metric_card("Accuracy", f"{metrics['accuracy'] * 100:.2f}%", "Overall correct predictions")
        with overview_cols[1]:
            render_metric_card("Precision", f"{metrics['precision_weighted'] * 100:.2f}%", "Weighted positive correctness")
        with overview_cols[2]:
            render_metric_card("Recall", f"{metrics['recall_weighted'] * 100:.2f}%", "Weighted event capture rate")
        with overview_cols[3]:
            render_metric_card("F1 Score", f"{metrics['f1_weighted'] * 100:.2f}%", "Balanced precision and recall")

        left, right = st.columns([1.15, 1], gap="large")

        with left:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            render_section_intro(
                "Classification report",
                "Read the class-level precision, recall, F1-score, and support together instead of relying on accuracy alone.",
            )
            st.dataframe(
                classification_report_df.style.format("{:.3f}"),
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            render_section_intro(
                "Project details",
                "These values are loaded from the saved training artifact.",
            )
            st.write(f"Model type: Random Forest Classifier")
            st.write(f"Dataset size: {artifact['dataset_size']} samples")
            st.write(f"Target column: {artifact['target_name']}")
            st.write(f"Features: {', '.join(artifact['features'])}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        render_section_intro(
            "Confusion matrix",
            "This heatmap shows where the model is separating classes well and where confusion still appears.",
        )
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.heatmap(
            confusion_matrix_df,
            annot=True,
            fmt="d",
            cmap=sns.light_palette("#0f766e", as_cmap=True),
            linewidths=0.5,
            linecolor="#f1f5f4",
            cbar=False,
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Model confusion matrix", loc="left", fontsize=13)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Weather Summary":
    render_hero(
        "Regional Conditions Snapshot",
        "A compact regional summary to complement the prediction workflow and make the app feel more like a monitoring console.",
        ["Regional Summary", "Readable Table", "Quick Status View"],
    )

    weather_data = pd.DataFrame(
        {
            "Region": ["North", "South", "East", "West"],
            "Avg Temp (deg C)": [28, 35, 30, 27],
            "Avg Humidity (%)": [55, 70, 60, 50],
            "Avg Rainfall (mm)": [150, 220, 180, 130],
            "Avg Wind Speed (km/h)": [40, 60, 50, 30],
            "Recent Disaster": ["Heatwave", "Flood", "Cyclone", "No Disaster"],
            "Status": ["Watch", "High Alert", "Monitor", "Stable"],
        }
    )

    top_col, bottom_col = st.columns([1.1, 0.9], gap="large")

    with top_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        render_section_intro(
            "Regional table",
            "A presentable summary table for demo or classroom use.",
        )
        st.dataframe(weather_data, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with bottom_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        render_section_intro(
            "Fast interpretation",
            "The south region is the most critical in this simulated view because rainfall and humidity are both elevated.",
        )
        st.write("High Alert: South")
        st.write("Best stability: West")
        st.write("Hottest region: South")
        st.write("Highest wind activity: South")
        st.markdown("</div>", unsafe_allow_html=True)

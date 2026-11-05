
import streamlit as st
import sqlite3
import uuid
import os
import textwrap
import html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import HumanMessage

from node import (
    llm,
    feature_parser,
    calculate_patent_analytics,
    generate_ai_recommendations,
    generate_ai_recommendations_report
)

from graph import workflow
from database import create_table, save_result
import templates as ui
from patent_retrieval import render_patent_retrieval


# ============================================================
# PARALLEL PROCESSING CONFIGURATION
# ============================================================

MAX_WORKERS = 5


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RelevanceIQ | iCuerious",
    page_icon="Icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def make_serializable(value):

    if hasattr(value, "model_dump") and callable(value.model_dump):
        return make_serializable(value.model_dump())

    elif hasattr(value, "dict") and callable(value.dict):
        return make_serializable(value.dict())

    if isinstance(value, dict):
        return {
            make_serializable(key): make_serializable(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_serializable(item) for item in value]

    if isinstance(value, np.ndarray):
        return make_serializable(value.tolist())

    if isinstance(value, np.generic):
        return make_serializable(value.item())

    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def normalize_dataframe_for_state(dataframe):

    dataframe = dataframe.copy().astype(object)
    dataframe = dataframe.where(pd.notna(dataframe), None)

    for column in dataframe.columns:
        dataframe[column] = dataframe[column].map(make_serializable)

    return dataframe


def prepare_retrieval_for_relevanceiq(retrieved_df):

    df = retrieved_df.copy()

    if "All Claims" not in df.columns:
        if "Claims" in df.columns:
            df["All Claims"] = df["Claims"]
        else:
            df["All Claims"] = ""

    # if "Independent Claim" not in df.columns:
    #     if "Independent Claims" in df.columns:
    #         df["Independent Claim"] = df["Independent Claims"]
    #     else:
    #         df["Independent Claim"] = ""

    if "Relevance Human" not in df.columns:
        df["Relevance Human"] = ""

    return df


# ============================================================
# PARALLEL PATENT PROCESSING
# ============================================================

def process_single_patent(
    index,
    row_data,
    product_description,
    relevance_framework,
    primary_product_features,
    secondary_product_features
):

    state = {

        "product_description": product_description,

        "publication_number":
            str(row_data.get("Publication Number", "")),

        "title":
            row_data.get("Title", ""),

        "abstract":
            row_data.get("Abstract", ""),

        "independent_claims":
            row_data.get("Independent Claims", ""),

        "all_claims":
            row_data.get("All Claims", ""),

        # "relevance_human":
        #     row_data.get("Relevance Human", ""),

        "relevance_framework":
            relevance_framework,

        "primary_product_features":
            primary_product_features,

        "secondary_product_features":
            secondary_product_features,

        "primary_patent_features": [],

        "secondary_patent_features": [],

        "comparison": [],

        "relevance": "",

        "confidence": 0,

        "rationale": "",

        "reasoning": "",

        "relevance_framework_only": "",

        "confidence_framework_only": 0,

        "rationale_framework_only": "",

        "reasoning_framework_only": "",

        "review_type": ""
    }

    state = make_serializable(state)

    result = workflow.invoke(state)

    return make_serializable(result)


# ============================================================
# HTML HELPER
# ============================================================

def render_html(content):

    content = textwrap.dedent(content)

    if hasattr(st, "html"):
        st.html(content)
    else:
        st.markdown(content, unsafe_allow_html=True)


def count_rating(df, column, value):

    if column not in df.columns:
        return 0

    return len(df[df[column] == value])


def render_cards(items):

    columns = st.columns(len(items))

    for column, (title, value) in zip(columns, items):

        with column:
            render_html(ui.info_card(title, value))


# ============================================================
# AI RECOMMENDATIONS STYLING
# ============================================================

render_html("""
<style>

.recommendation-section {
    margin-top: 18px;
    margin-bottom: 26px;
}

.recommendation-summary {
    background: linear-gradient(135deg, #f5f8fc, #edf3fa);
    border: 1px solid #dce5ef;
    border-left: 5px solid #f4512a;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 15px 0 28px 0;
    color: #172b45;
    line-height: 1.85;
    font-size: 15px;
}

.recommendation-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 13px;
    padding: 23px 26px;
    margin: 16px 0 22px 0;
    box-shadow: 0 3px 12px rgba(15, 35, 60, 0.055);
    color: #26364a;
    width: 100%;
    box-sizing: border-box;
    overflow-wrap: anywhere;
    word-break: normal;
}

.recommendation-card-title {
    font-size: 18px;
    font-weight: 700;
    color: #10243c;
    line-height: 1.55;
    margin-bottom: 15px;
    overflow-wrap: anywhere;
}

.recommendation-label {
    font-size: 12px;
    font-weight: 700;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 17px;
    margin-bottom: 7px;
}

.recommendation-body {
    font-size: 14px;
    line-height: 1.8;
    color: #34445a;
    overflow-wrap: anywhere;
}

.recommendation-action {
    background: #f4f7fb;
    border-left: 3px solid #f4512a;
    padding: 12px 16px;
    border-radius: 5px;
    margin-top: 8px;
    font-size: 14px;
    line-height: 1.75;
    color: #23364c;
}

.recommendation-publications {
    border-top: 1px solid #e8edf3;
    margin-top: 18px;
    padding-top: 13px;
    color: #718096;
    font-size: 12px;
    line-height: 1.7;
}

.recommendation-priority {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
}

.priority-highest {
    background: #fff0e9;
    color: #c2410c;
}

.priority-high {
    background: #fff7e6;
    color: #a16207;
}

.priority-medium {
    background: #eaf2ff;
    color: #2455a4;
}

.priority-normal {
    background: #edf2f7;
    color: #526174;
}

.recommendation-heading {
    font-size: 20px;
    font-weight: 750;
    color: #142b45;
    margin: 26px 0 12px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid #edf1f5;
}

.analytics-heading {
    font-size: 21px;
    font-weight: 750;
    color: #142b45;
    margin-top: 24px;
    margin-bottom: 15px;
}

.analytics-description {
    font-size: 13px;
    color: #718096;
    margin-bottom: 15px;
    line-height: 1.7;
}

</style>
""")


# ============================================================
# RECOMMENDATION DISPLAY HELPERS
# ============================================================

def get_priority_class(priority):

    priority = str(priority).strip().lower()

    if "highest" in priority:
        return "priority-highest"

    if "high" in priority:
        return "priority-high"

    if "medium" in priority:
        return "priority-medium"

    return "priority-normal"


def display_recommendation_card(item):

    if not isinstance(item, dict):

        render_html(f"""
        <div class="recommendation-card">
            <div class="recommendation-body">
                {html.escape(str(item))}
            </div>
        </div>
        """)

        return

    feature = item.get(
        "product_feature",
        item.get("feature", "Technical Feature")
    )

    priority = item.get("priority", "Normal")

    analysis = item.get(
        "analysis",
        item.get("technical_analysis", "")
    )

    recommendation = item.get(
        "recommendation",
        item.get("recommended_action", "")
    )

    publications = item.get(
        "supporting_publication_numbers",
        item.get("supporting_publications", [])
    )

    if isinstance(publications, str):
        publications = [publications]

    publication_text = ", ".join(
        str(p) for p in publications
    ) if publications else "Not specified"

    priority_class = get_priority_class(priority)

    render_html(f"""
    <div class="recommendation-card">

        <div class="recommendation-card-title">
            {html.escape(str(feature))}
        </div>

        <span class="recommendation-priority {priority_class}">
            {html.escape(str(priority).upper())} PRIORITY
        </span>

        <div class="recommendation-label">
            Technical Analysis
        </div>

        <div class="recommendation-body">
            {html.escape(str(analysis))}
        </div>

        <div class="recommendation-label">
            Recommended Action
        </div>

        <div class="recommendation-action">
            {html.escape(str(recommendation))}
        </div>

        <div class="recommendation-publications">
            <b>Supporting Publications:</b>
            {html.escape(publication_text)}
        </div>

    </div>
    """)


def display_recommendation_content(recommendations):

    if recommendations is None:

        st.info("No AI recommendations are available.")

        return

    if hasattr(recommendations, "model_dump"):
        recommendations = recommendations.model_dump()

    elif hasattr(recommendations, "dict"):
        recommendations = recommendations.dict()

    if isinstance(recommendations, str):

        render_html(f"""
        <div class="recommendation-summary">
            {html.escape(recommendations)}
        </div>
        """)

        return

    if isinstance(recommendations, list):

        for item in recommendations:
            display_recommendation_card(item)

        return

    if not isinstance(recommendations, dict):

        st.write(recommendations)

        return

    # --------------------------------------------------------
    # EXECUTIVE SUMMARY
    # --------------------------------------------------------

    executive_summary = recommendations.get(
        "executive_summary",
        recommendations.get("summary", "")
    )

    if executive_summary:

        render_html("""
        <div class="recommendation-heading">
            Executive Summary
        </div>
        """)

        render_html(f"""
        <div class="recommendation-summary">
            {html.escape(str(executive_summary))}
        </div>
        """)

    # --------------------------------------------------------
    # HIGHEST PRIORITY FEATURES
    # --------------------------------------------------------

    priority_features = recommendations.get(
        "highest_priority_features",
        recommendations.get("priority_features", [])
    )

    if priority_features:

        render_html("""
        <div class="recommendation-heading">
            Highest Priority Features
        </div>
        """)

        if isinstance(priority_features, dict):
            priority_features = list(priority_features.values())

        for item in priority_features:

            display_recommendation_card(item)

    # --------------------------------------------------------
    # OTHER RECOMMENDATION SECTIONS
    # --------------------------------------------------------

    handled_keys = {
        "executive_summary",
        "summary",
        "highest_priority_features",
        "priority_features"
    }

    for heading, content in recommendations.items():

        if heading in handled_keys:
            continue

        heading_text = str(heading).replace("_", " ").title()

        render_html(f"""
        <div class="recommendation-heading">
            {html.escape(heading_text)}
        </div>
        """)

        if isinstance(content, list):

            if content:

                for item in content:

                    if isinstance(item, dict):
                        display_recommendation_card(item)

                    else:

                        render_html(f"""
                        <div class="recommendation-card">
                            <div class="recommendation-body">
                                {html.escape(str(item))}
                            </div>
                        </div>
                        """)

            else:

                st.info("No items identified in this section.")

        elif isinstance(content, dict):

            for key, value in content.items():

                render_html(f"""
                <div class="recommendation-card">
                    <div class="recommendation-card-title">
                        {html.escape(str(key).replace("_", " ").title())}
                    </div>
                    <div class="recommendation-body">
                        {html.escape(str(value))}
                    </div>
                </div>
                """)

        else:

            render_html(f"""
            <div class="recommendation-summary">
                {html.escape(str(content))}
            </div>
            """)


# ============================================================
# PORTFOLIO ANALYTICS
# ============================================================


def display_portfolio_analytics(analytics):

    if analytics is None:
        st.info("No portfolio analytics are available.")
        return

    if hasattr(analytics, "model_dump"):
        analytics = analytics.model_dump()

    elif hasattr(analytics, "dict"):
        analytics = analytics.dict()

    if not isinstance(analytics, dict):
        st.write(analytics)
        return

    # ========================================================
    # PORTFOLIO METRICS
    # ========================================================

    total_patents = analytics.get(
        "total_patents",
        analytics.get("total_patent_count", 0)
    )

    rating_counts = analytics.get("rating_counts", {})

    if not isinstance(rating_counts, dict):
        rating_counts = {}

    metric_items = [
        ("Total Patents", total_patents),
        ("H", rating_counts.get("H", 0)),
        ("M+", rating_counts.get("M+", 0)),
        ("L", rating_counts.get("L", 0)),
        ("NR", rating_counts.get("NR", 0))
    ]

    render_cards(metric_items)

    # ========================================================
    # RATING DISTRIBUTION
    # ========================================================

    render_html("""
    <div class="analytics-heading">
        Patent Rating Distribution
    </div>
    """)

    rating_order = ["H", "M+", "L", "NR"]

    rating_df = pd.DataFrame({
        "Rating": rating_order,
        "Patent Count": [
            rating_counts.get(rating, 0)
            for rating in rating_order
        ]
    })

    rating_df["Patent Count"] = pd.to_numeric(
        rating_df["Patent Count"],
        errors="coerce"
    ).fillna(0)

    if rating_df["Patent Count"].sum() > 0:

        st.bar_chart(
            rating_df.set_index("Rating"),
            use_container_width=True
        )

    else:
        st.info("No rating distribution data is available.")

    # ========================================================
    # FEATURE-WISE PATENT CONCENTRATION
    # ========================================================

    feature_analytics = analytics.get(
        "feature_analytics",
        []
    )

    if isinstance(feature_analytics, pd.DataFrame):
        feature_analytics = feature_analytics.to_dict("records")

    if not feature_analytics:
        st.info("No feature-wise comparison data is available.")
        return

    feature_df = pd.DataFrame(feature_analytics)

    if "product_feature" not in feature_df.columns:
        st.info("Product feature information is unavailable.")
        return

    render_html("""
    <div class="analytics-heading">
        Product Feature-Wise Patent Concentration
    </div>

    <div class="analytics-description">
        Complete feature-level distribution of patents across
        H, M+, L and NR classifications. Every identified
        product feature is included in the analysis and graph.
    </div>
    """)

    # ========================================================
    # ENSURE ALL RATING COLUMNS EXIST
    # ========================================================

    rating_columns = ["H", "M+", "L", "NR"]

    for column in rating_columns:

        if column not in feature_df.columns:
            feature_df[column] = 0

        feature_df[column] = pd.to_numeric(
            feature_df[column],
            errors="coerce"
        ).fillna(0)

    feature_df["product_feature"] = (
        feature_df["product_feature"]
        .fillna("Unnamed Feature")
        .astype(str)
    )

    # ========================================================
    # CALCULATE FEATURE-LEVEL TOTALS
    # ========================================================

    feature_df["total_patents"] = (
        feature_df[rating_columns].sum(axis=1)
    )

    # Preserve additional analytics columns if supplied.
    if "high_relevance_count" not in feature_df.columns:

        feature_df["high_relevance_count"] = (
            feature_df["H"] + feature_df["M+"]
        )

    if "portfolio_concentration" not in feature_df.columns:

        total_patent_count = max(
            float(total_patents),
            1
        )

        feature_df["portfolio_concentration"] = (
            feature_df["total_patents"]
            / total_patent_count
        ) * 100

    if "h_share" not in feature_df.columns:

        feature_df["h_share"] = np.where(
            feature_df["total_patents"] > 0,
            (
                feature_df["H"]
                / feature_df["total_patents"]
            ) * 100,
            0
        )

    # ========================================================
    # COMPLETE FEATURE ANALYTICS TABLE
    # ========================================================

    display_columns = [
        "product_feature",
        "H",
        "M+",
        "L",
        "NR",
        "total_patents",
        "high_relevance_count",
        "portfolio_concentration",
        "h_share"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in feature_df.columns
    ]

    display_df = feature_df[available_columns].copy()

    display_df = display_df.rename(columns={
        "product_feature": "Product Feature",
        "total_patents": "Total Patents",
        "high_relevance_count": "H + M+",
        "portfolio_concentration": "Portfolio Concentration (%)",
        "h_share": "H Share (%)"
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(
            650,
            100 + len(display_df) * 42
        )
    )

    # ========================================================
    # ALL-FEATURE STACKED BAR GRAPH
    # ========================================================

    render_html("""
    <div class="analytics-heading">
        Patent Distribution Across All Product Features
    </div>

    <div class="analytics-description">
        Each bar represents a product feature. The coloured
        segments show the number of associated patents under
        H, M+, L and NR. No features are excluded.
    </div>
    """)

    chart_df = feature_df[
        ["product_feature"] + rating_columns
    ].copy()

    # --------------------------------------------------------
    # CORRECT SORTING
    # --------------------------------------------------------

    # Calculate row-wise totals instead of using column.sum()
    # inside pandas sort_values(key=...).

    chart_df["_total"] = chart_df[rating_columns].sum(axis=1)

    chart_df = chart_df.sort_values(
        by="_total",
        ascending=True,
        kind="stable"
    )

    chart_df["display_feature"] = (
        chart_df["product_feature"].apply(
            lambda value: textwrap.fill(
                str(value),
                width=42
            )
        )
    )

    # --------------------------------------------------------
    # GRAPH DIMENSIONS
    # --------------------------------------------------------

    number_of_features = len(chart_df)

    fig_height = max(
        5,
        number_of_features * 0.65
    )

    fig, ax = plt.subplots(
        figsize=(15, fig_height)
    )

    # --------------------------------------------------------
    # STACKED BAR PLOT
    # --------------------------------------------------------

    left_values = np.zeros(number_of_features)

    chart_colors = {
        "H": "#2474B5",
        "M+": "#F28C28",
        "L": "#45A66B",
        "NR": "#D9534F"
    }

    for column in rating_columns:

        values = chart_df[column].to_numpy(dtype=float)

        ax.barh(
            chart_df["display_feature"],
            values,
            left=left_values,
            label=column,
            color=chart_colors[column],
            height=0.68,
            edgecolor="white",
            linewidth=0.5
        )

        left_values += values

    # --------------------------------------------------------
    # GRAPH LABELS AND FORMATTING
    # --------------------------------------------------------

    ax.set_xlabel(
        "Number of Associated Patents",
        fontsize=11,
        labelpad=12
    )

    ax.set_ylabel(
        "Product Features",
        fontsize=11,
        labelpad=12
    )

    ax.set_title(
        "Patent Concentration Across All Product Features",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    ax.legend(
        title="Relevance Rating",
        loc="best",
        frameon=True,
        fontsize=9
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    ax.tick_params(
        axis="y",
        labelsize=9,
        pad=10
    )

    ax.tick_params(
        axis="x",
        labelsize=9
    )

    # --------------------------------------------------------
    # FIX LEFT-SIDE LABEL CLIPPING
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.43,
        right=0.97,
        top=0.94,
        bottom=0.09
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # ========================================================
    # JSON OUTPUT INTENTIONALLY REMOVED

# ============================================================
# CUSTOM CSS
# ============================================================

render_html(ui.load_css())


# ============================================================
# LOGIN
# ============================================================

def login():

    render_html(ui.login_card())

    username = st.text_input(
        "Username",
        placeholder="Enter username",
        key="login_username"
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password",
        key="login_password"
    )

    if st.button(
        "Login",
        use_container_width=True,
        key="login_button"
    ):

        users = st.secrets["users"]

        if username in users and password == users[username]:

            st.session_state.logged_in = True
            st.session_state.username = username

            st.rerun()

        else:

            st.error("Invalid username or password")


# ============================================================
# SESSION INITIALIZATION
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    login()
    st.stop()


if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

if "df" not in st.session_state:
    st.session_state.df = None

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "results" not in st.session_state:
    st.session_state.results = []

if "results_by_index" not in st.session_state:
    st.session_state.results_by_index = {}

if "failed_patents" not in st.session_state:
    st.session_state.failed_patents = {}

if "portfolio_results_by_index" not in st.session_state:
    st.session_state.portfolio_results_by_index = {}

if "portfolio_analytics" not in st.session_state:
    st.session_state.portfolio_analytics = None

if "ai_recommendations" not in st.session_state:
    st.session_state.ai_recommendations = None

if "portfolio_recommendations_report" not in st.session_state:
    st.session_state.portfolio_recommendations_report = None

if "product_description" not in st.session_state:
    st.session_state.product_description = ""

if "relevance_framework" not in st.session_state:
    st.session_state.relevance_framework = ""

if "primary_product_features" not in st.session_state:
    st.session_state.primary_product_features = []

if "secondary_product_features" not in st.session_state:
    st.session_state.secondary_product_features = []

if "product_features_approved" not in st.session_state:
    st.session_state.product_features_approved = False

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = False

if "active_tool" not in st.session_state:
    st.session_state.active_tool = "Patent Retrieval"


# ============================================================
# DATABASE
# ============================================================

create_table()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(ui.sidebar_brand())

    render_html(ui.sidebar_section("Tools"))

    if st.session_state.pop("navigate_to_relevanceiq", False):
        st.session_state["active_tool"] = "RelevanceIQ"

    active_tool = st.radio(
        "Select Tool",
        [
            "Patent Retrieval",
            "RelevanceIQ"
        ],
        key="active_tool",
        label_visibility="collapsed"
    )

    if active_tool == "RelevanceIQ":

        render_html(ui.sidebar_section("Application"))

        page = st.radio(
            "Navigation",
            [
                "Analysis",
                "Results",
                "Database"
            ],
            key="page",
            label_visibility="collapsed"
        )

    else:

        page = None

    render_html(ui.sidebar_user(st.session_state.username))

    if st.button(
        "Logout",
        use_container_width=True,
        key="logout_button"
    ):

        st.session_state.logged_in = False

        st.session_state.pop("username", None)

        st.rerun()


# ============================================================
# TOOL ROUTING
# ============================================================

if active_tool == "Patent Retrieval":

    render_patent_retrieval()

    st.stop()


# ============================================================
# TOP HEADER
# ============================================================

render_html(ui.top_header())


# ============================================================
# DATABASE PAGE
# ============================================================

if page == "Database":

    render_html(
        ui.hero(
            "RelevanceIQ",
            "Analysis Database",
            "Review previously generated patent relevance "
            "analyses stored by the application."
        )
    )

    conn = sqlite3.connect("RelevanceIQ.db")

    try:

        database_df = pd.read_sql_query(
            """
            SELECT *
            FROM FTO_result
            ORDER BY id DESC
            """,
            conn
        )

    except Exception as e:

        database_df = pd.DataFrame()

        st.error(f"Unable to load database records: {e}")

    finally:

        conn.close()

    if not database_df.empty:

        st.write(f"**{len(database_df)} records found**")

        st.dataframe(
            database_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download Database",
            database_df.to_csv(index=False),
            "RelevanceIQ_Database.csv",
            "text/csv",
            key="download_database"
        )

    else:

        st.info("No database records are available.")

    st.stop()


# ============================================================
# RESULTS PAGE
# ============================================================

if page == "Results":

    render_html(
        ui.hero(
            "RelevanceIQ",
            "Analysis Results",
            "Patent relevance results generated from the "
            "current analysis session."
        )
    )

    if st.session_state.results:

        result_df = pd.DataFrame(
            st.session_state.results
        )

        render_cards([
            ("Patents Analysed", len(result_df)),
            ("Highly Relevant", count_rating(result_df, "Relevance", "H")),
            ("M+", count_rating(result_df, "Relevance", "M+")),
            ("L", count_rating(result_df, "Relevance", "L")),
            ("NR", count_rating(result_df, "Relevance", "NR")),
        ])

        render_html(
            ui.section(
                "F",
                "Framework-Only Classification",
                "Classification based specifically on the "
                "user-provided relevance framework."
            )
        )

        render_cards([
            ("H", count_rating(result_df, "Relevance_Framework_Only", "H")),
            ("M+", count_rating(result_df, "Relevance_Framework_Only", "M+")),
            ("L", count_rating(result_df, "Relevance_Framework_Only", "L")),
            ("NR", count_rating(result_df, "Relevance_Framework_Only", "NR")),
        ])

        render_html(
            ui.section(
                "R",
                "Common Relevance Framework Results"
            )
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Results",
            result_df.to_csv(index=False),
            "RelevanceIQ_Results.csv",
            "text/csv",
            key="download_results"
        )

    else:

        st.info(
            "No analysis results are available yet. "
            "Run an analysis first."
        )

    st.stop()


# ============================================================
# ANALYSIS PAGE
# ============================================================

render_html(
    ui.hero(
        "Patent Intelligence • FTO",
        "Patent Relevance Analysis",
        "Evaluate patents against a target product using "
        "a defined relevance framework and human-reviewed "
        "technical feature extraction."
    )
)


# ============================================================
# INPUT SECTION
# ============================================================

render_html(
    ui.section(
        1,
        "Target Product",
        "Describe the technical product that you want to analyze."
    )
)


product_description = st.text_area(
    "Product Description",
    value=st.session_state.product_description,
    height=180,
    placeholder=(
        "Enter the technical description of the target product..."
    ),
    label_visibility="collapsed",
    key="product_description_input"
)


# ============================================================
# PATENT FILE
# ============================================================

render_html(
    ui.section(
        2,
        "Patent Dataset",
        "Upload the Excel file containing the patents to be evaluated."
    )
)

analysis_source = st.session_state.get(
    "analysis_source",
    "independent"
)

if analysis_source == "retrieval":

    uploaded_df = prepare_retrieval_for_relevanceiq(
        st.session_state["patent_search_candidates"]
    )

    product_description = st.session_state[
        "patent_search_product_description"
    ]

    st.success(
        f"{len(uploaded_df)} retrieved patents loaded for RelevanceIQ."
    )

else:

    uploaded_file = st.file_uploader(
        "Upload Patent Excel File",
        type=["xlsx"],
        label_visibility="collapsed",
        key="patent_file"
    )


col1, col2 = st.columns([3, 1])

with col1:

    render_html(ui.input_info())

with col2:

    template_path = "RelevanceIQ_Input_Data_Schema.xlsx"

    if os.path.exists(template_path):

        with open(template_path, "rb") as f:

            st.download_button(
                "Download Template",
                f.read(),
                "RelevanceIQ_Input_Data_Schema.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_template"
            )


# ============================================================
# FRAMEWORK
# ============================================================

render_html(
    ui.section(
        3,
        "Relevance Framework",
        "Define the criteria that will be used to determine "
        "patent relevance."
    )
)


relevance_framework = st.text_area(
    "Relevance Framework",
    value=st.session_state.relevance_framework,
    height=220,
    placeholder=(
        "Enter the relevance framework used to classify "
        "patents as H, M+, L or NR..."
    ),
    label_visibility="collapsed",
    key="relevance_framework_input"
)


# ============================================================
# RUN ANALYSIS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

run_col1, run_col2, run_col3 = st.columns([1, 2, 1])

with run_col2:

    run = st.button(
        "Run Relevance Analysis",
        use_container_width=True,
        key="run_analysis_button"
    )


if run:

    if analysis_source != "retrieval" and uploaded_file is None:

        st.warning(
            "Please upload an Excel file containing the patents."
        )

        st.stop()

    if not product_description.strip():

        st.warning("Please enter the product description.")

        st.stop()

    if not relevance_framework.strip():

        st.warning("Please provide the relevance framework.")

        st.stop()

    if analysis_source == "retrieval":

        uploaded_df = prepare_retrieval_for_relevanceiq(
            st.session_state["patent_search_candidates"]
        )

    else:

        try:

            uploaded_df = pd.read_excel(uploaded_file)

            uploaded_df = uploaded_df.astype(object)

            uploaded_df = uploaded_df.where(
                pd.notna(uploaded_df),
                None
            )

        except Exception as e:

            st.error(f"Unable to read the Excel file: {e}")

            st.stop()

    uploaded_df = normalize_dataframe_for_state(uploaded_df)

    if uploaded_df.empty:

        st.warning(
            "The uploaded Excel file does not contain "
            "any patent rows."
        )

        st.stop()

    required_columns = [
        "Publication Number",
        "Title",
        "Abstract",
        "Independent Claims",
        "All Claims",
       
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in uploaded_df.columns
    ]

    if missing_columns:

        st.error(
            "The uploaded Excel file is missing the following "
            f"required columns: {', '.join(missing_columns)}"
        )

        st.stop()

    # --------------------------------------------------------
    # SAVE SESSION DATA
    # --------------------------------------------------------

    st.session_state.df = uploaded_df

    st.session_state.product_description = product_description

    st.session_state.relevance_framework = relevance_framework

    st.session_state.results = []

    st.session_state.results_by_index = {}

    st.session_state.failed_patents = {}

    st.session_state.portfolio_results_by_index = {}

    st.session_state.portfolio_analytics = None

    st.session_state.ai_recommendations = None

    st.session_state.portfolio_recommendations_report = None

    st.session_state.current_index = 0

    st.session_state.primary_product_features = []

    st.session_state.secondary_product_features = []

    st.session_state.product_features_approved = False

    st.session_state.analysis_started = True

    st.session_state.analysis_error = False

    st.session_state.session_id = str(uuid.uuid4())

    st.rerun()


# ============================================================
# PRODUCT FEATURE EXTRACTION
# ============================================================

if (
    st.session_state.analysis_started
    and not st.session_state.product_features_approved
):

    if (
        not st.session_state.primary_product_features
        and not st.session_state.secondary_product_features
    ):

        render_html(
            ui.status_card(
                "Extracting Product Features",
                "RelevanceIQ is identifying the primary "
                "and secondary technical features of the "
                "target product for human review."
            )
        )

        product_prompt = f"""
You are an experienced patent analyst performing an FTO
(Freedom-to-Operate) technical analysis.

Extract the technical features explicitly disclosed in the
product description.

PRIMARY FEATURES:
- Core technical features and mechanisms.
- Features important for satisfying potential patent claim
  limitations.

SECONDARY FEATURES:
- Supporting, auxiliary, optional, control, communication,
  interface, or peripheral features.

Rules:
- Extract only explicitly disclosed features.
- Do not infer or invent features.
- Keep features concise and technically specific.

{feature_parser.get_format_instructions()}

Product Description:
{st.session_state.product_description}
"""

        response = llm.invoke(
            [
                HumanMessage(content=product_prompt)
            ]
        )

        result = feature_parser.parse(response.content)

        st.session_state.primary_product_features = (
            result.primary_features
        )

        st.session_state.secondary_product_features = (
            result.secondary_features
        )

        st.rerun()

    # --------------------------------------------------------
    # HUMAN REVIEW
    # --------------------------------------------------------

    render_html(
        ui.section(
            "✓",
            "Review Product Features",
            "Review and edit the extracted features before "
            "continuing with patent analysis."
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        edited_primary = st.text_area(
            "Primary Product Features",
            value="\n".join(
                st.session_state.primary_product_features
            ),
            height=250,
            key="product_primary_review"
        )

    with col2:

        edited_secondary = st.text_area(
            "Secondary Product Features",
            value="\n".join(
                st.session_state.secondary_product_features
            ),
            height=250,
            key="product_secondary_review"
        )

    approve_col1, approve_col2, approve_col3 = st.columns([1, 2, 1])

    with approve_col2:

        if st.button(
            "Approve Product Features",
            use_container_width=True,
            key="approve_product_features"
        ):

            st.session_state.primary_product_features = [
                x.strip()
                for x in edited_primary.split("\n")
                if x.strip()
            ]

            st.session_state.secondary_product_features = [
                x.strip()
                for x in edited_secondary.split("\n")
                if x.strip()
            ]

            st.session_state.product_features_approved = True

            st.rerun()

    st.stop()


# ============================================================
# PARALLEL PATENT PROCESSING
# ============================================================

if st.session_state.analysis_started:

    df = st.session_state.df

    total_patents = len(df)

    results_by_index = st.session_state.results_by_index

    failed_patents = st.session_state.failed_patents

    pending_indices = [
        i
        for i in range(total_patents)
        if i not in results_by_index
    ]

    # --------------------------------------------------------
    # PARALLEL EXECUTION
    # --------------------------------------------------------

    if pending_indices:

        render_html(
            ui.section(
                "P",
                "Parallel Patent Processing",
                f"Processing patents using {MAX_WORKERS} parallel workers."
            )
        )

        st.info(
            f"{len(pending_indices)} patents pending processing."
        )

        progress_bar = st.progress(
            len(results_by_index) / total_patents
        )

        status_text = st.empty()

        product_description = st.session_state.product_description

        relevance_framework = st.session_state.relevance_framework

        primary_product_features = list(
            st.session_state.primary_product_features
        )

        secondary_product_features = list(
            st.session_state.secondary_product_features
        )

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            future_to_index = {}

            for index in pending_indices:

                row_data = make_serializable(
                    df.iloc[index].to_dict()
                )

                future = executor.submit(
                    process_single_patent,
                    index,
                    row_data,
                    product_description,
                    relevance_framework,
                    primary_product_features,
                    secondary_product_features
                )

                future_to_index[future] = index

            # ------------------------------------------------
            # COLLECT COMPLETED RESULTS
            # ------------------------------------------------

            for future in as_completed(future_to_index):

                index = future_to_index[future]

                publication_number = str(
                    df.iloc[index]["Publication Number"]
                )

                try:

                    result = future.result()

                    # ----------------------------------------
                    # SAVE TO DATABASE
                    # ----------------------------------------

                    try:

                        save_result(result)

                    except Exception as db_error:

                        st.warning(
                            f"Database save failed for "
                            f"{publication_number}: {db_error}"
                        )

                    # ----------------------------------------
                    # PREPARE PATENT RESULT
                    # ----------------------------------------

                    patent_result = {

                        "Publication Number":
                            result["publication_number"],

                        "Title":
                            result["title"],

                        # "Relevance Human":
                        #     result["relevance_human"],

                        "Patent Primary Features":
                            result["primary_patent_features"],

                        "Patent Secondary Features":
                            result["secondary_patent_features"],

                        "Relevance":
                            result["relevance"],

                        "Confidence":
                            result["confidence"],

                        "Rationale":
                            result["rationale"],

                        "Reasoning":
                            result["reasoning"],

                        "Relevance_Framework_Only":
                            result["relevance_framework_only"],

                        "Confidence_Framework_Only":
                            result["confidence_framework_only"],

                        "Rationale_Framework_Only":
                            result["rationale_framework_only"],

                        "Reasoning_Framework_Only":
                            result["reasoning_framework_only"]

                    }

                    results_by_index[index] = make_serializable(
                        patent_result
                    )

                    # ----------------------------------------
                    # STORE COMPLETE PATENT RESULT
                    # ----------------------------------------

                    st.session_state.portfolio_results_by_index[index] = (
                        make_serializable(result)
                    )

                    failed_patents.pop(index, None)

                except Exception as e:

                    failed_patents[index] = str(e)

                    print(
                        f"Patent processing failed: "
                        f"{publication_number}"
                    )

                    print(e)

                # --------------------------------------------
                # UPDATE PROGRESS
                # --------------------------------------------

                total_completed = len(results_by_index)

                total_failed = len(failed_patents)

                total_processed = total_completed + total_failed

                progress_bar.progress(
                    min(total_processed / total_patents, 1.0)
                )

                status_text.write(
                    f"Completed: {total_completed}/{total_patents} | "
                    f"Failed: {total_failed} | "
                    f"Remaining: {total_patents - total_processed}"
                )

        # ----------------------------------------------------
        # UPDATE SESSION RESULTS
        # ----------------------------------------------------

        st.session_state.results = [
            results_by_index[i]
            for i in sorted(results_by_index)
        ]

        st.session_state.current_index = len(results_by_index)

    # --------------------------------------------------------
    # PROCESSING FINISHED
    # --------------------------------------------------------

    st.session_state.analysis_started = False

    result_df = pd.DataFrame(
        st.session_state.results
    )

    # --------------------------------------------------------
    # FAILED PATENTS
    # --------------------------------------------------------

    if failed_patents:

        st.warning(
            f"{len(failed_patents)} patent(s) failed "
            "and can be retried."
        )

        failed_df = pd.DataFrame([

            {
                "Patent Index": index + 1,

                "Publication Number":
                    df.iloc[index]["Publication Number"],

                "Error": error

            }

            for index, error in sorted(failed_patents.items())

        ])

        st.dataframe(
            failed_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "All patents have been processed successfully."
        )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    if not result_df.empty:

        render_html(
            ui.hero(
                "Analysis Complete",
                "Patent Analysis Results",
                "The completed patent analyses are available below."
            )
        )

        render_cards([
            ("Patents Analysed", len(result_df)),
            ("H", count_rating(result_df, "Relevance", "H")),
            ("M+", count_rating(result_df, "Relevance", "M+")),
            ("L", count_rating(result_df, "Relevance", "L")),
            ("NR", count_rating(result_df, "Relevance", "NR")),
        ])

        render_html(
            ui.section(
                "R",
                "Results"
            )
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Results",
            result_df.to_csv(index=False),
            "RelevanceIQ_Results.csv",
            "text/csv",
            key="download_final_results"
        )


# ============================================================
# AI GENERATED RECOMMENDATIONS
# ============================================================

if (
    not st.session_state.analysis_started
    and st.session_state.df is not None
    and len(st.session_state.portfolio_results_by_index)
        == len(st.session_state.df)
    and len(st.session_state.df) > 0
    and not st.session_state.failed_patents
):

    st.divider()

    render_html(
        ui.section(
            "AI",
            "AI Generated Recommendations",
            "Consolidated portfolio analytics and AI-generated "
            "recommendations based on the completed patent analysis."
        )
    )

    if st.button(
        "Generate AI Recommendations",
        use_container_width=True,
        key="generate_ai_recommendations"
    ):

        try:

            portfolio_results = [
                st.session_state.portfolio_results_by_index[i]
                for i in sorted(
                    st.session_state.portfolio_results_by_index
                )
            ]

            with st.spinner(
                "Analysing patent portfolio and generating recommendations..."
            ):

                # --------------------------------------------
                # GENERATE ANALYTICS AND RECOMMENDATIONS
                # --------------------------------------------

                analytics, recommendations = generate_ai_recommendations(
                    portfolio_results,
                    st.session_state.product_description
                )

                # --------------------------------------------
                # SAVE RESULTS IN SESSION STATE
                # --------------------------------------------

                st.session_state.portfolio_analytics = (
                    make_serializable(analytics)
                )

                st.session_state.ai_recommendations = (
                    make_serializable(recommendations)
                )

                # --------------------------------------------
                # GENERATE WORD REPORT
                # --------------------------------------------

                report = generate_ai_recommendations_report(
                    patent_results=portfolio_results,
                    product_description=(
                        st.session_state.product_description
                    ),
                    analytics=analytics,
                    recommendations=recommendations
                )

                st.session_state.portfolio_recommendations_report = report

            st.success(
                "AI recommendations and portfolio analytics generated successfully."
            )

        except Exception as e:

            st.error(
                f"Unable to generate AI recommendations: {e}"
            )

    # ========================================================
    # DISPLAY PORTFOLIO ANALYTICS
    # ========================================================

    if st.session_state.portfolio_analytics is not None:

        st.divider()

        render_html("""
        <div class="analytics-heading">
            Portfolio Analytics
        </div>
        """)

        display_portfolio_analytics(
            st.session_state.portfolio_analytics
        )

    # ========================================================
    # DISPLAY AI RECOMMENDATIONS
    # ========================================================

    if st.session_state.ai_recommendations is not None:

        st.divider()

        render_html("""
        <div class="analytics-heading">
            AI Recommendations
        </div>
        """)

        display_recommendation_content(
            st.session_state.ai_recommendations
        )

    # ========================================================
    # DOWNLOAD WORD REPORT
    # ========================================================

    if st.session_state.portfolio_recommendations_report is not None:

        st.divider()

        render_html("""
        <div class="analytics-heading">
            Download Portfolio Report
        </div>
        """)

        st.download_button(
            label="Download AI Recommendations Report",

            data=st.session_state.portfolio_recommendations_report,

            file_name="RelevanceIQ_AI_Recommendations.docx",

            mime=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),

            use_container_width=True,

            key="download_ai_recommendations"
        )


# ============================================================
# RETRY FAILED PATENTS
# ============================================================

if (
    not st.session_state.analysis_started
    and st.session_state.failed_patents
):

    if st.button(
        "Retry Failed Patents",
        use_container_width=True,
        key="retry_failed_patents"
    ):

        st.session_state.analysis_started = True

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

render_html(ui.footer())

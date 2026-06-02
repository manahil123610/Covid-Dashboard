import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 Global Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
PALETTE = {
    "bg":       "#0D1117",
    "card":     "#161B22",
    "border":   "#30363D",
    "accent1":  "#58A6FF",   # blue
    "accent2":  "#F78166",   # coral/red
    "accent3":  "#3FB950",   # green
    "accent4":  "#D2A8FF",   # purple
    "accent5":  "#FFA657",   # orange
    "text":     "#E6EDF3",
    "subtext":  "#8B949E",
}

CHART_COLORS = [
    PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
    PALETTE["accent4"], PALETTE["accent5"], "#79C0FF", "#FF7B72",
    "#56D364", "#BC8CFF", "#FFB86C",
]

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["card"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["subtext"],
    "ytick.color":       PALETTE["subtext"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["border"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  PALETTE["card"],
    "legend.edgecolor":  PALETTE["border"],
    "font.family":       "DejaVu Sans",
})

st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        background-color: {PALETTE["bg"]};
        color: {PALETTE["text"]};
    }}
    .stApp {{ background-color: {PALETTE["bg"]}; }}
    section[data-testid="stSidebar"] {{
        background-color: {PALETTE["card"]};
        border-right: 1px solid {PALETTE["border"]};
    }}
    .block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}

    /* KPI cards */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }}
    .kpi-card {{
        background: {PALETTE["card"]};
        border: 1px solid {PALETTE["border"]};
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }}
    .kpi-label {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {PALETTE["subtext"]};
        margin-bottom: 8px;
    }}
    .kpi-value {{
        font-size: 32px;
        font-weight: 700;
        color: {PALETTE["text"]};
        line-height: 1;
    }}
    .kpi-sub {{
        font-size: 12px;
        color: {PALETTE["subtext"]};
        margin-top: 6px;
    }}
    .kpi-accent {{ color: {PALETTE["accent1"]}; }}
    .kpi-danger {{ color: {PALETTE["accent2"]}; }}
    .kpi-success {{ color: {PALETTE["accent3"]}; }}
    .kpi-purple {{ color: {PALETTE["accent4"]}; }}

    /* Dashboard title */
    .dash-title {{
        font-size: 36px;
        font-weight: 800;
        color: {PALETTE["text"]};
        letter-spacing: -1px;
        margin-bottom: 4px;
    }}
    .dash-sub {{
        font-size: 14px;
        color: {PALETTE["subtext"]};
        margin-bottom: 24px;
    }}

    /* Section headers */
     .section-header {{
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: {PALETTE["text"]};
        border-bottom: 1px solid {PALETTE["accent1"]};
        padding-bottom: 8px;
        margin: 32px 0 16px 0;
    }}

    /* Streamlit widget overrides */
    .stSelectbox label, .stMultiSelect label,
    .stSlider label, .stTextInput label,
    .stDateInput label {{
        color: {PALETTE["subtext"]} !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }}
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background-color: {PALETTE["bg"]} !important;
        border-color: {PALETTE["border"]} !important;
        color: {PALETTE["text"]} !important;
    }}
    .stButton > button {{
        background: linear-gradient(135deg, {PALETTE["accent1"]}, {PALETTE["accent4"]});
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        padding: 10px;
    }}
    .stButton > button:hover {{
        opacity: 0.85;
        transform: translateY(-1px);
    }}
    div[data-testid="stMetric"] {{
        background: {PALETTE["card"]};
        border: 1px solid {PALETTE["border"]};
        border-radius: 10px;
        padding: 16px;
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("1780429023919_data.csv")
    df.columns = df.columns.str.strip()
    df["dateRep"] = pd.to_datetime(df["dateRep"], dayfirst=True, errors="coerce")
    df["cases"]  = pd.to_numeric(df["cases"],  errors="coerce").fillna(0).astype(int)
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce").fillna(0).astype(int)
    df["popData2019"] = pd.to_numeric(df["popData2019"], errors="coerce")
    df["cum14"] = pd.to_numeric(
        df["Cumulative_number_for_14_days_of_COVID-19_cases_per_100000"],
        errors="coerce"
    )
    df["cases"]  = df["cases"].clip(lower=0)
    df["deaths"] = df["deaths"].clip(lower=0)
    df["month_year"] = df["dateRep"].dt.to_period("M").astype(str)
    df["countriesAndTerritories"] = df["countriesAndTerritories"].str.replace("_", " ")
    df["mortality_rate"] = np.where(
        df["cases"] > 0, df["deaths"] / df["cases"] * 100, 0
    )
    return df

df_raw = load_data()


# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
        <div style="padding:16px 0 24px 0;">
            <div style="font-size:20px;font-weight:800;color:{PALETTE['text']};">🦠 COVID-19</div>
            <div style="font-size:11px;color:{PALETTE['subtext']};letter-spacing:2px;text-transform:uppercase;">Global Dashboard</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔍 Filters")

    # Date range filter
    date_min = df_raw["dateRep"].min().date()
    date_max = df_raw["dateRep"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d_start, d_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        d_start, d_end = pd.Timestamp(date_min), pd.Timestamp(date_max)

    # Continent filter
    continents_all = sorted(df_raw["continentExp"].dropna().unique())
    continents_sel = st.multiselect(
        "Continent(s)",
        options=continents_all,
        default=continents_all,
    )

    # Country filter
    countries_pool = sorted(
        df_raw[df_raw["continentExp"].isin(continents_sel)]["countriesAndTerritories"].unique()
    ) if continents_sel else sorted(df_raw["countriesAndTerritories"].unique())

    countries_sel = st.multiselect(
        "Country / Territory",
        options=countries_pool,
        default=[],
        placeholder="All (leave blank)"
    )

    # Numerical range slider – cases
    max_cases = int(df_raw.groupby("countriesAndTerritories")["cases"].sum().max())
    case_range = st.slider(
        "Total Cases Range (per country)",
        min_value=0,
        max_value=max_cases,
        value=(0, max_cases),
        step=1000,
        format="%d"
    )

    # Text search
    search_text = st.text_input("🔎 Search Country", placeholder="e.g. Germany")

    # Reset button
    reset = st.button("↺  Reset All Filters")

if reset:
    st.rerun()

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
df = df_raw.copy()
df = df[(df["dateRep"] >= d_start) & (df["dateRep"] <= d_end)]

if continents_sel:
    df = df[df["continentExp"].isin(continents_sel)]

if countries_sel:
    df = df[df["countriesAndTerritories"].isin(countries_sel)]

if search_text.strip():
    df = df[df["countriesAndTerritories"].str.contains(search_text.strip(), case=False, na=False)]

# Apply case range filter (on per-country totals)
country_totals = df.groupby("countriesAndTerritories")["cases"].sum()
valid_countries = country_totals[
    (country_totals >= case_range[0]) & (country_totals <= case_range[1])
].index
df = df[df["countriesAndTerritories"].isin(valid_countries)]


# ─────────────────────────────────────────────
# DASHBOARD HEADER
# ─────────────────────────────────────────────
st.markdown("""
    <div class="dash-title">🦠 COVID-19 Global Analytics Dashboard</div>
    <div class="dash-sub">ECDC EU/EEA Dataset · Comprehensive epidemiological analysis across countries and time</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("⚠️ No data matches the current filters. Please adjust your selections.")
    st.stop()


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
total_cases  = int(df["cases"].sum())
total_deaths = int(df["deaths"].sum())
total_countries = df["countriesAndTerritories"].nunique()
overall_mortality = (total_deaths / total_cases * 100) if total_cases > 0 else 0
avg_daily_cases = df.groupby("dateRep")["cases"].sum().mean()
peak_day = df.groupby("dateRep")["cases"].sum().idxmax()
peak_val = int(df.groupby("dateRep")["cases"].sum().max())

def fmt(n):
    if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Total Cases</div>
        <div class="kpi-value kpi-accent">{fmt(total_cases)}</div>
        <div class="kpi-sub">Confirmed infections</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Total Deaths</div>
        <div class="kpi-value kpi-danger">{fmt(total_deaths)}</div>
        <div class="kpi-sub">Fatalities recorded</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Mortality Rate</div>
        <div class="kpi-value kpi-purple">{overall_mortality:.2f}%</div>
        <div class="kpi-sub">Deaths / Cases</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Countries Affected</div>
        <div class="kpi-value kpi-success">{total_countries}</div>
        <div class="kpi-sub">Avg {fmt(int(avg_daily_cases))} cases/day</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER: styled fig
# ─────────────────────────────────────────────
def styled_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    return fig, ax

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, color=PALETTE["text"], fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=PALETTE["subtext"], fontsize=10)
    ax.set_ylabel(ylabel, color=PALETTE["subtext"], fontsize=10)
    ax.tick_params(colors=PALETTE["subtext"])
    ax.grid(True, linestyle="--", alpha=0.3, color=PALETTE["border"])


# ─────────────────────────────────────────────
# ROW 1: Line Chart + Bar Chart
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Temporal Trends</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("**3 — Line Chart: Daily Cases Over Time**")
    daily = df.groupby("dateRep")[["cases", "deaths"]].sum().reset_index().sort_values("dateRep")
    daily["cases_7d"]  = daily["cases"].rolling(7).mean()
    daily["deaths_7d"] = daily["deaths"].rolling(7).mean()

    fig, ax = styled_fig(9, 4)
    ax.fill_between(daily["dateRep"], daily["cases"], alpha=0.15, color=PALETTE["accent1"])
    ax.plot(daily["dateRep"], daily["cases"],   color=PALETTE["accent1"], alpha=0.4, linewidth=1, label="Daily Cases")
    ax.plot(daily["dateRep"], daily["cases_7d"], color=PALETTE["accent1"], linewidth=2.2, label="7-day Avg Cases")
    ax2 = ax.twinx()
    ax2.plot(daily["dateRep"], daily["deaths_7d"], color=PALETTE["accent2"], linewidth=2, linestyle="--", label="7-day Avg Deaths")
    ax2.set_ylabel("Deaths", color=PALETTE["subtext"], fontsize=10)
    ax2.tick_params(colors=PALETTE["subtext"])
    ax2.spines["right"].set_edgecolor(PALETTE["border"])
    ax2.set_facecolor(PALETTE["card"])
    style_ax(ax, "Daily COVID-19 Cases & Deaths Over Time", "Date", "Cases")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("**4 — Bar Chart: Top 15 Countries by Total Cases**")
    top15 = df.groupby("countriesAndTerritories")["cases"].sum().nlargest(15).reset_index()

    fig, ax = styled_fig(9, 4)
    bars = ax.barh(top15["countriesAndTerritories"], top15["cases"],
                   color=CHART_COLORS[:len(top15)], edgecolor=PALETTE["border"], linewidth=0.5)
    for bar, val in zip(bars, top15["cases"]):
        ax.text(bar.get_width() + top15["cases"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                fmt(int(val)), va="center", ha="left", color=PALETTE["subtext"], fontsize=8)
    ax.invert_yaxis()
    style_ax(ax, "Top 15 Countries — Total Cases", "Total Cases", "")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────
# ROW 2: Pie Chart + Histogram
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🥧 Distribution & Proportions</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    st.markdown("**1 — Pie Chart: Cases by Continent**")
    cont_data = df.groupby("continentExp")["cases"].sum().reset_index()
    cont_data = cont_data[cont_data["cases"] > 0].sort_values("cases", ascending=False)

    fig, ax = styled_fig(7, 5)
    wedges, texts, autotexts = ax.pie(
        cont_data["cases"],
        labels=cont_data["continentExp"],
        autopct="%1.1f%%",
        colors=CHART_COLORS[:len(cont_data)],
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor=PALETTE["bg"], linewidth=2),
    )
    for t in texts:     t.set_color(PALETTE["text"]); t.set_fontsize(10)
    for t in autotexts: t.set_color(PALETTE["bg"]);   t.set_fontsize(9); t.set_fontweight("bold")
    ax.set_title("Proportional Distribution of Cases by Continent",
                 color=PALETTE["text"], fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with col4:
    st.markdown("**2 — Histogram: Distribution of Daily Case Counts**")
    daily_cases = df.groupby("dateRep")["cases"].sum()
    daily_cases = daily_cases[daily_cases > 0]

    fig, ax = styled_fig(7, 5)
    n, bins, patches = ax.hist(daily_cases, bins=40, color=PALETTE["accent1"],
                                edgecolor=PALETTE["bg"], linewidth=0.5, alpha=0.85)
    # color gradient
    norm = plt.Normalize(n.min(), n.max())
    for count, patch in zip(n, patches):
        patch.set_facecolor(plt.cm.Blues(norm(count) * 0.7 + 0.3))
    ax.axvline(daily_cases.mean(), color=PALETTE["accent2"], linewidth=2,
               linestyle="--", label=f"Mean: {fmt(int(daily_cases.mean()))}")
    ax.axvline(daily_cases.median(), color=PALETTE["accent3"], linewidth=2,
               linestyle=":", label=f"Median: {fmt(int(daily_cases.median()))}")
    style_ax(ax, "Frequency Distribution of Daily Cases", "Daily Cases", "Frequency")
    ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────
# ROW 3: Scatter Plot + Box Plot
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🔬 Statistical Analysis</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)

with col5:
    st.markdown("**5 — Scatter Plot: Cases vs Deaths by Country**")
    scatter_df = df.groupby(["countriesAndTerritories", "continentExp"]).agg(
        cases=("cases", "sum"), deaths=("deaths", "sum")
    ).reset_index()
    scatter_df = scatter_df[(scatter_df["cases"] > 0) & (scatter_df["deaths"] >= 0)]
    continents_unique = scatter_df["continentExp"].unique()
    cont_color_map = {c: CHART_COLORS[i % len(CHART_COLORS)] for i, c in enumerate(continents_unique)}

    fig, ax = styled_fig(8, 5)
    for cont in continents_unique:
        sub = scatter_df[scatter_df["continentExp"] == cont]
        ax.scatter(sub["cases"], sub["deaths"],
                   color=cont_color_map[cont], alpha=0.75, s=40,
                   label=cont, edgecolors=PALETTE["bg"], linewidth=0.5)
    # regression line
    x, y = scatter_df["cases"].values, scatter_df["deaths"].values
    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 200)
        ax.plot(xline, m * xline + b, color=PALETTE["accent5"], linewidth=1.5,
                linestyle="--", alpha=0.8, label="Trend Line")
    style_ax(ax, "Cases vs Deaths per Country", "Total Cases", "Total Deaths")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with col6:
    st.markdown("**6 — Box Plot: Case Distribution by Continent**")
    box_data = [
        df[df["continentExp"] == c]["cases"].clip(0).values
        for c in continents_all if c in df["continentExp"].values
    ]
    labels_box = [c for c in continents_all if c in df["continentExp"].values]

    fig, ax = styled_fig(8, 5)
    bp = ax.boxplot(box_data, labels=labels_box, patch_artist=True,
                    medianprops=dict(color=PALETTE["accent5"], linewidth=2),
                    flierprops=dict(marker="o", markerfacecolor=PALETTE["accent2"],
                                    markersize=3, alpha=0.5, linestyle="none"),
                    whiskerprops=dict(color=PALETTE["subtext"]),
                    capprops=dict(color=PALETTE["subtext"]))
    for patch, color in zip(bp["boxes"], CHART_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    style_ax(ax, "Daily Case Distribution by Continent", "Continent", "Cases")
    ax.set_yscale("symlog")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────
# ROW 4: Heatmap + Area Chart
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Correlations & Cumulative Trends</div>', unsafe_allow_html=True)
col7, col8 = st.columns(2)

with col7:
    st.markdown("**7 — Heatmap: Monthly Cases by Top Countries**")
    top10_countries = df.groupby("countriesAndTerritories")["cases"].sum().nlargest(10).index
    heat_df = df[df["countriesAndTerritories"].isin(top10_countries)]
    pivot = heat_df.pivot_table(
        index="countriesAndTerritories", columns="month_year",
        values="cases", aggfunc="sum", fill_value=0
    )
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = styled_fig(10, 5)
    sns.heatmap(
        pivot, ax=ax, cmap="Blues",
        linewidths=0.3, linecolor=PALETTE["bg"],
        cbar_kws={"shrink": 0.8},
        fmt=".0f", annot=False,
    )
    ax.set_title("Monthly Cases — Top 10 Countries (Heatmap)",
                 color=PALETTE["text"], fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Month", color=PALETTE["subtext"], fontsize=10)
    ax.set_ylabel("", fontsize=10)
    plt.xticks(rotation=45, ha="right", color=PALETTE["subtext"])
    plt.yticks(color=PALETTE["subtext"])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with col8:
    st.markdown("**8 — Area Chart: Cumulative Cases by Continent**")
    area_df = df.groupby(["dateRep", "continentExp"])["cases"].sum().reset_index()
    area_pivot = area_df.pivot_table(index="dateRep", columns="continentExp",
                                      values="cases", fill_value=0)
    area_cumsum = area_pivot.cumsum()

    fig, ax = styled_fig(9, 5)
    conts_plot = area_cumsum.columns.tolist()
    for i, col_name in enumerate(conts_plot):
        ax.fill_between(area_cumsum.index, area_cumsum[col_name],
                        alpha=0.55, color=CHART_COLORS[i % len(CHART_COLORS)], label=col_name)
        ax.plot(area_cumsum.index, area_cumsum[col_name],
                color=CHART_COLORS[i % len(CHART_COLORS)], linewidth=1)
    style_ax(ax, "Cumulative COVID-19 Cases by Continent", "Date", "Cumulative Cases")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────
# ROW 5: Count Plot + Violin Plot
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🎻 Frequency & Density Analysis</div>', unsafe_allow_html=True)
col9, col10 = st.columns(2)

with col9:
    st.markdown("**9 — Count Plot: Records per Continent**")
    count_data = df["continentExp"].value_counts().reset_index()
    count_data.columns = ["Continent", "Count"]

    fig, ax = styled_fig(8, 5)
    bars = ax.bar(count_data["Continent"], count_data["Count"],
                  color=CHART_COLORS[:len(count_data)],
                  edgecolor=PALETTE["bg"], linewidth=0.5)
    for bar, val in zip(bars, count_data["Count"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + count_data["Count"].max() * 0.01,
                f"{val:,}", ha="center", va="bottom",
                color=PALETTE["subtext"], fontsize=8)
    style_ax(ax, "Record Count by Continent", "Continent", "Number of Records")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with col10:
    st.markdown("**10 — Violin Plot: Deaths Distribution by Continent**")
    violin_data = []
    violin_labels = []
    for c in continents_all:
        sub = df[df["continentExp"] == c]["deaths"].clip(0).values
        sub = sub[sub > 0]
        if len(sub) > 5:
            violin_data.append(sub)
            violin_labels.append(c)

    fig, ax = styled_fig(8, 5)
    if violin_data:
        parts = ax.violinplot(violin_data, positions=range(len(violin_labels)),
                               showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(CHART_COLORS[i % len(CHART_COLORS)])
            pc.set_alpha(0.7)
            pc.set_edgecolor(PALETTE["border"])
        parts["cmedians"].set_color(PALETTE["accent5"])
        parts["cmedians"].set_linewidth(2)
        parts["cmins"].set_color(PALETTE["subtext"])
        parts["cmaxes"].set_color(PALETTE["subtext"])
        parts["cbars"].set_color(PALETTE["subtext"])
        ax.set_xticks(range(len(violin_labels)))
        ax.set_xticklabels(violin_labels, rotation=20, ha="right")
    style_ax(ax, "Death Distribution & Density by Continent", "Continent", "Deaths")
    ax.set_yscale("symlog")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────
# BONUS: Correlation Heatmap
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🔗 Feature Correlations (Bonus)</div>', unsafe_allow_html=True)

num_cols = ["cases", "deaths", "popData2019", "cum14", "mortality_rate"]
corr_df = df[num_cols].dropna()
corr_matrix = corr_df.corr()

fig, ax = styled_fig(8, 5)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, ax=ax, mask=mask,
    cmap=sns.diverging_palette(220, 20, as_cmap=True),
    annot=True, fmt=".2f", annot_kws={"size": 10, "color": PALETTE["text"]},
    linewidths=0.5, linecolor=PALETTE["bg"],
    cbar_kws={"shrink": 0.8},
    vmin=-1, vmax=1,
)
ax.set_title("Feature Correlation Matrix",
             color=PALETTE["text"], fontsize=13, fontweight="bold", pad=10)
plt.xticks(color=PALETTE["subtext"])
plt.yticks(color=PALETTE["subtext"])
fig.tight_layout()
st.pyplot(fig)
plt.close()


# ─────────────────────────────────────────────
# RAW DATA TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Filtered Dataset Preview</div>', unsafe_allow_html=True)
show_cols = ["dateRep", "countriesAndTerritories", "continentExp",
             "cases", "deaths", "mortality_rate", "popData2019", "cum14"]
st.dataframe(
    df[show_cols].sort_values("dateRep", ascending=False).head(500),
    use_container_width=True,
    height=320,
)

st.markdown(f"""
<div style="text-align:center;color:{PALETTE['subtext']};font-size:11px;
            margin-top:32px;padding-top:16px;border-top:1px solid {PALETTE['border']};">
    COVID-19 Dashboard · Data Source: ECDC EU/EEA · Filtered {len(df):,} records across {df['countriesAndTerritories'].nunique()} countries
</div>
""", unsafe_allow_html=True)

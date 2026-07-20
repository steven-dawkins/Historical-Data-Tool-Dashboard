import math

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Historical Data Dashboard", layout="wide")

DATA_DIR = "./"
SECTORS = ["UK", "COLOMBIA"]


def _parse_dates(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    mask = result.isna()
    if mask.any():
        # "1990 Q1" / "1990Q2" / "2020-Q1" → decimal year (Q1=.0, Q2=.25, Q3=.5, Q4=.75)
        quarterly = series[mask].str.extract(r"(\d{4})[\s-]*[Qq](\d)", expand=True)
        valid = quarterly[0].notna()
        if valid.any():
            year = pd.to_numeric(quarterly.loc[valid, 0])
            quarter = pd.to_numeric(quarterly.loc[valid, 1])
            result.loc[quarterly.index[valid]] = year + (quarter - 1) * 0.25

    mask = result.isna()
    if mask.any():
        # ISO "YYYY-MM-DD" (daily/monthly) → decimal year via day-of-year fraction
        parsed = pd.to_datetime(series[mask], format="%Y-%m-%d", errors="coerce")
        valid = parsed.notna()
        if valid.any():
            idx = parsed.index[valid]
            dt = parsed.loc[idx]
            days_in_year = dt.dt.is_leap_year.map({True: 366, False: 365})
            result.loc[idx] = dt.dt.year + (dt.dt.dayofyear - 1) / days_in_year
    return result


def _apply_dual_axis(fig, plot_df: pd.DataFrame, series_col: str, series_order: list) -> None:
    """If there are exactly two series and their ranges differ by more than
    3x, split the second series onto a secondary y-axis, zero-based and
    scaled to a clean power-of-10 multiple of the first so equal visual
    height always means the same order-of-magnitude ratio between them."""
    if len(series_order) != 2:
        return

    s0_vals = plot_df[plot_df[series_col] == series_order[0]]["value"].dropna()
    s1_vals = plot_df[plot_df[series_col] == series_order[1]]["value"].dropna()
    r0 = s0_vals.max() - s0_vals.min() if len(s0_vals) else 0
    r1 = s1_vals.max() - s1_vals.min() if len(s1_vals) else 0
    max_r, min_r = max(r0, r1), min(r0, r1)
    if not (min_r > 0 and (max_r / min_r) > 3):
        return

    second = series_order[1]
    for trace in fig.data:
        if trace.name == second:
            trace.yaxis = "y2"

    top0 = max(s0_vals.max(), 0) if len(s0_vals) else 0
    top1 = max(s1_vals.max(), 0) if len(s1_vals) else 0
    headroom = 1.1
    # Fold the headroom into the power threshold so a ratio that lands a
    # hair above a clean power of 10 (e.g. 1.04e6) doesn't get bumped a
    # whole extra order of magnitude when the headroom alone already
    # covers the actual max.
    if top0 > 0 and top1 > 0:
        if top0 >= top1:
            power = max(1, math.ceil(math.log10(top0 / top1) - math.log10(headroom) - 1e-9))
            top1_range = top1 * headroom
            top0_range = top1_range * (10 ** power)
        else:
            power = max(1, math.ceil(math.log10(top1 / top0) - math.log10(headroom) - 1e-9))
            top0_range = top0 * headroom
            top1_range = top0_range * (10 ** power)
    else:
        top0_range = top0 * headroom if top0 else 1
        top1_range = top1 * headroom if top1 else 1

    # Don't clip negative data at a hard zero floor — extend the bottom of
    # each axis down to its own series' actual minimum.
    bottom0 = min(s0_vals.min(), 0) if len(s0_vals) else 0
    bottom1 = min(s1_vals.min(), 0) if len(s1_vals) else 0
    bottom0_range = bottom0 * headroom if bottom0 < 0 else 0
    bottom1_range = bottom1 * headroom if bottom1 < 0 else 0

    fig.update_layout(
        yaxis=dict(title=series_order[0], range=[bottom0_range, top0_range]),
        yaxis2=dict(title=second, overlaying="y", side="right", range=[bottom1_range, top1_range]),
    )


@st.cache_data
def load_data(sector: str) -> pd.DataFrame:
    files = [
        DATA_DIR + sector + "_data_alt.csv",
        DATA_DIR + sector + "_data_haver.csv",
    ]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["date"] = _parse_dates(df["date"].astype(str))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "categories" in df.columns:
        # haver-only rows carry the precomputed category under "categories";
        # alt rows carry it under "category" — fold into one column.
        df["category"] = df["category"].fillna(df["categories"]) if "category" in df.columns else df["categories"]
        df = df.drop(columns="categories")
    return df


title_col, sector_col = st.columns([4, 1])
with sector_col:
    SECTOR = st.selectbox("Sector", SECTORS)

df = load_data(SECTOR)

with title_col:
    st.title(f"Historical Data Dashboard — {SECTOR}")

row1_left, row1_right = st.columns([3, 2])
row2_left, row2_right = st.columns([3, 1])
row3_left, row3_right = st.columns([2, 3])

filtered = df

if "haver_long_source" in df.columns:
    haver_rows = df[df["provider"] == "haver-local"].drop_duplicates(subset=["ProviderMnemonic"])
    alt_mnemonics = set(df.loc[df["provider"] != "haver-local", "ProviderMnemonic"])
    source_counts = (
        haver_rows
        .assign(has_alt=haver_rows["ProviderMnemonic"].isin(alt_mnemonics))
        .groupby("haver_long_source")["has_alt"]
        .agg(n_alt="sum", n_haver="count")
    )

    sources = sorted(df["haver_long_source"].dropna().unique())
    with row1_right:
        selected_sources = st.pills(
            "Source",
            options=sources,
            selection_mode="multi",
            format_func=lambda s: f"{s} ({source_counts.loc[s, 'n_alt']}/{source_counts.loc[s, 'n_haver']})",
        )
    if selected_sources:
        haver_mnemonics = df.loc[
            (df["provider"] == "haver-local") & df["haver_long_source"].isin(selected_sources),
            "ProviderMnemonic",
        ]
        filtered = df[df["ProviderMnemonic"].isin(haver_mnemonics)]

# --- Provider filter ---
provider_options = sorted(df["provider"].dropna().unique())
provider_mnemonic_counts = (
    df.dropna(subset=["provider"]).groupby("provider")["ProviderMnemonic"].nunique()
)
with row2_right:
    selected_providers = st.pills(
        "Provider",
        options=provider_options,
        selection_mode="multi",
        format_func=lambda p: f"{p} ({provider_mnemonic_counts.get(p, 0)})",
    )
if selected_providers:
    provider_mnemonics = filtered.loc[
        filtered["provider"].isin(selected_providers), "ProviderMnemonic"
    ]
    filtered = filtered[filtered["ProviderMnemonic"].isin(provider_mnemonics)]


category_selected_mnemonics: list = []
category_pie_fig = None

with row2_left:
    st.subheader("Comparison table")
    if "category" not in filtered.columns:
        st.info("No comparison data in the loaded source files; category columns are missing.")
    elif not (filtered["provider"] == "haver-local").any():
        st.info("No haver-local data loaded; cannot categorize alternate sources.")
    else:
        diag_cols = [c for c in ["n_haver", "n_alt", "n_overlap", "corr", "ratio_median"] if c in filtered.columns]

        # Alt-side rows carry a category per (ProviderMnemonic, provider) pair.
        alt_categorized = (
            filtered[filtered["provider"] != "haver-local"]
            .dropna(subset=["category"])
            .drop_duplicates(subset=["ProviderMnemonic", "provider"])
            [["ProviderMnemonic", "provider", "category"] + diag_cols]
        )

        # Mnemonics with no alt pairing fall back to the haver-local category.
        haver_only = (
            filtered[
                (filtered["provider"] == "haver-local")
                & ~filtered["ProviderMnemonic"].isin(alt_categorized["ProviderMnemonic"])
            ]
            .dropna(subset=["category"])
            .drop_duplicates(subset=["ProviderMnemonic"])
            [["ProviderMnemonic", "category"]]
        )
        haver_only["provider"] = None

        category_df = (
            pd.concat([alt_categorized, haver_only], ignore_index=True)
            .sort_values(["category", "ProviderMnemonic"])
            .reset_index(drop=True)
        )
        category_counts_all = category_df["category"].value_counts()

        categories = sorted(category_df["category"].unique())
        selected_categories = st.pills(
            "Category",
            options=categories,
            selection_mode="multi",
            format_func=lambda c: f"{c} ({category_counts_all.loc[c]})",
        )
        if selected_categories:
            category_df = category_df[category_df["category"].isin(selected_categories)].reset_index(drop=True)

        category_counts = category_df["category"].value_counts()

        st.caption(" · ".join(f"{cat}: {n}" for cat, n in category_counts.items()) + " — select a row to plot and see diagnostics.")
        with row3_left:
            category_selection = st.dataframe(
                category_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                height=580,
                row_height=26,
                column_order=["ProviderMnemonic", "provider", "category"],
                column_config={"ProviderMnemonic": st.column_config.Column("haver_code")},
            )
        category_selected_rows = category_selection.selection.rows
        if category_selected_rows:
            category_selected_mnemonics = (
                category_df.iloc[category_selected_rows]["ProviderMnemonic"].unique().tolist()
            )
            diag = category_df.iloc[category_selected_rows[0]]
            if diag_cols:
                st.caption(" · ".join(f"{c}: {diag[c]}" for c in diag_cols))

        _CATEGORY_COLORS = {
            "Matches": "#0ca30c",
            "Backfilled by haver": "#2a78d6",
            "Different scale": "#fab219",
            "Different frequency": "#ec835a",
            "Complete mismatch": "#d03b3b",
            "Missing alternate source": "#898781",
            "Discontinued": "#4d4d4d",
        }
        category_pie_fig = px.pie(
            category_counts.reset_index(name="count"),
            names="category",
            values="count",
            title="ProviderMnemonic comparison outcomes",
            color="category",
            color_discrete_map=_CATEGORY_COLORS,
            category_orders={"category": list(_CATEGORY_COLORS.keys())},
        )
        category_pie_fig.update_traces(textinfo="percent+value")
        category_pie_fig.update_layout(height=430, margin=dict(t=40, b=10))

# Build pivot table
meta = (
    filtered
    .drop_duplicates(subset=["ProviderMnemonic", "provider"])
    [["ProviderMnemonic", "provider"]]
)
all_providers = sorted(filtered["provider"].dropna().unique())
pivot = (
    meta
    .groupby(["ProviderMnemonic"])["provider"]
    .apply(set)
    .reset_index()
)
other_providers = [p for p in all_providers if p != "haver-local"]
if "haver-local" in all_providers:
    pivot["haver-local"] = pivot["provider"].apply(lambda s: "haver-local" in s)
pivot["providers"] = pivot["provider"].apply(
    lambda s: ", ".join(p for p in other_providers if p in s)
)
pivot = pivot.drop(columns="provider").sort_values("ProviderMnemonic").reset_index(drop=True)

# --- Coverage by haver_long_source ---
if "haver_long_source" not in filtered.columns:
    st.info("Restart the app to load haver_long_source data.")
    long_source_map = pd.DataFrame(columns=["ProviderMnemonic", "haver_long_source"])
else:
    long_source_map = (
        filtered[filtered["provider"] == "haver-local"]
        .drop_duplicates(subset=["ProviderMnemonic"])
        [["ProviderMnemonic", "haver_long_source"]]
    )
pivot_with_source = pivot.merge(long_source_map, on="ProviderMnemonic", how="left")
has_other_col = pivot_with_source["providers"].str.strip().ne("")
source_summary = (
    pivot_with_source
    .assign(has_other=has_other_col)
    .groupby("haver_long_source")
    .agg(
        with_alternate=("has_other", "sum"),
        haver_only=("has_other", lambda x: (~x).sum()),
    )
    .astype(int)
    .sort_values("haver_only", ascending=False)
    .reset_index()
    .rename(columns={"haver_long_source": "Source", "with_alternate": "With alternate provider", "haver_only": "Haver only"})
)

with row1_left:
    if category_pie_fig is not None:
        st.plotly_chart(category_pie_fig, use_container_width=True)
    else:
        st.info("No category data to display.")

selected_mnemonics = sorted(set(category_selected_mnemonics))
if selected_mnemonics:
    chart_df = filtered[filtered["ProviderMnemonic"].isin(selected_mnemonics)]
else:
    chart_df = filtered

with row3_right:
    st.subheader("Chart")
    st.caption(f"{chart_df['ProviderMnemonic'].nunique():,} series · {len(chart_df):,} rows")

    if chart_df.empty:
        st.warning("No data matches the current filters.")
    else:
        plot_df = chart_df.dropna(subset=["date", "value"]).sort_values("date")

        n_series = plot_df["ProviderMnemonic"].nunique()
        if n_series > 50:
            st.info(f"{n_series} series — showing first 50. Select rows in the table to reduce.")
            top50 = plot_df["ProviderMnemonic"].unique()[:50]
            plot_df = plot_df[plot_df["ProviderMnemonic"].isin(top50)]

        plot_df = plot_df.copy()
        if plot_df["provider"].nunique() > 1:
            plot_df["_series"] = plot_df["indicator_id"] + "  [" + plot_df["provider"] + "]"
        else:
            plot_df["_series"] = plot_df["indicator_id"]

        series_order = list(plot_df["_series"].unique())

        fig = px.line(
            plot_df,
            x="date",
            y="value",
            color="_series",
            hover_data=["provider", "indicator_name", "haver_description"],
            labels={"date": "Year", "value": "Value", "_series": "Series"},
        )

        _apply_dual_axis(fig, plot_df, "_series", series_order)

        fig.update_layout(
            height=580,
            legend=dict(orientation="v", x=1.01, y=1),
            margin=dict(r=220),
        )
        st.plotly_chart(fig, use_container_width=True)

if not chart_df.empty:
    # Build from the full chart_df (not the 50-series-capped plot_df) so the
    # table lists every series from both sources.
    all_series = chart_df.copy()
    if all_series["provider"].nunique() > 1:
        all_series["_series"] = all_series["indicator_id"] + "  [" + all_series["provider"] + "]"
    else:
        all_series["_series"] = all_series["indicator_id"]
    series_info = (
        all_series
        .drop_duplicates(subset=["_series"])
        [["_series", "ProviderMnemonic", "provider", "indicator_id", "indicator_name"]]
        .rename(columns={"_series": "Series"})
        .sort_values("Series")
        .reset_index(drop=True)
    )
    series_info = series_info.merge(long_source_map, on="ProviderMnemonic", how="left")
    series_info = series_info[
        ["Series", "provider", "indicator_id", "indicator_name", "haver_long_source"]
    ]
    st.caption(f"{len(series_info):,} series")
    st.dataframe(series_info, use_container_width=True, hide_index=True)

st.subheader("Coverage by source")
st.dataframe(source_summary, use_container_width=True, hide_index=True, height=430)

# --- Raw data table ---
st.subheader("Side by side data")
raw = (
    chart_df
    .pivot_table(
        index=["date"],
        columns="provider",
        values="value",
        aggfunc="first",
    )
    .reset_index()
    .sort_values("date")
)
raw.columns.name = None
st.dataframe(raw, use_container_width=True, hide_index=True)

# --- Raw unformatted data table ---
st.subheader("Raw unformatted data")
raw_unformatted = chart_df.sort_values("date").drop(columns=["haver_description", "OeMnemonic"], errors="ignore").copy()
obj_cols = raw_unformatted.select_dtypes(include="object").columns
raw_unformatted[obj_cols] = raw_unformatted[obj_cols].fillna("")
st.dataframe(raw_unformatted, use_container_width=True, hide_index=True)

# --- Comparison table with a full chart per row ---
st.subheader("Comparison with charts")
if "category_df" in globals() and category_df is not None and not category_df.empty:
    plot_all = filtered.dropna(subset=["date", "value"]).sort_values("date")

    _ROW_CAP = 40
    rows_to_show = category_df
    if len(category_df) > _ROW_CAP:
        st.info(f"{len(category_df)} rows — showing first {_ROW_CAP}. Use the filters above to narrow down.")
        rows_to_show = category_df.head(_ROW_CAP)

    for _, row in rows_to_show.iterrows():
        mnemonic, prov, cat = row["ProviderMnemonic"], row["provider"], row["category"]
        col_meta, col_row_chart = st.columns([1, 3])

        with col_meta:
            st.markdown(f"**{mnemonic}**")
            st.caption(f"{prov or '—'} · {cat}")

        with col_row_chart:
            providers = ["haver-local"] + ([prov] if prov else [])
            row_df = plot_all[
                (plot_all["ProviderMnemonic"] == mnemonic)
                & (plot_all["provider"].isin(providers))
            ]
            if row_df.empty:
                st.caption("No data to plot.")
            else:
                row_series_order = list(row_df["provider"].unique())
                row_fig = px.line(
                    row_df,
                    x="date",
                    y="value",
                    color="provider",
                    category_orders={"provider": row_series_order},
                    labels={"date": "Year", "value": "Value", "provider": "Provider"},
                )
                _apply_dual_axis(row_fig, row_df, "provider", row_series_order)
                row_fig.update_layout(
                    height=180,
                    margin=dict(t=10, b=10, l=10, r=40 if len(row_series_order) == 2 else 10),
                    legend=dict(orientation="h", y=1.15, x=0),
                    showlegend=True,
                )
                st.plotly_chart(
                    row_fig,
                    use_container_width=True,
                    key=f"rowchart_{mnemonic}_{prov}",
                )
else:
    st.info("No comparison data available to chart.")

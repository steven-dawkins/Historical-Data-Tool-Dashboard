import math

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Historical Data Dashboard", layout="wide")

DATA_DIR = "./"
FILES = [
    DATA_DIR + "UK_data_alt.csv",
    DATA_DIR + "UK_data_haver.csv",
]


def _parse_dates(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    mask = result.isna()
    if mask.any():
        # "1990 Q1" / "1990Q2" → decimal year (Q1=.0, Q2=.25, Q3=.5, Q4=.75)
        quarterly = series[mask].str.extract(r"(\d{4})\s*[Qq](\d)", expand=True)
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


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in FILES], ignore_index=True)
    df["date"] = _parse_dates(df["date"].astype(str))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


df = load_data()

st.title("Historical Data Dashboard")

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


def _infer_step(dates: pd.Series) -> float | None:
    uniq = sorted(dates.dropna().unique())
    if len(uniq) < 2:
        return None
    diffs = pd.Series(uniq).diff().dropna()
    return diffs.median() if len(diffs) else None


def _categorize_pair(haver_g: pd.DataFrame, alt_g: pd.DataFrame) -> dict:
    n_haver, n_alt = len(haver_g), len(alt_g)
    result = {
        "n_haver": n_haver,
        "n_alt": n_alt,
        "n_overlap": None,
        "corr": None,
        "ratio_median": None,
    }

    if n_alt == 0:
        result["category"] = "Missing alternate source"
        return result

    haver_step = _infer_step(haver_g["date"])
    alt_step = _infer_step(alt_g["date"])
    if haver_step and alt_step:
        step_ratio = max(haver_step, alt_step) / min(haver_step, alt_step)
        if step_ratio > 1.8:
            # Different sampling frequency: confirm the two actually agree by
            # comparing annual averages rather than assuming a frequency gap
            # explains everything (a genuinely mismatched series can also
            # happen to sit at a different frequency).
            haver_annual = haver_g.assign(year=haver_g["date"].astype(int)).groupby("year")["value"].mean()
            alt_annual = alt_g.assign(year=alt_g["date"].astype(int)).groupby("year")["value"].mean()
            annual = pd.concat([haver_annual.rename("_haver"), alt_annual.rename("_alt")], axis=1).dropna()
            annual_valid = annual[(annual["_haver"] != 0) & annual["_haver"].notna() & annual["_alt"].notna()]

            if len(annual_valid) > 2:
                annual_ratios = annual_valid["_alt"] / annual_valid["_haver"]
                annual_ratio_median = annual_ratios.median()
                annual_ratio_cv = (
                    (annual_ratios.std() / abs(annual_ratio_median)) if annual_ratio_median else float("inf")
                )
                annual_corr = annual["_alt"].corr(annual["_haver"])
                result["n_overlap"] = len(annual)
                result["ratio_median"] = round(annual_ratio_median, 4)
                result["corr"] = round(annual_corr, 3) if annual_corr is not None else None

                if (annual_corr is not None and annual_corr < 0.5) or annual_ratio_cv > 0.25:
                    result["category"] = "Complete mismatch"
                    return result

                if not (0.9 <= annual_ratio_median <= 1.1):
                    result["category"] = "Different scale"
                    return result

            result["category"] = "Different frequency"
            return result

    merged = alt_g[["date", "value"]].merge(
        haver_g[["date", "value"]], on="date", suffixes=("_alt", "_haver")
    )
    n_overlap = len(merged)
    result["n_overlap"] = n_overlap

    if n_overlap == 0:
        haver_min, haver_max = haver_g["date"].min(), haver_g["date"].max()
        alt_min, alt_max = alt_g["date"].min(), alt_g["date"].max()
        if haver_max <= alt_min or alt_max <= haver_min:
            result["category"] = "Backfilled by haver"
        else:
            result["category"] = "Complete mismatch"
        return result

    valid = merged[
        merged["value_haver"].notna()
        & merged["value_alt"].notna()
        & (merged["value_haver"] != 0)
    ]
    if len(valid) == 0:
        result["category"] = "Complete mismatch"
        return result

    ratios = valid["value_alt"] / valid["value_haver"]
    ratio_median = ratios.median()
    ratio_cv = (ratios.std() / abs(ratio_median)) if ratio_median else float("inf")
    corr = merged["value_alt"].corr(merged["value_haver"]) if n_overlap > 2 else None
    result["ratio_median"] = round(ratio_median, 4)
    result["corr"] = round(corr, 3) if corr is not None else None

    if (corr is not None and corr < 0.5) or ratio_cv > 0.25:
        result["category"] = "Complete mismatch"
        return result

    if not (0.9 <= ratio_median <= 1.1):
        result["category"] = "Different scale"
        return result

    extra_haver = n_haver - n_overlap
    if extra_haver > 0 and extra_haver / n_haver > 0.1:
        result["category"] = "Backfilled by haver"
        return result

    result["category"] = "Matches"
    return result


category_selected_mnemonics: list = []
category_pie_fig = None

# col_cat_pie holds the line chart (swapped with the pie chart below).
col_cat_table, col_cat_pie = st.columns([2, 3])

with col_cat_table:
    st.subheader("Comparison table")
    if (filtered["provider"] == "haver-local").any():
        haver_all = filtered[filtered["provider"] == "haver-local"][["ProviderMnemonic", "date", "value"]]
        alt_all = filtered[filtered["provider"] != "haver-local"][["ProviderMnemonic", "provider", "date", "value"]]
        empty_alt = alt_all.iloc[0:0]

        category_rows = []
        for mnemonic, haver_g in haver_all.groupby("ProviderMnemonic"):
            alt_for_mnemonic = alt_all[alt_all["ProviderMnemonic"] == mnemonic]
            alt_providers = sorted(alt_for_mnemonic["provider"].dropna().unique())
            if not alt_providers:
                category_rows.append(
                    {"ProviderMnemonic": mnemonic, "provider": None, **_categorize_pair(haver_g, empty_alt)}
                )
            else:
                for prov in alt_providers:
                    alt_g = alt_for_mnemonic[alt_for_mnemonic["provider"] == prov]
                    category_rows.append(
                        {"ProviderMnemonic": mnemonic, "provider": prov, **_categorize_pair(haver_g, alt_g)}
                    )

        category_df = pd.DataFrame(category_rows).sort_values(["category", "ProviderMnemonic"]).reset_index(drop=True)
        category_counts = category_df["category"].value_counts()

        st.caption(" · ".join(f"{cat}: {n}" for cat, n in category_counts.items()) + " — select a row to plot and see diagnostics.")
        category_selection = st.dataframe(
            category_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=580,
            row_height=26,
            column_order=["ProviderMnemonic", "provider", "category"],
        )
        category_selected_rows = category_selection.selection.rows
        if category_selected_rows:
            category_selected_mnemonics = (
                category_df.iloc[category_selected_rows]["ProviderMnemonic"].unique().tolist()
            )
            diag = category_df.iloc[category_selected_rows[0]]
            st.caption(
                f"n_haver: {diag['n_haver']} · n_alt: {diag['n_alt']} · n_overlap: {diag['n_overlap']} "
                f"· corr: {diag['corr']} · ratio_median: {diag['ratio_median']}"
            )

        _CATEGORY_COLORS = {
            "Matches": "#0ca30c",
            "Backfilled by haver": "#2a78d6",
            "Different scale": "#fab219",
            "Different frequency": "#ec835a",
            "Complete mismatch": "#d03b3b",
            "Missing alternate source": "#898781",
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
    else:
        st.info("No haver-local data loaded; cannot categorize alternate sources.")

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

# Compute sum of absolute differences between haver-local and each other provider per mnemonic
if "haver-local" in all_providers and other_providers:
    haver_vals = (
        filtered[filtered["provider"] == "haver-local"]
        [["ProviderMnemonic", "date", "value"]]
        .rename(columns={"value": "_haver"})
    )
    other_vals = filtered[filtered["provider"].isin(other_providers)][["ProviderMnemonic", "date", "value", "provider"]]
    diff_df = other_vals.merge(haver_vals, on=["ProviderMnemonic", "date"])
    if diff_df.empty:
        diff_by_mnemonic = pd.Series(dtype=float)
    else:
        diff_by_mnemonic = (
            diff_df.groupby(["ProviderMnemonic", "provider"])
            .apply(lambda g: (g["value"] - g["_haver"]).abs().sum() / len(g), include_groups=False)
            .reset_index(name="diff")
            .groupby("ProviderMnemonic")["diff"]
            .sum()
        )
    pivot["sum_diff"] = pivot["ProviderMnemonic"].map(diff_by_mnemonic)

    # Compute period-on-period ratios then sum differences in those ratios
    haver_tmp = (
        filtered[filtered["provider"] == "haver-local"]
        .sort_values(["ProviderMnemonic", "date"])
        .copy()
    )
    haver_tmp["_haver_ratio"] = haver_tmp.groupby("ProviderMnemonic")["value"].transform(lambda s: s / s.shift(1))
    haver_ratios = haver_tmp[["ProviderMnemonic", "date", "_haver_ratio"]].dropna(subset=["_haver_ratio"])

    other_tmp = (
        filtered[filtered["provider"].isin(other_providers)]
        .sort_values(["ProviderMnemonic", "provider", "date"])
        .copy()
    )
    other_tmp["ratio"] = other_tmp.groupby(["ProviderMnemonic", "provider"])["value"].transform(lambda s: s / s.shift(1))
    other_ratios = other_tmp[["ProviderMnemonic", "provider", "date", "ratio"]].dropna(subset=["ratio"])
    ratio_diff_df = other_ratios.merge(haver_ratios, on=["ProviderMnemonic", "date"])
    if ratio_diff_df.empty:
        ratio_diff_by_mnemonic = pd.Series(dtype=float)
    else:
        ratio_diff_by_mnemonic = (
            ratio_diff_df.groupby(["ProviderMnemonic", "provider"])
            .apply(lambda g: (g["ratio"] - g["_haver_ratio"]).abs().sum() / len(g), include_groups=False)
            .reset_index(name="ratio_diff")
            .groupby("ProviderMnemonic")["ratio_diff"]
            .sum()
        )
    pivot["sum_diff_ratio"] = pivot["ProviderMnemonic"].map(ratio_diff_by_mnemonic)

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

# --- Side-by-side layout ---
col_table, col_chart = st.columns([2, 3])

with col_table:
    st.subheader("Coverage by source")
    st.dataframe(source_summary, use_container_width=True, hide_index=True, height=430)

with col_chart:
    if category_pie_fig is not None:
        st.plotly_chart(category_pie_fig, use_container_width=True)
    else:
        st.info("No category data to display.")

selected_mnemonics = sorted(set(category_selected_mnemonics))
if selected_mnemonics:
    chart_df = filtered[filtered["ProviderMnemonic"].isin(selected_mnemonics)]
else:
    chart_df = filtered

with col_cat_pie:
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

        if len(series_order) == 2:
            s0_vals = plot_df[plot_df["_series"] == series_order[0]]["value"].dropna()
            s1_vals = plot_df[plot_df["_series"] == series_order[1]]["value"].dropna()
            r0 = s0_vals.max() - s0_vals.min() if len(s0_vals) else 0
            r1 = s1_vals.max() - s1_vals.min() if len(s1_vals) else 0
            max_r, min_r = max(r0, r1), min(r0, r1)
            use_dual_axis = min_r > 0 and (max_r / min_r) > 3

            if use_dual_axis:
                second = series_order[1]
                for trace in fig.data:
                    if trace.name == second:
                        trace.yaxis = "y2"

                # Zero-base both axes and force their tops to a clean power-of-10
                # multiple of each other, so equal visual height always means the
                # same order-of-magnitude ratio between the two series.
                top0 = max(s0_vals.max(), 0) if len(s0_vals) else 0
                top1 = max(s1_vals.max(), 0) if len(s1_vals) else 0
                headroom = 1.1
                # Fold the headroom into the power threshold so a ratio that lands
                # a hair above a clean power of 10 (e.g. 1.04e6) doesn't get bumped
                # a whole extra order of magnitude when the headroom alone already
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

                # Don't clip negative data at a hard zero floor — extend the
                # bottom of each axis down to its own series' actual minimum.
                bottom0 = min(s0_vals.min(), 0) if len(s0_vals) else 0
                bottom1 = min(s1_vals.min(), 0) if len(s1_vals) else 0
                bottom0_range = bottom0 * headroom if bottom0 < 0 else 0
                bottom1_range = bottom1 * headroom if bottom1 < 0 else 0

                fig.update_layout(
                    yaxis=dict(title=series_order[0], range=[bottom0_range, top0_range]),
                    yaxis2=dict(title=second, overlaying="y", side="right", range=[bottom1_range, top1_range]),
                )

        fig.update_layout(
            height=580,
            legend=dict(orientation="v", x=1.01, y=1),
            margin=dict(r=220),
        )
        st.plotly_chart(fig, use_container_width=True)

if not chart_df.empty:
    series_info = (
        plot_df
        .drop_duplicates(subset=["_series"])
        [["_series", "indicator_id", "indicator_name"]]
        .rename(columns={"_series": "Series", "indicator_id": "indicator_id", "indicator_name": "indicator_name"})
        .set_index("Series")
        .loc[series_order]
        .reset_index()
    )
    st.dataframe(series_info, use_container_width=True, hide_index=True)

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

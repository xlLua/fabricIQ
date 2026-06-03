import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="fabricIQ Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_products() -> pd.DataFrame:
    return pd.read_csv("DimProducts.csv")


@st.cache_data
def load_stores() -> pd.DataFrame:
    return pd.read_csv("DimStore.csv")


@st.cache_data
def load_freezers() -> pd.DataFrame:
    return pd.read_csv("Freezer.csv")


@st.cache_data
def load_sales() -> pd.DataFrame:
    return pd.read_csv("FactSales.csv", parse_dates=["SaleDate"])


@st.cache_data
def load_telemetry() -> pd.DataFrame:
    df = pd.read_csv("FreezerTelemetry.csv", parse_dates=["timestamp"])
    # Strip UTC timezone to avoid tz-naive/tz-aware comparison errors
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.rename(columns={"storeId": "StoreId", "freezerId": "FreezerId"})
    return df


# ── Enrichment ────────────────────────────────────────────────────────────────

def build_enriched_sales(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    return (
        sales
        .merge(
            products[["ProductId", "ProductName", "Category", "Subcategory"]],
            on="ProductId", how="left",
        )
        .merge(
            stores[["StoreId", "StoreName", "City", "Region", "Latitude", "Longitude"]],
            on="StoreId", how="left",
        )
    )


def build_enriched_telemetry(
    telemetry: pd.DataFrame,
    stores: pd.DataFrame,
    freezers: pd.DataFrame,
) -> pd.DataFrame:
    # Telemetry already carries StoreId, so we exclude it from freezers to avoid collision
    return (
        telemetry
        .merge(
            freezers[["FreezerId", "Model", "minSafeTempC"]],
            on="FreezerId", how="left",
        )
        .merge(
            stores[["StoreId", "StoreName", "City"]],
            on="StoreId", how="left",
        )
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
):
    with st.sidebar:
        st.title("fabricIQ")
        st.caption("Cold-chain retail intelligence")
        st.divider()

        min_date = sales["SaleDate"].min().date()
        max_date = sales["SaleDate"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        all_stores = sorted(stores["StoreName"].unique())
        sel_stores = st.multiselect("Stores", all_stores, default=all_stores)

        all_cats = sorted(products["Category"].unique())
        sel_cats = st.multiselect("Category", all_cats, default=all_cats)

    return date_range, sel_stores, sel_cats


# ── Filter helpers ────────────────────────────────────────────────────────────

def apply_filters(
    df: pd.DataFrame,
    date_range,
    sel_stores: list,
    sel_cats: list,
) -> pd.DataFrame:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1])
    mask = (
        df["SaleDate"].between(start, end)
        & df["StoreName"].isin(sel_stores)
        & df["Category"].isin(sel_cats)
    )
    return df[mask]


def apply_telemetry_filters(
    df: pd.DataFrame,
    date_range,
    sel_stores: list,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    store_ids = stores.loc[stores["StoreName"].isin(sel_stores), "StoreId"]
    mask = df["timestamp"].between(start, end) & df["StoreId"].isin(store_ids)
    return df[mask]


# ── Section renderers ─────────────────────────────────────────────────────────

def render_kpis(fdf: pd.DataFrame) -> None:
    total_rev = fdf["RevenueUSD"].sum()
    total_units = fdf["Units"].sum()
    avg_daily_rev = fdf.groupby("SaleDate")["RevenueUSD"].sum().mean()
    rev_per_unit = total_rev / total_units if total_units else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${total_rev:,.0f}")
    c2.metric("Total Units Sold", f"{total_units:,}")
    c3.metric("Avg Daily Revenue", f"${avg_daily_rev:,.0f}")
    c4.metric("Revenue Per Unit", f"${rev_per_unit:.2f}")


def render_sales_trends(fdf: pd.DataFrame) -> None:
    st.subheader("Sales Trends")
    show_breakdown = st.toggle("Break down by store", value=False)
    tab_daily, tab_roll = st.tabs(["Daily Revenue", "7-Day Rolling Average"])

    with tab_daily:
        if show_breakdown:
            daily = (
                fdf.groupby(["SaleDate", "StoreName"])["RevenueUSD"]
                .sum().reset_index()
            )
            fig = px.line(
                daily, x="SaleDate", y="RevenueUSD", color="StoreName",
                labels={"SaleDate": "Date", "RevenueUSD": "Revenue (USD)", "StoreName": "Store"},
                title="Daily Revenue by Store",
            )
        else:
            daily = fdf.groupby("SaleDate")["RevenueUSD"].sum().reset_index()
            fig = px.line(
                daily, x="SaleDate", y="RevenueUSD",
                labels={"SaleDate": "Date", "RevenueUSD": "Revenue (USD)"},
                title="Daily Revenue",
            )
        st.plotly_chart(fig, use_container_width=True)

    with tab_roll:
        daily_total = (
            fdf.groupby("SaleDate")["RevenueUSD"].sum()
            .reset_index().sort_values("SaleDate")
        )
        daily_total["7-Day Avg"] = (
            daily_total["RevenueUSD"].rolling(window=7, min_periods=1).mean()
        )
        daily_total = daily_total.rename(columns={"RevenueUSD": "Daily Revenue"})
        fig2 = px.line(
            daily_total, x="SaleDate", y=["Daily Revenue", "7-Day Avg"],
            labels={"value": "Revenue (USD)", "SaleDate": "Date", "variable": "Series"},
            title="Revenue with 7-Day Rolling Average",
        )
        st.plotly_chart(fig2, use_container_width=True)


def render_store_performance(fdf: pd.DataFrame, stores_df: pd.DataFrame) -> None:
    st.subheader("Store Performance")
    tab_bar, tab_map, tab_heat = st.tabs(["Ranked Stores", "Map", "Revenue Heatmap"])

    with tab_bar:
        store_rev = (
            fdf.groupby("StoreName")["RevenueUSD"]
            .sum().reset_index()
            .sort_values("RevenueUSD", ascending=True)
        )
        fig = px.bar(
            store_rev, x="RevenueUSD", y="StoreName",
            orientation="h",
            color="RevenueUSD",
            color_continuous_scale="Blues",
            labels={"RevenueUSD": "Revenue (USD)", "StoreName": "Store"},
            title="Revenue by Store (Ranked)",
            text="RevenueUSD",
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_map:
        store_rev_map = (
            fdf.groupby("StoreId")["RevenueUSD"].sum().reset_index()
            .merge(
                stores_df[["StoreId", "StoreName", "City", "Latitude", "Longitude"]],
                on="StoreId", how="right",
            )
            .fillna({"RevenueUSD": 0})
        )
        store_rev_map["RevenueLabel"] = store_rev_map["RevenueUSD"].apply(
            lambda v: f"${v:,.0f}"
        )
        fig = px.scatter_mapbox(
            store_rev_map,
            lat="Latitude", lon="Longitude",
            size="RevenueUSD",
            color="RevenueUSD",
            hover_name="StoreName",
            hover_data={"City": True, "RevenueLabel": True,
                        "RevenueUSD": False, "Latitude": False, "Longitude": False},
            color_continuous_scale="Viridis",
            size_max=40,
            zoom=4,
            center={"lat": 50.5, "lon": 7.0},
            mapbox_style="open-street-map",
            title="Store Revenue Map",
        )
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

    with tab_heat:
        heat_df = (
            fdf.groupby(["StoreName", "ProductName"])["RevenueUSD"]
            .sum().reset_index()
        )
        pivot = heat_df.pivot(
            index="StoreName", columns="ProductName", values="RevenueUSD"
        ).fillna(0)
        fig = px.imshow(
            pivot,
            color_continuous_scale="YlOrRd",
            labels={"color": "Revenue (USD)"},
            title="Store × Product Revenue Heatmap",
            aspect="auto",
            text_auto=",.0f",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_product_analytics(fdf: pd.DataFrame) -> None:
    st.subheader("Product Analytics")
    tab_top, tab_pie, tab_scatter = st.tabs(
        ["Top Products", "Category Mix", "Units vs Revenue"]
    )

    with tab_top:
        prod_rev = (
            fdf.groupby(["ProductName", "Category"])["RevenueUSD"]
            .sum().reset_index()
            .sort_values("RevenueUSD", ascending=False)
        )
        fig = px.bar(
            prod_rev, x="ProductName", y="RevenueUSD",
            color="Category",
            labels={"RevenueUSD": "Revenue (USD)", "ProductName": "Product"},
            title="Revenue by Product",
            text="RevenueUSD",
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(xaxis_tickangle=-30, uniformtext_minsize=8)
        st.plotly_chart(fig, use_container_width=True)

    with tab_pie:
        cat_rev = fdf.groupby("Category")["RevenueUSD"].sum().reset_index()
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = px.pie(
                cat_rev, names="Category", values="RevenueUSD",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Revenue Share by Category",
            )
            fig.update_traces(textinfo="percent+label+value",
                              texttemplate="%{label}<br>%{percent:.1%}<br>$%{value:,.0f}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sub_rev = fdf.groupby("Subcategory")["RevenueUSD"].sum().reset_index()
            fig2 = px.pie(
                sub_rev, names="Subcategory", values="RevenueUSD",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Revenue Share by Subcategory",
            )
            fig2.update_traces(textinfo="percent+label")
            st.plotly_chart(fig2, use_container_width=True)

    with tab_scatter:
        prod_agg = (
            fdf.groupby(["ProductName", "Category"])
            .agg(TotalUnits=("Units", "sum"), TotalRevenue=("RevenueUSD", "sum"))
            .reset_index()
        )
        fig = px.scatter(
            prod_agg, x="TotalUnits", y="TotalRevenue",
            color="Category",
            size="TotalRevenue",
            hover_name="ProductName",
            text="ProductName",
            labels={"TotalUnits": "Total Units Sold", "TotalRevenue": "Total Revenue (USD)"},
            title="Units vs Revenue by Product",
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)


SAFE_TEMP = -18.0


def render_cold_chain(ftdf: pd.DataFrame, freezers_df: pd.DataFrame) -> None:
    st.subheader("Cold Chain Monitoring")

    if ftdf.empty:
        st.info("No telemetry data for the selected filters.")
        return

    # Compliance summary
    compliance = (
        ftdf.groupby("FreezerId")
        .agg(
            TotalReadings=("temperatureC", "count"),
            AlertReadings=("temperatureC", lambda x: (x > SAFE_TEMP).sum()),
            MinTemp=("temperatureC", "min"),
            MaxTemp=("temperatureC", "max"),
            AvgTemp=("temperatureC", "mean"),
            StoreName=("StoreName", "first"),
        )
        .reset_index()
    )
    compliance["CompliancePct"] = (
        (1 - compliance["AlertReadings"] / compliance["TotalReadings"]) * 100
    ).round(1)
    compliance["Status"] = compliance["AlertReadings"].apply(
        lambda n: "✅ OK" if n == 0 else "⚠️ ALERT"
    )

    st.markdown("**Freezer Compliance Summary**")
    st.dataframe(
        compliance[[
            "FreezerId", "StoreName", "Status",
            "TotalReadings", "AlertReadings", "CompliancePct",
            "MinTemp", "MaxTemp", "AvgTemp",
        ]].rename(columns={
            "FreezerId": "Freezer", "StoreName": "Store",
            "TotalReadings": "Readings", "AlertReadings": "Alerts",
            "CompliancePct": "Compliance %",
            "MinTemp": "Min °C", "MaxTemp": "Max °C", "AvgTemp": "Avg °C",
        }).style.format({
            "Min °C": "{:.1f}", "Max °C": "{:.1f}", "Avg °C": "{:.1f}",
            "Compliance %": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    tab_temp, tab_hum, tab_door = st.tabs(["Temperature", "Humidity", "Door Events"])

    with tab_temp:
        all_freezers = sorted(ftdf["FreezerId"].unique())
        sel_freezers = st.multiselect(
            "Select freezers", all_freezers, default=all_freezers,
            key="freezer_temp_select",
        )
        temp_df = ftdf[ftdf["FreezerId"].isin(sel_freezers)].sort_values("timestamp")
        fig = px.line(
            temp_df, x="timestamp", y="temperatureC", color="FreezerId",
            labels={"timestamp": "Time", "temperatureC": "Temperature (°C)", "FreezerId": "Freezer"},
            title="Freezer Temperature Over Time",
        )
        fig.add_hline(
            y=SAFE_TEMP, line_dash="dash", line_color="red",
            annotation_text=f"{SAFE_TEMP}°C safety threshold",
            annotation_position="top left",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_hum:
        hum_df = ftdf.sort_values("timestamp")
        fig = px.line(
            hum_df, x="timestamp", y="humidityPct", color="FreezerId",
            labels={"timestamp": "Time", "humidityPct": "Humidity (%)", "FreezerId": "Freezer"},
            title="Freezer Humidity Over Time",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_door:
        door_df = ftdf[ftdf["doorOpen"] == 1]
        if door_df.empty:
            st.info("No door-open events in the selected period.")
        else:
            door_counts = (
                door_df.groupby(["FreezerId", "StoreName"])["doorOpen"]
                .count().reset_index()
                .rename(columns={"doorOpen": "DoorOpenEvents"})
                .sort_values("DoorOpenEvents", ascending=False)
            )
            fig = px.bar(
                door_counts, x="FreezerId", y="DoorOpenEvents",
                color="StoreName",
                labels={"DoorOpenEvents": "Door Open Events", "FreezerId": "Freezer", "StoreName": "Store"},
                title="Door Open Event Count by Freezer",
                text="DoorOpenEvents",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    products = load_products()
    stores = load_stores()
    freezers = load_freezers()
    sales = load_sales()
    telemetry = load_telemetry()

    enriched_sales = build_enriched_sales(sales, products, stores)
    enriched_telem = build_enriched_telemetry(telemetry, stores, freezers)

    date_range, sel_stores, sel_cats = render_sidebar(sales, stores, products)

    if len(date_range) != 2:
        st.warning("Please select a start and end date.")
        st.stop()

    fdf = apply_filters(enriched_sales, date_range, sel_stores, sel_cats)
    ftdf = apply_telemetry_filters(enriched_telem, date_range, sel_stores, stores)

    st.title("fabricIQ — Retail & Cold Chain Dashboard")
    if fdf.empty:
        st.warning("No sales data matches the current filters.")
        return

    st.caption(
        f"Showing **{len(fdf):,}** transactions · "
        f"{fdf['SaleDate'].min().date()} to {fdf['SaleDate'].max().date()} · "
        f"{fdf['StoreName'].nunique()} store(s) · {fdf['ProductName'].nunique()} product(s)"
    )

    render_kpis(fdf)
    st.divider()
    render_sales_trends(fdf)
    st.divider()
    render_store_performance(fdf, stores)
    st.divider()
    render_product_analytics(fdf)
    st.divider()
    render_cold_chain(ftdf, freezers)


if __name__ == "__main__":
    main()

import hashlib

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

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


# ── Relationship graph ────────────────────────────────────────────────────────

NODE_COLORS = {
    "store": "#1f77b4",          # blue
    "product": "#2ca02c",        # green
    "freezer_ok": "#17becf",     # teal
    "freezer_alert": "#d62728",  # red
}


def _scale(values, lo: float, hi: float) -> pd.Series:
    """Min-max scale a numeric sequence into [lo, hi]; constant input -> midpoint."""
    s = pd.Series(list(values), dtype="float64")
    if s.empty:
        return s
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series([(lo + hi) / 2] * len(s), index=s.index)
    return lo + (s - mn) / (mx - mn) * (hi - lo)


def _layered_layout(G: "nx.Graph") -> dict:
    """Deterministic columnar layout: products | stores | freezers."""
    cols = {"product": -1.0, "store": 0.0, "freezer_ok": 1.0, "freezer_alert": 1.0}
    buckets: dict = {}
    for n, d in G.nodes(data=True):
        x = cols.get(d["ntype"], 0.0)
        buckets.setdefault(x, []).append(n)
    pos = {}
    for x, nodes in buckets.items():
        nodes = sorted(nodes)
        n = len(nodes)
        for i, node in enumerate(nodes):
            y = 0.5 if n == 1 else i / (n - 1)
            pos[node] = (x, y)
    return pos


def render_relationship_graph(fdf, ftdf, stores, freezers, products) -> None:
    st.subheader("Relationship Graph")

    c1, c2, c3 = st.columns([1.3, 1, 2])
    with c1:
        layout_mode = st.radio("Layout", ["Layered", "Radial"], horizontal=True)
    with c2:
        show_freezers = st.checkbox("Show freezers", value=True)

    # Aggregations (reuse the same groupby idioms as the chart sections)
    store_rev = (
        fdf.groupby(["StoreId", "StoreName", "City"])
        .agg(Rev=("RevenueUSD", "sum"), Units=("Units", "sum"))
        .reset_index()
    )
    prod_rev = (
        fdf.groupby(["ProductId", "ProductName", "Category"])
        .agg(Rev=("RevenueUSD", "sum"), Units=("Units", "sum"))
        .reset_index()
    )
    sp = fdf.groupby(["StoreId", "ProductId"])["RevenueUSD"].sum().reset_index()

    max_edge = int(sp["RevenueUSD"].max()) if not sp.empty else 0
    with c3:
        if max_edge > 0:
            min_edge_rev = st.slider(
                "Min sales-edge revenue ($)", 0, max_edge, 0,
                step=max(1, max_edge // 50),
            )
        else:
            min_edge_rev = 0
    sp = sp[sp["RevenueUSD"] >= min_edge_rev]

    # Freezers in the selected stores + alert status from telemetry
    sel_store_ids = set(store_rev["StoreId"])
    fz = freezers[freezers["StoreId"].isin(sel_store_ids)].copy()
    if not ftdf.empty:
        fz_alerts = (
            ftdf.groupby("FreezerId")
            .agg(Readings=("temperatureC", "count"),
                 Alerts=("temperatureC", lambda x: (x > SAFE_TEMP).sum()))
            .reset_index()
        )
        fz = fz.merge(fz_alerts, on="FreezerId", how="left")
    if "Readings" not in fz.columns:
        fz["Readings"] = 0
        fz["Alerts"] = 0
    fz[["Readings", "Alerts"]] = fz[["Readings", "Alerts"]].fillna(0)

    # Build graph
    G = nx.Graph()
    for (_, row), sz in zip(store_rev.iterrows(), _scale(store_rev["Rev"], 22, 55)):
        G.add_node(
            f"S::{row.StoreId}", ntype="store",
            label=row.StoreName.replace("Lakeshore Retail ", ""), size=sz,
            hover=(f"🏪 <b>{row.StoreName}</b><br>City: {row.City}"
                   f"<br>Revenue: ${row.Rev:,.0f}<br>Units: {int(row.Units):,}"),
        )
    for (_, row), sz in zip(prod_rev.iterrows(), _scale(prod_rev["Rev"], 18, 50)):
        G.add_node(
            f"P::{row.ProductId}", ntype="product", label=row.ProductName, size=sz,
            hover=(f"🍦 <b>{row.ProductName}</b><br>Category: {row.Category}"
                   f"<br>Revenue: ${row.Rev:,.0f}<br>Units: {int(row.Units):,}"),
        )
    if show_freezers:
        for _, row in fz.iterrows():
            alert = row.Alerts > 0
            comp = (1 - row.Alerts / row.Readings) * 100 if row.Readings else 100.0
            G.add_node(
                f"F::{row.FreezerId}",
                ntype="freezer_alert" if alert else "freezer_ok",
                label=row.FreezerId, size=16,
                hover=(f"🧊 <b>{row.FreezerId}</b><br>Model: {row.Model}"
                       f"<br>Compliance: {comp:.1f}%"
                       f"<br>Status: {'⚠️ ALERT' if alert else '✅ OK'}"),
            )

    for _, row in sp.iterrows():
        s_nid, p_nid = f"S::{row.StoreId}", f"P::{row.ProductId}"
        if G.has_node(s_nid) and G.has_node(p_nid):
            G.add_edge(s_nid, p_nid, weight=float(row.RevenueUSD), etype="sales")
    if show_freezers:
        for _, row in fz.iterrows():
            s_nid, f_nid = f"S::{row.StoreId}", f"F::{row.FreezerId}"
            if G.has_node(s_nid) and G.has_node(f_nid):
                G.add_edge(s_nid, f_nid, weight=1.0, etype="contains")

    if G.number_of_nodes() == 0:
        st.info("No entities match the current filters.")
        return

    # ── Selection state ───────────────────────────────────────────────────────
    # Filter-aware key: embedding a hash of the current node set means any filter
    # change yields a fresh widget, so a selection on a now-filtered node drops
    # automatically. Deterministic hashlib hash (not Python's salted hash) so the
    # key is stable across reruns and inside the WASM runtime.
    node_sig = hashlib.md5("|".join(sorted(G.nodes)).encode()).hexdigest()[:8]
    nonce = st.session_state.get("relgraph_nonce", 0)
    chart_key = f"relgraph_{node_sig}_{nonce}"

    selected_node = None
    sel = st.session_state.get(chart_key)
    if sel and sel.get("selection", {}).get("points"):
        for p in sel["selection"]["points"]:
            cd = p.get("customdata")
            cd = cd[0] if isinstance(cd, (list, tuple)) else cd
            if cd in G:  # validate against the current filtered graph
                selected_node = cd
                break

    # None => no selection => render exactly as before (no dimming)
    keep = (None if selected_node is None
            else {selected_node} | set(G.neighbors(selected_node)))

    # ── Layout ────────────────────────────────────────────────────────────────
    # The sales subgraph is complete bipartite (every store sells every product), so a
    # force-directed layout collapses the core into a blob. Layered (columns) and Radial
    # (concentric rings by type) both place nodes deterministically and stay readable.
    if layout_mode == "Radial":
        rings = [
            [n for n, d in G.nodes(data=True) if d["ntype"] == "store"],
            [n for n, d in G.nodes(data=True) if d["ntype"] == "product"],
            [n for n, d in G.nodes(data=True)
             if d["ntype"] in ("freezer_ok", "freezer_alert")],
        ]
        rings = [r for r in rings if r]
        pos = nx.shell_layout(G, nlist=rings)
    else:
        pos = _layered_layout(G)

    fig = go.Figure()

    # Sales edges — per-edge so width/opacity can vary with revenue; when a node is
    # selected, incident edges keep their colour and the rest fade out.
    sales_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["etype"] == "sales"]
    if sales_edges:
        weights = [d["weight"] for _, _, d in sales_edges]
        wmin, wmax = min(weights), max(weights)
        for (u, v, d), w in zip(sales_edges, _scale(weights, 1.0, 8.0)):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            incident = keep is not None and selected_node in (u, v)
            if keep is not None and not incident:
                color = "rgba(200,200,200,0.05)"
            else:
                op = (0.25 + 0.55 * (d["weight"] - wmin) / (wmax - wmin)) if wmax > wmin else 0.6
                color = f"rgba(120,120,120,{op:.2f})"
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(width=float(w), color=color),
                hoverinfo="skip", showlegend=False,
            ))

    # Containment edges — dotted; split into incident/faint when a node is selected
    contain_edges = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "contains"]

    def _contain_trace(edges, color):
        cx, cy = [], []
        for u, v in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            cx += [x0, x1, None]
            cy += [y0, y1, None]
        return go.Scatter(x=cx, y=cy, mode="lines",
                          line=dict(width=1, color=color, dash="dot"),
                          hoverinfo="skip", showlegend=False)

    if contain_edges:
        if keep is None:
            fig.add_trace(_contain_trace(contain_edges, "rgba(150,150,200,0.5)"))
        else:
            inc = [(u, v) for u, v in contain_edges if selected_node in (u, v)]
            rest = [(u, v) for u, v in contain_edges if selected_node not in (u, v)]
            if rest:
                fig.add_trace(_contain_trace(rest, "rgba(150,150,200,0.06)"))
            if inc:
                fig.add_trace(_contain_trace(inc, "rgba(120,120,200,0.9)"))

    # Node traces, one per type (clean legend that doubles as compliance key).
    # Per-point opacity/line arrays dim non-neighbours and ring the clicked node.
    trace_specs = [("store", "🏪 Store", NODE_COLORS["store"]),
                   ("product", "🍦 Product", NODE_COLORS["product"])]
    if show_freezers:
        trace_specs += [("freezer_ok", "🧊 Freezer OK", NODE_COLORS["freezer_ok"]),
                        ("freezer_alert", "⚠️ Freezer Alert", NODE_COLORS["freezer_alert"])]
    for ntype, legend, color in trace_specs:
        nodes = [n for n, dd in G.nodes(data=True) if dd["ntype"] == ntype]
        if not nodes:
            continue
        on = [keep is None or n in keep for n in nodes]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers+text",
            marker=dict(
                size=[G.nodes[n]["size"] for n in nodes], color=color,
                opacity=[1.0 if o else 0.12 for o in on],
                line=dict(
                    width=[3 if n == selected_node else 1 for n in nodes],
                    color=["#FFD700" if n == selected_node else "white" for n in nodes],
                ),
            ),
            text=[G.nodes[n]["label"] if o else "" for n, o in zip(nodes, on)],
            textposition="top center", textfont=dict(size=9),
            customdata=nodes,
            hovertext=[G.nodes[n]["hover"] for n in nodes],
            hoverinfo="text", name=legend,
        ))

    fig.update_layout(
        title="Store ↔ Product ↔ Freezer Network",
        height=650, hovermode="closest",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key,
                    on_select="rerun", selection_mode="points")

    if selected_node is not None:
        bcol, ccol = st.columns([1, 4])
        with bcol:
            if st.button("✖ Clear selection", use_container_width=True):
                st.session_state["relgraph_nonce"] = nonce + 1
                st.rerun()
        with ccol:
            st.caption(f"Highlighting **{G.nodes[selected_node]['label']}** and its "
                       f"{len(keep) - 1} direct relationship(s).")

    st.caption(
        "Node size ∝ revenue. Grey edges = sales (thickness & opacity ∝ revenue); "
        "dotted edges = freezer-in-store. Drag the slider to hide low-revenue sales links. "
        "**Layered** reads products | stores | freezers as columns; **Radial** rings them "
        "stores → products → freezers from the centre out. "
        "**Click any node** to highlight its relationships."
    )


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
    render_relationship_graph(fdf, ftdf, stores, freezers, products)
    st.divider()
    render_product_analytics(fdf)
    st.divider()
    render_cold_chain(ftdf, freezers)


if __name__ == "__main__":
    main()

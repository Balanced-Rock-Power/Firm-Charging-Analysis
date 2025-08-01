import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import seaborn as sns
import matplotlib as plt
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objects as go


def get_colorado_season(dt):
    month = dt.month
    if month in [11, 12, 1, 2]:
        return "Winter"
    elif month in [3, 4]:
        return "Spring"
    elif month in [5, 6, 7, 8]:
        return "Summer"
    elif month in [9, 10]:
        return "Fall"

st.set_page_config(page_title="Taelor Analysis Dashboard", layout="wide")

# ------------------------------
# Sidebar - Navigation
# ------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Add BESS and NG", "Seasonality & Trends", "Optimization Insights"])

# ------------------------------
# Page 1: Overview
# ------------------------------
if page == "Overview":
    st.title("📈 Taelor Analysis Dashboard")
    st.markdown("""
    This dashboard presents a summary of insights derived from two Jupyter Notebooks:
    - **Seasonality & Trend Analysis**
    - **Recent Updates & Deeper Exploration**

    The visuals and metrics help highlight patterns, customer behavior, and seasonal performance.
    """)

    # --------- LOAD & PROCESS PV_MPP DATA ----------
    df = pd.read_csv("simulations/PV_MPP.csv")
    df["Datetime"] = pd.date_range(start="2020-01-01 00:00", periods=len(df), freq="h")
    df["EnergyAvailable_kWh"] = df["EArrayMpp"]
    df["EInvOut_Calculated"] = df["EArrayMpp"] - df["Inv Loss"]
    df["EInvOut_Difference"] = df["EInvOut"] - df["EInvOut_Calculated"]
    df["Total_Losses"] = df["Inv Loss"] + df["EACohml"] + df["EMVohml"] + df["EMVtrfl"]
    df["PVEnergy_kW"] = df["EArrayMpp"] - df["Total_Losses"]
    df["Date"] = df["Datetime"].dt.date

    st.subheader("🔆 Daily PV Energy Output")
    daily_energy = df.groupby("Date")["EnergyAvailable_kWh"].sum().reset_index()
    fig1 = px.line(daily_energy, x="Date", y="EnergyAvailable_kWh",
                   title="Daily PV Energy (kWh)",
                   labels={"EnergyAvailable_kWh": "kWh"})
    fig1.update_layout(xaxis_title="Date", yaxis_title="Daily Energy (kWh)", hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("⚡ Daily PV Losses Breakdown")
    daily_loss = df.groupby("Date")[["Inv Loss", "EACohml", "EMVohml", "EMVtrfl"]].sum().reset_index()
    fig2 = go.Figure()
    for col in ["Inv Loss", "EACohml", "EMVohml", "EMVtrfl"]:
        fig2.add_trace(go.Bar(name=col, x=daily_loss["Date"], y=daily_loss[col]))
    fig2.update_layout(
        barmode='stack',
        title="Daily PV Loss Breakdown (kWh)",
        xaxis_title="Date",
        yaxis_title="Loss Energy (kWh)",
        hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📉 Hourly PV Output: Raw vs Net")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["Datetime"], y=df["EArrayMpp"], name="Raw PV Output (kW)", line=dict(width=1)))
    fig3.add_trace(go.Scatter(x=df["Datetime"], y=df["PVEnergy_kW"], name="Net PV Output (kW)", line=dict(width=1)))
    fig3.update_layout(
        title="Hourly PV Output: Raw vs Net",
        xaxis_title="Datetime",
        yaxis_title="Power (kW)",
        hovermode="x unified"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📐 PV Scaling Simulation (No BESS)")
    firm_power_mw   = 100
    firm_power_kw   = firm_power_mw * 1000
    base_system_mw  = 100

    def simulate_pv_only(df, firm_power_kw, scaling_factors, base_system_mw):
        results = []
        for scale in scaling_factors:
            df['Scaled_PV_kW'] = df['EArrayMpp'] * scale
            hours_met = (df['Scaled_PV_kW'] >= firm_power_kw).sum()
            success_rate = hours_met / len(df)

            results.append({
                'ScalingFactor' : scale,
                'PV_Size_MWdc'  : scale * base_system_mw,
                'HoursMet'      : hours_met,
                'SuccessRate'   : success_rate
            })
        return pd.DataFrame(results)

    scaling_factors = [x / 10 for x in range(10, 200)]
    pv_only_results = simulate_pv_only(df.copy(), firm_power_kw, scaling_factors, base_system_mw)

    fig4 = px.line(pv_only_results,
                   x="PV_Size_MWdc",
                   y="SuccessRate",
                   markers=True,
                   title="PV Size vs. 100 MW Firm Delivery (No BESS)",
                   labels={"PV_Size_MWdc": "PV Size (MWdc)", "SuccessRate": "Success Rate"})
    fig4.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="100% Delivery")
    fig4.update_layout(xaxis_title="PV Size (MWdc)",
                       yaxis_title="Success Rate (Fraction of Year)",
                       hovermode="x unified")
    st.plotly_chart(fig4, use_container_width=True)

# ------------------------------
# Page 2: Add BESS and NG
# ------------------------------
elif page == "Add BESS and NG":
    st.title("⚡ Success Rate from PV + BESS + NG Combinations")
    st.markdown("""
    This section explores how different configurations of **PV** (Photovoltaics), **BESS** (Battery Energy Storage System), and **NG** (Natural Gas) affect the ability to deliver a **firm 100 MW load**.

    **Success Rate** = Fraction of hours where full demand was met (≤ 0.01 MW unmet).

    Use the selector below to explore different types of visualizations:
    """)

    # Load and prepare data
    df_summary = pd.read_excel("simulations/pv_bess_ng_sweep_summary_with_cost.xlsx")
    df_summary["Season"] = df_summary["Season"].str.strip()

    # Aggregate to get overall success rate
    df_grouped = df_summary.groupby(["PV_MWdc", "BESS_MW", "NG_MW"]).agg({
        "SuccessRate": "mean"
    }).reset_index()

    st.subheader("📊 Select Analysis Type")
    analysis_type = st.selectbox("Choose what to explore:", [
        "Vary BESS (PV fixed, NG = 0)",
        "Vary NG (PV and BESS fixed)",
        "PV + BESS Heatmap (NG = 0)",
        "PV + BESS + NG Heatmaps"
    ])

    if analysis_type == "Vary BESS (PV fixed, NG = 0)":
        pv_selected = st.selectbox("Select Fixed PV Size (MWdc)", sorted(df_grouped["PV_MWdc"].unique()))
        st.subheader(f"🔋 Success Rate vs BESS Capacity (PV = {pv_selected} MWdc, NG = 0)")
        df_bess = df_grouped[(df_grouped["PV_MWdc"] == pv_selected) & (df_grouped["NG_MW"] == 0)]
        fig = px.line(df_bess, x="BESS_MW", y="SuccessRate", markers=True)
        fig.update_layout(xaxis_title="BESS Power (MW)", yaxis_title="Success Rate", hovermode="x unified", yaxis_range=[0,1])
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Vary NG (PV and BESS fixed)":
        pv = st.selectbox("Select Fixed PV Size (MWdc)", sorted(df_grouped["PV_MWdc"].unique()))
        bess = st.selectbox("Select Fixed BESS Size (MW)", sorted(df_grouped["BESS_MW"].unique()))
        st.subheader(f"⛽ Success Rate vs NG Capacity (PV = {pv} MWdc, BESS = {bess} MW)")
        df_ng = df_grouped[(df_grouped["PV_MWdc"] == pv) & (df_grouped["BESS_MW"] == bess)]
        fig = px.line(df_ng, x="NG_MW", y="SuccessRate", markers=True)
        fig.update_layout(xaxis_title="NG Capacity (MW)", yaxis_title="Success Rate", hovermode="x unified", yaxis_range=[0,1])
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "PV + BESS Heatmap (NG = 0)":
        st.subheader("🟢 Success Rate Heatmap — PV vs BESS (NG = 0 MW)")
        df_pb = df_grouped[df_grouped["NG_MW"] == 0]
        heatmap = df_pb.pivot_table(index="BESS_MW", columns="PV_MWdc", values="SuccessRate")

        fig = px.imshow(
            heatmap,
            text_auto=True,
            aspect="auto",
            labels=dict(x="PV (MWdc)", y="BESS (MW)", color="Success Rate"),
            color_continuous_scale=[(0.0, "red"), (0.5, "yellow"), (1.0, "green")],
            zmin=0, zmax=1
        )
        fig.update_layout(
            coloraxis_colorbar=dict(title="Success Rate"),
            xaxis_title="PV Capacity (MWdc)",
            yaxis_title="BESS Capacity (MW)"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "PV + BESS + NG Heatmaps":
        st.subheader("🔥 PV + BESS Success Rate Heatmaps by NG Capacity")
        ng_levels = sorted(df_grouped["NG_MW"].unique())
        for ng in ng_levels:
            st.markdown(f"**NG = {ng} MW**")
            df_ng = df_grouped[df_grouped["NG_MW"] == ng]
            heatmap = df_ng.pivot_table(index="BESS_MW", columns="PV_MWdc", values="SuccessRate")

            fig = px.imshow(
                heatmap,
                text_auto=True,
                aspect="auto",
                labels=dict(x="PV (MWdc)", y="BESS (MW)", color="Success Rate"),
                color_continuous_scale=[(0.0, "red"), (0.5, "yellow"), (1.0, "green")],
                zmin=0, zmax=1
            )
            fig.update_layout(
                title=f"Success Rate Heatmap — NG = {ng} MW",
                xaxis_title="PV Capacity (MWdc)",
                yaxis_title="BESS Capacity (MW)",
                coloraxis_colorbar=dict(title="Success Rate")
            )
            st.plotly_chart(fig, use_container_width=True)




# ------------------------------
# Page 3: Seasonality & Trends
# ------------------------------
elif page == "Seasonality & Trends":
    st.title("📈 Seasonality & Trends in Firm Power Delivery")
    st.markdown("""
    This section shows how the **energy delivery from PV, BESS, and NG** varies across seasons for a selected system configuration.

    All data is sourced from the pre-computed summary file.
    """)

    import plotly.express as px
    import plotly.graph_objects as go

    # Load and normalize column names
    df_summary = pd.read_excel("simulations/pv_bess_ng_sweep_summary_with_cost.xlsx")
    df_summary = df_summary.rename(columns={
        "BESS_Energy_Utilization_%": "BESS_Utilization",
        "Deficit_kWh": "Unmet_Energy_MWh"
    })

    # Dropdowns for configuration
    pv_options = sorted(df_summary["PV_MWdc"].unique())
    bess_options = sorted(df_summary["BESS_MW"].unique())
    ng_options = sorted(df_summary["NG_MW"].unique())
    season_options = ["Winter", "Spring", "Summer", "Fall"]

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_pv = st.selectbox("Select PV (MWdc)", pv_options)
    with col2:
        selected_bess = st.selectbox("Select BESS (MW)", bess_options)
    with col3:
        selected_ng = st.selectbox("Select NG (MW)", ng_options)

    selected_seasons = st.multiselect("Select Seasons to Compare", season_options, default=season_options)

    # Filter for selected config and season
    df_config = df_summary[
        (df_summary["PV_MWdc"] == selected_pv) &
        (df_summary["BESS_MW"] == selected_bess) &
        (df_summary["NG_MW"] == selected_ng) &
        (df_summary["Season"].isin(selected_seasons))
    ]

    if df_config.empty:
        st.warning("No data found for selected configuration and seasons.")
    else:
        st.subheader(f"🌞 Colorado-Seasonal Energy Summary — PV={selected_pv}, BESS={selected_bess}, NG={selected_ng}")

        df_energy = df_config[["Season", "PV_MWh", "BESS_MWh", "NG_MWh", "Unmet_Energy_MWh"]].copy()
        df_energy = df_energy.melt(id_vars="Season", var_name="Source", value_name="Energy_MWh")

        color_map = {
            "PV_MWh": "#4B8BBE",
            "BESS_MWh": "#6A5ACD",
            "NG_MWh": "#708090",
            "Unmet_Energy_MWh": "#00CED1"
        }

        fig_energy = px.bar(df_energy, x="Season", y="Energy_MWh", color="Source", barmode="group",
                            labels={"Energy_MWh": "Energy (MWh)"},
                            color_discrete_map=color_map,
                            text_auto=True)

        avg_demand = df_config[["PV_MWh", "BESS_MWh", "NG_MWh", "Unmet_Energy_MWh"]].sum(axis=1).mean()
        fig_energy.add_hline(
            y=avg_demand,
            line_dash="dash",
            annotation_text=f"CAPEX: ${df_config['Total_Cost_$'].iloc[0] / 1e6:.2f}M",
            annotation_position="top left"
        )

        st.plotly_chart(fig_energy, use_container_width=True)

        # 🔋 Utilization chart
        st.subheader("🔋 Seasonal Utilization by Source")
        df_viz = df_config[["Season", "PV_Utilization", "BESS_Utilization", "NG_Utilization"]].copy()
        df_viz = df_viz.melt(id_vars="Season", var_name="Source", value_name="Utilization")
        df_viz["Utilization"] = df_viz["Utilization"].astype(float)

        util_color_map = {
            "PV_Utilization": "#4B8BBE",
            "BESS_Utilization": "#6A5ACD",
            "NG_Utilization": "#708090"
        }

        fig_util = px.bar(df_viz, x="Season", y="Utilization", color="Source",
                          barmode="group", text_auto=True,
                          color_discrete_map=util_color_map,
                          labels={"Utilization": "Utilization (%)"})
        fig_util.update_layout(yaxis_range=[0, 110])
        st.plotly_chart(fig_util, use_container_width=True)

        # 📋 Seasonal Performance Table
        st.subheader("📋 Seasonal System Performance")
        df_perf = df_config[[
            "Season", "PV_MWh", "BESS_MWh", "NG_MWh", "Unmet_Energy_MWh", "Unmet_h", "SuccessRate",
            "PV_Utilization", "BESS_Utilization", "NG_Utilization", "BESS_Cycles"
        ]].copy()

        for col in ["PV_Utilization", "BESS_Utilization", "NG_Utilization"]:
            df_perf[col] = df_perf[col].map(lambda x: f"{x:.2f}%")
        df_perf["SuccessRate"] = df_perf["SuccessRate"].map(lambda x: f"{x * 100:.2f}%")

        st.dataframe(df_perf.set_index("Season"))
        
        
                # --- HOURLY DISPATCH COMBINED PLOT ---
        st.subheader("🕒 Hourly Dispatch Patterns")

        scenario_file = f"simulations/scenario_PV{selected_pv}_BESS{selected_bess}_NG{selected_ng}.csv"
        if not os.path.exists(scenario_file):
            st.warning(f"Scenario file not found: {scenario_file}")
        else:
            df_hourly = pd.read_csv(scenario_file, parse_dates=["Datetime"])
            df_hourly["Date"] = df_hourly["Datetime"].dt.date
            df_hourly["Season"] = df_hourly["Datetime"].apply(get_colorado_season)

            # Granularity selector
            granularity = st.radio("Select Granularity", ["Full Year", "By Season", "By Day"], horizontal=True)

            if granularity == "By Season":
                selected_season = st.selectbox("Choose Season", df_hourly["Season"].unique())
                df_filtered = df_hourly[df_hourly["Season"] == selected_season].copy()

            elif granularity == "By Day":
                selected_day = st.date_input("Select a Day", value=df_hourly["Date"].min(),
                                             min_value=df_hourly["Date"].min(),
                                             max_value=df_hourly["Date"].max())
                df_filtered = df_hourly[df_hourly["Date"] == selected_day].copy()

            else:
                df_filtered = df_hourly.copy()

            # Plot combined dispatch
            fig_combined = px.line(df_filtered, x="Datetime", y=["NetPV_MW", "Discharge_MW", "NG_MW"],
                                   labels={"value": "MW", "variable": "Source"},
                                   color_discrete_map={
                                       "NetPV_MW": "skyblue",
                                       "Discharge_MW": "limegreen",
                                       "NG_MW": "orangered"
                                   })

            fig_combined.add_scatter(
                x=df_filtered["Datetime"],
                y=[100] * len(df_filtered),
                mode="lines",
                name="Demand (100 MW)",
                line=dict(dash="dot", color="gray")
            )

            fig_combined.update_layout(
                title="Combined Dispatch — PV / BESS / NG",
                yaxis_title="Power (MW)",
                xaxis_title="Datetime",
                legend_title="Source",
                height=500
            )
            st.plotly_chart(fig_combined, use_container_width=True)



# ---------------- Optimization Insights Page ---------------- #
# ---------------- Optimization Insights Page ---------------- #
elif page == "Optimization Insights":
    st.title("📌 Optimization Insights — Configurations & Trade-offs")

    st.markdown("""
    This section explores **optimal system configurations** using:
    - Ranked configs
    - Pareto frontier (cost vs. success)
    - Constraint filtering
    - Sensitivity analysis
    - Clustering insights
    - Multi-objective optimization
    """)

    # Load Data
    df_summary = pd.read_excel("simulations/pv_bess_ng_sweep_summary_with_cost.xlsx")

    # Setup tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Top Configs", "Pareto Frontier", "Constraint Filter",
        "Sensitivity", "Cluster View", "Multi-Objective Optimizer"
    ])

    # ---------------- Tab 1: Top Configs ---------------- #
    with tab1:
        required_success = st.slider("Success Rate Threshold (%)", 80, 100, 99) / 100
        top_n = st.slider("How many top configs to show?", 1, 10, 5)

        configs = df_summary.groupby(["PV_MWdc", "BESS_MW", "NG_MW"])
        valid_rows = []

        for (pv, bess, ng), group in configs:
            if len(group["Season"].unique()) == 4 and (group["SuccessRate"] >= required_success).all():
                row = group.iloc[0].copy()
                row["Total_Cost_$"] = row["PV_Cost"] + row["BESS_Cost"] + row["NG_Cost"]
                valid_rows.append(row)

        df_valid = pd.DataFrame(valid_rows)

        if not df_valid.empty:
            df_valid = df_valid.sort_values("Total_Cost_$").head(top_n)
            st.dataframe(df_valid.reset_index(drop=True))

            fig_pie = px.pie(df_valid, values="Total_Cost_$", names="SuccessRate",
                             title="Cost Distribution by Config Success Rate")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("❌ No configurations met the criteria. Try lowering the success rate.")

    # ---------------- Tab 2: Pareto Frontier ---------------- #
    with tab2:
        df_config = df_summary.groupby(["PV_MWdc", "BESS_MW", "NG_MW"])[["SuccessRate", "PV_Cost", "BESS_Cost", "NG_Cost"]].mean().reset_index()
        df_config["Total_Cost_$"] = df_config[["PV_Cost", "BESS_Cost", "NG_Cost"]].sum(axis=1)

        def pareto_front(df):
            df_sorted = df.sort_values("Total_Cost_$")
            pareto = []
            max_sr = -1
            for _, row in df_sorted.iterrows():
                if row["SuccessRate"] > max_sr:
                    pareto.append(row)
                    max_sr = row["SuccessRate"]
            return pd.DataFrame(pareto)

        pareto_df = pareto_front(df_config)
        fig = px.scatter(df_config, x="Total_Cost_$", y="SuccessRate",
                         color_discrete_sequence=["gray"], opacity=0.4,
                         title="All Configs: Cost vs. Success Rate")
        fig.add_scatter(x=pareto_df["Total_Cost_$"], y=pareto_df["SuccessRate"],
                        mode="markers+lines", name="Pareto Front",
                        marker=dict(color="green", size=10))
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- Tab 3: Constraint Filter ---------------- #
    with tab3:
        max_cost = st.number_input("Max Total Cost ($)", value=150_000_000)
        max_bess_cycles = st.number_input("Max BESS Cycles", value=500)

        filtered = df_summary.copy()
        filtered["Total_Cost_$"] = filtered["PV_Cost"] + filtered["BESS_Cost"] + filtered["NG_Cost"]

        constraints_met = filtered[
            (filtered["Total_Cost_$"] <= max_cost) &
            (filtered["BESS_Cycles"] <= max_bess_cycles)
        ]
        st.write(f"🔎 {len(constraints_met)} configs meet the constraints")
        st.dataframe(constraints_met.head(20))

    # ---------------- Tab 4: Sensitivity ---------------- #
    with tab4:
        variable = st.selectbox("Vary which component?", ["PV_MWdc", "BESS_MW", "NG_MW"])
        fixed_ng = st.selectbox("Fix NG Capacity", sorted(df_summary["NG_MW"].unique()))

        df_sens = df_summary[df_summary["NG_MW"] == fixed_ng]
        grouped = df_sens.groupby(variable)[["SuccessRate", "PV_Cost", "BESS_Cost", "NG_Cost"]].mean().reset_index()
        grouped["Total_Cost_$"] = grouped[["PV_Cost", "BESS_Cost", "NG_Cost"]].sum(axis=1)

        fig1 = px.line(grouped, x=variable, y="SuccessRate", markers=True, title=f"Success Rate vs {variable}")
        fig2 = px.line(grouped, x=variable, y="Total_Cost_$", markers=True, title=f"Total Cost vs {variable}")

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- Tab 5: Clustering ---------------- #
    with tab5:
        df_cluster = df_summary.copy()
        df_cluster["Total_Cost_$"] = df_cluster[["PV_Cost", "BESS_Cost", "NG_Cost"]].sum(axis=1)

        features = df_cluster[["PV_MWdc", "BESS_MW", "NG_MW"]]
        kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
        df_cluster["Cluster"] = kmeans.labels_

        fig = px.scatter_3d(df_cluster, x="PV_MWdc", y="BESS_MW", z="NG_MW",
                            color=df_cluster["Cluster"].astype(str),
                            title="Clustered Configurations (Design Archetypes)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Cluster Summary")
        st.dataframe(df_cluster.groupby("Cluster")[["SuccessRate", "Total_Cost_$"]].mean().round(2))

    # ---------------- Tab 6: Multi-Objective Optimizer ---------------- #
    with tab6:
        st.markdown("### 🎯 Optimal Trade-off: Max Success, Min Cost")

        sort_by = st.radio("Sort configs by:", ["SuccessRate (High)", "Total_Cost_$ (Low)"])
        max_cycles = st.slider("Max BESS Cycles (optional)", 0, 1000, 500)
        show_n = st.slider("How many configurations to show?", 5, 50, 10)

        df_opt = df_summary.copy()
        df_opt["Total_Cost_$"] = df_opt["PV_Cost"] + df_opt["BESS_Cost"] + df_opt["NG_Cost"]

        # Keep configs with all 4 seasons and within cycle limit
        df_valid = df_opt.groupby(["PV_MWdc", "BESS_MW", "NG_MW"]).filter(
            lambda g: len(g["Season"].unique()) == 4 and g["BESS_Cycles"].max() <= max_cycles
        )

        # Pareto filtering
        df_grouped = df_valid.groupby(["PV_MWdc", "BESS_MW", "NG_MW"]).mean(numeric_only=True).reset_index()

        pareto = []
        for _, row in df_grouped.sort_values("Total_Cost_$").iterrows():
            if all(row["SuccessRate"] >= p["SuccessRate"] for p in pareto):
                pareto.append(row)

        df_pareto = pd.DataFrame(pareto)
        if sort_by == "SuccessRate (High)":
            df_pareto = df_pareto.sort_values("SuccessRate", ascending=False)
        else:
            df_pareto = df_pareto.sort_values("Total_Cost_$")

        st.dataframe(df_pareto.head(show_n).reset_index(drop=True))

        fig = px.scatter(df_grouped, x="Total_Cost_$", y="SuccessRate", opacity=0.3,
                         title="Pareto Front: Cost vs Success Rate")
        fig.add_scatter(x=df_pareto["Total_Cost_$"], y=df_pareto["SuccessRate"],
                        mode="markers+lines", name="Pareto Front", marker=dict(size=10, color="green"))
        st.plotly_chart(fig, use_container_width=True)

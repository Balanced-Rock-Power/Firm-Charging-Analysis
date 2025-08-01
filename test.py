import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime, date

# Page configuration
st.set_page_config(page_title="Firm Charging Analysis", page_icon="⚡", layout="wide")

# Initialize session state
if 'simulation_complete' not in st.session_state:
    st.session_state.simulation_complete = False
if 'summary_df' not in st.session_state:
    st.session_state.summary_df = None
if 'hourly_tabs' not in st.session_state:
    st.session_state.hourly_tabs = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

# Enhanced Sidebar Configuration
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("---")

# System Parameters
with st.sidebar.expander("🎯 System Parameters", expanded=True):
    poi_kw = st.number_input("POI Target (kW)", min_value=1000, max_value=1000000, value=100000, step=1000)
    
    col1, col2 = st.columns(2)
    with col1:
        rte = st.slider("Battery Efficiency", min_value=0.5, max_value=1.0, value=0.88, step=0.01)
    with col2:
        duration_h = st.number_input("Duration (h)", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
    
    charge_limit_kw = st.number_input("Max Charging (kW)", min_value=10000, max_value=500000, value=200000, step=10000)

# Natural Gas Parameters
with st.sidebar.expander("🔥 Natural Gas Firming", expanded=True):
    include_ng = st.checkbox("Include Natural Gas", value=False, help="Add NG as backup firming source")
    
    if include_ng:
        st.markdown("**🔧 NG Engineering Parameters**")
        
        ng_type = st.selectbox("NG Technology", 
                              ["Simple Cycle", "Combined Cycle", "Aeroderivative"],
                              index=0, help="Technology affects efficiency and startup time")
        
        col1, col2 = st.columns(2)
        with col1:
            ng_heat_rate = st.number_input("Heat Rate (Btu/kWh)", 
                                         min_value=6000, max_value=12000, 
                                         value=9500 if ng_type == "Simple Cycle" else 7000 if ng_type == "Combined Cycle" else 8500,
                                         step=100, help="Fuel efficiency")
            ng_min_load_pct = st.slider("Min Load (%)", min_value=20, max_value=60, 
                                       value=40 if ng_type in ["Simple Cycle", "Combined Cycle"] else 30, step=5,
                                       help="Minimum stable operating level")
        
        with col2:
            ng_startup_min = st.number_input("Startup Time (min)", min_value=2, max_value=60, 
                                           value=5 if ng_type == "Aeroderivative" else 8 if ng_type == "Simple Cycle" else 30,
                                           step=1, help="Time from cold start to full load")
            ng_ramp_rate = st.number_input("Ramp Rate (%/min)", min_value=10, max_value=100, 
                                         value=50 if ng_type == "Aeroderivative" else 25 if ng_type == "Simple Cycle" else 15,
                                         step=5, help="Load change rate capability")
        
        st.markdown("**📏 NG Sizing Range**")
        col1, col2, col3 = st.columns(3)
        with col1:
            ng_min = st.number_input("NG Min (MW)", min_value=10, max_value=200, value=25, step=5)
        with col2:
            ng_max = st.number_input("NG Max (MW)", min_value=50, max_value=500, value=150, step=25)
        with col3:
            ng_step = st.number_input("NG Step (MW)", min_value=5, max_value=50, value=25, step=5)
    else:
        ng_min = ng_max = ng_step = ng_heat_rate = ng_min_load_pct = ng_startup_min = ng_ramp_rate = 0
        ng_type = None

# System Losses
with st.sidebar.expander("⚡ System Losses", expanded=False):
    st.markdown("**Cable Losses (%)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        loss_lv = st.number_input("LV", min_value=0.0, max_value=5.0, value=0.2, step=0.1) / 100
    with col2:
        loss_mv = st.number_input("MV", min_value=0.0, max_value=5.0, value=0.1, step=0.1) / 100
    with col3:
        loss_hv = st.number_input("HV", min_value=0.0, max_value=5.0, value=0.05, step=0.01) / 100
    
    st.markdown("**Transformer Losses**")
    col1, col2 = st.columns(2)
    with col1:
        mvt_nl = st.number_input("MVT No Load (kW)", min_value=0.0, max_value=50.0, value=4.218, step=0.1)
        mvt_fl = st.number_input("MVT Full Load (kW)", min_value=0.0, max_value=100.0, value=40.110, step=0.1)
        mvt_count = st.number_input("MVT Count", min_value=1, max_value=100, value=30, step=1)
    with col2:
        gsu_nl = st.number_input("GSU No Load (kW)", min_value=0.0, max_value=200.0, value=95.0, step=1.0)
        gsu_fl = st.number_input("GSU Full Load (kW)", min_value=0.0, max_value=1000.0, value=400.0, step=10.0)

# Analysis Range
with st.sidebar.expander("🔍 Analysis Range", expanded=True):
    st.markdown("**PV Range (MW)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        pv_min = st.number_input("Min", min_value=50, max_value=500, value=100, step=50)
    with col2:
        pv_max = st.number_input("Max", min_value=500, max_value=2000, value=1400, step=100)
    with col3:
        pv_step = st.number_input("Step", min_value=50, max_value=200, value=100, step=50)
    
    st.markdown("**BESS Range (MW)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        bess_min = st.number_input("Min", min_value=25, max_value=200, value=50, step=25, key="bess_min")
    with col2:
        bess_max = st.number_input("Max", min_value=200, max_value=1500, value=950, step=50, key="bess_max")
    with col3:
        bess_step = st.number_input("Step", min_value=25, max_value=100, value=50, step=25, key="bess_step")
    
    # Analysis summary
    if include_ng:
        ng_configs = len(range(ng_min, ng_max + ng_step, ng_step))
        total_configs = len(range(pv_min, pv_max + pv_step, pv_step)) * len(range(bess_min, bess_max + bess_step, bess_step)) * ng_configs
        st.info(f"**Total Configurations:** {total_configs:,} (PV×BESS×NG)")
        st.info(f"**NG Sizes:** {ng_configs} configurations ({ng_min}-{ng_max} MW)")
    else:
        total_configs = len(range(pv_min, pv_max + pv_step, pv_step)) * len(range(bess_min, bess_max + bess_step, bess_step))
        st.info(f"**Total Configurations:** {total_configs:,}")

hourly_save_thresh = st.sidebar.slider("Hourly Save Threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.05)

# Title and Introduction
if include_ng:
    st.title("⚡ Firm Charging Analysis with Natural Gas")
    st.markdown("**Enhanced PV + BESS + NG** - Find optimal hybrid system sizes for reliable power delivery")
    
    with st.expander("🔥 **Natural Gas Integration Overview**", expanded=False):
        st.markdown(f"""
        **System Configuration**: PV + BESS + Natural Gas ({ng_min}-{ng_max} MW {ng_type})
        
        **Dispatch Priority**:
        1. 🌞 **Solar PV** - Primary generation (zero marginal cost)
        2. 🔋 **Battery Storage** - Secondary dispatch when PV insufficient  
        3. 🔥 **Natural Gas** - Tertiary backup for firm delivery
        
        **NG Plant Specifications**:
        - **Technology**: {ng_type} ({ng_heat_rate:,} Btu/kWh efficiency)
        - **Startup Time**: {ng_startup_min} minutes from cold start
        - **Operating Range**: {ng_min_load_pct}%-100% of capacity
        - **Sizing Range**: {ng_min}-{ng_max} MW ({ng_step} MW steps)
        """)
else:
    st.title("⚡ Firm Charging Analysis")
    st.markdown("Find optimal PV and BESS sizes for reliable power delivery")

# File Upload and Data Analysis
st.header("📁 PV Generation Data Analysis")

uploaded_file = st.file_uploader("Select CSV file with PV generation data", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Basic file info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        if 'EArrayMpp' in df.columns:
            st.metric("Annual Energy", f"{df['EArrayMpp'].sum():,.0f} kWh")
    with col4:
        if 'EArrayMpp' in df.columns:
            st.metric("Peak Power", f"{df['EArrayMpp'].max():,.0f} kW")
    
    # Data processing
    df["Datetime"] = pd.date_range(start="2020-01-01 00:00", periods=len(df), freq="h")
    df["EnergyAvailable_kWh"] = df["EArrayMpp"]
    
    if all(col in df.columns for col in ["Inv Loss", "EACohml", "EMVohml", "EMVtrfl"]):
        df["Total_Losses"] = (df["Inv Loss"] + df["EACohml"] + df["EMVohml"] + df["EMVtrfl"])
        df["PVEnergy_kW"] = df["EArrayMpp"] - df["Total_Losses"]
        avg_loss_rate = (df["Total_Losses"] / df["EArrayMpp"]).mean() * 100
        st.success(f"✅ Loss data applied (Avg loss: {avg_loss_rate:.1f}%)")
    else:
        df["PVEnergy_kW"] = df["EArrayMpp"]
        st.warning("⚠️ Using gross PV generation (no loss data)")
    
    st.session_state.processed_df = df
    
    # Quick data visualization
    st.subheader("📊 Generation Profile Analysis")
    
    df["Date"] = df["Datetime"].dt.date
    df["Hour"] = df["Datetime"].dt.hour
    
    analysis_type = st.selectbox("Select Analysis Type:", 
                                ["Daily Profile", "Monthly Summary", "Hourly Pattern"])
    
    if analysis_type == "Daily Profile":
        daily_energy = df.groupby("Date")["EnergyAvailable_kWh"].sum()
        fig_daily = px.line(x=daily_energy.index, y=daily_energy.values,
                           title="Daily PV Energy Generation",
                           labels={'x': 'Date', 'y': 'Daily Energy (kWh)'})
        fig_daily.add_hline(y=daily_energy.mean(), line_dash="dash", 
                           annotation_text=f"Average: {daily_energy.mean():,.0f} kWh/day")
        st.plotly_chart(fig_daily, use_container_width=True)
    
    elif analysis_type == "Monthly Summary":
        df["MonthName"] = df["Datetime"].dt.strftime('%b')
        monthly_energy = df.groupby("MonthName")["EnergyAvailable_kWh"].sum().reindex(
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )
        fig_monthly = px.bar(x=monthly_energy.index, y=monthly_energy.values,
                           title="Monthly PV Energy Generation",
                           labels={'x': 'Month', 'y': 'Monthly Energy (kWh)'})
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    elif analysis_type == "Hourly Pattern":
        hourly_avg = df.groupby("Hour")["EnergyAvailable_kWh"].agg(['mean', 'max', 'min']).reset_index()
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(x=hourly_avg["Hour"], y=hourly_avg["mean"],
                                      mode='lines+markers', name='Average', line=dict(width=3)))
        fig_hourly.add_trace(go.Scatter(x=hourly_avg["Hour"], y=hourly_avg["max"],
                                      mode='lines', name='Maximum', opacity=0.7))
        fig_hourly.add_trace(go.Scatter(x=hourly_avg["Hour"], y=hourly_avg["min"],
                                      mode='lines', name='Minimum', opacity=0.7))
        fig_hourly.update_layout(title="Hourly Generation Pattern",
                               xaxis_title="Hour of Day", yaxis_title="Generation (kW)")
        st.plotly_chart(fig_hourly, use_container_width=True)

# Main Simulation
if st.button("🚀 Run Analysis", type="primary") and uploaded_file is not None:
    
    # Configuration
    cfg = {
        "poi_kw": poi_kw, "rte": rte, "duration_h": duration_h, "charge_limit_kw": charge_limit_kw,
        "loss": {"lv": loss_lv, "mv": loss_mv, "hv": loss_hv,
                "mvt": {"nl": mvt_nl, "fl": mvt_fl, "count": mvt_count},
                "gsu": {"nl": gsu_nl, "fl": gsu_fl}},
        "sweep": {"pv_sizes_mw": list(range(pv_min, pv_max + pv_step, pv_step)),
                 "bess_powers_mw": list(range(bess_min, bess_max + bess_step, bess_step)),
                 "ng_sizes_mw": list(range(ng_min, ng_max + ng_step, ng_step)) if include_ng else [0]},
        "hourly_save_thresh": hourly_save_thresh,
        "ng": {"enabled": include_ng, "type": ng_type, "heat_rate": ng_heat_rate,
               "min_load_pct": ng_min_load_pct, "startup_min": ng_startup_min, "ramp_rate": ng_ramp_rate}
    }
    
    # Calculate efficiency factors
    ηc = ηd = np.sqrt(cfg["rte"])
    lv, mv, hv = cfg["loss"]["lv"], cfg["loss"]["mv"], cfg["loss"]["hv"]
    mvt_nl, mvt_fl, mvt_cnt = cfg["loss"]["mvt"].values()
    gsu_nl, gsu_fl = cfg["loss"]["gsu"].values()
    
    # Baseline data
    pv_base = df["PVEnergy_kW"].values
    hours = df["Datetime"].values
    N = len(hours)
    POI = cfg["poi_kw"]
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    summary = []
    hourly_tabs = {}
    
    pv_sizes = cfg["sweep"]["pv_sizes_mw"]
    bess_sizes = cfg["sweep"]["bess_powers_mw"]
    ng_sizes = cfg["sweep"]["ng_sizes_mw"]
    total_iterations = len(pv_sizes) * len(bess_sizes) * len(ng_sizes)
    current_iteration = 0
    
    # Main simulation loop
    for pv_mw in pv_sizes:
        scale = pv_mw / 100
        pv_kw = pv_base * scale
        pv_cap = pv_mw * 1_000
        
        mvt_kw = mvt_cnt * (mvt_nl + mvt_fl * (POI / pv_cap))
        gsu_kw = gsu_nl + gsu_fl * (POI / pv_cap)
        hv_kw = hv * POI
        line_pct = lv + mv
        
        for bess_mw in bess_sizes:
            P_max = bess_mw * 1_000
            E_max = P_max * cfg["duration_h"]
            
            for ng_mw in ng_sizes:
                current_iteration += 1
                progress = current_iteration / total_iterations
                progress_bar.progress(progress)
                status_text.text(f"Processing: PV {pv_mw} MW, BESS {bess_mw} MW, NG {ng_mw} MW ({current_iteration}/{total_iterations})")
                
                soc = 0
                
                # NG parameters
                ng_enabled = cfg["ng"]["enabled"] and ng_mw > 0
                ng_capacity_kw = ng_mw * 1_000 if ng_enabled else 0
                ng_min_load_kw = ng_capacity_kw * (cfg["ng"]["min_load_pct"] / 100) if ng_enabled else 0
                ng_running = False
                ng_startup_counter = 0
                
                # Results tracking
                succ = surp = defi = unav = max_soc = 0
                ng_runtime_h = ng_fuel_used_mmbtu = ng_starts = ng_emissions_total = 0
                
                hlog = {k: [] for k in
                        ["Datetime","NetPV_kW","Delivered_kW","Charge_kW",
                         "Discharge_kW","SOC_kWh","Unmet_kW","NG_Output_kW","NG_Fuel_MMBtu"]}
                
                for t, pv in enumerate(pv_kw):
                    net_pv = max(pv - (mvt_kw + gsu_kw + line_pct*pv + hv_kw), 0)
                    gap = POI - net_pv
                    ng_output = 0
                    ng_fuel_hour = 0
                    
                    if gap <= 0:  # Surplus - charge battery
                        avail = -gap
                        ch_kw = min(avail, cfg["charge_limit_kw"], P_max)
                        ch_kwh = ch_kw * ηc
                        accept = min(ch_kwh, E_max - soc)
                        soc += accept
                        surp += (avail - accept/ηc)
                        dis_kw = unmet = 0
                        delivered = POI
                        succ += 1
                        
                        # Turn off NG if running
                        if ng_running:
                            ng_running = False
                            ng_startup_counter = 0
                    
                    else:  # Deficit - discharge battery then NG
                        need = gap
                        
                        # Step 1: Try BESS discharge
                        available_discharge = min(need / ηd, soc, P_max)
                        dis_kwh = available_discharge
                        soc -= dis_kwh
                        dis_kw = dis_kwh
                        bess_contribution = dis_kwh * ηd
                        
                        remaining_gap = need - bess_contribution
                        
                        # Step 2: NG dispatch if still short and NG enabled
                        if remaining_gap > 1 and ng_enabled:
                            
                            # NG startup logic
                            if not ng_running:
                                ng_startup_counter += 1
                                if ng_startup_counter >= cfg["ng"]["startup_min"] / 60:
                                    ng_running = True
                                    ng_startup_counter = 0
                                    ng_starts += 1
                            
                            if ng_running:
                                # Determine NG output (respect minimum load)
                                ng_needed = min(remaining_gap, ng_capacity_kw)
                                if ng_needed >= ng_min_load_kw:
                                    ng_output = ng_needed
                                elif remaining_gap > ng_min_load_kw * 0.5:
                                    ng_output = ng_min_load_kw
                                
                                # Calculate fuel consumption and emissions
                                if ng_output > 0:
                                    ng_fuel_hour = (ng_output * cfg["ng"]["heat_rate"]) / 1_000_000  # MMBtu
                                    ng_fuel_used_mmbtu += ng_fuel_hour
                                    ng_runtime_h += 1
                                    # Calculate emissions (117 lb CO2/MMBtu for natural gas)
                                    ng_emissions_factor = 117.0
                                    ng_emissions_hour = ng_fuel_hour * ng_emissions_factor
                                    ng_emissions_total += ng_emissions_hour
                        
                        delivered = net_pv + bess_contribution + ng_output
                        
                        if delivered >= POI:
                            succ += 1
                            unmet = 0
                        else:
                            unmet = POI - delivered
                            defi += unmet
                            if soc <= 0 and ng_output == 0:
                                unav += 1
                        
                        ch_kw = 0
                    
                    max_soc = max(max_soc, soc)
                    
                    # Enhanced logging with NG data
                    hlog["Datetime"].append(hours[t])
                    hlog["NetPV_kW"].append(net_pv)
                    hlog["Delivered_kW"].append(delivered)
                    hlog["Charge_kW"].append(ch_kw)
                    hlog["Discharge_kW"].append(dis_kw)
                    hlog["SOC_kWh"].append(soc)
                    hlog["Unmet_kW"].append(unmet)
                    hlog["NG_Output_kW"].append(ng_output)
                    hlog["NG_Fuel_MMBtu"].append(ng_fuel_hour)
                
                sr = succ / N
                
                # Results with NG metrics
                result = {
                    "PV_MWdc": pv_mw, "BESS_MW": bess_mw, "NG_MW": ng_mw, "BESS_MWh": E_max/1_000,
                    "SuccessRate": sr, "Surplus_kWh": surp, "Deficit_kWh": defi,
                    "Unavail_h": unav, "MaxSOC_kWh": max_soc
                }
                
                if ng_enabled:
                    ng_capacity_factor = ng_runtime_h / N if N > 0 else 0
                    result.update({
                        "NG_Runtime_h": ng_runtime_h,
                        "NG_Fuel_MMBtu": ng_fuel_used_mmbtu,
                        "NG_CapacityFactor": ng_capacity_factor,
                        "NG_Starts": ng_starts,
                        "NG_Emissions_lb": ng_emissions_total
                    })
                
                summary.append(result)
                
                # Save hourly data with proper key format
                if sr >= cfg["hourly_save_thresh"]:
                    hourly_tabs[(pv_mw, bess_mw, ng_mw)] = hlog
    
    # Save results
    st.session_state.summary_df = pd.DataFrame(summary)
    st.session_state.hourly_tabs = hourly_tabs
    st.session_state.simulation_complete = True
    
    progress_bar.progress(1.0)
    status_text.text("✅ Analysis complete!")
    st.success("🎉 Simulation completed successfully!")

elif uploaded_file is None:
    st.info("Please upload a PV profile CSV file to begin analysis")

# Results Section
if st.session_state.simulation_complete and st.session_state.summary_df is not None:
    summary_df = st.session_state.summary_df
    
    st.header("📊 Analysis Results")
    
    # Summary metrics
    if include_ng and "NG_Runtime_h" in summary_df.columns:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col5:
            avg_ng_cf = summary_df["NG_CapacityFactor"].mean()
            st.metric("Avg NG Capacity Factor", f"{avg_ng_cf:.1%}")
    else:
        col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Configurations", len(summary_df))
    with col2:
        avg_success = summary_df["SuccessRate"].mean()
        st.metric("Average Success Rate", f"{avg_success:.1%}")
    with col3:
        best_success = summary_df["SuccessRate"].max()
        st.metric("Best Success Rate", f"{best_success:.1%}")
    with col4:
        configs_90 = len(summary_df[summary_df["SuccessRate"] >= 0.9])
        st.metric("Configs ≥90% Success", configs_90)
    
    # Performance Analysis
    st.subheader("📈 Performance Analysis")
    st.markdown("#### Success Rate Heatmap")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Handle NG dimension for heatmap
        if include_ng and "NG_MW" in summary_df.columns:
            ng_sizes_available = sorted(summary_df["NG_MW"].unique())
            selected_ng_for_heatmap = st.selectbox(
                "Select NG Size for Heatmap", 
                ng_sizes_available,
                help="Choose NG size to display in the PV vs BESS heatmap"
            )
            
            # Create heatmap for selected NG size
            ng_subset = summary_df[summary_df["NG_MW"] == selected_ng_for_heatmap]
            
            if len(ng_subset) > 0:
                pivot = ng_subset.pivot(index="BESS_MW", columns="PV_MWdc", values="SuccessRate")
                title = f"Success Rate: PV vs BESS (NG = {selected_ng_for_heatmap} MW)"
            else:
                st.warning(f"No data available for NG size {selected_ng_for_heatmap} MW")
                pivot = None
        else:
            # Original heatmap for PV+BESS only (no NG)
            pivot = summary_df.pivot_table(
                index="BESS_MW", 
                columns="PV_MWdc", 
                values="SuccessRate",
                aggfunc='max'
            )
            title = "Success Rate by PV and BESS Size"
        
        if pivot is not None:
            # Create heatmap with annotations
            fig_heatmap = px.imshow(
                pivot,
                title=title,
                labels=dict(x="PV Size (MWdc)", y="BESS Power (MW)", color="Success Rate"),
                color_continuous_scale="RdYlGn",
                aspect="auto"
            )
            
            # Add text annotations
            annotations = []
            for i, row in enumerate(pivot.index):
                for j, col in enumerate(pivot.columns):
                    value = pivot.iloc[i, j]
                    if not pd.isna(value):
                        annotations.append(
                            dict(
                                x=j, y=i,
                                text=f"{value:.0%}" if value >= 0.1 else f"{value:.1%}",
                                showarrow=False,
                                font=dict(color="white" if value < 0.5 else "black", size=8)
                            )
                        )
            
            fig_heatmap.update_layout(annotations=annotations, height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        st.markdown("**Heatmap Guide:**")
        st.markdown("🟩 **Green**: High success rate (>80%)")
        st.markdown("🟨 **Yellow**: Medium success rate (50-80%)")  
        st.markdown("🟥 **Red**: Low success rate (<50%)")
        
        # Quick stats
        best_config = summary_df.loc[summary_df["SuccessRate"].idxmax()]
        st.markdown("**Best Configuration:**")
        st.write(f"PV: {best_config['PV_MWdc']} MW")
        st.write(f"BESS: {best_config['BESS_MW']} MW")
        if include_ng and "NG_MW" in best_config:
            st.write(f"NG: {best_config['NG_MW']} MW")
        st.write(f"Success: {best_config['SuccessRate']:.1%}")
    
    # FIXED Natural Gas Analysis (if enabled)
    if include_ng and "NG_Runtime_h" in summary_df.columns:
        st.markdown("#### 🔥 **Natural Gas Performance Analysis**")
        
        # Add analysis scope selector
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            analysis_scope = st.selectbox(
                "Analysis Scope", 
                ["By NG Size", "Overall Statistics"],
                help="Choose how to view NG performance data"
            )
        
        if analysis_scope == "By NG Size":
            # Filter by specific NG size
            with col2:
                ng_sizes_available = sorted(summary_df["NG_MW"].unique())
                ng_sizes_nonzero = [ng for ng in ng_sizes_available if ng > 0]
                
                if ng_sizes_nonzero:
                    selected_ng_analysis = st.selectbox(
                        "Select NG Size (MW)", 
                        ng_sizes_nonzero,
                        help="Choose NG size to analyze across all PV/BESS combinations"
                    )
                else:
                    st.warning("No NG configurations found")
                    selected_ng_analysis = None
            
            with col3:
                if selected_ng_analysis:
                    ng_count = len(summary_df[summary_df["NG_MW"] == selected_ng_analysis])
                    st.info(f"📊 **Analyzing {selected_ng_analysis} MW NG across {ng_count} PV/BESS combinations**")
            
            if selected_ng_analysis:
                # Filter data for selected NG size
                ng_analysis_data = summary_df[summary_df["NG_MW"] == selected_ng_analysis]
                
                # Show statistics for this NG size
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_runtime = ng_analysis_data["NG_Runtime_h"].mean()
                    max_runtime = ng_analysis_data["NG_Runtime_h"].max()
                    st.metric("Avg NG Runtime", f"{avg_runtime:,.0f} h/year", 
                             help=f"Average across all configs. Max: {max_runtime:,.0f} h")
                
                with col2:
                    avg_fuel = ng_analysis_data["NG_Fuel_MMBtu"].mean()
                    st.metric("Avg Fuel per Config", f"{avg_fuel:,.0f} MMBtu/year",
                             help="Per configuration annually")
                
                with col3:
                    avg_emissions = ng_analysis_data["NG_Emissions_lb"].mean()
                    st.metric("Avg Emissions per Config", f"{avg_emissions:,.0f} lb CO₂/year",
                             help="Per configuration annually")
                
                with col4:
                    avg_cf = ng_analysis_data["NG_CapacityFactor"].mean()
                    cf_range = f"{ng_analysis_data['NG_CapacityFactor'].min():.1%}-{ng_analysis_data['NG_CapacityFactor'].max():.1%}"
                    st.metric("Avg Capacity Factor", f"{avg_cf:.1%}",
                             help=f"Range across configs: {cf_range}")
                
                # Show distribution charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Runtime distribution
                    fig_runtime = px.histogram(ng_analysis_data, x="NG_Runtime_h", 
                                             title=f"{selected_ng_analysis} MW NG Runtime Distribution",
                                             labels={"NG_Runtime_h": "Runtime (hours/year)"})
                    fig_runtime.add_vline(x=avg_runtime, line_dash="dash", line_color="red",
                                         annotation_text=f"Avg: {avg_runtime:,.0f} h")
                    st.plotly_chart(fig_runtime, use_container_width=True)
                
                with col2:
                    # Capacity factor vs success rate
                    fig_cf_success = px.scatter(ng_analysis_data, x="NG_CapacityFactor", y="SuccessRate",
                                              color="BESS_MW", size="PV_MWdc",
                                              title=f"{selected_ng_analysis} MW NG: CF vs Success Rate")
                    st.plotly_chart(fig_cf_success, use_container_width=True)
        
        elif analysis_scope == "Overall Statistics":
            # Show summary statistics across all NG configurations
            with col2:
                ng_configs = summary_df[summary_df["NG_MW"] > 0]
                total_ng_configs = len(ng_configs)
                st.write(f"**Total NG Configs:** {total_ng_configs}")
            
            with col3:
                st.info(f"📊 **Summary statistics across {total_ng_configs} NG-enabled configurations**")
            
            # Summary statistics table
            st.markdown("**📈 NG Performance Statistics Across All Configurations:**")
            
            ng_configs = summary_df[summary_df["NG_MW"] > 0]
            
            if not ng_configs.empty:
                # Create summary table
                stats_data = []
                
                for ng_size in sorted(ng_configs["NG_MW"].unique()):
                    ng_subset = ng_configs[ng_configs["NG_MW"] == ng_size]
                    
                    stats_data.append({
                        "NG_Size_MW": ng_size,
                        "Configurations": len(ng_subset),
                        "Avg_Runtime_h": ng_subset["NG_Runtime_h"].mean(),
                        "Avg_Fuel_MMBtu": ng_subset["NG_Fuel_MMBtu"].mean(),
                        "Avg_Emissions_lb": ng_subset["NG_Emissions_lb"].mean(),
                        "Avg_Capacity_Factor": ng_subset["NG_CapacityFactor"].mean(),
                        "Avg_Success_Rate": ng_subset["SuccessRate"].mean()
                    })
                
                stats_df = pd.DataFrame(stats_data)
                
                # Format and display table
                formatted_stats = stats_df.style.format({
                    'Avg_Runtime_h': '{:,.0f}',
                    'Avg_Fuel_MMBtu': '{:,.0f}',
                    'Avg_Emissions_lb': '{:,.0f}',
                    'Avg_Capacity_Factor': '{:.1%}',
                    'Avg_Success_Rate': '{:.1%}'
                })
                
                st.dataframe(formatted_stats, use_container_width=True)
                
                # Key insights
                st.markdown("**🔍 Key Insights:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_configs = len(summary_df)
                    ng_configs_count = len(ng_configs)
                    st.write(f"• **NG Configurations:** {ng_configs_count} of {total_configs} total")
                    
                    avg_runtime_all = ng_configs["NG_Runtime_h"].mean()
                    st.write(f"• **Average NG Runtime:** {avg_runtime_all:,.0f} hours/year")
                
                with col2:
                    best_ng_size = stats_df.loc[stats_df["Avg_Success_Rate"].idxmax(), "NG_Size_MW"]
                    best_success = stats_df.loc[stats_df["Avg_Success_Rate"].idxmax(), "Avg_Success_Rate"]
                    st.write(f"• **Best NG Size:** {best_ng_size} MW")
                    st.write(f"• **Best Success Rate:** {best_success:.1%}")
                
                with col3:
                    min_emissions_idx = stats_df["Avg_Emissions_lb"].idxmin()
                    most_efficient_ng = stats_df.loc[min_emissions_idx, "NG_Size_MW"]
                    min_emissions = stats_df.loc[min_emissions_idx, "Avg_Emissions_lb"]
                    st.write(f"• **Most Efficient:** {most_efficient_ng} MW NG")
                    st.write(f"• **Lowest Emissions:** {min_emissions:,.0f} lb CO₂/year")
            
            else:
                st.warning("No NG configurations found in the analysis")
        
        # Important note about metrics
        st.markdown("---")
        st.info("""
        **📝 Important Note:** The metrics shown above are per-configuration or averaged across configurations, 
        not cumulative totals. Each configuration represents one combination of PV + BESS + NG sizes 
        analyzed over a full year (8,760 hours).
        """)
    
    # Results table with enhanced filtering
    st.subheader("🔍 Configuration Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_success = st.slider("Min Success Rate", 0.0, 1.0, 0.8, 0.05)
    with col2:
        max_deficit = st.number_input("Max Deficit (kWh)", 0, int(summary_df["Deficit_kWh"].max()), 
                                     int(summary_df["Deficit_kWh"].max()))
    with col3:
        sort_by = st.selectbox("Sort by", ["SuccessRate", "Deficit_kWh", "Surplus_kWh", "PV_MWdc", "BESS_MW"])
    
    filtered_df = summary_df[
        (summary_df["SuccessRate"] >= min_success) &
        (summary_df["Deficit_kWh"] <= max_deficit)
    ].sort_values(sort_by, ascending=False)
    
    # Dynamic column selection
    display_cols = ["PV_MWdc", "BESS_MW", "SuccessRate", "Deficit_kWh", "Surplus_kWh"]
    format_dict = {
        'SuccessRate': '{:.1%}',
        'Deficit_kWh': '{:,.0f}',
        'Surplus_kWh': '{:,.0f}',
        'MaxSOC_kWh': '{:,.0f}'
    }
    
    if include_ng and "NG_MW" in filtered_df.columns:
        display_cols.insert(2, "NG_MW")
        if "NG_Runtime_h" in filtered_df.columns:
            display_cols.append("NG_Runtime_h")
            format_dict['NG_Runtime_h'] = '{:,.0f}'
    
    display_cols.append("MaxSOC_kWh")
    
    st.dataframe(filtered_df[display_cols].style.format(format_dict), use_container_width=True)
    st.info(f"Showing {len(filtered_df):,} of {len(summary_df):,} configurations")
    
    # IMPROVED Energy Balance Analysis
    st.subheader("⚡ Energy Balance Analysis")

    if include_ng and "NG_MW" in summary_df.columns:
        # Enhanced energy balance analysis for NG-integrated systems
        st.markdown("#### 🔄 **Multi-Source Energy Balance**")
        
        # Add NG size filter for cleaner visualization
        col1, col2 = st.columns([1, 3])
        with col1:
            ng_sizes_for_charts = sorted(summary_df["NG_MW"].unique())
            selected_ng_chart = st.selectbox(
                "NG Size for Charts", 
                ng_sizes_for_charts,
                help="Select NG size to display in energy balance charts"
            )
        with col2:
            st.info(f"📊 **Showing energy balance for systems with {selected_ng_chart} MW Natural Gas**")
        
        # Filter data for selected NG size
        chart_data = summary_df[summary_df["NG_MW"] == selected_ng_chart]
        
        # Enhanced visualization with cleaner lines
        col1, col2 = st.columns(2)
        
        with col1:
            fig_deficit = go.Figure()
            
            # Select fewer BESS sizes for cleaner visualization
            bess_sizes_chart = sorted(chart_data["BESS_MW"].unique())
            bess_step_chart = max(1, len(bess_sizes_chart) // 4)  # Show max 4 lines
            bess_sizes_to_show = bess_sizes_chart[::bess_step_chart]
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, bp in enumerate(bess_sizes_to_show):
                s = chart_data[chart_data["BESS_MW"] == bp].sort_values("PV_MWdc")
                if len(s) > 0:
                    fig_deficit.add_trace(go.Scatter(
                        x=s["PV_MWdc"], y=s["Deficit_kWh"],
                        mode='lines+markers', 
                        name=f"{bp} MW BESS",
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6),
                        hovertemplate="PV: %{x} MW<br>BESS: " + f"{bp} MW<br>" + 
                                    "NG: " + f"{selected_ng_chart} MW<br>" +
                                    "Deficit: %{y:,.0f} kWh<extra></extra>"
                    ))
            
            fig_deficit.update_layout(
                title=f"Deficit Energy vs PV Size (NG: {selected_ng_chart} MW)", 
                xaxis_title="PV Size (MWdc)", 
                yaxis_title="Deficit Energy (kWh)",
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_deficit, use_container_width=True)
        
        with col2:
            fig_surplus = go.Figure()
            
            for i, bp in enumerate(bess_sizes_to_show):
                s = chart_data[chart_data["BESS_MW"] == bp].sort_values("PV_MWdc")
                if len(s) > 0:
                    fig_surplus.add_trace(go.Scatter(
                        x=s["PV_MWdc"], y=s["Surplus_kWh"],
                        mode='lines+markers', 
                        name=f"{bp} MW BESS",
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6),
                        hovertemplate="PV: %{x} MW<br>BESS: " + f"{bp} MW<br>" + 
                                    "NG: " + f"{selected_ng_chart} MW<br>" +
                                    "Surplus: %{y:,.0f} kWh<extra></extra>"
                    ))
            
            fig_surplus.update_layout(
                title=f"Surplus Energy vs PV Size (NG: {selected_ng_chart} MW)",
                xaxis_title="PV Size (MWdc)", 
                yaxis_title="Surplus Energy (kWh)",
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_surplus, use_container_width=True)

    else:
        # Original energy balance charts for PV+BESS only systems
        col1, col2 = st.columns(2)
        
        with col1:
            fig_deficit = go.Figure()
            bess_sizes_to_show = sorted(summary_df["BESS_MW"].unique())[::max(1, len(summary_df["BESS_MW"].unique())//5)]
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            
            for i, bp in enumerate(bess_sizes_to_show):
                s = summary_df[summary_df["BESS_MW"] == bp].sort_values("PV_MWdc")
                fig_deficit.add_trace(go.Scatter(
                    x=s["PV_MWdc"], y=s["Deficit_kWh"],
                    mode='lines+markers', 
                    name=f"{bp} MW BESS",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                    hovertemplate="PV: %{x} MW<br>BESS: " + f"{bp} MW<br>" +
                                "Deficit: %{y:,.0f} kWh<extra></extra>"
                ))
            
            fig_deficit.update_layout(
                title="Deficit Energy vs PV Size", 
                xaxis_title="PV Size (MWdc)", 
                yaxis_title="Deficit Energy (kWh)",
                height=400
            )
            st.plotly_chart(fig_deficit, use_container_width=True)
        
        with col2:
            fig_surplus = go.Figure()
            for i, bp in enumerate(bess_sizes_to_show):
                s = summary_df[summary_df["BESS_MW"] == bp].sort_values("PV_MWdc")
                fig_surplus.add_trace(go.Scatter(
                    x=s["PV_MWdc"], y=s["Surplus_kWh"],
                    mode='lines+markers', 
                    name=f"{bp} MW BESS",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                    hovertemplate="PV: %{x} MW<br>BESS: " + f"{bp} MW<br>" +
                                "Surplus: %{y:,.0f} kWh<extra></extra>"
                ))
            
            fig_surplus.update_layout(
                title="Surplus Energy vs PV Size",
                xaxis_title="PV Size (MWdc)", 
                yaxis_title="Surplus Energy (kWh)",
                height=400
            )
            st.plotly_chart(fig_surplus, use_container_width=True)

# Single Configuration Analysis with IMPROVED layout
if st.session_state.simulation_complete:
    st.header("🔍 Single Configuration Analysis")
    st.markdown("Analyze performance of a specific configuration")
    
    # Configuration selection with improved NG support and better display
    if include_ng and "NG_MW" in summary_df.columns:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        
        with col1:
            selected_pv = st.selectbox("Select PV Size (MW)", 
                                      sorted(summary_df["PV_MWdc"].unique()),
                                      index=5)
        
        with col2:
            available_bess = summary_df[summary_df["PV_MWdc"] == selected_pv]["BESS_MW"].unique()
            selected_bess = st.selectbox("Select BESS Size (MW)", 
                                        sorted(available_bess),
                                        index=len(available_bess)//2 if len(available_bess) > 0 else 0)
        
        with col3:
            available_ng = summary_df[
                (summary_df["PV_MWdc"] == selected_pv) & 
                (summary_df["BESS_MW"] == selected_bess)
            ]["NG_MW"].unique()
            selected_ng = st.selectbox("Select NG Size (MW)", 
                                     sorted(available_ng),
                                     index=0)
        
        config_key = (selected_pv, selected_bess, selected_ng)
        
        with col4:
            # Improved configuration display with two lines
            st.markdown("**🎯 Selected Configuration:**")
            st.markdown(f"**Primary System:** {selected_pv} MW PV + {selected_bess} MW BESS")
            st.markdown(f"**Backup System:** {selected_ng} MW Natural Gas")
        
        # Get specific configuration
        specific_config = summary_df[
            (summary_df["PV_MWdc"] == selected_pv) & 
            (summary_df["BESS_MW"] == selected_bess) &
            (summary_df["NG_MW"] == selected_ng)
        ]

    else:
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            selected_pv = st.selectbox("Select PV Size (MW)", 
                                      sorted(summary_df["PV_MWdc"].unique()),
                                      index=5)
        
        with col2:
            available_bess = summary_df[summary_df["PV_MWdc"] == selected_pv]["BESS_MW"].unique()
            selected_bess = st.selectbox("Select BESS Size (MW)", 
                                        sorted(available_bess),
                                        index=len(available_bess)//2 if len(available_bess) > 0 else 0)
        
        config_key = (selected_pv, selected_bess, 0)  # Default NG = 0
        
        with col3:
            # Clean configuration display for PV+BESS only
            st.markdown("**🎯 Selected Configuration:**")
            st.markdown(f"**System:** {selected_pv} MW PV + {selected_bess} MW BESS")
        
        # Get specific configuration
        specific_config = summary_df[
            (summary_df["PV_MWdc"] == selected_pv) & 
            (summary_df["BESS_MW"] == selected_bess)
        ]
        selected_ng = 0

    # Enhanced metrics display with better organization
    if not specific_config.empty:
        config = specific_config.iloc[0]
        
        # Create organized metric groups
        st.markdown("#### 📊 **Performance Metrics**")
        
        # Primary metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Success Rate", f"{config['SuccessRate']:.1%}", 
                     help="Percentage of hours POI target was met")
        with col2:
            st.metric("Annual Deficit", f"{config['Deficit_kWh']:,.0f} kWh",
                     help="Total energy shortfall when POI target not met")
        with col3:
            st.metric("Annual Surplus", f"{config['Surplus_kWh']:,.0f} kWh",
                     help="Excess energy that couldn't be stored")
        with col4:
            st.metric("Max SOC", f"{config['MaxSOC_kWh']:,.0f} kWh",
                     help="Maximum battery state of charge reached")
        
        # NG metrics row (if applicable)
        if include_ng and "NG_MW" in summary_df.columns and selected_ng > 0 and "NG_Runtime_h" in config:
            st.markdown("#### 🔥 **Natural Gas Metrics**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("NG Runtime", f"{config['NG_Runtime_h']:.0f} h",
                         help="Total hours NG plant was operating")
            with col2:
                ng_cf = config.get('NG_CapacityFactor', 0)
                st.metric("NG Capacity Factor", f"{ng_cf:.1%}",
                         help="Percentage of time NG plant was running")
            with col3:
                ng_fuel = config.get('NG_Fuel_MMBtu', 0)
                st.metric("Fuel Consumption", f"{ng_fuel:,.0f} MMBtu",
                         help="Total natural gas fuel consumed")
            with col4:
                ng_emissions = config.get('NG_Emissions_lb', 0)
                st.metric("CO₂ Emissions", f"{ng_emissions:,.0f} lb",
                         help="Total CO₂ emissions from NG combustion")
    
    # Debug info and hourly analysis
    st.write("🔍 **DEBUG - Hourly Data Search:**")
    st.write(f"**Looking for key:** {config_key}")
    st.write(f"**Available keys (first 5):** {list(st.session_state.hourly_tabs.keys())[:5]}")
    st.write(f"**Total hourly datasets:** {len(st.session_state.hourly_tabs)}")
    st.write(f"**Key exists:** {config_key in st.session_state.hourly_tabs}")
    
    # Show hourly data if available
    if config_key in st.session_state.hourly_tabs:
        if include_ng and selected_ng > 0:
            st.subheader(f"📈 Hourly Performance: {selected_pv} MW PV + {selected_bess} MW BESS + {selected_ng} MW NG")
        else:
            st.subheader(f"📈 Hourly Performance: {selected_pv} MW PV + {selected_bess} MW BESS")
        
        hourly_data = pd.DataFrame(st.session_state.hourly_tabs[config_key])
        hourly_data["Date"] = pd.to_datetime(hourly_data["Datetime"]).dt.date
        
        # Date range selection for detailed view
        col1, col2 = st.columns(2)
        with col1:
            start_date_detail = st.date_input("Start Date", value=date(2020, 6, 1), 
                                            min_value=date(2020, 1, 1), max_value=date(2020, 12, 31),
                                            key="detail_start")
        with col2:
            end_date_detail = st.date_input("End Date", value=date(2020, 6, 7),
                                          min_value=date(2020, 1, 1), max_value=date(2020, 12, 31),
                                          key="detail_end")
        
        if start_date_detail <= end_date_detail:
            mask = (hourly_data["Date"] >= start_date_detail) & (hourly_data["Date"] <= end_date_detail)
            filtered_hourly = hourly_data[mask]
            
            if len(filtered_hourly) > 0:
                # Enhanced detailed hourly chart with NG
                fig_detail = go.Figure()
                
                # PV generation (base)
                fig_detail.add_trace(go.Scatter(
                    x=filtered_hourly["Datetime"], y=filtered_hourly["NetPV_kW"],
                    mode='lines', name='Net PV', line=dict(color='orange'),
                    fill='tozeroy', stackgroup='one'
                ))
                
                # BESS discharge (positive values)
                bess_discharge = filtered_hourly["Discharge_kW"] * 0.88  # Apply efficiency
                fig_detail.add_trace(go.Scatter(
                    x=filtered_hourly["Datetime"], y=bess_discharge,
                    mode='lines', name='BESS Discharge', line=dict(color='blue'),
                    stackgroup='one'
                ))
                
                # NG output if available and used
                if include_ng and "NG_Output_kW" in filtered_hourly.columns and selected_ng > 0:
                    ng_output = filtered_hourly["NG_Output_kW"]
                    if ng_output.sum() > 0:  # Only show if NG was actually used
                        fig_detail.add_trace(go.Scatter(
                            x=filtered_hourly["Datetime"], y=ng_output,
                            mode='lines', name='Natural Gas', line=dict(color='red'),
                            stackgroup='one'
                        ))
                
                # Total delivered power (reference line)
                fig_detail.add_trace(go.Scatter(
                    x=filtered_hourly["Datetime"], y=filtered_hourly["Delivered_kW"],
                    mode='lines', name='Total Delivered', line=dict(color='green', width=3),
                    stackgroup=None  # Don't stack this line
                ))
                
                # POI target line
                fig_detail.add_hline(y=poi_kw, line_dash="dash", line_color="red",
                                   annotation_text=f"POI Target: {poi_kw:,} kW")
                
                fig_detail.update_layout(
                    title=f"Multi-Source Dispatch: {start_date_detail} to {end_date_detail}",
                    xaxis_title="Time",
                    yaxis_title="Power (kW)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_detail, use_container_width=True)
                
                # Battery SOC and NG performance
                col1, col2 = st.columns(2)
                
                with col1:
                    # Battery SOC chart
                    fig_soc = go.Figure()
                    fig_soc.add_trace(go.Scatter(
                        x=filtered_hourly["Datetime"], y=filtered_hourly["SOC_kWh"],
                        mode='lines', name='Battery SOC', line=dict(color='blue'),
                        fill='tonexty'
                    ))
                    
                    # Add max capacity line
                    max_capacity = selected_bess * 1000 * duration_h
                    fig_soc.add_hline(y=max_capacity, line_dash="dash", line_color="gray",
                                    annotation_text=f"Max Capacity: {max_capacity:,.0f} kWh")
                    
                    fig_soc.update_layout(
                        title="Battery State of Charge",
                        xaxis_title="Time",
                        yaxis_title="SOC (kWh)",
                        height=300
                    )
                    
                    st.plotly_chart(fig_soc, use_container_width=True)
                
                with col2:
                    # NG output and fuel consumption if available
                    if include_ng and "NG_Output_kW" in filtered_hourly.columns and selected_ng > 0:
                        ng_output = filtered_hourly["NG_Output_kW"]
                        if ng_output.sum() > 0:
                            fig_ng = go.Figure()
                            # NG Power Output
                            fig_ng.add_trace(go.Scatter(
                                x=filtered_hourly["Datetime"], y=ng_output,
                                mode='lines', name='NG Output', line=dict(color='red'),
                                yaxis='y'
                            ))
                            
                            # NG Fuel Rate (if available)
                            if "NG_Fuel_MMBtu" in filtered_hourly.columns:
                                fig_ng.add_trace(go.Scatter(
                                    x=filtered_hourly["Datetime"], y=filtered_hourly["NG_Fuel_MMBtu"],
                                    mode='lines', name='Fuel Rate', line=dict(color='orange'),
                                    yaxis='y2'
                                ))
                                
                                # Create subplot with secondary y-axis
                                fig_ng.update_layout(
                                    title="Natural Gas Performance",
                                    xaxis_title="Time",
                                    yaxis=dict(title="Power (kW)", side="left"),
                                    yaxis2=dict(title="Fuel Rate (MMBtu/h)", side="right", overlaying="y"),
                                    height=300
                                )
                            else:
                                fig_ng.update_layout(
                                    title="Natural Gas Output",
                                    xaxis_title="Time",
                                    yaxis_title="Power (kW)",
                                    height=300
                                )
                            
                            st.plotly_chart(fig_ng, use_container_width=True)
                        else:
                            st.info("🔥 NG plant was not dispatched during this period")
                    else:
                        # Show charging pattern instead
                        fig_charge = go.Figure()
                        fig_charge.add_trace(go.Scatter(
                            x=filtered_hourly["Datetime"], y=filtered_hourly["Charge_kW"],
                            mode='lines', name='Battery Charging', line=dict(color='green'),
                            fill='tonexty'
                        ))
                        
                        fig_charge.update_layout(
                            title="Battery Charging Pattern",
                            xaxis_title="Time",
                            yaxis_title="Charging Power (kW)",
                            height=300
                        )
                        
                        st.plotly_chart(fig_charge, use_container_width=True)
            else:
                st.warning("No data available for the selected date range")
    
    else:
        st.info(f"💡 Detailed hourly data not available for this configuration")
        st.write(f"**Success rate threshold:** {hourly_save_thresh:.0%}")
        if not specific_config.empty:
            actual_sr = specific_config.iloc[0]["SuccessRate"]
            st.write(f"**This config's success rate:** {actual_sr:.1%}")
            if actual_sr < hourly_save_thresh:
                st.write("❌ **Reason:** Success rate below threshold")
            else:
                st.write("❌ **Reason:** Key format mismatch")

# CORRECTED Advanced Optimization Analysis
if st.session_state.simulation_complete:
    st.header("🎯 Advanced Optimization Analysis")
    
    df = summary_df.copy()
    
    # Enhanced normalization and penalty calculations
    df["Deficit_norm"] = df["Deficit_kWh"] / df["Deficit_kWh"].max() if df["Deficit_kWh"].max() > 0 else 0
    df["PV_norm"] = df["PV_MWdc"] / df["PV_MWdc"].max()
    df["BESS_norm"] = df["BESS_MW"] / df["BESS_MW"].max()
    
    # ROBUST NG handling with proper error checking
    ng_columns_available = "NG_MW" in df.columns and include_ng
    ng_runtime_available = "NG_Runtime_h" in df.columns and include_ng
    ng_emissions_available = "NG_Emissions_lb" in df.columns and include_ng
    
    if ng_columns_available:
        # Basic NG calculations
        df["NG_norm"] = df["NG_MW"] / df["NG_MW"].max() if df["NG_MW"].max() > 0 else 0
        
        # NG capacity factor (with error handling)
        if ng_runtime_available:
            df["NG_CF"] = df["NG_Runtime_h"] / 8760
            df["NG_runtime_penalty"] = (df["NG_CF"] ** 2) * 3.0  # Quadratic penalty
        else:
            df["NG_CF"] = 0
            df["NG_runtime_penalty"] = 0
        
        # NG emissions penalty (with error handling)
        if ng_emissions_available:
            df["NG_emissions_penalty"] = (df["NG_Emissions_lb"] / df["NG_Emissions_lb"].max()) * 2.0 if df["NG_Emissions_lb"].max() > 0 else 0
        else:
            df["NG_emissions_penalty"] = 0
        
        # Combined NG penalty
        df["total_ng_penalty"] = df["NG_norm"] + df["NG_runtime_penalty"] + df["NG_emissions_penalty"]
        
        # Corrected EES with stronger penalties
        df["EES"] = df["SuccessRate"] / (
            df["Deficit_norm"] + 
            df["PV_norm"] + 
            df["BESS_norm"] + 
            df["total_ng_penalty"] + 
            0.1
        )
        
        # Calculate renewable energy fraction
        total_annual_energy = poi_kw * 8760
        if ng_runtime_available:
            ng_annual_energy = df["NG_Runtime_h"] * df["NG_MW"] * 1000
            df["Renewable_Fraction"] = np.maximum(0, 1 - (ng_annual_energy / total_annual_energy))
        else:
            df["Renewable_Fraction"] = 1.0  # 100% renewable if no NG data
        
        # Alternative scores
        df["Green_Score"] = df["SuccessRate"] * df["Renewable_Fraction"] / (df["PV_norm"] + df["BESS_norm"] + 0.1)
        df["Performance_Score"] = df["SuccessRate"]
        
        if ng_emissions_available:
            df["Carbon_Intensity"] = df["NG_Emissions_lb"] / (poi_kw * df["SuccessRate"] * 8760 / 1000)
        else:
            df["Carbon_Intensity"] = 0
        
    else:
        # No NG - simple EES calculation
        df["EES"] = df["SuccessRate"] / (df["Deficit_norm"] + df["PV_norm"] + df["BESS_norm"] + 0.1)
        df["Green_Score"] = df["EES"]
        df["Performance_Score"] = df["SuccessRate"]
        df["Renewable_Fraction"] = 1.0
        df["NG_CF"] = 0
        df["Carbon_Intensity"] = 0
    
    # Get top configurations
    top_ees = df.sort_values("EES", ascending=False).head(10)
    
    if ng_columns_available:
        top_green = df.sort_values("Green_Score", ascending=False).head(10)
        top_performance = df.sort_values("Performance_Score", ascending=False).head(10)
        if ng_runtime_available:
            low_ng_configs = df[df["NG_CF"] < 0.3].sort_values("EES", ascending=False).head(10)
        else:
            low_ng_configs = pd.DataFrame()
    
    # Enhanced Pareto analysis
    pareto_configs = []
    for _, config in df.iterrows():
        dominated = False
        for _, other in df.iterrows():
            conditions = [
                other["SuccessRate"] >= config["SuccessRate"],
                other["Deficit_kWh"] <= config["Deficit_kWh"],
                other["PV_MWdc"] <= config["PV_MWdc"],
                other["BESS_MW"] <= config["BESS_MW"]
            ]
            
            if ng_columns_available:
                conditions.extend([
                    other["NG_MW"] <= config["NG_MW"]
                ])
                if ng_runtime_available:
                    conditions.append(other["NG_Runtime_h"] <= config["NG_Runtime_h"])
                if ng_emissions_available:
                    conditions.append(other["NG_Emissions_lb"] <= config["NG_Emissions_lb"])
            
            if all(conditions):
                strict_better = any([
                    other["SuccessRate"] > config["SuccessRate"],
                    other["Deficit_kWh"] < config["Deficit_kWh"],
                    other["PV_MWdc"] < config["PV_MWdc"],
                    other["BESS_MW"] < config["BESS_MW"]
                ])
                
                if ng_columns_available:
                    strict_better = strict_better or other["NG_MW"] < config["NG_MW"]
                    if ng_runtime_available:
                        strict_better = strict_better or other["NG_Runtime_h"] < config["NG_Runtime_h"]
                    if ng_emissions_available:
                        strict_better = strict_better or other["NG_Emissions_lb"] < config["NG_Emissions_lb"]
                
                if strict_better:
                    dominated = True
                    break
        
        if not dominated:
            pareto_configs.append(config)
    
    pareto_df = pd.DataFrame(pareto_configs).sort_values("SuccessRate", ascending=False) if pareto_configs else pd.DataFrame()
    
    # Optimization method selection
    if ng_columns_available:
        opt_methods = [
            "Engineering Efficiency Score (Corrected)", 
            "Green Score (Renewable Focus)",
            "Performance Score (Reliability Focus)",
            "Pareto Optimal Solutions"
        ]
        if ng_runtime_available and len(low_ng_configs) > 0:
            opt_methods.insert(-1, "Low NG Usage (<30% CF)")
    else:
        opt_methods = ["Engineering Efficiency Score", "Pareto Optimal Solutions"]
    
    opt_method = st.selectbox("Select Optimization Method:", opt_methods)
    
    if opt_method == "Engineering Efficiency Score (Corrected)" or opt_method == "Engineering Efficiency Score":
        st.subheader("⚖️ Engineering Efficiency Score (EES)")
        if ng_columns_available:
            st.markdown("**Purpose**: Balances performance with system complexity, **heavily penalizing NG overuse**")
        else:
            st.markdown("**Purpose**: Balances high performance with reasonable system complexity")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            display_cols = ["PV_MWdc", "BESS_MW", "SuccessRate", "Deficit_kWh", "EES"]
            format_dict = {
                'SuccessRate': '{:.1%}',
                'Deficit_kWh': '{:,.0f}',
                'EES': '{:.3f}'
            }
            
            if ng_columns_available:
                display_cols.insert(2, "NG_MW")
                display_cols.extend(["NG_CF", "Renewable_Fraction"])
                format_dict.update({
                    'NG_CF': '{:.1%}',
                    'Renewable_Fraction': '{:.1%}'
                })
            
            st.dataframe(top_ees[display_cols].style.format(format_dict), use_container_width=True)
        
        with col2:
            best_ees = top_ees.iloc[0]
            ng_info = ""
            if ng_columns_available:
                ng_cf = best_ees.get('NG_CF', 0)
                renewable_frac = best_ees.get('Renewable_Fraction', 1)
                ng_info = f"""
- **NG**: {best_ees['NG_MW']} MW
- **NG Capacity Factor**: {ng_cf:.1%}
- **Renewable Fraction**: {renewable_frac:.1%}"""
            
            st.success(f"""
            **🏆 Best EES Configuration:**
            - **PV**: {best_ees['PV_MWdc']} MW
            - **BESS**: {best_ees['BESS_MW']} MW{ng_info}
            - **Success Rate**: {best_ees['SuccessRate']:.1%}
            - **EES Score**: {best_ees['EES']:.3f}
            
            **Why This Works:**
            Optimal balance with minimal NG dependency
            """)
    
    elif opt_method == "Green Score (Renewable Focus)" and ng_columns_available:
        st.subheader("🌱 Green Score - Renewable Energy Focus")
        st.markdown("**Purpose**: Maximizes renewable energy fraction while maintaining high performance")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            display_cols = ["PV_MWdc", "BESS_MW", "NG_MW", "SuccessRate", "NG_CF", "Renewable_Fraction", "Green_Score"]
            format_dict = {
                'SuccessRate': '{:.1%}',
                'NG_CF': '{:.1%}',
                'Renewable_Fraction': '{:.1%}',
                'Green_Score': '{:.3f}'
            }
            
            st.dataframe(top_green[display_cols].style.format(format_dict), use_container_width=True)
        
        with col2:
            best_green = top_green.iloc[0]
            st.success(f"""
            **🌱 Best Green Configuration:**
            - **PV**: {best_green['PV_MWdc']} MW
            - **BESS**: {best_green['BESS_MW']} MW
            - **NG**: {best_green['NG_MW']} MW
            - **Success Rate**: {best_green['SuccessRate']:.1%}
            - **Renewable Fraction**: {best_green['Renewable_Fraction']:.1%}
            - **NG Capacity Factor**: {best_green['NG_CF']:.1%}
            
            **Environmental Benefit:**
            Minimizes fossil fuel dependency
            """)
    
    elif opt_method == "Performance Score (Reliability Focus)" and ng_columns_available:
        st.subheader("⚡ Performance Score - Reliability Focus")
        st.markdown("**Purpose**: Maximizes success rate regardless of NG usage")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            display_cols = ["PV_MWdc", "BESS_MW", "NG_MW", "SuccessRate", "NG_CF", "Carbon_Intensity"]
            format_dict = {
                'SuccessRate': '{:.1%}',
                'NG_CF': '{:.1%}',
                'Carbon_Intensity': '{:.0f}'
            }
            
            st.dataframe(top_performance[display_cols].style.format(format_dict), use_container_width=True)
        
        with col2:
            best_performance = top_performance.iloc[0]
            st.info(f"""
            **⚡ Best Performance Configuration:**
            - **PV**: {best_performance['PV_MWdc']} MW
            - **BESS**: {best_performance['BESS_MW']} MW
            - **NG**: {best_performance['NG_MW']} MW
            - **Success Rate**: {best_performance['SuccessRate']:.1%}
            - **NG Capacity Factor**: {best_performance['NG_CF']:.1%}
            - **Carbon Intensity**: {best_performance['Carbon_Intensity']:.0f} lb CO₂/MWh
            
            **Trade-off:**
            High reliability but higher emissions
            """)
    
    elif opt_method == "Low NG Usage (<30% CF)" and ng_columns_available and ng_runtime_available:
        st.subheader("🔋 Low NG Usage - Renewable-Dominated Systems")
        st.markdown("**Purpose**: Systems where NG provides <30% of operating time (backup only)")
        
        if len(low_ng_configs) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                display_cols = ["PV_MWdc", "BESS_MW", "NG_MW", "SuccessRate", "NG_CF", "Renewable_Fraction", "EES"]
                format_dict = {
                    'SuccessRate': '{:.1%}',
                    'NG_CF': '{:.1%}',
                    'Renewable_Fraction': '{:.1%}',
                    'EES': '{:.3f}'
                }
                
                st.dataframe(low_ng_configs[display_cols].style.format(format_dict), use_container_width=True)
            
            with col2:
                best_low_ng = low_ng_configs.iloc[0]
                st.success(f"""
                **🔋 Best Low-NG Configuration:**
                - **PV**: {best_low_ng['PV_MWdc']} MW
                - **BESS**: {best_low_ng['BESS_MW']} MW
                - **NG**: {best_low_ng['NG_MW']} MW
                - **Success Rate**: {best_low_ng['SuccessRate']:.1%}
                - **NG Capacity Factor**: {best_low_ng['NG_CF']:.1%}
                - **Renewable Fraction**: {best_low_ng['Renewable_Fraction']:.1%}
                
                **Advantage:**
                True renewable firming with minimal backup
                """)
        else:
            st.warning("No configurations found with NG capacity factor <30%")
    
    elif "Pareto Optimal" in opt_method:
        st.subheader("📊 Pareto Optimal Solutions") 
        st.markdown("**Purpose**: Configurations where no objective can be improved without worsening another")
        
        if not pareto_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                display_cols = ["PV_MWdc", "BESS_MW", "SuccessRate", "Deficit_kWh"]
                format_dict = {
                    'SuccessRate': '{:.1%}',
                    'Deficit_kWh': '{:,.0f}'
                }
                
                if ng_columns_available:
                    display_cols.insert(2, "NG_MW")
                    if ng_runtime_available:
                        display_cols.append("NG_CF")
                        format_dict['NG_CF'] = '{:.1%}'
                
                st.dataframe(pareto_df[display_cols].style.format(format_dict), use_container_width=True)
            
            with col2:
                st.info(f"""
                **📈 Pareto Analysis Results:**
                - **Total Solutions**: {len(pareto_df)}
                - **Efficiency**: {len(pareto_df)/len(df)*100:.1f}% of all configs
                
                **Best Pareto Solution:**
                - **PV**: {pareto_df.iloc[0]['PV_MWdc']} MW
                - **BESS**: {pareto_df.iloc[0]['BESS_MW']} MW
                - **Success**: {pareto_df.iloc[0]['SuccessRate']:.1%}
                """)
                
                # Pareto visualization
                if ng_columns_available and ng_runtime_available:
                    fig_pareto = px.scatter_3d(df, x="Deficit_kWh", y="SuccessRate", z="NG_CF",
                                             color="EES", title="3D Pareto Frontier",
                                             labels={"NG_CF": "NG Capacity Factor"})
                    fig_pareto.add_scatter3d(x=pareto_df["Deficit_kWh"], y=pareto_df["SuccessRate"], z=pareto_df["NG_CF"],
                                           mode='markers', marker=dict(color='red', size=8),
                                           name='Pareto Optimal')
                else:
                    fig_pareto = px.scatter(df, x="Deficit_kWh", y="SuccessRate", 
                                          title="Pareto Frontier", opacity=0.6)
                    fig_pareto.add_scatter(x=pareto_df["Deficit_kWh"], y=pareto_df["SuccessRate"],
                                         mode='markers', marker=dict(color='red', size=8),
                                         name='Pareto Optimal')
                st.plotly_chart(fig_pareto, use_container_width=True)
        else:
            st.warning("No Pareto optimal solutions found.")
    
    # Validation and insights
    if ng_columns_available and ng_runtime_available:
        st.markdown("---")
        st.subheader("🔍 Optimization Validation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_ng_count = len(df[df["NG_CF"] > 0.5])
            medium_ng_count = len(df[(df["NG_CF"] > 0.3) & (df["NG_CF"] <= 0.5)])
            low_ng_count = len(df[df["NG_CF"] <= 0.3])
            
            st.markdown("**🔥 NG Usage Distribution:**")
            st.write(f"• High NG (>50% CF): {high_ng_count} configs")
            st.write(f"• Medium NG (30-50% CF): {medium_ng_count} configs")
            st.write(f"• Low NG (<30% CF): {low_ng_count} configs")
        
        with col2:
            avg_renewable_frac = df["Renewable_Fraction"].mean()
            best_renewable_frac = df["Renewable_Fraction"].max()
            
            st.markdown("**🌱 Renewable Performance:**")
            st.write(f"• Avg Renewable Fraction: {avg_renewable_frac:.1%}")
            st.write(f"• Best Renewable Fraction: {best_renewable_frac:.1%}")
            
            high_renewable_count = len(df[df["Renewable_Fraction"] > 0.8])
            st.write(f"• Configs >80% Renewable: {high_renewable_count}")
        
        with col3:
            if ng_emissions_available:
                avg_carbon = df["Carbon_Intensity"].mean()
                min_carbon = df["Carbon_Intensity"].min()
                
                st.markdown("**💨 Carbon Performance:**")
                st.write(f"• Avg Carbon: {avg_carbon:.0f} lb CO₂/MWh")
                st.write(f"• Min Carbon: {min_carbon:.0f} lb CO₂/MWh")
                
                low_carbon_count = len(df[df["Carbon_Intensity"] < 100])
                st.write(f"• Low Carbon Configs: {low_carbon_count}")
            else:
                st.markdown("**💨 Carbon Performance:**")
                st.write("Emissions data not available")
        
        # Recommendations
        st.markdown("**💡 Optimization Insights:**")
        if high_ng_count > low_ng_count:
            st.warning("⚠️ **Many configurations are NG-heavy.** Consider increasing PV/BESS sizes or accepting lower success rates for greener solutions.")
        else:
            st.success("✅ **Good balance found.** Most configurations use NG as backup only.")

# Export functionality
if st.session_state.simulation_complete:
    st.header("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Complete Data"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='All_Results', index=False)
                top_ees.to_excel(writer, sheet_name='Top_EES', index=False)
                
                if not pareto_df.empty:
                    pareto_df.to_excel(writer, sheet_name='Pareto_Solutions', index=False)
                
                # Add hourly data for top 5 configurations
                for i, row in top_ees.head(5).iterrows():
                    if include_ng and "NG_MW" in row:
                        key = (row['PV_MWdc'], row['BESS_MW'], row['NG_MW'])
                    else:
                        key = (row['PV_MWdc'], row['BESS_MW'], 0)
                    
                    if key in st.session_state.hourly_tabs:
                        sheet_name = f"PV_{row['PV_MWdc']}_BESS_{row['BESS_MW']}"
                        if include_ng and "NG_MW" in row:
                            sheet_name += f"_NG_{row['NG_MW']}"
                        sheet_name = sheet_name[:31]  # Excel sheet name limit
                        pd.DataFrame(st.session_state.hourly_tabs[key]).to_excel(writer, sheet_name=sheet_name, index=False)
            
            st.download_button(
                label="📥 Download Excel",
                data=output.getvalue(),
                file_name="Firm_Charging_Complete_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📈 Export Summary Report"):
            best_ees = top_ees.iloc[0]
            ng_summary = ""
            if include_ng and "NG_MW" in best_ees:
                ng_summary = f"""
Natural Gas Integration:
- NG Size: {best_ees['NG_MW']} MW
- NG Runtime: {best_ees.get('NG_Runtime_h', 0):,.0f} hours
- NG Fuel Used: {best_ees.get('NG_Fuel_MMBtu', 0):,.0f} MMBtu
- NG Emissions: {best_ees.get('NG_Emissions_lb', 0):,.0f} lb CO₂
"""
            
            summary_text = f"""
Firm Charging Analysis Summary

System Configuration:
- POI Target: {poi_kw:,} kW
- Battery Efficiency: {rte:.1%}
- Duration: {duration_h} hours
- Natural Gas: {'Enabled' if include_ng else 'Disabled'}

Analysis Results:
- Total Configurations: {len(summary_df):,}
- Best Success Rate: {summary_df['SuccessRate'].max():.1%}
- Average Success Rate: {summary_df['SuccessRate'].mean():.1%}

Best Configuration (EES):
- PV Size: {best_ees['PV_MWdc']} MW
- BESS Size: {best_ees['BESS_MW']} MW
- Success Rate: {best_ees['SuccessRate']:.1%}
- Annual Deficit: {best_ees['Deficit_kWh']:,.0f} kWh
- Annual Surplus: {best_ees['Surplus_kWh']:,.0f} kWh{ng_summary}

Optimization Results:
- Engineering Efficiency Score: {best_ees['EES']:.3f}
- Pareto Solutions Found: {len(pareto_df) if not pareto_df.empty else 0}
            """
            
            st.download_button(
                label="📥 Download Summary",
                data=summary_text,
                file_name="Analysis_Summary.txt",
                mime="text/plain"
            )
    
    with col3:
        st.metric("Export Status", "✅ Ready")

# Footer
st.markdown("---")
st.markdown("""
### 📚 **Firm Charging Analysis Tool**

This application analyzes optimal sizing for firm power delivery using:
- **Solar PV** (primary generation)
- **Battery Energy Storage** (secondary dispatch)  
- **Natural Gas** (optional tertiary backup)

**Key Features:**
- Multi-dimensional optimization with Engineering Efficiency Score
- Pareto optimal solution identification
- Detailed hourly dispatch analysis
- Natural gas integration with engineering parameters
- Comprehensive export capabilities

*Upload PV data, configure system parameters, and optimize for reliable firm power delivery*
""")
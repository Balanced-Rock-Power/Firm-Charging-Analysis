import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------
# Industry Standard Constants
# --------------------------------------------------------------
HOURS_PER_YEAR = 8760
DEFAULT_BESS_POWER_MW = 200  # MW power rating
DEFAULT_BESS_RTE = 0.92  # Round-trip efficiency (updated to industry standard)
DEFAULT_BESS_DEGRADATION = 0.0  # Annual degradation (can be enhanced later)

# Multi-layer efficiency modeling (from industry analysis)
DEFAULT_HV_TRANSFORMER_EFF = 0.992  # HV transformer efficiency
DEFAULT_GEN_TIE_EFF = 0.9993  # Gen-tie line efficiency  
DEFAULT_BESS_AC_COLLECTION_EFF = 0.995  # BESS AC collection efficiency
DEFAULT_PV_DERATING = 0.99  # PV system derating factor

# Natural Gas Plant Engineering Parameters
DEFAULT_NG_HEAT_RATE = 7800  # BTU/kWh (modern combined cycle: 6500-8500)
DEFAULT_NG_PLANT_EFFICIENCY = 0.45  # 45% (HHV basis, typical for modern CCGT)
DEFAULT_NG_PARASITIC_LOAD = 0.04  # 4% parasitic load (station service power)
DEFAULT_NG_MIN_LOAD_FACTOR = 0.30  # 30% minimum stable load
DEFAULT_NG_AVAILABILITY = 0.92  # 92% availability factor (planned + unplanned outages)
DEFAULT_NG_STARTUP_TIME = 1  # Hours for hot start
DEFAULT_NG_STARTUP_ENERGY = 50  # MWh fuel equivalent for startup
DEFAULT_NG_RAMP_RATE = 0.10  # 10% of capacity per minute ramp rate

# PVsyst file structure constants
PVSYST_SHEET_NAME = "8760_All"
PVSYST_SKIP_ROWS = 10
PVSYST_EGRID_COLUMN = "E_Grid"

# --------------------------------------------------------------
# Enhanced PV Profile Loader with Validation
# --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_pv_profile(xlsx_file: bytes, firm_mw_target: float) -> tuple[np.ndarray, dict]:
    """
    Load and validate PVsyst 8760 profile data.
    Returns: (normalized_profile_kw, validation_info)
    """
    try:
        xls = pd.ExcelFile(xlsx_file)
        
        # Validate sheet exists
        if PVSYST_SHEET_NAME not in xls.sheet_names:
            st.error(f"Sheet '{PVSYST_SHEET_NAME}' not found. Available sheets: {xls.sheet_names}")
            st.stop()
        
        # Load and clean data
        df = xls.parse(PVSYST_SHEET_NAME, skiprows=PVSYST_SKIP_ROWS).drop([0, 1]).reset_index(drop=True)
        
        # Validate E_Grid column exists
        if PVSYST_EGRID_COLUMN not in df.columns:
            st.error(f"Column '{PVSYST_EGRID_COLUMN}' not found. Available columns: {list(df.columns)}")
            st.stop()
        
        # Convert to numeric and validate
        df[PVSYST_EGRID_COLUMN] = pd.to_numeric(df[PVSYST_EGRID_COLUMN], errors="coerce")
        
        if df[PVSYST_EGRID_COLUMN].isna().any():
            st.warning("Some E_Grid values could not be converted to numbers. These will be treated as zero.")
            df[PVSYST_EGRID_COLUMN] = df[PVSYST_EGRID_COLUMN].fillna(0)
        
        # Validate data length
        if len(df) != HOURS_PER_YEAR:
            st.warning(f"Expected {HOURS_PER_YEAR} hours, got {len(df)} hours. Results may be inaccurate.")
        
        egrid_values = df[PVSYST_EGRID_COLUMN].values
        max_pv_kw = egrid_values.max()
        
        # Validation info
        validation_info = {
            "max_pv_kw": max_pv_kw,
            "annual_pv_mwh": egrid_values.sum() / 1000,
            "capacity_factor": (egrid_values.mean() / max_pv_kw * 100) if max_pv_kw > 0 else 0,
            "zero_hours": (egrid_values == 0).sum(),
            "data_points": len(egrid_values)
        }
        
        # Scale to firm target (this gives us the base 1x scaling)
        firm_kw_target = firm_mw_target * 1000
        scaled_profile = egrid_values * (firm_kw_target / max_pv_kw) if max_pv_kw > 0 else egrid_values
        
        return scaled_profile, validation_info
        
    except Exception as e:
        st.error(f"Error loading PV profile: {str(e)}")
        st.stop()

# --------------------------------------------------------------
# Enhanced Dispatch Engine with Detailed Tracking
# --------------------------------------------------------------
def run_firm_capacity_dispatch(
    pv_profile_kw: np.ndarray,
    pv_scale_factor: float,
    firm_mw_target: float,
    enable_bess: bool,
    bess_energy_mwh: float,
    enable_ng: bool,
    ng_capacity_mw: float,
    bess_discharge_power_mw: float = DEFAULT_BESS_POWER_MW,
    bess_charge_power_mw: float = DEFAULT_BESS_POWER_MW * 1.1,
    bess_rte: float = DEFAULT_BESS_RTE,
    hv_transformer_eff: float = DEFAULT_HV_TRANSFORMER_EFF,
    gen_tie_eff: float = DEFAULT_GEN_TIE_EFF,
    bess_ac_collection_eff: float = DEFAULT_BESS_AC_COLLECTION_EFF,
    ng_heat_rate: float = DEFAULT_NG_HEAT_RATE,
    ng_plant_efficiency: float = DEFAULT_NG_PLANT_EFFICIENCY,
    ng_parasitic_load: float = DEFAULT_NG_PARASITIC_LOAD,
    ng_min_load_factor: float = DEFAULT_NG_MIN_LOAD_FACTOR,
    ng_availability: float = DEFAULT_NG_AVAILABILITY,
    ng_startup_time: float = DEFAULT_NG_STARTUP_TIME,
    ng_startup_energy: float = DEFAULT_NG_STARTUP_ENERGY,
    ng_ramp_rate: float = DEFAULT_NG_RAMP_RATE
) -> tuple[dict, pd.DataFrame]:
    """
    Enhanced dispatch simulation with comprehensive NG plant engineering modeling.
    
    Priority Logic: PV → BESS → Natural Gas (with engineering constraints)
    Efficiency Layers: HV Transformer → Gen-Tie → BESS AC Collection → BESS RTE
    NG Plant Modeling: Heat rate, parasitic loads, min load, startups, ramp rates
    
    Returns: (summary_metrics, hourly_dataframe)
    """
    
    # Input validation
    if firm_mw_target <= 0:
        st.error("Firm capacity target must be positive")
        st.stop()
    
    if pv_scale_factor < 0:
        st.error("PV scale factor cannot be negative")
        st.stop()
    
    # Convert to consistent units (kW for calculations)
    pv_output_kw_dc = pv_profile_kw * pv_scale_factor
    firm_kw_target = firm_mw_target * 1000
    
    # Calculate BESS capacities accounting for efficiency losses
    total_bess_efficiency = hv_transformer_eff * gen_tie_eff * bess_ac_collection_eff
    bess_energy_kwh_lv = (bess_energy_mwh * 1000) / total_bess_efficiency if enable_bess else 0
    
    # Power ratings at different points in system
    bess_discharge_power_kw_poi = bess_discharge_power_mw * 1000 if enable_bess else 0
    bess_charge_power_kw_poi = bess_charge_power_mw * 1000 if enable_bess else 0
    
    # Adjust power for losses to get LV terminal ratings
    bess_discharge_power_kw_lv = bess_discharge_power_kw_poi / total_bess_efficiency if enable_bess else 0
    bess_charge_power_kw_lv = bess_charge_power_kw_poi / total_bess_efficiency if enable_bess else 0
    
    # NG Plant Parameters
    ng_capacity_kw_gross = ng_capacity_mw * 1000 if enable_ng else 0
    ng_capacity_kw_net = ng_capacity_kw_gross * (1 - ng_parasitic_load) if enable_ng else 0
    ng_min_output_kw = ng_capacity_kw_net * ng_min_load_factor if enable_ng else 0
    ng_max_ramp_kw_per_hour = ng_capacity_kw_net * ng_ramp_rate * 60  # Convert per minute to per hour
    
    # Initialize states
    soc_kwh = 0.0
    ng_previous_output_kw = 0.0  # Previous hour NG output for ramp constraints
    ng_online_hours = 0  # Track consecutive online hours
    ng_startup_count = 0  # Track number of startups
    
    # Generate availability mask (simplified - could be enhanced with maintenance schedules)
    np.random.seed(42)  # For reproducible results
    availability_mask = np.random.random(HOURS_PER_YEAR) < ng_availability
    
    # Track hourly results
    hourly_results = []
    
    # Dispatch simulation
    for hour, pv_kw_dc in enumerate(pv_output_kw_dc, start=1):
        
        # Apply PV system derating to get AC output
        pv_kw_ac = pv_kw_dc * DEFAULT_PV_DERATING
        
        # Check NG availability for this hour
        ng_available = availability_mask[hour - 1] if enable_ng else False
        
        # Initialize hour record
        hour_record = {
            "Hour": hour,
            "PV_DC_kW": pv_kw_dc,
            "PV_AC_kW": pv_kw_ac,
            "BESS_Discharge_kW_POI": 0,
            "BESS_Charge_kW_POI": 0,
            "BESS_Discharge_kW_LV": 0,
            "BESS_Charge_kW_LV": 0,
            "NG_Output_Net_kW": 0,
            "NG_Output_Gross_kW": 0,
            "NG_Fuel_Consumption_MMBtu": 0,
            "NG_Startup_Event": False,
            "NG_Available": ng_available,
            "NG_Parasitic_Load_kW": 0,
            "Firm_Delivered_kW": 0,
            "Energy_Surplus_kW": 0,
            "Energy_Deficit_kW": 0,
            "BESS_SOC_kWh": soc_kwh,
            "BESS_SOC_Percent": (soc_kwh / bess_energy_kwh_lv * 100) if bess_energy_kwh_lv > 0 else 0,
            "System_Efficiency": total_bess_efficiency if enable_bess else 1.0
        }
        
        if pv_kw_ac >= firm_kw_target:
            # SURPLUS SCENARIO: PV exceeds firm target
            surplus_kw = pv_kw_ac - firm_kw_target
            
            # Try to charge battery with surplus
            if enable_bess and surplus_kw > 0:
                max_charge_by_power = min(surplus_kw, bess_charge_power_kw_lv)
                max_charge_by_capacity = (bess_energy_kwh_lv - soc_kwh) / bess_rte
                actual_charge_kw_lv = min(max_charge_by_power, max_charge_by_capacity)
                
                if actual_charge_kw_lv > 0:
                    soc_kwh += actual_charge_kw_lv * bess_rte
                    surplus_kw -= actual_charge_kw_lv
                    hour_record["BESS_Charge_kW_LV"] = actual_charge_kw_lv
                    hour_record["BESS_Charge_kW_POI"] = actual_charge_kw_lv * total_bess_efficiency
            
            # NG plant shutdown logic (if online but not needed)
            if enable_ng and ng_available and ng_previous_output_kw > 0:
                # Check if we can shut down (considering ramp constraints and minimum run time)
                if ng_online_hours >= 2:  # Minimum 2-hour run time (simplified)
                    ng_previous_output_kw = 0
                    ng_online_hours = 0
                else:
                    # Continue running at minimum load
                    ng_net_output_kw = ng_min_output_kw
                    ng_gross_output_kw = ng_net_output_kw / (1 - ng_parasitic_load)
                    
                    hour_record["NG_Output_Net_kW"] = ng_net_output_kw
                    hour_record["NG_Output_Gross_kW"] = ng_gross_output_kw
                    hour_record["NG_Parasitic_Load_kW"] = ng_gross_output_kw * ng_parasitic_load
                    
                    # Calculate fuel consumption with part-load efficiency penalty
                    load_factor = ng_gross_output_kw / ng_capacity_kw_gross
                    part_load_efficiency = ng_plant_efficiency * (0.85 + 0.15 * load_factor)  # Simplified curve
                    fuel_mmbtuh = ng_gross_output_kw * ng_heat_rate / 1000  # Convert to MMBtu/h
                    hour_record["NG_Fuel_Consumption_MMBtu"] = fuel_mmbtuh
                    
                    ng_online_hours += 1
                    ng_previous_output_kw = ng_net_output_kw
            
            hour_record["Energy_Surplus_kW"] = surplus_kw
            firm_delivered_kw = firm_kw_target
            
        else:
            # DEFICIT SCENARIO: PV below firm target
            deficit_kw = firm_kw_target - pv_kw_ac
            sources_kw = pv_kw_ac
            
            # Try to discharge battery to cover deficit
            if enable_bess and deficit_kw > 0:
                max_discharge_by_power = min(deficit_kw / total_bess_efficiency, bess_discharge_power_kw_lv)
                max_discharge_by_energy = soc_kwh
                actual_discharge_kw_lv = min(max_discharge_by_power, max_discharge_by_energy)
                
                if actual_discharge_kw_lv > 0:
                    soc_kwh -= actual_discharge_kw_lv
                    discharge_at_poi = actual_discharge_kw_lv * total_bess_efficiency
                    deficit_kw -= discharge_at_poi
                    sources_kw += discharge_at_poi
                    hour_record["BESS_Discharge_kW_LV"] = actual_discharge_kw_lv
                    hour_record["BESS_Discharge_kW_POI"] = discharge_at_poi
            
            # Try to use natural gas for remaining deficit (with engineering constraints)
            if enable_ng and ng_available and deficit_kw > 0:
                
                # Determine required NG output
                required_ng_net_kw = min(deficit_kw, ng_capacity_kw_net)
                
                # Handle startup if plant was offline
                startup_occurred = False
                if ng_previous_output_kw == 0 and required_ng_net_kw > 0:
                    # Plant startup required
                    startup_occurred = True
                    ng_startup_count += 1
                    ng_online_hours = 1
                    hour_record["NG_Startup_Event"] = True
                    
                    # During startup hour, may have reduced output
                    if ng_startup_time <= 1.0:
                        # Fast start - can reach full load this hour
                        target_ng_net_kw = required_ng_net_kw
                    else:
                        # Slow start - limited output during startup
                        target_ng_net_kw = min(required_ng_net_kw, ng_capacity_kw_net * 0.5)
                else:
                    # Plant already online - apply ramp rate constraints
                    max_increase = ng_max_ramp_kw_per_hour
                    target_ng_net_kw = min(required_ng_net_kw, ng_previous_output_kw + max_increase)
                    ng_online_hours += 1
                
                # Apply minimum load constraint
                if target_ng_net_kw > 0:
                    actual_ng_net_kw = max(target_ng_net_kw, ng_min_output_kw)
                    actual_ng_gross_kw = actual_ng_net_kw / (1 - ng_parasitic_load)
                    
                    # Update deficit and sources
                    ng_contribution = min(actual_ng_net_kw, deficit_kw)
                    deficit_kw -= ng_contribution
                    sources_kw += ng_contribution
                    
                    # Record NG outputs
                    hour_record["NG_Output_Net_kW"] = actual_ng_net_kw
                    hour_record["NG_Output_Gross_kW"] = actual_ng_gross_kw
                    hour_record["NG_Parasitic_Load_kW"] = actual_ng_gross_kw * ng_parasitic_load
                    
                    # Calculate fuel consumption with part-load efficiency
                    load_factor = actual_ng_gross_kw / ng_capacity_kw_gross
                    part_load_efficiency = ng_plant_efficiency * (0.85 + 0.15 * load_factor)
                    fuel_mmbtuh = actual_ng_gross_kw * ng_heat_rate / 1000
                    
                    # Add startup fuel consumption
                    if startup_occurred:
                        startup_fuel_mmbtuh = ng_startup_energy * ng_heat_rate / 1000
                        fuel_mmbtuh += startup_fuel_mmbtuh
                    
                    hour_record["NG_Fuel_Consumption_MMBtu"] = fuel_mmbtuh
                    ng_previous_output_kw = actual_ng_net_kw
                else:
                    ng_previous_output_kw = 0
                    ng_online_hours = 0
            
            hour_record["Energy_Deficit_kW"] = max(0, deficit_kw)
            firm_delivered_kw = min(sources_kw, firm_kw_target)
        
        hour_record["Firm_Delivered_kW"] = firm_delivered_kw
        hour_record["BESS_SOC_kWh"] = soc_kwh
        hour_record["BESS_SOC_Percent"] = (soc_kwh / bess_energy_kwh_lv * 100) if bess_energy_kwh_lv > 0 else 0
        
        hourly_results.append(hour_record)
    
    # Create results dataframe
    df_results = pd.DataFrame(hourly_results)
    
    # Calculate enhanced summary metrics
    total_firm_energy_target_mwh = firm_mw_target * HOURS_PER_YEAR
    total_firm_delivered_mwh = df_results["Firm_Delivered_kW"].sum() / 1000
    hours_at_target = (df_results["Energy_Deficit_kW"] == 0).sum()
    
    # Enhanced NG analytics
    ng_operating_hours = (df_results["NG_Output_Net_kW"] > 0).sum()
    ng_startups = df_results["NG_Startup_Event"].sum()
    total_ng_fuel_mmbtu = df_results["NG_Fuel_Consumption_MMBtu"].sum()
    avg_ng_efficiency = 0
    if df_results["NG_Output_Gross_kW"].sum() > 0:
        avg_ng_efficiency = (df_results["NG_Output_Gross_kW"].sum() * 3.412) / total_ng_fuel_mmbtu * 100  # Convert to %
    
    # Battery cycle analysis
    total_discharge_mwh = df_results["BESS_Discharge_kW_LV"].sum() / 1000
    battery_cycles = total_discharge_mwh / bess_energy_mwh if bess_energy_mwh > 0 else 0
    
    summary_metrics = {
        # Firm Capacity Performance
        "Hours_Meeting_Target": hours_at_target,
        "Hours_Meeting_Target_Percent": round(hours_at_target / HOURS_PER_YEAR * 100, 2),
        "Energy_Delivered_Percent": round(total_firm_delivered_mwh / total_firm_energy_target_mwh * 100, 2),
        
        # Energy Breakdown (MWh)
        "PV_DC_Energy_MWh": round(df_results["PV_DC_kW"].sum() / 1000, 1),
        "PV_AC_Energy_MWh": round(df_results["PV_AC_kW"].sum() / 1000, 1),
        "BESS_Discharge_MWh": round(df_results["BESS_Discharge_kW_POI"].sum() / 1000, 1),
        "BESS_Charge_MWh": round(df_results["BESS_Charge_kW_POI"].sum() / 1000, 1),
        "NG_Energy_Net_MWh": round(df_results["NG_Output_Net_kW"].sum() / 1000, 1),
        "NG_Energy_Gross_MWh": round(df_results["NG_Output_Gross_kW"].sum() / 1000, 1),
        "Surplus_Energy_MWh": round(df_results["Energy_Surplus_kW"].sum() / 1000, 1),
        "Deficit_Energy_MWh": round(df_results["Energy_Deficit_kW"].sum() / 1000, 1),
        
        # Enhanced NG Plant Analytics
        "NG_Operating_Hours": ng_operating_hours,
        "NG_Startups": ng_startups,
        "NG_Capacity_Factor": round(df_results["NG_Output_Net_kW"].mean() / ng_capacity_kw_net * 100, 1) if ng_capacity_kw_net > 0 else 0,
        "NG_Fuel_Consumption_MMBtu": round(total_ng_fuel_mmbtu, 0),
        "NG_Average_Efficiency": round(avg_ng_efficiency, 1),
        "NG_Parasitic_Load_MWh": round(df_results["NG_Parasitic_Load_kW"].sum() / 1000, 1),
        "NG_Heat_Rate_Avg": ng_heat_rate,
        
        # Enhanced Battery Analytics
        "BESS_Cycles": round(battery_cycles, 1),
        "BESS_Capacity_Factor": round(df_results["BESS_Discharge_kW_POI"].mean() / bess_discharge_power_kw_poi * 100, 1) if bess_discharge_power_kw_poi > 0 else 0,
        "Max_BESS_SOC_Percent": round(df_results["BESS_SOC_Percent"].max(), 1),
        "Min_BESS_SOC_Percent": round(df_results["BESS_SOC_Percent"].min(), 1),
        "Avg_BESS_SOC_Percent": round(df_results["BESS_SOC_Percent"].mean(), 1),
        
        # System Performance
        "PV_Capacity_Factor": round(df_results["PV_AC_kW"].mean() / (pv_profile_kw.max() * pv_scale_factor * DEFAULT_PV_DERATING) * 100, 1),
        "System_Efficiency_Avg": round(total_bess_efficiency * 100, 2) if enable_bess else 100.0
    }
    
    return summary_metrics, df_results

# --------------------------------------------------------------
# Streamlit Application Layout
# --------------------------------------------------------------

st.set_page_config(
    page_title="Firm Capacity Analysis Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Utility-Scale Firm Capacity Analysis Tool")
st.markdown("**Analyze PV + BESS + Natural Gas dispatch to meet firm capacity commitments**")

# File upload section
st.subheader("📊 PVsyst Data Input")
uploaded_file = st.file_uploader(
    "Upload PVsyst 8760 output file (.xlsx)",
    type=["xlsx"],
    help="Upload your PVsyst simulation file containing 8760 E_Grid values"
)

if not uploaded_file:
    st.info("📁 Please upload a PVsyst 8760 output file to begin analysis")
    st.markdown("""
    **Expected file format:**
    - Excel file (.xlsx) 
    - Sheet name: '8760_All'
    - Column: 'E_Grid' (energy to grid in kW)
    - 8760 hourly values (skip first 10 rows as headers)
    """)
    st.stop()

# Load and validate PV data
with st.spinner("Loading and validating PV profile..."):
    firm_mw_target = st.number_input(
        "Firm Capacity Target at POI (MW)",
        min_value=1.0,
        max_value=1000.0,
        value=100.0,
        step=5.0,
        help="Target firm capacity to deliver at Point of Interconnection"
    )
    
    pv_profile_kw, validation_info = load_pv_profile(uploaded_file, firm_mw_target)

# Display PV validation info
with st.expander("📋 PV Profile Validation", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max PV Output", f"{validation_info['max_pv_kw']/1000:.1f} MW")
    col2.metric("Annual Energy", f"{validation_info['annual_pv_mwh']:.0f} MWh")
    col3.metric("Capacity Factor", f"{validation_info['capacity_factor']:.1f}%")
    col4.metric("Zero Output Hours", f"{validation_info['zero_hours']}")

# --------------------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------------------
st.sidebar.header("⚙️ System Configuration")

# Resource selection
st.sidebar.subheader("🔋 Resource Mix")
resource_config = st.sidebar.radio(
    "Select Resource Configuration",
    options=["PV Only", "PV + BESS", "PV + BESS + Natural Gas"],
    help="Choose which resources to include in the dispatch analysis"
)

enable_bess = resource_config in ["PV + BESS", "PV + BESS + Natural Gas"]
enable_ng = resource_config == "PV + BESS + Natural Gas"

# PV Configuration
st.sidebar.subheader("☀️ Solar PV")
pv_scale_factor = st.sidebar.slider(
    "PV Scale Factor (×)",
    min_value=0.1,
    max_value=5.0,
    value=1.4,
    step=0.1,
    help="Multiplier for PV capacity (1.0 = base PVsyst sizing)"
)

# BESS Configuration
st.sidebar.subheader("🔋 Battery Energy Storage")
bess_energy_mwh = st.sidebar.slider(
    "BESS Energy Capacity (MWh)",
    min_value=0,
    max_value=2000,
    value=400,
    step=50,
    disabled=not enable_bess,
    help="Battery energy storage capacity at POI"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    bess_discharge_power_mw = st.sidebar.slider(
        "BESS Discharge Power (MW)",
        min_value=10,
        max_value=500,
        value=DEFAULT_BESS_POWER_MW,
        step=10,
        disabled=not enable_bess,
        help="Maximum discharge power rating"
    )
with col2:
    bess_charge_power_mw = st.sidebar.slider(
        "BESS Charge Power (MW)", 
        min_value=10,
        max_value=500,
        value=int(DEFAULT_BESS_POWER_MW * 1.1),  # Typically 10% higher
        step=10,
        disabled=not enable_bess,
        help="Maximum charge power rating"
    )

bess_rte = st.sidebar.slider(
    "BESS Round-Trip Efficiency",
    min_value=0.75,
    max_value=0.95,
    value=DEFAULT_BESS_RTE,
    step=0.01,
    disabled=not enable_bess,
    help="Battery round-trip efficiency (DC-DC)"
)

# Advanced efficiency modeling
with st.sidebar.expander("⚙️ Advanced System Efficiencies"):
    hv_transformer_eff = st.slider(
        "HV Transformer Efficiency",
        min_value=0.95,
        max_value=1.0,
        value=DEFAULT_HV_TRANSFORMER_EFF,
        step=0.001,
        format="%.3f",
        disabled=not enable_bess,
        help="High voltage transformer efficiency"
    )
    gen_tie_eff = st.slider(
        "Gen-Tie Line Efficiency", 
        min_value=0.95,
        max_value=1.0,
        value=DEFAULT_GEN_TIE_EFF,
        step=0.0001,
        format="%.4f", 
        disabled=not enable_bess,
        help="Generation tie line efficiency"
    )
    bess_ac_collection_eff = st.slider(
        "BESS AC Collection Efficiency",
        min_value=0.95,
        max_value=1.0, 
        value=DEFAULT_BESS_AC_COLLECTION_EFF,
        step=0.001,
        format="%.3f",
        disabled=not enable_bess,
        help="BESS AC collection system efficiency"
    )

# Natural Gas Configuration
st.sidebar.subheader("🔥 Natural Gas Backup Generation")
ng_capacity_mw = st.sidebar.slider(
    "NG Generator Capacity (MW)",
    min_value=0,
    max_value=200,
    value=50,
    step=10,
    disabled=not enable_ng,
    help="Natural gas generator nameplate capacity (gross)"
)

# NG Plant Engineering Parameters
with st.sidebar.expander("⚙️ NG Plant Engineering Parameters"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        ng_heat_rate = st.slider(
            "Heat Rate (BTU/kWh)",
            min_value=6500,
            max_value=12000,
            value=DEFAULT_NG_HEAT_RATE,
            step=100,
            disabled=not enable_ng,
            help="Fuel heat rate (HHV basis). CCGT: 6500-8500, Simple Cycle: 9000-12000"
        )
        
        ng_plant_efficiency = st.slider(
            "Plant Efficiency (%)",
            min_value=25.0,
            max_value=65.0,
            value=DEFAULT_NG_PLANT_EFFICIENCY * 100,
            step=1.0,
            disabled=not enable_ng,
            help="Net plant efficiency (HHV basis). CCGT: 40-60%, Simple Cycle: 25-40%"
        ) / 100
        
        ng_parasitic_load = st.slider(
            "Parasitic Load (%)",
            min_value=1.0,
            max_value=10.0,
            value=DEFAULT_NG_PARASITIC_LOAD * 100,
            step=0.5,
            disabled=not enable_ng,
            help="Station service power as % of gross output"
        ) / 100
        
        ng_availability = st.slider(
            "Availability Factor (%)",
            min_value=80.0,
            max_value=98.0,
            value=DEFAULT_NG_AVAILABILITY * 100,
            step=1.0,
            disabled=not enable_ng,
            help="Annual availability accounting for planned/unplanned outages"
        ) / 100
    
    with col2:
        ng_min_load_factor = st.slider(
            "Minimum Load (%)",
            min_value=10.0,
            max_value=50.0,
            value=DEFAULT_NG_MIN_LOAD_FACTOR * 100,
            step=5.0,
            disabled=not enable_ng,
            help="Minimum stable operating load as % of capacity"
        ) / 100
        
        ng_startup_time = st.slider(
            "Startup Time (hours)",
            min_value=0.25,
            max_value=8.0,
            value=float(DEFAULT_NG_STARTUP_TIME),
            step=0.25,
            disabled=not enable_ng,
            help="Time required for hot start (cold start takes longer)"
        )
        
        ng_startup_energy = st.slider(
            "Startup Energy (MWh)",
            min_value=10,
            max_value=200,
            value=DEFAULT_NG_STARTUP_ENERGY,
            step=10,
            disabled=not enable_ng,
            help="Fuel energy equivalent consumed during startup"
        )
        
        ng_ramp_rate = st.slider(
            "Ramp Rate (%/min)",
            min_value=2.0,
            max_value=20.0,
            value=DEFAULT_NG_RAMP_RATE * 100,
            step=1.0,
            disabled=not enable_ng,
            help="Maximum ramp rate as % of capacity per minute"
        ) / 100

# Run simulation
st.sidebar.markdown("---")
run_simulation = st.sidebar.button("🚀 Run Firm Capacity Analysis", type="primary")

if not run_simulation:
    st.info("👈 Configure your system and click 'Run Analysis' to begin simulation")
    st.stop()

# --------------------------------------------------------------
# Run Dispatch Simulation
# --------------------------------------------------------------
with st.spinner("Running enhanced dispatch simulation with NG plant engineering..."):
    summary_metrics, hourly_df = run_firm_capacity_dispatch(
        pv_profile_kw=pv_profile_kw,
        pv_scale_factor=pv_scale_factor,
        firm_mw_target=firm_mw_target,
        enable_bess=enable_bess,
        bess_energy_mwh=bess_energy_mwh,
        enable_ng=enable_ng,
        ng_capacity_mw=ng_capacity_mw,
        bess_discharge_power_mw=bess_discharge_power_mw if enable_bess else DEFAULT_BESS_POWER_MW,
        bess_charge_power_mw=bess_charge_power_mw if enable_bess else DEFAULT_BESS_POWER_MW,
        bess_rte=bess_rte,
        hv_transformer_eff=hv_transformer_eff if enable_bess else DEFAULT_HV_TRANSFORMER_EFF,
        gen_tie_eff=gen_tie_eff if enable_bess else DEFAULT_GEN_TIE_EFF,
        bess_ac_collection_eff=bess_ac_collection_eff if enable_bess else DEFAULT_BESS_AC_COLLECTION_EFF,
        ng_heat_rate=ng_heat_rate if enable_ng else DEFAULT_NG_HEAT_RATE,
        ng_plant_efficiency=ng_plant_efficiency if enable_ng else DEFAULT_NG_PLANT_EFFICIENCY,
        ng_parasitic_load=ng_parasitic_load if enable_ng else DEFAULT_NG_PARASITIC_LOAD,
        ng_min_load_factor=ng_min_load_factor if enable_ng else DEFAULT_NG_MIN_LOAD_FACTOR,
        ng_availability=ng_availability if enable_ng else DEFAULT_NG_AVAILABILITY,
        ng_startup_time=ng_startup_time if enable_ng else DEFAULT_NG_STARTUP_TIME,
        ng_startup_energy=ng_startup_energy if enable_ng else DEFAULT_NG_STARTUP_ENERGY,
        ng_ramp_rate=ng_ramp_rate if enable_ng else DEFAULT_NG_RAMP_RATE
    )

# --------------------------------------------------------------
# Results Display
# --------------------------------------------------------------

# Key Performance Indicators
st.subheader("📊 Firm Capacity Performance")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

kpi_col1.metric(
    "Hours Meeting Target",
    f"{summary_metrics['Hours_Meeting_Target']:,}/{HOURS_PER_YEAR:,}",
    f"{summary_metrics['Hours_Meeting_Target_Percent']}%"
)

kpi_col2.metric(
    "Energy Delivered",
    f"{summary_metrics['Energy_Delivered_Percent']}%",
    f"{summary_metrics['Deficit_Energy_MWh']} MWh deficit"
)

if enable_bess:
    kpi_col3.metric(
        "Battery Performance",
        f"{summary_metrics['BESS_Cycles']} cycles/yr",
        f"Avg SOC: {summary_metrics['Avg_BESS_SOC_Percent']}%"
    )

if enable_ng:
    kpi_col4.metric(
        "NG Plant Performance",
        f"{summary_metrics['NG_Capacity_Factor']}%",
        f"{summary_metrics['NG_Operating_Hours']} hours online"
    )

# Enhanced system performance indicators
performance_indicators = []
if enable_bess:
    performance_indicators.append(f"**System Efficiency**: {summary_metrics['System_Efficiency_Avg']}%")
performance_indicators.append(f"**PV Capacity Factor**: {summary_metrics['PV_Capacity_Factor']}%")
if enable_bess:
    performance_indicators.append(f"**BESS Capacity Factor**: {summary_metrics['BESS_Capacity_Factor']}%")
if enable_ng:
    performance_indicators.append(f"**NG Plant Efficiency**: {summary_metrics['NG_Average_Efficiency']}%")
    performance_indicators.append(f"**NG Startups**: {summary_metrics['NG_Startups']}")

if performance_indicators:
    st.info(" | ".join(performance_indicators))

# Natural Gas Plant Engineering Analysis
if enable_ng and summary_metrics['NG_Energy_Net_MWh'] > 0:
    st.subheader("🔥 Natural Gas Plant Engineering Analysis")
    
    ng_col1, ng_col2, ng_col3, ng_col4, ng_col5 = st.columns(5)
    
    ng_col1.metric(
        "Fuel Consumption",
        f"{summary_metrics['NG_Fuel_Consumption_MMBtu']:,.0f}",
        "MMBtu"
    )
    
    ng_col2.metric(
        "Gross vs Net Energy",
        f"{summary_metrics['NG_Energy_Gross_MWh']:.1f} MWh",
        f"Net: {summary_metrics['NG_Energy_Net_MWh']:.1f} MWh"
    )
    
    ng_col3.metric(
        "Parasitic Losses",
        f"{summary_metrics['NG_Parasitic_Load_MWh']:.1f} MWh",
        f"{(summary_metrics['NG_Parasitic_Load_MWh']/summary_metrics['NG_Energy_Gross_MWh']*100):.1f}%" if summary_metrics['NG_Energy_Gross_MWh'] > 0 else "0%"
    )
    
    ng_col4.metric(
        "Average Heat Rate",
        f"{summary_metrics['NG_Heat_Rate_Avg']:,.0f}",
        "BTU/kWh"
    )
    
    ng_col5.metric(
        "Plant Starts",
        f"{summary_metrics['NG_Startups']}",
        f"{summary_metrics['NG_Operating_Hours']/summary_metrics['NG_Startups']:.1f} hrs/start" if summary_metrics['NG_Startups'] > 0 else "N/A"
    )

# Energy Mix Breakdown
st.subheader("⚡ Annual Energy Breakdown")

# Enhanced energy breakdown with engineering details
energy_sources = []
energy_values = []
colors = []

energy_sources.append("PV Generation (AC)")
energy_values.append(summary_metrics["PV_AC_Energy_MWh"])
colors.append("#FFD700")

if enable_bess and summary_metrics["BESS_Discharge_MWh"] > 0:
    energy_sources.append("BESS Discharge")
    energy_values.append(summary_metrics["BESS_Discharge_MWh"])
    colors.append("#1E90FF")

if enable_ng and summary_metrics["NG_Energy_Net_MWh"] > 0:
    energy_sources.append("NG Plant (Net)")
    energy_values.append(summary_metrics["NG_Energy_Net_MWh"])
    colors.append("#FF6347")

if summary_metrics["Surplus_Energy_MWh"] > 0:
    energy_sources.append("Surplus (Curtailed)")
    energy_values.append(summary_metrics["Surplus_Energy_MWh"])
    colors.append("#90EE90")

energy_fig = px.bar(
    x=energy_sources,
    y=energy_values,
    color=energy_sources,
    title="Annual Energy Production by Source (Net Deliverable Energy)",
    labels={"x": "Energy Source", "y": "Energy (MWh)"},
    color_discrete_sequence=colors
)
energy_fig.update_layout(showlegend=False, height=400)
st.plotly_chart(energy_fig, use_container_width=True)

# Enhanced system losses analysis
with st.expander("🔌 Comprehensive System Losses Analysis"):
    loss_col1, loss_col2, loss_col3, loss_col4 = st.columns(4)
    
    # PV losses
    pv_dc_ac_loss = summary_metrics["PV_DC_Energy_MWh"] - summary_metrics["PV_AC_Energy_MWh"]
    loss_col1.metric(
        "PV DC→AC Losses", 
        f"{pv_dc_ac_loss:.1f} MWh", 
        f"{(pv_dc_ac_loss/summary_metrics['PV_DC_Energy_MWh']*100):.1f}%" if summary_metrics["PV_DC_Energy_MWh"] > 0 else "0%"
    )
    
    # BESS losses
    if enable_bess:
        bess_charge_discharge_loss = summary_metrics["BESS_Charge_MWh"] - summary_metrics["BESS_Discharge_MWh"]
        loss_col2.metric(
            "BESS RTE Losses", 
            f"{bess_charge_discharge_loss:.1f} MWh", 
            f"RTE: {bess_rte*100:.1f}%"
        )
        
        loss_col3.metric(
            "System Efficiency", 
            f"{summary_metrics['System_Efficiency_Avg']:.2f}%", 
            "Multi-layer losses"
        )
    
    # NG plant losses
    if enable_ng and summary_metrics["NG_Energy_Gross_MWh"] > 0:
        ng_gross_net_loss = summary_metrics["NG_Energy_Gross_MWh"] - summary_metrics["NG_Energy_Net_MWh"]
        loss_col4.metric(
            "NG Parasitic Losses",
            f"{ng_gross_net_loss:.1f} MWh",
            f"{(ng_gross_net_loss/summary_metrics['NG_Energy_Gross_MWh']*100):.1f}%"
        )

# 8760 Dispatch Profile with enhanced NG visualization
st.subheader("📈 8760 Hour Dispatch Profile with Engineering Details")

# Create enhanced stacked area chart
dispatch_fig = go.Figure()

# Add PV (use AC output)
dispatch_fig.add_trace(go.Scatter(
    x=hourly_df["Hour"],
    y=hourly_df["PV_AC_kW"]/1000,
    fill='tonexty',
    mode='none',
    name='PV Generation (AC)',
    fillcolor='rgba(255, 215, 0, 0.7)',
    hovertemplate='Hour: %{x}<br>PV AC: %{y:.1f} MW<extra></extra>'
))

if enable_bess:
    # Add BESS discharge on top of PV (use POI values)
    dispatch_fig.add_trace(go.Scatter(
        x=hourly_df["Hour"],
        y=(hourly_df["PV_AC_kW"] + hourly_df["BESS_Discharge_kW_POI"])/1000,
        fill='tonexty',
        mode='none',
        name='BESS Discharge',
        fillcolor='rgba(30, 144, 255, 0.7)',
        hovertemplate='Hour: %{x}<br>BESS: %{customdata:.1f} MW<extra></extra>',
        customdata=hourly_df["BESS_Discharge_kW_POI"]/1000
    ))

if enable_ng:
    # Add NG net output on top of PV + BESS
    total_without_ng = hourly_df["PV_AC_kW"] + hourly_df["BESS_Discharge_kW_POI"]
    dispatch_fig.add_trace(go.Scatter(
        x=hourly_df["Hour"],
        y=(total_without_ng + hourly_df["NG_Output_Net_kW"])/1000,
        fill='tonexty',
        mode='none',
        name='NG Plant (Net)',
        fillcolor='rgba(255, 99, 71, 0.7)',
        hovertemplate='Hour: %{x}<br>NG Net: %{customdata:.1f} MW<br>NG Gross: %{customdata2:.1f} MW<extra></extra>',
        customdata=hourly_df["NG_Output_Net_kW"]/1000,
        customdata2=hourly_df["NG_Output_Gross_kW"]/1000
    ))

# Add firm target line
dispatch_fig.add_trace(go.Scatter(
    x=hourly_df["Hour"],
    y=[firm_mw_target] * len(hourly_df),
    mode='lines',
    name='Firm Target',
    line=dict(color='black', width=2, dash='dash'),
    hovertemplate='Firm Target: %{y} MW<extra></extra>'
))

# Add NG startup events if any
if enable_ng and hourly_df["NG_Startup_Event"].any():
    startup_hours = hourly_df[hourly_df["NG_Startup_Event"]]["Hour"]
    dispatch_fig.add_trace(go.Scatter(
        x=startup_hours,
        y=[firm_mw_target * 1.1] * len(startup_hours),
        mode='markers',
        name='NG Startups',
        marker=dict(color='red', size=8, symbol='triangle-up'),
        hovertemplate='NG Startup<br>Hour: %{x}<extra></extra>'
    ))

dispatch_fig.update_layout(
    title="Enhanced Dispatch Profile with NG Plant Engineering Constraints",
    xaxis_title="Hour of Year",
    yaxis_title="Power Output (MW)",
    hovermode='x unified',
    height=500
)

st.plotly_chart(dispatch_fig, use_container_width=True)

# Battery SOC Profile (if BESS enabled)
if enable_bess:
    st.subheader("🔋 Battery State of Charge Profile")
    
    soc_fig = px.line(
        hourly_df,
        x="Hour",
        y="BESS_SOC_Percent",
        title="Battery State of Charge Throughout Year",
        labels={"BESS_SOC_Percent": "SOC (%)", "Hour": "Hour of Year"}
    )
    soc_fig.add_hline(y=100, line_dash="dash", line_color="red", 
                      annotation_text="Full Capacity")
    soc_fig.add_hline(y=0, line_dash="dash", line_color="red", 
                      annotation_text="Empty")
    soc_fig.update_layout(height=400)
    st.plotly_chart(soc_fig, use_container_width=True)

# Deficit Analysis
if summary_metrics["Deficit_Energy_MWh"] > 0:
    st.subheader("⚠️ Deficit Analysis")
    
    deficit_hours = hourly_df[hourly_df["Energy_Deficit_kW"] > 0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Deficit Hours", len(deficit_hours))
        st.metric("Average Deficit per Hour", f"{deficit_hours['Energy_Deficit_kW'].mean()/1000:.1f} MW")
        st.metric("Max Hourly Deficit", f"{deficit_hours['Energy_Deficit_kW'].max()/1000:.1f} MW")
    
    with col2:
        # Monthly deficit breakdown
        deficit_hours_copy = deficit_hours.copy()
        deficit_hours_copy['Month'] = ((deficit_hours_copy['Hour'] - 1) // (HOURS_PER_YEAR // 12)) + 1
        monthly_deficit = deficit_hours_copy.groupby('Month')['Energy_Deficit_kW'].sum() / 1000
        
        monthly_fig = px.bar(
            x=monthly_deficit.index,
            y=monthly_deficit.values,
            title="Monthly Deficit Energy (MWh)",
            labels={"x": "Month", "y": "Deficit (MWh)"}
        )
        st.plotly_chart(monthly_fig, use_container_width=True)

# Data Export
st.subheader("📥 Export Results")
with st.expander("Download Simulation Data"):
    
    # Prepare enhanced summary for export
    summary_export = {
        **summary_metrics,
        "Configuration": {
            "PV_Scale_Factor": pv_scale_factor,
            "Firm_Target_MW": firm_mw_target,
            "BESS_Enabled": enable_bess,
            "BESS_Energy_MWh": bess_energy_mwh if enable_bess else 0,
            "BESS_Discharge_Power_MW": bess_discharge_power_mw if enable_bess else 0,
            "BESS_Charge_Power_MW": bess_charge_power_mw if enable_bess else 0,
            "BESS_RTE": bess_rte if enable_bess else 0,
            "NG_Enabled": enable_ng,
            "NG_Capacity_MW": ng_capacity_mw if enable_ng else 0,
            "NG_Heat_Rate_BTU_per_kWh": ng_heat_rate if enable_ng else 0,
            "NG_Plant_Efficiency": ng_plant_efficiency if enable_ng else 0,
            "NG_Parasitic_Load_Percent": ng_parasitic_load * 100 if enable_ng else 0,
            "NG_Min_Load_Factor_Percent": ng_min_load_factor * 100 if enable_ng else 0,
            "NG_Availability_Percent": ng_availability * 100 if enable_ng else 0
        }
    }
    
    summary_df = pd.DataFrame([summary_export]).T
    summary_df.columns = ['Value']
    summary_df.index.name = 'Metric'
    
    col1, col2 = st.columns(2)
    
    with col1:
        summary_csv = summary_df.to_csv()
        st.download_button(
            label="📊 Download Enhanced Summary (CSV)",
            data=summary_csv,
            file_name=f"firm_capacity_summary_enhanced_{resource_config.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )
    
    with col2:
        hourly_csv = hourly_df.to_csv(index=False)
        st.download_button(
            label="📈 Download Hourly Engineering Data (CSV)",
            data=hourly_csv,
            file_name=f"firm_capacity_hourly_engineering_{resource_config.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )
    
    # Additional NG-specific export if NG is enabled
    if enable_ng and summary_metrics.get('NG_Energy_Net_MWh', 0) > 0:
        ng_analysis_data = {
            "NG Plant Performance Summary": {
                "Operating_Hours": summary_metrics['NG_Operating_Hours'],
                "Capacity_Factor_Percent": summary_metrics['NG_Capacity_Factor'],
                "Total_Startups": summary_metrics['NG_Startups'],
                "Average_Runtime_per_Start_Hours": summary_metrics['NG_Operating_Hours']/summary_metrics['NG_Startups'] if summary_metrics['NG_Startups'] > 0 else 0,
                "Fuel_Consumption_MMBtu": summary_metrics['NG_Fuel_Consumption_MMBtu'],
                "Average_Plant_Efficiency_Percent": summary_metrics['NG_Average_Efficiency'],
                "Gross_Generation_MWh": summary_metrics['NG_Energy_Gross_MWh'],
                "Net_Generation_MWh": summary_metrics['NG_Energy_Net_MWh'],
                "Parasitic_Load_MWh": summary_metrics['NG_Parasitic_Load_MWh'],
                "Heat_Rate_BTU_per_kWh": summary_metrics['NG_Heat_Rate_Avg']
            }
        }
        
        ng_df = pd.DataFrame([ng_analysis_data["NG Plant Performance Summary"]]).T
        ng_df.columns = ['Value']
        ng_df.index.name = 'NG_Plant_Metric'
        
        ng_csv = ng_df.to_csv()
        st.download_button(
            label="🔥 Download NG Plant Engineering Analysis (CSV)",
            data=ng_csv,
            file_name=f"ng_plant_engineering_analysis_{resource_config.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )

# Add reset button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset to Defaults"):
    st.rerun()

# Sensitivity Analysis Section (inspired by industry research)
st.sidebar.markdown("---")
if st.sidebar.button("📊 Quick Sensitivity Analysis"):
    st.subheader("🔬 Quick Sensitivity Analysis")
    
    # Define sensitivity ranges based on industry analysis
    pv_scales = [0.8, 1.0, 1.2, 1.4, 1.6, 2.0]
    bess_sizes = [200, 400, 600, 800, 1000] if enable_bess else [0]
    
    sensitivity_results = []
    
    progress_bar = st.progress(0)
    total_scenarios = len(pv_scales) * len(bess_sizes)
    scenario_count = 0
    
    for pv_scale in pv_scales:
        for bess_size in bess_sizes:
            scenario_count += 1
            progress_bar.progress(scenario_count / total_scenarios)
            
            # Run quick analysis for this scenario
            summary, _ = run_firm_capacity_dispatch(
                pv_profile_kw=pv_profile_kw,
                pv_scale_factor=pv_scale,
                firm_mw_target=firm_mw_target,
                enable_bess=enable_bess and bess_size > 0,
                bess_energy_mwh=bess_size,
                enable_ng=enable_ng,
                ng_capacity_mw=ng_capacity_mw,
                bess_discharge_power_mw=bess_discharge_power_mw if enable_bess else DEFAULT_BESS_POWER_MW,
                bess_charge_power_mw=bess_charge_power_mw if enable_bess else DEFAULT_BESS_POWER_MW,
                bess_rte=bess_rte,
                hv_transformer_eff=hv_transformer_eff if enable_bess else DEFAULT_HV_TRANSFORMER_EFF,
                gen_tie_eff=gen_tie_eff if enable_bess else DEFAULT_GEN_TIE_EFF,
                bess_ac_collection_eff=bess_ac_collection_eff if enable_bess else DEFAULT_BESS_AC_COLLECTION_EFF,
                ng_heat_rate=ng_heat_rate if enable_ng else DEFAULT_NG_HEAT_RATE,
                ng_plant_efficiency=ng_plant_efficiency if enable_ng else DEFAULT_NG_PLANT_EFFICIENCY,
                ng_parasitic_load=ng_parasitic_load if enable_ng else DEFAULT_NG_PARASITIC_LOAD,
                ng_min_load_factor=ng_min_load_factor if enable_ng else DEFAULT_NG_MIN_LOAD_FACTOR,
                ng_availability=ng_availability if enable_ng else DEFAULT_NG_AVAILABILITY,
                ng_startup_time=ng_startup_time if enable_ng else DEFAULT_NG_STARTUP_TIME,
                ng_startup_energy=ng_startup_energy if enable_ng else DEFAULT_NG_STARTUP_ENERGY,
                ng_ramp_rate=ng_ramp_rate if enable_ng else DEFAULT_NG_RAMP_RATE
            )
            
            sensitivity_results.append({
                "PV_Scale": pv_scale,
                "BESS_Size_MWh": bess_size,
                "Hours_Met_%": summary["Hours_Meeting_Target_Percent"],
                "Energy_Met_%": summary["Energy_Delivered_Percent"],
                "Deficit_MWh": summary["Deficit_Energy_MWh"],
                "Surplus_MWh": summary["Surplus_Energy_MWh"],
                "BESS_Cycles": summary["BESS_Cycles"] if enable_bess and bess_size > 0 else 0
            })
    
    progress_bar.empty()
    
    # Create sensitivity analysis visualization
    sens_df = pd.DataFrame(sensitivity_results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Heatmap of Hours Met %
        if enable_bess:
            pivot_hours = sens_df.pivot(index="BESS_Size_MWh", columns="PV_Scale", values="Hours_Met_%")
            fig_hours = px.imshow(
                pivot_hours,
                title="Hours Meeting Target (%) - Sensitivity Analysis",
                labels=dict(x="PV Scale Factor", y="BESS Size (MWh)", color="Hours Met %"),
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_hours, use_container_width=True)
        else:
            # Line chart for PV only
            pv_only_df = sens_df[sens_df["BESS_Size_MWh"] == 0]
            fig_pv = px.line(
                pv_only_df,
                x="PV_Scale",
                y="Hours_Met_%",
                title="PV Only - Hours Meeting Target (%)",
                markers=True
            )
            st.plotly_chart(fig_pv, use_container_width=True)
    
    with col2:
        # Deficit analysis
        if enable_bess:
            pivot_deficit = sens_df.pivot(index="BESS_Size_MWh", columns="PV_Scale", values="Deficit_MWh")
            fig_deficit = px.imshow(
                pivot_deficit,
                title="Annual Deficit (MWh) - Sensitivity Analysis",
                labels=dict(x="PV Scale Factor", y="BESS Size (MWh)", color="Deficit MWh"),
                color_continuous_scale="Reds_r"
            )
            st.plotly_chart(fig_deficit, use_container_width=True)
        else:
            # Deficit line chart for PV only
            fig_def = px.line(
                pv_only_df,
                x="PV_Scale", 
                y="Deficit_MWh",
                title="PV Only - Annual Deficit (MWh)",
                markers=True
            )
            st.plotly_chart(fig_def, use_container_width=True)
    
    # Summary table
    st.subheader("📋 Sensitivity Analysis Results")
    
    # Find optimal scenarios
    best_hours = sens_df.loc[sens_df["Hours_Met_%"].idxmax()]
    best_energy = sens_df.loc[sens_df["Energy_Met_%"].idxmax()]
    min_deficit = sens_df.loc[sens_df["Deficit_MWh"].idxmin()]
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric(
            "Best Hours Met",
            f"{best_hours['Hours_Met_%']:.1f}%",
            f"PV: {best_hours['PV_Scale']}×, BESS: {best_hours['BESS_Size_MWh']:.0f} MWh"
        )
    
    with result_col2:
        st.metric(
            "Best Energy Delivered", 
            f"{best_energy['Energy_Met_%']:.1f}%",
            f"PV: {best_energy['PV_Scale']}×, BESS: {best_energy['BESS_Size_MWh']:.0f} MWh"
        )
    
    with result_col3:
        st.metric(
            "Minimum Deficit",
            f"{min_deficit['Deficit_MWh']:.0f} MWh",
            f"PV: {min_deficit['PV_Scale']}×, BESS: {min_deficit['BESS_Size_MWh']:.0f} MWh"
        )
    
    # Detailed results table
    with st.expander("📊 Detailed Sensitivity Results"):
        st.dataframe(
            sens_df.round(1),
            use_container_width=True,
            height=300
        )
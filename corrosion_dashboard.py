import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# OILFIELD WATER QUALITY & CORROSION ANALYSIS DASHBOARD
# Based on the user's brine_corrosion_model.py
# ============================================================

st.set_page_config(
    page_title="Oilfield Water Quality & Corrosion Analysis",
    page_icon="💧",
    layout="wide"
)

# -------------------- Model constants ------------------------
IRON_EQUIVALENT_WEIGHT = 27.92       # g/equivalent, Fe -> Fe2+
STEEL_DENSITY = 7.86                 # g/cm3
REFERENCE_TEMPERATURE_C = 25.0
APPARENT_ACTIVATION_ENERGY_J_MOL = 40000.0  # assumed, illustrative
GAS_CONSTANT = 8.314462618
FARADAY = 96485.0
E0_FE = -0.440
FE2_ACTIVITY = 1e-6

DEFAULT_DATA = pd.DataFrame([
    {
        "sample": "Distilled water",
        "salt": "Reference",
        "concentration_ppm": 0.0,
        "pH": 7.87,
        "pH_status": "Measured",
        "conductivity_uScm": 259.0,
        "salinity_ppt": 0.17,
        "tds_mgL": 148.0
    },
    {
        "sample": "NaCl 1000 ppm",
        "salt": "NaCl",
        "concentration_ppm": 1000.0,
        "pH": 7.0,
        "pH_status": "Assumed",
        "conductivity_uScm": 1750.0,
        "salinity_ppt": 1.13,
        "tds_mgL": 920.0
    },
    {
        "sample": "KCl 1000 ppm",
        "salt": "KCl",
        "concentration_ppm": 1000.0,
        "pH": 7.0,
        "pH_status": "Assumed",
        "conductivity_uScm": 1550.0,
        "salinity_ppt": 0.99,
        "tds_mgL": 810.0
    },
    {
        "sample": "Na2SO4 1000 ppm",
        "salt": "Na2SO4",
        "concentration_ppm": 1000.0,
        "pH": 7.0,
        "pH_status": "Assumed",
        "conductivity_uScm": 1320.0,
        "salinity_ppt": 0.83,
        "tds_mgL": 700.0
    }
])

SALT_CHEMISTRY = {
    "NaCl": (58.44, ("Na+", 22.99, 1, 1), ("Cl-", 35.45, 1, 1)),
    "KCl": (74.55, ("K+", 39.10, 1, 1), ("Cl-", 35.45, 1, 1)),
    "Na2SO4": (142.04, ("Na+", 22.99, 1, 2), ("SO4 2-", 96.06, 2, 1)),
}
ICORR_SCENARIOS = {
    "Low": 5.0,
    "Moderate": 10.0,
    "High": 20.0,
    "Very high": 50.0
}

# -------------------- Calculation functions -----------------
def corrosion_rate_mm_year(icorr):
    """Faraday-law conversion used in the supplied model."""
    return 0.00327 * np.asarray(icorr, dtype=float) * IRON_EQUIVALENT_WEIGHT / STEEL_DENSITY


def severity_band(cr):
    # Reproduces the thresholds coded in the supplied model.
    if cr < 0.025:
        return "Low"
    if cr <= 0.12:
        return "Moderate"
    if cr <= 0.25:
        return "High"
    return "Severe"
def arrhenius_factor(temp_c, activation_energy=APPARENT_ACTIVATION_ENERGY_J_MOL):
    t_ref = REFERENCE_TEMPERATURE_C + 273.15
    t = np.asarray(temp_c, dtype=float) + 273.15
    return np.exp((activation_energy / GAS_CONSTANT) * (1.0 / t_ref - 1.0 / t))


def build_ion_table(data):
    ref_ec = float(data.loc[data["salt"] == "Reference", "conductivity_uScm"].iloc[0])
    rows = []
    for _, r in data[data["salt"] != "Reference"].iterrows():
        if r["salt"] not in SALT_CHEMISTRY or r["concentration_ppm"] <= 0:
            continue
        m_salt, (cat, m_cat, z_cat, n_cat), (an, m_an, z_an, n_an) = SALT_CHEMISTRY[r["salt"]]
        mol_l = float(r["concentration_ppm"]) / 1000.0 / m_salt
        ionic_strength = 0.5 * (
            n_cat * mol_l * z_cat ** 2 + n_an * mol_l * z_an ** 2
        )
        rows.append({
            "Sample": r["sample"],
            "Salt concentration (mmol/L)": mol_l * 1000,
            "Cation": cat,
            "Cation (mg/L)": n_cat * mol_l * m_cat * 1000,
            "Anion": an,
            "Anion (mg/L)": n_an * mol_l * m_an * 1000,
            "Chloride (mg/L)": n_an * mol_l * m_an * 1000 if an == "Cl-" else 0.0,
            "Sulfate (mg/L)": n_an * mol_l * m_an * 1000 if an.startswith("SO4") else 0.0,
            "Ionic strength (mol/L)": ionic_strength,
            "Net EC per mmol/L": (float(r["conductivity_uScm"]) - ref_ec) / (mol_l * 1000),
            "Data type": "Calculated from salt formula + transcribed EC"
        })
    return pd.DataFrame(rows)

def make_csv_download(frame):
    return frame.to_csv(index=False).encode("utf-8")


# -------------------- Header and sidebar --------------------
st.title("💧 Oilfield Water Quality & Corrosion Analysis")
st.caption("Simulation-based analysis of distilled water and prepared brine solutions")

st.sidebar.header("Model controls")
st.sidebar.write("Adjust the scenario inputs below. Changes update the dashboard.")

if "water_data" not in st.session_state:
    st.session_state.water_data = DEFAULT_DATA.copy()

if st.sidebar.button("Restore original sample data"):
    st.session_state.water_data = DEFAULT_DATA.copy()
    st.rerun()

st.sidebar.subheader("Corrosion scenario")
scenario_name = st.sidebar.selectbox(
    "Reference corrosion-current scenario",
    list(ICORR_SCENARIOS.keys()),
    index=1
)
preset_icorr = ICORR_SCENARIOS[scenario_name]

icorr = st.sidebar.number_input(
    "Assumed corrosion current density, Icorr (µA/cm²)",
    min_value=0.1,
    max_value=500.0,
    value=float(preset_icorr),
    step=0.5,
    format="%.2f",
    help=(
        "This is a user-entered illustrative assumption, not a measured value. "
        "Choose a reference scenario above, then edit this number as needed."
    )
)
st.sidebar.caption(
    "Reference values: Low = 5, Moderate = 10, High = 20, Very high = 50 µA/cm². "
    "The selected Icorr is used for the theoretical corrosion calculations."
)
temperature = st.sidebar.slider("Temperature (°C)", 20, 60, 25, 1)
exposure_years = st.sidebar.selectbox("Exposure period (years)", [1, 2, 5, 10], index=2)
activation_energy_kj = st.sidebar.slider(
    "Apparent activation energy (kJ/mol; assumed)",
    min_value=10, max_value=80, value=40, step=5
)

# -------------------- Data editing ---------------------------
st.subheader("1. Water sample data")
st.info(
    "The initial values below are transcribed from the supplied Python model. "
    "The pH of the three salt solutions is assumed to be 7.0, not measured. "
    "Edit values only when you have verified readings from your laboratory."
)

edited_data = st.data_editor(
    st.session_state.water_data,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=["sample", "salt", "concentration_ppm", "pH_status"],
    column_config={
        "pH": st.column_config.NumberColumn("pH", min_value=0.0, max_value=14.0, format="%.2f"),
        "conductivity_uScm": st.column_config.NumberColumn("Conductivity (µS/cm)", min_value=0.01, format="%.2f"),
        "salinity_ppt": st.column_config.NumberColumn("Salinity (ppt)", min_value=0.0, format="%.3f"),
        "tds_mgL": st.column_config.NumberColumn("TDS (mg/L)", min_value=0.0, format="%.1f")
    }
)
st.session_state.water_data = edited_data.copy()

data = edited_data.copy()
numeric_cols = ["pH", "conductivity_uScm", "salinity_ppt", "tds_mgL", "concentration_ppm"]
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

if data[numeric_cols].isna().any().any():
    st.error("Please enter valid numeric values in all water-quality fields.")
    st.stop()
if (data["conductivity_uScm"] <= 0).any():
    st.error("Conductivity must be greater than zero.")
    st.stop()
if (data[["salinity_ppt", "tds_mgL"]] < 0).any().any():
    st.error("Salinity and TDS cannot be negative.")
    st.stop()
if (data["pH"] < 0).any() or (data["pH"] > 14).any():
    st.error("pH must be between 0 and 14.")
    st.stop()
if (data["salt"] == "Reference").sum() != 1:
    st.error("The dataset must contain exactly one distilled-water reference row.")
    st.stop()

reference = data.loc[data["salt"] == "Reference"].iloc[0]
data["TDS_to_EC_ratio"] = data["tds_mgL"] / data["conductivity_uScm"]
data["resistivity_ohm_cm"] = 1e6 / data["conductivity_uScm"]
for col in ["conductivity_uScm", "salinity_ppt", "tds_mgL"]:
    data[col + "_change_pct_vs_ref"] = (data[col] - reference[col]) / reference[col] * 100.0
    data[col + "_net_of_water"] = data[col] - reference[col]
data["pH_difference_vs_ref"] = data["pH"] - reference["pH"]

# -------------------- Key metrics ----------------------------
brines = data[data["salt"] != "Reference"].copy()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Water samples", len(data))
m2.metric("Brine samples", len(brines))
m3.metric("Highest conductivity", f"{data['conductivity_uScm'].max():,.0f} µS/cm")
m4.metric("Highest TDS", f"{data['tds_mgL'].max():,.0f} mg/L")

# -------------------- Tabs -----------------------------------
tab_overview, tab_water, tab_ions, tab_sensitivity, tab_corrosion, tab_downloads = st.tabs([
    "Overview",
    "Water quality",
    "Ion chemistry",
    "Sensitivity analysis",
    "Corrosion scenarios",
    "Download results"
])

with tab_overview:
    st.subheader("Project overview")
    st.write(
        "This dashboard compares measured/transcribed water-quality readings for "
        "distilled water and three prepared 1000 ppm salt solutions. It also provides "
        "separate sensitivity indicators and illustrative corrosion scenarios."
    )
    st.warning(
        "Interpretation boundary: water-quality readings alone do not determine a "
        "corrosion rate. The corrosion-current densities and Arrhenius activation energy "
        "used here are assumptions for scenario analysis, not laboratory measurements "
        "or validated field predictions."
    )
    st.markdown("**Data status**")
    status_view = data[["sample", "pH", "pH_status", "conductivity_uScm", "salinity_ppt", "tds_mgL"]].copy()
    status_view.columns = ["Sample", "pH", "pH status", "Conductivity (µS/cm)", "Salinity (ppt)", "TDS (mg/L)"]
    st.dataframe(status_view, use_container_width=True, hide_index=True)
    st.markdown("**Quick observations from the current input data**")
    highest_ec = data.loc[data["conductivity_uScm"].idxmax(), "sample"]
    highest_tds = data.loc[data["tds_mgL"].idxmax(), "sample"]
    st.write(f"- Highest conductivity in the current dataset: **{highest_ec}**.")
    st.write(f"- Highest TDS in the current dataset: **{highest_tds}**.")
    st.write("- The salt-solution pH values are marked as assumed and should be replaced with measured pH when available.")

with tab_water:
    st.subheader("Water-quality comparison")
    parameter = st.selectbox(
        "Choose a parameter",
        ["pH", "conductivity_uScm", "salinity_ppt", "tds_mgL"],
        format_func=lambda x: {
            "pH": "pH",
            "conductivity_uScm": "Conductivity (µS/cm)",
            "salinity_ppt": "Salinity (ppt)",
            "tds_mgL": "TDS (mg/L)"
        }[x]
    )
    chart_data = data.set_index("sample")[[parameter]]
    st.bar_chart(chart_data, use_container_width=True)
    display = data[["sample", parameter, "pH_status"]].copy()
    display.columns = ["Sample", parameter, "pH status"]
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("**Comparison against distilled-water reference**")
    compare_col = st.selectbox(
        "Parameter for reference comparison",
        ["conductivity_uScm", "salinity_ppt", "tds_mgL"],
        format_func=lambda x: {
            "conductivity_uScm": "Conductivity change (%)",
            "salinity_ppt": "Salinity change (%)",
            "tds_mgL": "TDS change (%)"
        }[x]
    )
    pct_col = compare_col + "_change_pct_vs_ref"
    ref_chart = data.set_index("sample")[[pct_col]]
    st.bar_chart(ref_chart, use_container_width=True)
    st.caption("Percentage change is calculated relative to the distilled-water reading.")


ratio_view = data[["sample", "TDS_to_EC_ratio"]].copy()
    ratio_view.columns = ["Sample", "TDS / EC ratio"]
    st.markdown("**TDS-to-conductivity ratio check**")
    st.dataframe(ratio_view.round(3), use_container_width=True, hide_index=True)
    if ((data["TDS_to_EC_ratio"] < 0.4) | (data["TDS_to_EC_ratio"] > 0.8)).any():
        st.warning("At least one TDS/EC ratio is outside the 0.4–0.8 screening range used in the supplied script. Recheck units and meter readings.")
    else:
        st.success("All TDS/EC ratios are within the 0.4–0.8 screening range used in the supplied script.")
    st.caption("Confirm instrument units: conductivity in µS/cm, salinity in ppt, and TDS in mg/L.")

with tab_ions:
    st.subheader("Calculated ion chemistry")
    st.write("Ion concentrations and ionic strength are calculated from salt formula and nominal concentration; they are not direct ion measurements.")
    ion_df = build_ion_table(data)
    if ion_df.empty:
        st.info("No recognised salt-solution rows are available for ion calculations.")
    else:
        st.dataframe(ion_df.round(4), use_container_width=True, hide_index=True)
        ion_choice = st.selectbox("Ion quantity to compare", ["Chloride (mg/L)", "Sulfate (mg/L)", "Ionic strength (mol/L)"])
        st.bar_chart(ion_df.set_index("Sample")[[ion_choice]], use_container_width=True)
    st.caption("Chloride and sulfate chemistry provide context for water chemistry; this table does not calculate corrosion rate.")

with tab_sensitivity:
    st.subheader("Sensitivity indicators")
    st.markdown("**A. pH sensitivity (thermodynamic indicator)**")
    pH_values = np.linspace(4, 9, 51)
    temp_k = REFERENCE_TEMPERATURE_C + 273.15
    slope = 2.303 * GAS_CONSTANT * temp_k / FARADAY
    e_h2 = -slope * pH_values
    e_fe = E0_FE + (slope / 2) * np.log10(FE2_ACTIVITY)
    pH_sweep = pd.DataFrame({
        "pH": pH_values,
        "Hydrogen ion concentration (mol/L)": 10.0 ** (-pH_values),
        "Driving force indicator (V)": e_h2 - e_fe
    })
    st.line_chart(pH_sweep.set_index("pH")[["Driving force indicator (V)"]], use_container_width=True)
    st.line_chart(pH_sweep.set_index("pH")[["Hydrogen ion concentration (mol/L)"]], use_container_width=True)
    st.caption("These are thermodynamic indicators based on the assumptions in the supplied model, not corrosion-rate predictions.")

st.markdown("**B. Conductivity and solution resistivity**")
    k_values = np.logspace(np.log10(10), np.log10(5000), 60)
    conductivity_sweep = pd.DataFrame({
        "Conductivity (µS/cm)": k_values,
        "Resistivity (ohm·cm)": 1e6 / k_values
    })
    st.line_chart(conductivity_sweep.set_index("Conductivity (µS/cm)"), use_container_width=True)
    sample_resistivity = data[["sample", "conductivity_uScm", "resistivity_ohm_cm"]].copy()
    sample_resistivity.columns = ["Sample", "Conductivity (µS/cm)", "Resistivity (ohm·cm)"]
    st.dataframe(sample_resistivity.round(3), use_container_width=True, hide_index=True)

with tab_corrosion:
    st.subheader("Illustrative corrosion scenarios")
    st.warning(
        "These rates are calculated from an assumed corrosion-current density using "
        "Faraday's law. They are not measured rates and are not inferred from the salt "
        "composition. The same current-density scenarios are applied independently of sample."
    )
    scenario_table = pd.DataFrame({
        "Scenario": list(ICORR_SCENARIOS.keys()),
        "Assumed Icorr (µA/cm²)": list(ICORR_SCENARIOS.values())
    })

    # Add the user's editable Icorr value when it is not one of the four presets.
    preset_values = list(ICORR_SCENARIOS.values())
    if not any(np.isclose(icorr, value) for value in preset_values):
        scenario_table.loc[len(scenario_table)] = ["User-defined", float(icorr)]

    scenario_table["Theoretical rate (mm/year)"] = corrosion_rate_mm_year(
        scenario_table["Assumed Icorr (µA/cm²)"].to_numpy()
    )
    scenario_table["Rate (mpy)"] = scenario_table["Theoretical rate (mm/year)"] / 0.0254
    scenario_table["Screening band (as coded)"] = scenario_table["Theoretical rate (mm/year)"].apply(severity_band)
    st.dataframe(scenario_table.round(4), use_container_width=True, hide_index=True)

    selected_rate = float(corrosion_rate_mm_year(icorr))
    st.info(
        f"Current selected input: Icorr = {icorr:.2f} µA/cm². "
        "This is an assumed scenario input, not an experimentally measured corrosion current density."
    )
    selected_factor = float(arrhenius_factor(temperature, activation_energy_kj * 1000.0))
    temperature_adjusted_rate = selected_rate * selected_factor
    selected_loss = temperature_adjusted_rate * exposure_years

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected assumed Icorr", f"{icorr:.1f} µA/cm²")
    c2.metric("Scenario rate at selected temperature", f"{temperature_adjusted_rate:.4f} mm/year")
    c3.metric(f"Theoretical loss over {exposure_years} years", f"{selected_loss:.4f} mm")

    st.markdown("**Temperature sensitivity for the selected scenario**")
    temps = np.arange(20, 61, 5)
    temp_results = pd.DataFrame({
        "Temperature (°C)": temps,
        "Relative Arrhenius factor": arrhenius_factor(temps, activation_energy_kj * 1000.0)
    })
    temp_results["Scenario rate (mm/year)"] = selected_rate * temp_results["Relative Arrhenius factor"]
    st.line_chart(temp_results.set_index("Temperature (°C)")[["Scenario rate (mm/year)"]], use_container_width=True)
    st.dataframe(temp_results.round(4), use_container_width=True, hide_index=True)

    st.markdown("**Theoretical thickness loss over exposure time**")
    years = np.array([1, 2, 5, 10])
    loss_results = pd.DataFrame({
        "Exposure time (years)": years,
        "Thickness loss (mm)": temperature_adjusted_rate * years
    })
    st.line_chart(loss_results.set_index("Exposure time (years)"), use_container_width=True)
    st.caption(
        f"Arrhenius scaling uses an assumed apparent activation energy of {activation_energy_kj} kJ/mol "
        "and is illustrative. In real open systems, dissolved oxygen solubility can fall as temperature rises."
    )
    st.caption(
        "The severity-band labels reproduce the thresholds coded in the supplied script; "
        "verify the applicable standard and project basis before describing them as a formal NACE classification."
    )

with tab_downloads:
    st.subheader("Download tables for your report")
    downloads = {
        "Input water-quality data": data,
        "Reference comparison": data[[
            "sample", "pH", "pH_status", "conductivity_uScm",
            "conductivity_uScm_change_pct_vs_ref", "conductivity_uScm_net_of_water",
            "tds_mgL", "tds_mgL_change_pct_vs_ref", "resistivity_ohm_cm"
        ]],
        "Calculated ion chemistry": build_ion_table(data),
        "Corrosion scenarios": scenario_table
    }
    for label, frame in downloads.items():
        st.markdown(f"**{label}**")
        if frame.empty:
            st.caption("No rows available.")
        else:
            safe_name = label.lower().replace(" ", "_")
            st.download_button(
                label=f"Download {label} (CSV)",
                data=make_csv_download(frame),
                file_name=f"{safe_name}.csv",
                mime="text/csv",
                key=f"download_{safe_name}"
            )


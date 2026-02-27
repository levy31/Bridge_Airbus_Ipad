import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import os
from datetime import datetime
import numpy as np
import requests

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Airbus Audit Master - Final", initial_sidebar_state="auto")
st.title("🎯 Airbus Audit : The Complete Strategic Cockpit")

# Lecture des clés API depuis les secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

# Diagnostic des clés (affiché dans la sidebar)
st.sidebar.markdown("---")
st.sidebar.caption("🔑 **API Keys status**")
st.sidebar.write(f"Gemini: {'✅' if GEMINI_API_KEY else '❌'} {GEMINI_API_KEY[:4] if GEMINI_API_KEY else ''}...")
st.sidebar.write(f"Groq: {'✅' if GROQ_API_KEY else '❌'} {GROQ_API_KEY[:4] if GROQ_API_KEY else ''}...")

if not GEMINI_API_KEY and not GROQ_API_KEY:
    st.error("No API keys found. Please configure at least one provider in secrets.")
    st.stop()

# Configuration Gemini si la clé est présente
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- 2. FONCTION D'APPEL GROQ ---
def call_groq_api(prompt):
    """Appelle l'API Groq avec la clé stockée dans les secrets."""
    if not GROQ_API_KEY:
        st.error("Groq API key is missing.")
        return None
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.3-70b-versatile",  # modèle gratuit et rapide
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.7
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"Groq error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Groq exception: {e}")
        return None

# --- 3. INITIALISATION DES DATES EN SESSION STATE ---
if "dates" not in st.session_state:
    st.session_state.dates = {
        "Devis_Alpha": datetime(2026, 6, 10).date(),
        "Devis_Beta": datetime(2024, 3, 20).date(),
        "Devis_Gamma": datetime(2022, 1, 15).date()
    }

# --- 4. SIDEBAR : DATES ---
st.sidebar.header("📅 Issue Dates (Free Chronology)")
files_list = ["Devis_Alpha", "Devis_Beta", "Devis_Gamma"]
for f in files_list:
    new_date = st.sidebar.date_input(
        f"Date for {f}",
        value=st.session_state.dates[f],
        key=f"date_input_{f}"
    )
    if new_date != st.session_state.dates[f]:
        st.session_state.dates[f] = new_date

dates = st.session_state.dates

# --- 5. CHOIX DU FOURNISSEUR IA ---
st.sidebar.header("🤖 IA Provider")
provider_options = []
if GEMINI_API_KEY:
    provider_options.append("Gemini")
if GROQ_API_KEY:
    provider_options.append("Groq")

if not provider_options:
    st.sidebar.error("No IA provider available. Please check your secrets.")
    st.stop()

ia_provider = st.sidebar.radio("Choose IA provider", provider_options, index=0)

# --- 6. MAPPING MANAGEMENT ---
MAPPING_ACTIF = "mapping_actif.csv"
st.sidebar.header("🗂️ Mapping Management")
uploaded_mapping = st.sidebar.file_uploader("Load a mapping file (CSV)", type=["csv"])
if uploaded_mapping is not None:
    with open(MAPPING_ACTIF, "wb") as f:
        f.write(uploaded_mapping.getbuffer())
    st.sidebar.success(f"File {uploaded_mapping.name} loaded as active mapping.")
    st.rerun()

if st.sidebar.button("🔄 Reset mapping (start from scratch)"):
    if os.path.exists(MAPPING_ACTIF):
        os.remove(MAPPING_ACTIF)
    st.sidebar.success("Mapping reset.")
    st.rerun()

# --- 7. LOAD QUOTES ---
raw_dfs = {}
all_wps_list = []

for f_name in files_list:
    path = f"{f_name}.xlsx"
    if os.path.exists(path):
        df = pd.read_excel(path, engine='openpyxl')
        if 'Work_Package' not in df.columns or 'Cout_Total' not in df.columns:
            st.error(f"File {f_name}.xlsx must contain columns 'Work_Package' and 'Cout_Total'.")
            st.stop()
        raw_dfs[f_name] = df
        for wp in df['Work_Package'].unique():
            all_wps_list.append({"System": f_name, "Original WP": wp})
    else:
        st.warning(f"File {f_name}.xlsx not found. Please place it in the directory.")

if not raw_dfs:
    st.warning("Please place Devis_Alpha.xlsx, Devis_Beta.xlsx and Devis_Gamma.xlsx in the directory.")
    st.stop()

# --- 8. LOAD OR CREATE ACTIVE MAPPING ---
df_init = pd.DataFrame(all_wps_list)

if os.path.exists(MAPPING_ACTIF):
    df_saved = pd.read_csv(MAPPING_ACTIF)
    required_cols = ["System", "Original WP", "Common Name", "Complexity", "Comments"]
    if not all(col in df_saved.columns for col in required_cols):
        st.error(f"The mapping file must contain columns: {required_cols}")
        st.stop()
    df_mapping = pd.merge(df_init, df_saved, on=['System', 'Original WP'], how='left')
else:
    df_mapping = df_init.copy()

for col in ["Common Name", "Complexity", "Comments"]:
    if col not in df_mapping.columns:
        df_mapping[col] = 1.0 if col == "Complexity" else ""
    else:
        if col == "Complexity":
            df_mapping[col] = df_mapping[col].fillna(1.0)
        else:
            df_mapping[col] = df_mapping[col].fillna("")

st.subheader("1. Technical Normalization Matrix")
edited_mapping = st.data_editor(df_mapping, hide_index=True, width='stretch')

if not edited_mapping.equals(df_mapping):
    edited_mapping.to_csv(MAPPING_ACTIF, index=False)
    st.success(f"Mapping saved to {MAPPING_ACTIF}")
    st.rerun()

# --- 9. CALCULATIONS AND CHRONOLOGY ---
map_dict = {(r["System"], r["Original WP"]): r["Common Name"] for _, r in edited_mapping.iterrows()}
comp_dict = {(r["System"], r["Original WP"]): r["Complexity"] for _, r in edited_mapping.iterrows()}

all_data = []
for name, df in raw_dfs.items():
    df_c = df.copy()
    df_c['System'] = name
    df_c['Date'] = pd.to_datetime(dates[name])
    df_c['WP_ST'] = df_c['Work_Package'].apply(lambda x: map_dict.get((name, x), x))
    df_c['Ratio'] = df_c['Work_Package'].apply(lambda x: comp_dict.get((name, x), 1.0))
    df_c['Normalized_Cost'] = df_c['Cout_Total'] / df_c['Ratio']
    all_data.append(df_c)

df_global = pd.concat(all_data).sort_values(by="Date")
pivot_raw = df_global.pivot_table(index='WP_ST', columns='System', values='Cout_Total', aggfunc='sum').fillna(0)
pivot_norm = df_global.pivot_table(index='WP_ST', columns='System', values='Normalized_Cost', aggfunc='sum').fillna(0)
chrono_order = df_global[['System', 'Date']].drop_duplicates().sort_values('Date')['System'].tolist()

def draw_bridge(pivot_df, base_sys, target_sys):
    if base_sys not in pivot_df.columns or target_sys not in pivot_df.columns:
        return go.Figure()
    v_base = pivot_df[base_sys].sum()
    labels = [base_sys]
    values = [v_base]
    measures = ["absolute"]
    for wp in pivot_df.index:
        diff = pivot_df.loc[wp, target_sys] - pivot_df.loc[wp, base_sys]
        if abs(diff) > 0.1:
            labels.append(wp)
            values.append(diff)
            measures.append("relative")
    labels.append(f"Total {target_sys}")
    values.append(pivot_df[target_sys].sum())
    measures.append("total")
    return go.Figure(go.Waterfall(measure=measures, x=labels, y=values)).update_layout(title=f"Bridge: {base_sys} → {target_sys}")

# --- 10. THE 6 SNAPSHOTS ---
st.divider()
st.subheader("2. Performance Snapshots (RAW DATA)")
c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(px.bar(df_global, x="WP_ST", y="Cout_Total", color="System", barmode="group", title="Raw Volume"),
                    use_container_width=True, key="raw_volume")
with c2:
    st.plotly_chart(px.line(df_global, x="Date", y="Cout_Total", color="WP_ST", markers=True, title="Raw Timeline"),
                    use_container_width=True, key="raw_timeline")
with c3:
    col_base, col_target = st.columns(2)
    with col_base:
        base_r = st.selectbox("Base", files_list, index=0, key="base_r")
    with col_target:
        target_r = st.selectbox("Target", [s for s in files_list if s != base_r], index=0, key="target_r")
    if base_r != target_r:
        st.plotly_chart(draw_bridge(pivot_raw, base_r, target_r), use_container_width=True, key=f"raw_bridge_{base_r}_{target_r}")
    else:
        st.warning("Please choose two different systems.")

st.subheader("3. Performance Snapshots (NORMALIZED DATA)")
c4, c5, c6 = st.columns(3)
with c4:
    st.plotly_chart(px.bar(df_global, x="WP_ST", y="Normalized_Cost", color="System", barmode="group", title="Normalized Volume"),
                    use_container_width=True, key="norm_volume")
with c5:
    st.plotly_chart(px.line(df_global, x="Date", y="Normalized_Cost", color="WP_ST", markers=True, title="Normalized Timeline"),
                    use_container_width=True, key="norm_timeline")
with c6:
    col_base_n, col_target_n = st.columns(2)
    with col_base_n:
        base_n = st.selectbox("Base", files_list, index=0, key="base_n")
    with col_target_n:
        target_n = st.selectbox("Target", [s for s in files_list if s != base_n], index=0, key="target_n")
    if base_n != target_n:
        st.plotly_chart(draw_bridge(pivot_norm, base_n, target_n), use_container_width=True, key=f"norm_bridge_{base_n}_{target_n}")
    else:
        st.warning("Please choose two different systems.")

# ========== GLOBAL DRIFT ==========
st.divider()
st.subheader("4. Global complexity drift (total normalized cost)")

total_norm_by_system = df_global.groupby('System')['Normalized_Cost'].sum().reset_index()
date_map = df_global[['System', 'Date']].drop_duplicates()
total_norm_by_system = total_norm_by_system.merge(date_map, on='System').sort_values('Date')

if len(total_norm_by_system) >= 2:
    x_vals = (total_norm_by_system['Date'] - total_norm_by_system['Date'].min()).dt.days.values
    y_vals = total_norm_by_system['Normalized_Cost'].values
    coeffs = np.polyfit(x_vals, y_vals, 1)
    slope_per_day = coeffs[0]
    
    slope_per_month = slope_per_day * 30.44
    slope_per_year = slope_per_day * 365

    x_pred = np.array([0, x_vals.max()])
    y_pred = np.polyval(coeffs, x_pred)
    trend_dates = [total_norm_by_system['Date'].min(), total_norm_by_system['Date'].max()]

    fig_global_drift = go.Figure()
    fig_global_drift.add_trace(go.Scatter(x=total_norm_by_system['Date'], y=total_norm_by_system['Normalized_Cost'],
                                           mode='markers+lines', marker=dict(size=12), line=dict(width=2),
                                           name='Total normalized cost'))
    fig_global_drift.add_trace(go.Scatter(x=trend_dates, y=y_pred, mode='lines', line=dict(dash='dash', color='red'),
                                           name='Trend line'))
    fig_global_drift.update_layout(title="Evolution of total normalized cost", xaxis_title="Date", yaxis_title="€", hovermode='x unified')
    st.plotly_chart(fig_global_drift, use_container_width=True, key="global_drift")

    if slope_per_day != 0:
        first_cost = total_norm_by_system['Normalized_Cost'].iloc[0]
        annualized_pct = (slope_per_day * 365 / first_cost) * 100
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Daily drift", f"{slope_per_day:+.2f} €/day")
        with col_m2:
            st.metric("Monthly drift", f"{slope_per_month:+.2f} €/month")
        with col_m3:
            st.metric("Annualized drift", f"{annualized_pct:+.1f} % / year")
else:
    st.warning("Not enough points to calculate a global trend.")
    slope_per_day, slope_per_month, first_cost, annualized_pct = 0, 0, 0, 0

# ========== INDIVIDUAL WP DRIFT ==========
st.subheader("5. Individual Work Package drift")

wp_drift_dict = {}
for wp in df_global['WP_ST'].unique():
    wp_data = df_global[df_global['WP_ST'] == wp].sort_values('Date')
    if len(wp_data) >= 2:
        x_wp = (wp_data['Date'] - wp_data['Date'].min()).dt.days.values
        y_wp = wp_data['Normalized_Cost'].values
        coeffs_wp = np.polyfit(x_wp, y_wp, 1)
        slope_per_day_wp = coeffs_wp[0]
        first_cost_wp = y_wp[0]
        slope_per_month_wp = slope_per_day_wp * 30.44
        annual_pct_wp = (slope_per_day_wp * 365 / first_cost_wp) * 100 if first_cost_wp != 0 else np.nan
        wp_drift_dict[wp] = {
            'pente_jour': slope_per_day_wp,
            'pente_mois': slope_per_month_wp,
            'annual_pct': annual_pct_wp,
            'data': wp_data[['Date', 'Normalized_Cost', 'System']]
        }
    else:
        wp_drift_dict[wp] = {
            'pente_jour': None,
            'pente_mois': None,
            'annual_pct': None,
            'data': wp_data[['Date', 'Normalized_Cost', 'System']]
        }

wp_drift_list = []
for wp, vals in wp_drift_dict.items():
    if vals['pente_jour'] is not None:
        wp_drift_list.append({
            'Work Package': wp,
            'Slope (€/day)': round(vals['pente_jour'], 2),
            'Slope (€/month)': round(vals['pente_mois'], 2),
            'Annualized drift (%)': round(vals['annual_pct'], 1) if not np.isnan(vals['annual_pct']) else 'N/A'
        })
df_wp_drift = pd.DataFrame(wp_drift_list)
if not df_wp_drift.empty:
    st.dataframe(df_wp_drift, use_container_width=True, hide_index=True)
else:
    st.info("Not enough data to calculate slopes (need at least two versions per WP).")

if wp_drift_dict:
    all_wps = sorted(wp_drift_dict.keys())
    selected_wp = st.selectbox("Choose a Work Package:", all_wps, key="wp_selector")
    wp_info = wp_drift_dict[selected_wp]
    wp_data = wp_info['data']

    fig_wp = go.Figure()
    fig_wp.add_trace(go.Scatter(x=wp_data['Date'], y=wp_data['Normalized_Cost'],
                                 mode='markers+lines' if len(wp_data) > 1 else 'markers',
                                 marker=dict(size=10), line=dict(width=2) if len(wp_data) > 1 else None,
                                 name='Normalized cost', text=wp_data['System'],
                                 hovertemplate='<b>%{text}</b><br>%{x}<br>%{y:.2f} €<extra></extra>'))
    if wp_info['pente_jour'] is not None:
        x_wp_vals = (wp_data['Date'] - wp_data['Date'].min()).dt.days.values
        y_wp_vals = wp_data['Normalized_Cost'].values
        coeffs_wp_indiv = np.polyfit(x_wp_vals, y_wp_vals, 1)
        slope_wp_indiv = coeffs_wp_indiv[0]
        x_pred_wp = np.array([0, x_wp_vals.max()])
        y_pred_wp = np.polyval(coeffs_wp_indiv, x_pred_wp)
        trend_dates_wp = [wp_data['Date'].min(), wp_data['Date'].max()]
        fig_wp.add_trace(go.Scatter(x=trend_dates_wp, y=y_pred_wp, mode='lines', line=dict(dash='dash', color='red'),
                                     name='Trend line'))
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.metric("Daily drift", f"{wp_info['pente_jour']:+.2f} €/day")
        with col_w2:
            st.metric("Monthly drift", f"{wp_info['pente_mois']:+.2f} €/month")
        with col_w3:
            st.metric("Annualized drift", f"{wp_info['annual_pct']:+.1f} % / year")
    else:
        st.caption("⚠️ This Work Package appears in only one version – no trend can be calculated.")
    fig_wp.update_layout(title=f"Drift of WP: {selected_wp}", xaxis_title="Date", yaxis_title="Normalized cost (€)")
    st.plotly_chart(fig_wp, use_container_width=True, key=f"wp_drift_{selected_wp}")

# ========== ORACLE VALIDATION MODE ==========
st.sidebar.header("🧪 Validation Mode")
oracle_file = st.sidebar.file_uploader("Load oracle file", type=["xlsx", "csv"])

if oracle_file:
    # Read Dates sheet
    try:
        xl = pd.ExcelFile(oracle_file, engine='openpyxl')
        if 'Dates' in xl.sheet_names:
            df_dates_oracle = pd.read_excel(oracle_file, sheet_name='Dates', engine='openpyxl')
            new_dates = {}
            for _, row in df_dates_oracle.iterrows():
                system = row.get('Système') or row.get('System')
                date_val = pd.to_datetime(row['Date']).date()
                if system and system in files_list:
                    new_dates[system] = date_val

            if new_dates:
                if st.session_state.dates != new_dates:
                    st.sidebar.warning("Oracle dates differ from current dates.")
                    if st.sidebar.button("📅 Apply oracle dates"):
                        for sys, date in new_dates.items():
                            st.session_state.dates[sys] = date
                        st.sidebar.success("Dates updated. Reloading...")
                        st.rerun()
                else:
                    st.sidebar.success("✅ Dates already match the oracle.")
        else:
            st.sidebar.info("Oracle file does not contain a 'Dates' sheet. Manually entered dates will be used.")
    except Exception as e:
        st.sidebar.error(f"Error reading dates: {e}")

    # --- Drift comparison ---
    st.divider()
    st.subheader("🔬 Comparison with expected scenario (Oracle)")
    try:
        df_oracle_global = pd.read_excel(oracle_file, sheet_name="Attendus_Global", engine='openpyxl')
        df_oracle_wp = pd.read_excel(oracle_file, sheet_name="Attendus_WP", engine='openpyxl')
        expected_analysis = pd.read_excel(oracle_file, sheet_name="Analyse_IA_attendue", engine='openpyxl').iloc[0,0]

        # Global comparison
        st.markdown("#### Global comparison")
        if len(total_norm_by_system) >= 2:
            computed_global = ((total_norm_by_system['Normalized_Cost'].iloc[-1] / total_norm_by_system['Normalized_Cost'].iloc[0]) - 1) * 100
        else:
            computed_global = 0
        expected_global_str = df_oracle_global[df_oracle_global['Metrique'] == 'Dérive_globale_normalisee_%']['Valeur'].values[0]
        expected_global = float(expected_global_str.replace('%', '').replace('+', ''))
        diff_global = computed_global - expected_global

        col1, col2, col3 = st.columns(3)
        col1.metric("Computed", f"{computed_global:.1f}%")
        col2.metric("Expected", expected_global_str)
        col3.metric("Difference", f"{diff_global:+.1f} pts", delta_color="off" if abs(diff_global) < 5 else "inverse")

        # Per WP comparison
        st.markdown("#### Per Work Package comparison")
        comparisons = []
        for _, row in df_oracle_wp.iterrows():
            wp_oracle = row['WP']
            expected_drift = row['Dérive_normalisee_%_attendue']
            expected_alert = row['Alerte']
            comment = row['Commentaire_attendu']

            found_wp = None
            for wp_calc in df_wp_drift['Work Package']:
                if wp_calc.strip().lower() == wp_oracle.strip().lower():
                    found_wp = wp_calc
                    break

            if found_wp is not None:
                computed_drift = float(df_wp_drift[df_wp_drift['Work Package'] == found_wp]['Annualized drift (%)'].iloc[0])
                if expected_drift != 'N/A':
                    expected_val = float(expected_drift)
                    diff = computed_drift - expected_val
                    status = "✅" if abs(diff) < 5 else "⚠️" if abs(diff) < 10 else "❌"
                else:
                    diff = None
                    status = "ℹ️ New WP"
            else:
                computed_drift = None
                diff = None
                status = "❌ Not found"

            comparisons.append({
                'WP (oracle)': wp_oracle,
                'Expected': str(expected_drift),
                'Computed': f"{computed_drift:.1f}%" if computed_drift is not None else "N/A",
                'Difference': f"{diff:+.1f}" if diff is not None else "N/A",
                'Expected alert': expected_alert,
                'Status': status,
                'Comment': comment
            })
        st.dataframe(pd.DataFrame(comparisons), hide_index=True, use_container_width=True)

        # IA analysis comparison
        st.markdown("#### IA analysis comparison")
        col_ia1, col_ia2 = st.columns(2)
        col_ia1.markdown("**📝 Expected analysis:**")
        col_ia1.info(expected_analysis)
        col_ia2.markdown("**🤖 Produced analysis:**")
        if 'ai_audit' in st.session_state:
            col_ia2.info(st.session_state.ai_audit)
        else:
            col_ia2.warning("Please run the full strategic audit first (section 7).")
    except Exception as e:
        st.error(f"Error reading oracle sheets: {e}")
        st.info("Make sure the Excel file contains sheets 'Attendus_Global', 'Attendus_WP' and 'Analyse_IA_attendue' with the correct format.")

# ========== IA DRIFT ANALYSIS ==========
st.divider()
st.subheader("6. IA analysis of observed drifts")
if st.button("🤖 Analyze drifts with IA", key="btn_drift_analysis"):
    # Construction du prompt
    global_trend = f"global slope = {slope_per_day:.2f} €/day, i.e. {annualized_pct:.1f}% per year (initial total cost = {first_cost:.2f} €)"
    wp_trends = []
    for wp, vals in wp_drift_dict.items():
        if vals['pente_jour'] is not None:
            wp_trends.append(f"- {wp}: slope = {vals['pente_jour']:.2f} €/day, annualized = {vals['annual_pct']:.1f}% per year")
        else:
            wp_trends.append(f"- {wp}: appears only in {len(vals['data'])} version(s) – no trend calculable")
    wp_trends_str = "\n".join(wp_trends)

    raw_points = df_global[['System', 'Date', 'WP_ST', 'Normalized_Cost']].to_string()

    prompt = f"""
    You are an expert in aerospace project analysis. We have calculated normalized cost drifts (after complexity adjustment) for three versions of a project (Alpha, Beta, Gamma) with their actual dates.
    
    Here are the drift indicators:
    
    **Global trend** (sum of all WPs):
    {global_trend}
    
    **Per Work Package trends**:
    {wp_trends_str}
    
    **Detailed data** (optional, for reference):
    {raw_points}
    
    Questions to answer in your analysis:
    1. Which Work Packages show the strongest upward drift? Which are stable or decreasing?
    2. Is the global drift concerning? How does it compare to normal inflation (say 2-3% per year)?
    3. Are there possible correlations between dates and the magnitude of drifts (e.g., an acceleration after a certain date)?
    4. What strategic recommendations would you make given these trends?
    
    Answer concisely but precisely, based on the figures provided.
    """
    
    with st.spinner(f"IA ({ia_provider}) analyzing drifts..."):
        if ia_provider == "Gemini":
            if not GEMINI_API_KEY:
                st.error("Gemini API key not configured.")
            else:
                try:
                    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    model_name = next((m for m in available_models if "flash" in m), available_models[0])
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    st.session_state.ai_drift_analysis = response.text
                except Exception as e:
                    st.error(f"Gemini error: {e}")
        elif ia_provider == "Groq":
            if not GROQ_API_KEY:
                st.error("Groq API key not configured.")
            else:
                response_text = call_groq_api(prompt)
                if response_text:
                    st.session_state.ai_drift_analysis = response_text

if 'ai_drift_analysis' in st.session_state:
    st.markdown(st.session_state.ai_drift_analysis)

# --- 7. COMPLETE STRATEGIC AUDIT ---
st.divider()
st.subheader("7. Complete Strategic Audit")
if st.button("🧠 Execute Full Strategic Audit (including data and drifts)", key="btn_full_audit"):
    summary = f"""
    ACTUAL PROJECT DATA (real costs, complexity, dates):
    - Chronological order: {' -> '.join(chrono_order)}
    - Dates: { {k: v.strftime('%Y-%m-%d') for k,v in dates.items()} }
    - Normalized costs per WP (Alpha, Beta, Gamma):
    {pivot_norm.to_string()}
    - Complexity factors per WP:
    {edited_mapping[['Common Name', 'Complexity']].to_string()}
    - Comments provided:
    {edited_mapping[['Common Name', 'Comments']].to_string()}
    
    COMPUTED DRIFT INDICATORS:
    - Global annualized drift: {annualized_pct:.1f}% per year (based on total normalized cost)
    - Per WP drift table:
    {df_wp_drift.to_string()}
    """
    prompt = f"""
    Act as a Senior Airbus Project Controller. Analyze the cost drift based on the real data below.
    {summary}
    
    Provide a concise audit covering:
    1. Key Work Packages with significant cost variations between versions.
    2. Impact of the chronological sequence on budget stability.
    3. Whether the provided comments justify the observed drifts.
    4. Strategic recommendations and negotiation points.
    
    Do NOT invent any numerical scores. Base your analysis strictly on the figures above.
    """
    
    with st.spinner(f"Senior Airbus audit ({ia_provider}) in progress..."):
        if ia_provider == "Gemini":
            if not GEMINI_API_KEY:
                st.error("Gemini API key not configured.")
            else:
                try:
                    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    model_name = next((m for m in available_models if "flash" in m), available_models[0])
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    st.session_state.ai_audit = response.text
                except Exception as e:
                    st.error(f"Gemini error: {e}")
        elif ia_provider == "Groq":
            if not GROQ_API_KEY:
                st.error("Groq API key not configured.")
            else:
                response_text = call_groq_api(prompt)
                if response_text:
                    st.session_state.ai_audit = response_text

if 'ai_audit' in st.session_state:
    st.info(st.session_state.ai_audit)

# --- 8. AUDIT GUIDE ---
st.divider()
st.subheader("📚 Audit Guide")
st.markdown("""
* **The 6 graphs**: Comprehensive analysis of volumes, timelines and gaps (raw and normalized).
* **Global drift**: Graph of total normalized cost evolution with trend line – the slope visualizes the average temporal drift.
* **Per WP drift**: Table of individual slopes (€/day, €/month, and annualized %) for each Work Package, and individual graph with selector (all WPs listed).
* **IA analysis**: Two buttons – one for focused drift analysis, one for a complete strategic audit including data and comments. Choose your preferred IA provider in the sidebar.
* **Validation mode**: Load an oracle file to automatically compare your results with a reference scenario. If the oracle contains a 'Dates' sheet, you can apply those dates with one click to ensure consistency.
""")
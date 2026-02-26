import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import os
import io
import re
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Airbus Expert Audit")
st.title("🎯 Airbus Expert Audit - Strategic Analysis")

if "GEMINI_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        st.success("✅ Connexion API établie.")
    except Exception as e:
        st.error(f"❌ Erreur configuration : {e}")
        st.stop()
else:
    st.error("❌ Clé absente des secrets.")
    st.stop()

# --- 2. SIDEBAR : DATES ---
st.sidebar.header("📅 Internal Issue Dates")
dates = {}
files_list = ["Devis_Alpha", "Devis_Beta", "Devis_Gamma"]
for f in files_list:
    dates[f] = st.sidebar.date_input(f"Date for {f}", datetime.now())

# --- 3. CHARGEMENT DES DONNÉES ---
MAPPING_FILE = "mapping_audit.csv"
raw_dfs, all_wps_list = {}, []

for f_name in files_list:
    path = f"{f_name}.xlsx"
    if os.path.exists(path):
        df = pd.read_excel(path)
        raw_dfs[f_name] = df
        for wp in df['Work_Package'].unique(): 
            all_wps_list.append({"System": f_name, "Original WP": wp})

if not all_wps_list:
    st.warning("⚠️ En attente des fichiers Excel (Alpha, Beta, Gamma).")
    st.stop()

# --- 4. MATRICE DE NORMALISATION ---
df_init = pd.DataFrame(all_wps_list)
if os.path.exists(MAPPING_FILE):
    df_saved = pd.read_csv(MAPPING_FILE)
    df_mapping = pd.merge(df_init, df_saved, on=['System', 'Original WP'], how='left')
else:
    df_mapping = df_init

for col in ["Common Name", "Complexity", "Comments"]:
    if col not in df_mapping.columns: 
        df_mapping[col] = 1.0 if col == "Complexity" else ""

st.subheader("1. Technical Normalization Matrix")
edited_mapping = st.data_editor(df_mapping, hide_index=True, width='stretch')
if not edited_mapping.equals(df_mapping): 
    edited_mapping.to_csv(MAPPING_FILE, index=False)
    st.rerun()

# --- 5. CALCULS ---
map_dict = {(r["System"], r["Original WP"]): r["Common Name"] for _, r in edited_mapping.iterrows()}
comp_dict = {(r["System"], r["Original WP"]): r["Complexity"] for _, r in edited_mapping.iterrows()}

all_data = []
for name, df in raw_dfs.items():
    df_c = df.copy()
    df_c['System'] = name
    df_c['WP_ST'] = df_c['Work_Package'].apply(lambda x: map_dict.get((name, x), x))
    df_c['Ratio'] = df_c['Work_Package'].apply(lambda x: comp_dict.get((name, x), 1.0))
    df_c['Normalized_Cost'] = df_c['Cout_Total'] / df_c['Ratio']
    all_data.append(df_c)

df_global = pd.concat(all_data)
pivot_data = df_global.pivot_table(index='WP_ST', columns='System', values='Normalized_Cost', aggfunc='sum').fillna(0)

# --- 6. VISUALISATION ---
st.divider()
st.subheader("2. Performance Snapshots")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.bar(df_global, x="WP_ST", y="Normalized_Cost", color="System", barmode="group"), use_container_width=True)
with c2:
    target = st.selectbox("Compare Alpha vs:", [s for s in files_list if s != "Devis_Alpha"])
    v_base = pivot_data["Devis_Alpha"].sum()
    labels, values, measures = ["Alpha"], [v_base], ["absolute"]
    for wp in pivot_data.index:
        diff = pivot_data.loc[wp, target] - pivot_data.loc[wp, "Devis_Alpha"]
        if abs(diff) > 0.1: 
            labels.append(wp); values.append(diff); measures.append("relative")
    labels.append(f"Total {target}"); values.append(pivot_data[target].sum()); measures.append("total")
    fig_bridge = go.Figure(go.Waterfall(measure=measures, x=labels, y=values))
    fig_bridge.update_layout(title="Cost Bridge Analysis", showlegend=False)
    st.plotly_chart(fig_bridge, use_container_width=True)

# --- 7. IA (AUTO-DÉTECTION DU MODÈLE POUR ÉVITER 404) ---
st.divider()
st.subheader("3. AI Deep Audit & Strategic Matrices")

if st.button("🧠 Execute Full Strategic Analysis"):
    try:
        # On cherche dynamiquement le nom exact du modèle Flash autorisé pour ta clé
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        flash_model_name = next((m for m in available_models if 'flash' in m.lower()), "gemini-1.5-flash")
        
        model = genai.GenerativeModel(flash_model_name)
        
        timeline_info = ", ".join([f"{k} issued on {v}" for k,v in dates.items()])
        
        # TON PROMPT SENIOR AIRBUS
        prompt = f"""
        Act as a Senior Airbus Project Controller. Analyze internal cost drift.
        
        CONTEXT:
        - Timeline: {timeline_info}
        - Normalized Costs Data: {pivot_data.to_string()}
        - Technical Mapping & Comments: {edited_mapping[['System', 'Common Name', 'Comments']].to_string()}
        
        REQUIRED AUDIT (English):
        1. DETAILED WP ANALYSIS: Specifically mention the Work Packages identified in the data.
        2. TEMPORAL EVOLUTION: Analyze strategy based on issue dates.
        3. JUSTIFICATION AUDIT: Do the 'Comments' provided justify these specific WP drifts?
        4. STRATEGIC CONCLUSION: Identify specific negotiation leverage points based on efficiency vs trend.

        CRITICAL DATA FORMAT:
        At the very end of your response, you MUST provide these two CSV blocks:
        
        [GLOBAL_DRIFT]
        System,Efficiency,Trend
        Devis_Alpha,score,score
        Devis_Beta,score,score
        Devis_Gamma,score,score
        
        [WP_DRIFT_DETAIL]
        WP,System,Efficiency,Trend
        """
        
        with st.spinner(f"Analyse avec {flash_model_name}..."):
            response = model.generate_content(prompt)
            st.session_state.ai_txt = response.text
    except Exception as e:
        st.error(f"Erreur durant l'audit : {e}")

if 'ai_txt' in st.session_state:
    res = st.session_state.ai_txt
    clean = res.replace("```csv", "").replace("```", "").replace("`", "")
    st.info(clean.split("[GLOBAL_DRIFT]")[0])
    
    col_g1, col_g2 = st.columns(2)
    try:
        g_block = re.search(r'\[GLOBAL_DRIFT\](.*?)\[WP_DRIFT_DETAIL\]', clean, re.DOTALL).group(1).strip()
        df_g = pd.read_csv(io.StringIO(g_block))
        fig_g = px.scatter(df_g, x="Efficiency", y="Trend", text="System", color="System", size=[40]*len(df_g))
        fig_g.update_layout(xaxis_range=[0,11], yaxis_range=[0,11])
        fig_g.add_hline(y=5, line_dash="dot"); fig_g.add_vline(x=5, line_dash="dot")
        col_g1.plotly_chart(fig_g, use_container_width=True)

        wp_block = re.search(r'\[WP_DRIFT_DETAIL\](.*)', clean, re.DOTALL).group(1).strip()
        df_wp = pd.read_csv(io.StringIO(wp_block))
        df_wp.columns = [c.strip() for c in df_wp.columns]
        sel_wp = st.selectbox("Detailed Analysis for WP:", df_wp['WP'].unique())
        df_filt = df_wp[df_wp['WP'] == sel_wp]
        fig_wp = px.scatter(df_filt, x="Efficiency", y="Trend", text="System", color="System", size=[35]*len(df_filt))
        fig_wp.update_layout(xaxis_range=[0,11], yaxis_range=[0,11])
        fig_wp.add_hline(y=5, line_dash="dot"); fig_wp.add_vline(x=5, line_dash="dot")
        col_g2.plotly_chart(fig_wp, use_container_width=True)
    except:
        st.warning("Audit textuel prêt.")
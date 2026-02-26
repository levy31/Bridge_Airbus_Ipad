import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import os
import io
import re
from datetime import datetime

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="Airbus Expert Audit")
GENAI_KEY = "AIzaSyAEMku4tPlRSVa279yoSDKbMWSVQKNxXL0"
genai.configure(api_key=GENAI_KEY)

def get_valid_model_name():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
                return m.name
    except: pass
    return 'models/gemini-1.5-flash'

# --- SIDEBAR DATES ---
st.sidebar.header("📅 Internal Issue Dates")
dates = {}
files_list = ["Devis_Alpha", "Devis_Beta", "Devis_Gamma"]
for f in files_list:
    dates[f] = st.sidebar.date_input(f"Date for {f}", datetime.now())

# --- DATA LOADING ---
MAPPING_FILE = "mapping_audit.csv"
raw_dfs, all_wps_list = {}, []
for f_name in files_list:
    path = f"{f_name}.xlsx"
    if os.path.exists(path):
        df = pd.read_excel(path)
        raw_dfs[f_name] = df
        for wp in df['Work_Package'].unique(): 
            all_wps_list.append({"System": f_name, "Original WP": wp})

if all_wps_list:
    st.title("🎯 Airbus Expert Audit - Global & WP Trend Selector")
    
    # --- 1. HARMONIZATION ---
    df_init = pd.DataFrame(all_wps_list)
    if os.path.exists(MAPPING_FILE):
        df_saved = pd.read_csv(MAPPING_FILE)
        df_mapping = pd.merge(df_init, df_saved, on=['System', 'Original WP'], how='left')
    else:
        df_mapping = df_init

    for col in ["Common Name", "Complexity", "Comments"]:
        if col not in df_mapping.columns: df_mapping[col] = 1.0 if col == "Complexity" else ""
    
    st.subheader("1. Technical Normalization Matrix")
    edited_mapping = st.data_editor(df_mapping, hide_index=True, width='stretch')
    if not edited_mapping.equals(df_mapping): 
        edited_mapping.to_csv(MAPPING_FILE, index=False)
        st.rerun()

    # --- CALCULS ---
    map_dict = {(r["System"], r["Original WP"]): r["Common Name"] for _, r in edited_mapping.iterrows()}
    comp_dict = {(r["System"], r["Original WP"]): r["Complexity"] for _, r in edited_mapping.iterrows()}
    
    all_data = []
    for name, df in raw_dfs.items():
        df_c = df.copy()
        df_c['System'] = name
        df_c['WP_STD'] = df_c['Work_Package'].apply(lambda x: map_dict.get((name, x), x))
        df_c['Ratio'] = df_c['Work_Package'].apply(lambda x: comp_dict.get((name, x), 1.0))
        df_c['Normalized_Cost'] = df_c['Cout_Total'] / df_c['Ratio']
        all_data.append(df_c)
    
    df_global = pd.concat(all_data)
    pivot_data = df_global.pivot_table(index='WP_STD', columns='System', values='Normalized_Cost', aggfunc='sum').fillna(0)

    # --- 2. PERFORMANCE SNAPSHOTS ---
    st.divider()
    st.subheader("2. Performance Snapshots")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(df_global, x="WP_STD", y="Normalized_Cost", color="System", barmode="group", title="Normalized Costs per WP"), use_container_width=True)
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
        fig_bridge.update_layout(title=f"Cost Bridge: Alpha vs {target}", showlegend=False)
        st.plotly_chart(fig_bridge, use_container_width=True)

    # --- 3. AI AUDIT ---
    st.divider()
    st.subheader("3. AI Deep Audit & Dual-Level Drift Matrices")
    
    if st.button("🧠 Execute Full Drift Analysis"):
        model_name = get_valid_model_name()
        model = genai.GenerativeModel(model_name)
        timeline_info = ", ".join([f"{k} on {v}" for k,v in dates.items()])
        
        prompt = f"""
        Act as a Senior Airbus Project Controller. Analyze internal cost drift.
        
        CONTEXT:
        - Timeline: {timeline_info}
        - Normalized Costs: {pivot_data.to_string()}
        - Technical Comments: {edited_mapping[['System', 'Common Name', 'Comments']].to_string()}
        
        REQUIRED AUDIT (English):
        1. DETAILED WP ANALYSIS: Specifically mention the Work Packages identified in the data.
        2. TEMPORAL EVOLUTION: Analyze strategy based on issue dates.
        3. JUSTIFICATION AUDIT: Do 'Comments' justify these specific WP drifts?
        4. STRATEGIC CONCLUSION: Negotiation leverage points.

        CRITICAL DATA FORMAT (MUST BE RAW CSV AT THE END):
        [GLOBAL_DRIFT]
        System,Efficiency,Trend
        Devis_Alpha,score,score
        Devis_Beta,score,score
        Devis_Gamma,score,score
        
        [WP_DRIFT_DETAIL]
        WP,System,Efficiency,Trend
        (List EVERY WP for EVERY System with scores 1-10)
        """
        
        with st.spinner("Analyzing..."):
            try:
                response = model.generate_content(prompt)
                st.session_state.ai_txt = response.text
            except Exception as e:
                st.error(f"AI Error: {e}")

    if 'ai_txt' in st.session_state:
        full_text = st.session_state.ai_txt
        # Nettoyage agressif des balises markdown
        clean_text = full_text.replace("```csv", "").replace("```", "").replace("`", "")
        
        report_part = clean_text.split("[GLOBAL_DRIFT]")[0]
        st.info(report_part)
        
        col_g1, col_g2 = st.columns(2)
        
        # 1. GRAPH GLOBAL
        with col_g1:
            try:
                g_block = re.search(r'\[GLOBAL_DRIFT\](.*?)\[WP_DRIFT_DETAIL\]', clean_text, re.DOTALL).group(1).strip()
                df_g = pd.read_csv(io.StringIO(g_block))
                df_g.columns = [c.strip() for c in df_g.columns]
                fig_g = px.scatter(df_g, x="Efficiency", y="Trend", text="System", color="System", size=[40]*len(df_g), title="GLOBAL Strategy")
                fig_g.update_layout(xaxis_range=[0,11], yaxis_range=[0,11])
                fig_g.add_hline(y=5, line_dash="dot"); fig_g.add_vline(x=5, line_dash="dot")
                st.plotly_chart(fig_g, use_container_width=True)
            except:
                st.error("⚠️ Global Drift Data format error. Check AI output.")

        # 2. GRAPH WP AVEC MENU
        with col_g2:
            try:
                wp_block = re.search(r'\[WP_DRIFT_DETAIL\](.*)', clean_text, re.DOTALL).group(1).strip()
                df_wp = pd.read_csv(io.StringIO(wp_block))
                df_wp.columns = [c.strip() for c in df_wp.columns]
                
                list_wps = df_wp['WP'].unique()
                sel_wp = st.selectbox("Select WP to analyze:", list_wps)
                
                df_filt = df_wp[df_wp['WP'] == sel_wp]
                fig_wp = px.scatter(df_filt, x="Efficiency", y="Trend", text="System", color="System", size=[35]*len(df_filt), title=f"WP Drift: {sel_wp}")
                fig_wp.update_layout(xaxis_range=[0,11], yaxis_range=[0,11])
                fig_wp.add_hline(y=5, line_dash="dot"); fig_wp.add_vline(x=5, line_dash="dot")
                st.plotly_chart(fig_wp, use_container_width=True)
            except:
                st.error("⚠️ WP Detail Data format error. Check AI output.")
        
        # DEBUG OPTION
        with st.expander("🔍 Debug AI Output"):
            st.text(full_text)
else:
    st.error("Missing Excel files.")
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import os
import io
import re
from datetime import datetime
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Airbus Audit Master - Final")
st.title("🎯 Airbus Audit : The Complete Strategic Cockpit")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("API Key missing.")
    st.stop()

# --- 2. INITIALISATION DES DATES EN SESSION STATE ---
if "dates" not in st.session_state:
    st.session_state.dates = {
        "Devis_Alpha": datetime(2025, 1, 1).date(),
        "Devis_Beta": datetime(2025, 3, 1).date(),
        "Devis_Gamma": datetime(2025, 6, 1).date()
    }

# --- 3. SIDEBAR : DATES ---
st.sidebar.header("📅 Issue Dates (Free Chronology)")
files_list = ["Devis_Alpha", "Devis_Beta", "Devis_Gamma"]
for f in files_list:
    nouvelle_date = st.sidebar.date_input(
        f"Date for {f}",
        value=st.session_state.dates[f],
        key=f"date_input_{f}"
    )
    if nouvelle_date != st.session_state.dates[f]:
        st.session_state.dates[f] = nouvelle_date
        # Pas de rerun immédiat, les calculs se feront avec la nouvelle valeur au prochain cycle

# Raccourci pour le dictionnaire dates utilisé dans le reste du code
dates = st.session_state.dates

# --- 4. GESTION DU MAPPING ---
MAPPING_ACTIF = "mapping_actif.csv"

st.sidebar.header("🗂️ Gestion du mapping")
uploaded_mapping = st.sidebar.file_uploader("Charger un fichier mapping (CSV)", type=["csv"])
if uploaded_mapping is not None:
    with open(MAPPING_ACTIF, "wb") as f:
        f.write(uploaded_mapping.getbuffer())
    st.sidebar.success(f"Fichier {uploaded_mapping.name} chargé comme mapping actif.")
    st.rerun()

if st.sidebar.button("🔄 Réinitialiser le mapping (partir de zéro)"):
    if os.path.exists(MAPPING_ACTIF):
        os.remove(MAPPING_ACTIF)
    st.sidebar.success("Mapping réinitialisé.")
    st.rerun()

# --- 5. CHARGEMENT DES DEVIS ---
raw_dfs = {}
all_wps_list = []

for f_name in files_list:
    path = f"{f_name}.xlsx"
    if os.path.exists(path):
        df = pd.read_excel(path, engine='openpyxl')
        if 'Work_Package' not in df.columns or 'Cout_Total' not in df.columns:
            st.error(f"Le fichier {f_name}.xlsx doit contenir les colonnes 'Work_Package' et 'Cout_Total'.")
            st.stop()
        raw_dfs[f_name] = df
        for wp in df['Work_Package'].unique():
            all_wps_list.append({"System": f_name, "Original WP": wp})
    else:
        st.warning(f"Fichier {f_name}.xlsx non trouvé. Placez-le dans le répertoire.")

if not raw_dfs:
    st.warning("Veuillez placer Devis_Alpha.xlsx, Devis_Beta.xlsx et Devis_Gamma.xlsx dans le répertoire.")
    st.stop()

# --- 6. CHARGEMENT OU CRÉATION DU MAPPING ACTIF ---
df_init = pd.DataFrame(all_wps_list)
with st.expander("Diagnostic - Nombre de WP bruts"):
    st.write(f"Total WP bruts (tous systèmes) : {len(df_init)}")

if os.path.exists(MAPPING_ACTIF):
    df_saved = pd.read_csv(MAPPING_ACTIF)
    with st.expander("Diagnostic - Mapping chargé"):
        st.write(f"Lignes dans le mapping : {len(df_saved)}")
    required_cols = ["System", "Original WP", "Common Name", "Complexity", "Comments"]
    if not all(col in df_saved.columns for col in required_cols):
        st.error(f"Le fichier de mapping doit contenir les colonnes : {required_cols}")
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
    st.success(f"Mapping sauvegardé dans {MAPPING_ACTIF}")
    st.rerun()

# --- 7. CALCULS ET CHRONOLOGIE ---
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
with st.expander("Diagnostic - Après mapping"):
    st.write(f"Nombre de WP après mapping (WP_ST) : {df_global['WP_ST'].nunique()}")
    st.dataframe(df_global[['System', 'Date', 'WP_ST', 'Cout_Total', 'Normalized_Cost']].head(20))

pivot_raw = df_global.pivot_table(index='WP_ST', columns='System', values='Cout_Total', aggfunc='sum').fillna(0)
pivot_norm = df_global.pivot_table(index='WP_ST', columns='System', values='Normalized_Cost', aggfunc='sum').fillna(0)
chrono_order = df_global[['System', 'Date']].drop_duplicates().sort_values('Date')['System'].tolist()

def draw_bridge(pivot_df, title_text, target_sys, base_sys="Devis_Alpha"):
    if base_sys not in pivot_df.columns or target_sys not in pivot_df.columns:
        return go.Figure()
    v_base = pivot_df[base_sys].sum()
    labels, values, measures = [base_sys], [v_base], ["absolute"]
    for wp in pivot_df.index:
        diff = pivot_df.loc[wp, target_sys] - pivot_df.loc[wp, base_sys]
        if abs(diff) > 0.1:
            labels.append(wp)
            values.append(diff)
            measures.append("relative")
    labels.append(f"Total {target_sys}")
    values.append(pivot_df[target_sys].sum())
    measures.append("total")
    return go.Figure(go.Waterfall(measure=measures, x=labels, y=values)).update_layout(title=title_text)

# --- 8. LES 6 SNAPSHOTS ---
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
    target_r = st.selectbox("Compare Raw Alpha vs:", [s for s in pivot_raw.columns if s != "Devis_Alpha"], key="r_sel")
    st.plotly_chart(draw_bridge(pivot_raw, f"Raw Bridge: {target_r}", target_r),
                    use_container_width=True, key=f"raw_bridge_{target_r}")

st.subheader("3. Performance Snapshots (NORMALIZED DATA)")
c4, c5, c6 = st.columns(3)
with c4:
    st.plotly_chart(px.bar(df_global, x="WP_ST", y="Normalized_Cost", color="System", barmode="group", title="Normalized Volume"),
                    use_container_width=True, key="norm_volume")
with c5:
    st.plotly_chart(px.line(df_global, x="Date", y="Normalized_Cost", color="WP_ST", markers=True, title="Normalized Timeline"),
                    use_container_width=True, key="norm_timeline")
with c6:
    target_n = st.selectbox("Compare Normalized Alpha vs:", [s for s in pivot_norm.columns if s != "Devis_Alpha"], key="n_sel")
    st.plotly_chart(draw_bridge(pivot_norm, f"Normalized Bridge: {target_n}", target_n),
                    use_container_width=True, key=f"norm_bridge_{target_n}")

# ========== DÉRIVE GLOBALE ==========
st.divider()
st.subheader("4. Dérive globale de sur‑complexité (coût normalisé total)")

total_norm_by_system = df_global.groupby('System')['Normalized_Cost'].sum().reset_index()
date_map = df_global[['System', 'Date']].drop_duplicates()
total_norm_by_system = total_norm_by_system.merge(date_map, on='System').sort_values('Date')

if len(total_norm_by_system) >= 2:
    x_vals = (total_norm_by_system['Date'] - total_norm_by_system['Date'].min()).dt.days.values
    y_vals = total_norm_by_system['Normalized_Cost'].values
    coeffs = np.polyfit(x_vals, y_vals, 1)
    slope = coeffs[0]

    x_pred = np.array([0, x_vals.max()])
    y_pred = np.polyval(coeffs, x_pred)
    trend_dates = [total_norm_by_system['Date'].min(), total_norm_by_system['Date'].max()]

    fig_global_drift = go.Figure()
    fig_global_drift.add_trace(go.Scatter(x=total_norm_by_system['Date'], y=total_norm_by_system['Normalized_Cost'],
                                           mode='markers+lines', marker=dict(size=12), line=dict(width=2),
                                           name='Coût total normalisé'))
    fig_global_drift.add_trace(go.Scatter(x=trend_dates, y=y_pred, mode='lines', line=dict(dash='dash', color='red'),
                                           name=f'Tendance (pente = {slope:.2f} €/jour)'))
    fig_global_drift.update_layout(title="Évolution du coût total normalisé", xaxis_title="Date", yaxis_title="€", hovermode='x unified')
    st.plotly_chart(fig_global_drift, use_container_width=True, key="global_drift")

    if slope != 0:
        first_cost = total_norm_by_system['Normalized_Cost'].iloc[0]
        annualized_pct = (slope * 365 / first_cost) * 100
        st.metric("Dérive annualisée", f"{annualized_pct:.1f} % / an")
else:
    st.warning("Pas assez de points pour une tendance globale.")
    slope, first_cost, annualized_pct = 0, 0, 0

# ========== DÉRIVE PAR WORK PACKAGE ==========
st.subheader("5. Dérive individuelle par Work Package")

wp_drift_dict = {}
for wp in df_global['WP_ST'].unique():
    wp_data = df_global[df_global['WP_ST'] == wp].sort_values('Date')
    if len(wp_data) >= 2:
        x_wp = (wp_data['Date'] - wp_data['Date'].min()).dt.days.values
        y_wp = wp_data['Normalized_Cost'].values
        coeffs_wp = np.polyfit(x_wp, y_wp, 1)
        slope_wp = coeffs_wp[0]
        first_cost_wp = y_wp[0]
        annual_pct_wp = (slope_wp * 365 / first_cost_wp) * 100 if first_cost_wp != 0 else np.nan
        wp_drift_dict[wp] = {'pente': slope_wp, 'annual_pct': annual_pct_wp, 'data': wp_data[['Date', 'Normalized_Cost', 'System']]}
    else:
        wp_drift_dict[wp] = {'pente': None, 'annual_pct': None, 'data': wp_data[['Date', 'Normalized_Cost', 'System']]}

wp_drift_list = []
for wp, vals in wp_drift_dict.items():
    if vals['pente'] is not None:
        wp_drift_list.append({
            'Work Package': wp,
            'Pente (€/jour)': round(vals['pente'], 2),
            'Dérive annualisée (%)': round(vals['annual_pct'], 1) if not np.isnan(vals['annual_pct']) else 'N/A'
        })
df_wp_drift = pd.DataFrame(wp_drift_list)
if not df_wp_drift.empty:
    st.dataframe(df_wp_drift, use_container_width=True, hide_index=True)
else:
    st.info("Pas assez de données pour calculer des pentes.")

if wp_drift_dict:
    all_wps = sorted(wp_drift_dict.keys())
    selected_wp = st.selectbox("Choisissez un Work Package :", all_wps, key="wp_selector")
    wp_info = wp_drift_dict[selected_wp]
    wp_data = wp_info['data']

    fig_wp = go.Figure()
    fig_wp.add_trace(go.Scatter(x=wp_data['Date'], y=wp_data['Normalized_Cost'],
                                 mode='markers+lines' if len(wp_data) > 1 else 'markers',
                                 marker=dict(size=10), line=dict(width=2) if len(wp_data) > 1 else None,
                                 name='Coût normalisé', text=wp_data['System'],
                                 hovertemplate='<b>%{text}</b><br>%{x}<br>%{y:.2f} €<extra></extra>'))
    if wp_info['pente'] is not None:
        x_wp_vals = (wp_data['Date'] - wp_data['Date'].min()).dt.days.values
        y_wp_vals = wp_data['Normalized_Cost'].values
        coeffs_wp_indiv = np.polyfit(x_wp_vals, y_wp_vals, 1)
        slope_wp_indiv = coeffs_wp_indiv[0]
        x_pred_wp = np.array([0, x_wp_vals.max()])
        y_pred_wp = np.polyval(coeffs_wp_indiv, x_pred_wp)
        trend_dates_wp = [wp_data['Date'].min(), wp_data['Date'].max()]
        fig_wp.add_trace(go.Scatter(x=trend_dates_wp, y=y_pred_wp, mode='lines', line=dict(dash='dash', color='red'),
                                     name=f'Tendance (pente = {slope_wp_indiv:.2f} €/jour)'))
        st.caption(f"Dérive annualisée : {wp_info['annual_pct']:.1f} % / an")
    else:
        st.caption("⚠️ WP avec une seule version – pas de tendance.")
    fig_wp.update_layout(title=f"Dérive du WP : {selected_wp}", xaxis_title="Date", yaxis_title="Coût normalisé (€)")
    st.plotly_chart(fig_wp, use_container_width=True, key=f"wp_drift_{selected_wp}")

# ========== MODE VALIDATION ORACLE ==========
st.sidebar.header("🧪 Mode Validation")
oracle_file = st.sidebar.file_uploader("Charger fichier oracle", type=["xlsx", "csv"])

if oracle_file:
    # Lecture de l'onglet Dates
    try:
        xl = pd.ExcelFile(oracle_file, engine='openpyxl')
        if 'Dates' in xl.sheet_names:
            df_dates_oracle = pd.read_excel(oracle_file, sheet_name='Dates', engine='openpyxl')
            nouvelles_dates = {}
            for _, row in df_dates_oracle.iterrows():
                # Adapter les noms de colonnes possibles
                systeme = row.get('Système') or row.get('System')
                date_val = pd.to_datetime(row['Date']).date()
                if systeme and systeme in files_list:
                    nouvelles_dates[systeme] = date_val

            if nouvelles_dates:
                if st.session_state.dates != nouvelles_dates:
                    st.sidebar.warning("Les dates de l'oracle sont différentes des dates actuelles.")
                    if st.sidebar.button("📅 Appliquer les dates de l'oracle"):
                        for sys, date in nouvelles_dates.items():
                            st.session_state.dates[sys] = date
                        st.sidebar.success("Dates mises à jour. Rechargement...")
                        st.rerun()
                else:
                    st.sidebar.success("✅ Les dates correspondent déjà à l'oracle.")
        else:
            st.sidebar.info("Le fichier oracle ne contient pas d'onglet 'Dates'. Les dates saisies manuellement seront utilisées.")
    except Exception as e:
        st.sidebar.error(f"Erreur lecture des dates : {e}")

    # --- Comparaison des dérives ---
    st.divider()
    st.subheader("🔬 Comparaison avec le scénario attendu (Oracle)")
    try:
        # Chargement des autres onglets
        df_oracle_global = pd.read_excel(oracle_file, sheet_name="Attendus_Global", engine='openpyxl')
        df_oracle_wp = pd.read_excel(oracle_file, sheet_name="Attendus_WP", engine='openpyxl')
        analyse_attendue = pd.read_excel(oracle_file, sheet_name="Analyse_IA_attendue", engine='openpyxl').iloc[0,0]

        # Affichage des listes de WP pour diagnostic (optionnel)
        with st.expander("Diagnostic - Comparaison des WP"):
            st.write("**WP dans l'oracle :**", df_oracle_wp['WP'].tolist())
            st.write("**WP calculés par l'app :**", df_wp_drift['Work Package'].tolist() if not df_wp_drift.empty else "Aucun")

        # Comparaison globale
        st.markdown("#### Comparaison globale")
        if len(total_norm_by_system) >= 2:
            derive_calculee = ((total_norm_by_system['Normalized_Cost'].iloc[-1] / total_norm_by_system['Normalized_Cost'].iloc[0]) - 1) * 100
        else:
            derive_calculee = 0
        derive_attendue_str = df_oracle_global[df_oracle_global['Metrique'] == 'Dérive_globale_normalisee_%']['Valeur'].values[0]
        derive_attendue = float(derive_attendue_str.replace('%', '').replace('+', ''))
        ecart_global = derive_calculee - derive_attendue

        col1, col2, col3 = st.columns(3)
        col1.metric("Calculée", f"{derive_calculee:.1f}%")
        col2.metric("Attendue", derive_attendue_str)
        col3.metric("Écart", f"{ecart_global:+.1f} pts", delta_color="off" if abs(ecart_global) < 5 else "inverse")

        # Comparaison par WP
        st.markdown("#### Comparaison par Work Package")
        comparaisons = []
        for _, row in df_oracle_wp.iterrows():
            wp_oracle = row['WP']
            derive_attendue = row['Dérive_normalisee_%_attendue']
            alerte_attendue = row['Alerte']
            commentaire = row['Commentaire_attendu']

            # Recherche insensible à la casse et aux espaces
            wp_calc = None
            for wp_calc_nom in df_wp_drift['Work Package']:
                if wp_calc_nom.strip().lower() == wp_oracle.strip().lower():
                    wp_calc = wp_calc_nom
                    break

            if wp_calc is not None:
                derive_calc = float(df_wp_drift[df_wp_drift['Work Package'] == wp_calc]['Dérive annualisée (%)'].iloc[0])
                if derive_attendue != 'N/A':
                    derive_attendue_val = float(derive_attendue)
                    ecart = derive_calc - derive_attendue_val
                    statut = "✅" if abs(ecart) < 5 else "⚠️" if abs(ecart) < 10 else "❌"
                else:
                    ecart = None
                    statut = "ℹ️ Nouveau WP"
            else:
                derive_calc = None
                ecart = None
                statut = "❌ Non trouvé"

            comparaisons.append({
                'WP (oracle)': wp_oracle,
                'Attendu': str(derive_attendue),
                'Calculé': f"{derive_calc:.1f}%" if derive_calc is not None else "N/A",
                'Écart': f"{ecart:+.1f}" if ecart is not None else "N/A",
                'Alerte attendue': alerte_attendue,
                'Statut': statut,
                'Commentaire': commentaire
            })
        st.dataframe(pd.DataFrame(comparaisons), hide_index=True, use_container_width=True)

        # Comparaison de l'analyse IA
        st.markdown("#### Comparaison de l'analyse IA")
        col_ia1, col_ia2 = st.columns(2)
        col_ia1.markdown("**📝 Analyse attendue :**")
        col_ia1.info(analyse_attendue)
        col_ia2.markdown("**🤖 Analyse produite :**")
        if 'ai_audit' in st.session_state:
            col_ia2.info(st.session_state.ai_audit)
        else:
            col_ia2.warning("Lancez d'abord l'audit IA complet (section 7).")
    except Exception as e:
        st.error(f"Erreur lors de la lecture des onglets de l'oracle : {e}")
        st.info("Assurez-vous que le fichier Excel contient les onglets 'Attendus_Global', 'Attendus_WP' et 'Analyse_IA_attendue'.")

# ========== ANALYSE IA DES DÉRIVES ==========
st.divider()
st.subheader("6. Analyse IA des dérives observées")
if st.button("🤖 Analyser les dérives avec l'IA", key="btn_drift_analysis"):
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model_name = next((m for m in available_models if "flash" in m), available_models[0])
        model = genai.GenerativeModel(model_name)

        global_trend = f"pente globale = {slope:.2f} €/jour, soit {annualized_pct:.1f}% par an (coût initial total = {first_cost:.2f} €)"
        wp_trends = []
        for wp, vals in wp_drift_dict.items():
            if vals['pente'] is not None:
                wp_trends.append(f"- {wp}: pente = {vals['pente']:.2f} €/jour, annualisé = {vals['annual_pct']:.1f}% par an")
            else:
                wp_trends.append(f"- {wp}: présent seulement dans {len(vals['data'])} version(s) – pas de tendance calculable")
        wp_trends_str = "\n".join(wp_trends)

        raw_points = df_global[['System', 'Date', 'WP_ST', 'Normalized_Cost']].to_string()

        prompt = f"""
        Tu es un expert en analyse de projets aéronautiques. On a calculé les dérives de coûts normalisés (après ajustement par la complexité) pour trois versions d'un projet (Alpha, Beta, Gamma) avec leurs dates réelles.
        
        Voici les indicateurs de dérive :
        
        **Tendance globale** (somme de tous les WP) :
        {global_trend}
        
        **Tendances par Work Package** :
        {wp_trends_str}
        
        **Données détaillées** (optionnelles, pour référence) :
        {raw_points}
        
        Questions auxquelles répondre dans ton analyse :
        1. Quels sont les Work Packages qui présentent la plus forte dérive à la hausse ? Lesquels sont stables ou en baisse ?
        2. La dérive globale est-elle préoccupante ? Comment se compare-t-elle à une inflation normale (disons 2-3% par an) ?
        3. Y a-t-il des corrélations possibles entre les dates et l'ampleur des dérives (par exemple, une accélération après une certaine date) ?
        4. Quelles recommandations stratégiques pourriez-vous faire au vu de ces tendances ?
        
        Réponds de manière concise mais précise, en t'appuyant sur les chiffres fournis.
        """
        with st.spinner("L'IA analyse les dérives..."):
            response = model.generate_content(prompt)
            st.session_state.ai_drift_analysis = response.text
    except Exception as e:
        st.error(f"Erreur IA : {e}")

if 'ai_drift_analysis' in st.session_state:
    st.markdown(st.session_state.ai_drift_analysis)

# --- 7. IA : AUDIT STRATÉGIQUE COMPLET ---
st.divider()
st.subheader("7. AI Strategic Audit (complet)")
if st.button("🧠 Execute Full Strategic Audit (incluant données et dérives)", key="btn_full_audit"):
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model_name = next((m for m in available_models if "flash" in m), available_models[0])
        model = genai.GenerativeModel(model_name)

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
        with st.spinner("Audit Senior Airbus en cours..."):
            response = model.generate_content(prompt)
            st.session_state.ai_audit = response.text
    except Exception as e:
        st.error(f"Erreur IA : {e}")

if 'ai_audit' in st.session_state:
    st.info(st.session_state.ai_audit)

# --- 8. GUIDE D'AUDIT ---
st.divider()
st.subheader("📚 Guide d'Audit")
st.markdown("""
* **Les 6 graphiques** : Analyse exhaustive des volumes, timelines et écarts (bruts et normalisés).
* **Dérive globale** : Graphique de l'évolution du coût total normalisé avec droite de tendance – la pente visualise la dérive temporelle moyenne.
* **Dérive par WP** : Tableau des pentes individuelles (€/jour et % annualisé) pour chaque Work Package, et graphique individuel avec sélecteur (tous les WP listés).
* **Analyse IA** : Deux boutons – l'un pour une analyse ciblée des dérives, l'autre pour un audit stratégique complet incluant les données et commentaires.
* **Mode validation** : Chargez un fichier oracle pour comparer automatiquement vos résultats avec un scénario de référence. Si l'oracle contient un onglet 'Dates', vous pouvez appliquer ces dates en un clic pour garantir la cohérence.
""")
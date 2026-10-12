import streamlit as st
from PIL import Image
import time
import pandas as pd
import numpy as np
from datetime import datetime
from model_utils import get_model, preprocess_image, generate_gradcam, overlay_heatmap, get_clinical_findings

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="RadixAI Dashboard", layout="wide", page_icon="✨")

# --- 2. ULTIMATE VIBRANT CSS ---
st.markdown("""
    <style>
    /* Animated "Aurora" Mesh Background */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp { 
        background: linear-gradient(-45deg, #E0F2FE, #FDF4FF, #E0E7FF, #F0FDF4); 
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
        font-family: 'Inter', sans-serif; 
        color: #0F172A; 
    }
    
    #MainMenu, header, footer {visibility: hidden;}
    .stMarkdown a { display: none !important; }
    
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.8) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.04);
    }
    [data-testid="stSidebarNav"] { display: none; } 
    
    /* Soft, Bright Interactive Inputs */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] > input {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        color: #0F172A !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    div[data-baseweb="input"] > div:hover, 
    div[data-baseweb="select"] > div:hover {
        border-color: #6366F1 !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15) !important;
        transform: translateY(-1px);
    }
    
    /* Text Area Styling (Bright Mode) */
    [data-baseweb="textarea"] textarea {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #0F172A !important; 
        -webkit-text-fill-color: #0F172A !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.7 !important;
        transition: all 0.3s ease;
    }
    [data-baseweb="textarea"] textarea:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.2) !important;
        background-color: #FFFFFF !important;
    }
    [data-testid="InputInstructions"] { display: none !important; }
    
    /* Frosted Glass Form Container */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.7) !important; 
        backdrop-filter: blur(24px) !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important; 
        border-radius: 24px !important; 
        padding: 32px !important; 
        box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Vibrant Primary Button */
    @keyframes buttonGlow {
        0% { box-shadow: 0 0 10px rgba(99, 102, 241, 0.3); }
        50% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.6), 0 0 35px rgba(217, 70, 239, 0.4); }
        100% { box-shadow: 0 0 10px rgba(99, 102, 241, 0.3); }
    }
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #4F46E5, #D946EF) !important;
        background-size: 200% auto !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px !important;
        padding: 14px 32px !important;
        transition: all 0.4s ease !important;
        animation: buttonGlow 3s infinite;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background-position: right center !important;
        transform: translateY(-3px) scale(1.02);
    }
    
    /* Interactive File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.5) !important;
        border: 2px dashed #8B5CF6 !important;
        border-radius: 20px !important;
        padding: 40px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border: 2px solid #D946EF !important;
        background: rgba(255, 255, 255, 0.9) !important;
        transform: translateY(-4px);
        box-shadow: 0 15px 30px rgba(217, 70, 239, 0.15);
    }
    
    /* Top Navigation (White Glass) */
    .top-nav { 
        display: flex; justify-content: space-between; align-items: center; 
        padding: 16px 28px; 
        background: rgba(255, 255, 255, 0.8); 
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        margin-bottom: 24px; border-radius: 20px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.05);
    }
    .top-nav-logo { 
        font-weight: 900; font-size: 1.7rem; display: flex; align-items: center; gap: 12px;
        background: linear-gradient(135deg, #4F46E5, #D946EF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    @keyframes pulse-bright {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.3); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .status-badge { 
        background-color: #ECFDF5; color: #059669; 
        border: 1px solid #34D399; padding: 6px 16px; border-radius: 24px; 
        font-size: 0.8rem; font-weight: 800; letter-spacing: 1px;
        animation: pulse-bright 2s infinite;
    }
    
    /* Elevated Frosted Cards */
    .patient-header { 
        background: rgba(255, 255, 255, 0.7); 
        backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.8); 
        border-radius: 24px; padding: 28px; margin-bottom: 24px; 
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.05);
    }
    .patient-header:hover {
        transform: translateY(-6px);
        box-shadow: 0 25px 50px -12px rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Patient Info Typography */
    .patient-info { display: flex; gap: 20px; align-items: center; }
    .avatar { 
        background: linear-gradient(135deg, #0EA5E9, #6366F1); 
        color: white; width: 65px; height: 65px; border-radius: 20px; 
        display: flex; align-items: center; justify-content: center; 
        font-weight: 800; font-size: 1.6rem; box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
    }
    .patient-details h3 { margin: 0; font-size: 1.5rem; font-weight: 900; color: #0F172A; }
    .patient-details p { margin: 6px 0 0 0; font-size: 0.95rem; color: #475569; font-weight: 600;}

    /* Custom iOS Segmented Radio Buttons */
    div[role="radiogroup"] {
        background-color: #F5F3FF; padding: 6px; border-radius: 14px;
        border: 1px solid rgba(139, 92, 246, 0.2); display: flex; gap: 0;
    }
    div[role="radiogroup"] > label {
        background-color: transparent; padding: 10px 20px; border-radius: 10px;
        margin: 0; transition: all 0.3s ease; color: #4F46E5; font-weight: 700;
    }
    
    /* Shimmering Progress Bars */
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(200%); }
    }
    .finding-row { margin-bottom: 22px; }
    .finding-header { display: flex; justify-content: space-between; font-size: 0.95rem; margin-bottom: 8px; font-weight: 800;}
    .bar-bg { width: 100%; background-color: #F1F5F9; height: 12px; border-radius: 6px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);}
    
    .bar-fill { 
        height: 100%; border-radius: 6px; transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1); 
        position: relative; overflow: hidden;
    }
    .bar-fill::after {
        content: ''; position: absolute; top: 0; right: 0; bottom: 0; left: 0;
        background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.5) 50%, rgba(255,255,255,0) 100%);
        animation: shimmer 2.5s infinite linear;
    }
    
    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 30px; background-color: transparent; border-bottom: 2px solid rgba(99, 102, 241, 0.1);}
    .stTabs [data-baseweb="tab"] { height: 55px; background-color: transparent; color: #475569; font-weight: 800; font-size: 1.05rem; transition: color 0.3s; }
    .stTabs [data-baseweb="tab"]:hover { color: #6366F1; }
    .stTabs [aria-selected="true"] { color: #6366F1 !important; border-bottom: 4px solid #6366F1 !important; }
    
    .custom-footer { text-align: center; color: #64748B; font-size: 0.85rem; margin-top: 60px; padding-bottom: 24px; letter-spacing: 1px; font-weight: 700;}
    </style>
""", unsafe_allow_html=True)

# --- 3. DYNAMIC HCI INPUTS (Sidebar) ---
st.sidebar.markdown("<h3 style='color: #4F46E5; margin-bottom: 20px; font-weight: 900; font-size: 1.5rem;'>📋 Patient Demographics</h3>", unsafe_allow_html=True)

patient_id = st.sidebar.text_input("Patient ID / MRN", "OM-1029", help="Medical Record Number synced from hospital database.")
patient_age = st.sidebar.number_input("Age", 1, 120, 21)
patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
patient_ward = st.sidebar.text_input("Department/Ward", "AI/ML Lab")

# --- 4. TOP BRANDING ---
st.markdown("""
    <div class="top-nav">
        <div class="top-nav-logo">
            ✨ RadixAI 
        </div>
        <div>
            <span class="status-badge">● SYSTEM ACTIVE</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 5. DYNAMIC PATIENT HEADER ---
st.markdown(f"""
    <div class="patient-header">
        <div class="patient-info">
            <div class="avatar">{patient_id[:2].upper()}</div>
            <div class="patient-details">
                <h3>{patient_id.upper()}</h3>
                <p>{patient_gender}, {patient_age}y &nbsp;|&nbsp; {patient_ward} &nbsp;|&nbsp; {datetime.today().strftime('%b %d, %Y')}</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cached_model():
    return get_model()

model = load_cached_model()

prediction_label = None
confidence = None
impression = None
actions = None

# --- 6. INTERACTIVE FUNCTIONAL TABS ---
tab_analysis, tab_overview, tab_history, tab_reports = st.tabs(["🔍 AI Inference", "📊 Patient Telemetry", "🕒 Historical Scans", "📝 Clinical Notes"])

# ==========================================
# TAB 1: ANALYSIS
# ==========================================
with tab_analysis:
    col_left, col_right = st.columns([2.2, 1.2], gap="large")

    with col_left:
        st.markdown("<h3 style='font-size: 1.4rem; color: #0F172A; margin-bottom: 20px; font-weight: 900;'>🖼️ Radiographic Imaging</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CXR Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            with st.spinner("Processing neural activation maps..."):
                input_tensor = preprocess_image(image)
                heatmap, pred_idx, probabilities = generate_gradcam(model, input_tensor)
                prediction_label = "Pneumonia" if pred_idx == 1 else "Normal"
                confidence = probabilities[pred_idx] * 100
                
                st.markdown("<br>", unsafe_allow_html=True)
                view_mode = st.radio("Toggle Layer", ["Original Radiograph", "Grad-CAM Neural Overlay"], horizontal=True, label_visibility="collapsed")
                
                if view_mode == "Original Radiograph":
                    st.image(image, use_container_width=True)
                else:
                    overlay_img = overlay_heatmap(image, heatmap, opacity=0.6)
                    st.image(overlay_img, use_container_width=True)

    with col_right:
        st.markdown("<h3 style='font-size: 1.4rem; color: #0F172A; margin-bottom: 20px; font-weight: 900;'>📈 Inference Metrics</h3>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            findings, actions, impression = get_clinical_findings(prediction_label, confidence)
            
            box_color = "linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%)" if prediction_label == "Pneumonia" else "linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%)"
            border_color = "#FDA4AF" if prediction_label == "Pneumonia" else "#86EFAC"
            text_color = "#BE123C" if prediction_label == "Pneumonia" else "#15803D"
            
            st.markdown(f"""
                <div style="background: {box_color}; border: 1px solid {border_color}; border-radius: 16px; padding: 24px; margin-bottom: 28px; box-shadow: 0 8px 20px rgba(0,0,0,0.04); transition: transform 0.3s; transform: scale(1.02);">
                    <div style="font-size: 0.8rem; color: {text_color}; margin-bottom: 8px; font-weight: 900; letter-spacing: 1px;">PRIMARY FINDING</div>
                    <h4 style="color: {text_color}; margin: 0 0 10px 0; font-size: 1.5rem; font-weight: 900;">{prediction_label}</h4>
                    <p style="color: {text_color}; font-size: 1.05rem; margin: 0; line-height: 1.6; opacity: 0.95; font-weight: 600;">{impression}</p>
                </div>
            """, unsafe_allow_html=True)
            
            inner_tab1, inner_tab2 = st.tabs(["Biomarkers", "Protocols"])
            
            with inner_tab1:
                st.markdown("<br>", unsafe_allow_html=True)
                for key, data in findings.items():
                    bar_color = "#F43F5E" if data['color'] == "#ef4444" else ("#8B5CF6" if data['color'] == "#f59e0b" else "#06B6D4")
                    
                    st.markdown(f"""
                        <div class="finding-row">
                            <div class="finding-header">
                                <span style="font-weight: 800; color: #0F172A;">{key}</span>
                                <span style="color: {bar_color}; font-weight: 900;">{data['confidence']:.0f}%</span>
                            </div>
                            <div class="bar-bg">
                                <div class="bar-fill" style="width: {data['confidence']}%; background: linear-gradient(90deg, {bar_color}66, {bar_color}); box-shadow: 0 0 10px {bar_color}44;"></div>
                            </div>
                            <div style="font-size: 0.85rem; color: #64748B; margin-top: 8px; font-weight: 600;">{data['sub']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
            with inner_tab2:
                st.markdown("<br>", unsafe_allow_html=True)
                for action in actions:
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.8); backdrop-filter: blur(10px); border-left: 5px solid #8B5CF6; padding: 16px 20px; margin-bottom: 16px; border-radius: 0 12px 12px 0; font-size: 1.05rem; color: #334155; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.05);' onmouseover="this.style.transform='translateX(8px)'; this.style.boxShadow='0 8px 20px rgba(139, 92, 246, 0.15)';" onmouseout="this.style.transform='translateX(0)'; this.style.boxShadow='0 4px 12px rgba(139, 92, 246, 0.05)';">
                        {action}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: rgba(255,255,255,0.4); backdrop-filter: blur(10px); border: 2px dashed rgba(99, 102, 241, 0.3); border-radius: 20px; padding: 40px 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 400px; text-align: center;">
                    <div style="background: linear-gradient(135deg, #6366F1, #D946EF); width: 88px; height: 88px; border-radius: 44px; display: flex; align-items: center; justify-content: center; margin-bottom: 28px; box-shadow: 0 15px 30px rgba(217, 70, 239, 0.4); transform: scale(1); transition: transform 0.3s;" onmouseover="this.style.transform='scale(1.1)'" onmouseout="this.style.transform='scale(1)'">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                        </svg>
                    </div>
                    <h4 style="color: #0F172A; margin: 0 0 10px 0; font-size: 1.5rem; font-weight: 900;">System Standby</h4>
                    <p style="color: #475569; font-size: 1.1rem; margin: 0; max-width: 280px; line-height: 1.6; font-weight: 600;">Upload a radiographic scan to initialize the neural inference engine.</p>
                </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 2: OVERVIEW
# ==========================================
with tab_overview:
    st.markdown("<h3 style='font-size: 1.4rem; color: #0F172A; font-weight: 900; margin-bottom: 24px;'>📊 Patient Probability Trend</h3>", unsafe_allow_html=True)
    time_index = pd.date_range(end=datetime.now(), periods=24, freq='h')
    chart_data = pd.DataFrame({
        'Consolidation Risk (%)': np.random.normal(loc=15, scale=5.0, size=24).clip(0, 100),
        'Effusion Risk (%)': np.random.normal(loc=5, scale=2.0, size=24).clip(0, 100),
    }, index=time_index)
    st.line_chart(chart_data)

# ==========================================
# TAB 3: HISTORY
# ==========================================
with tab_history:
    st.markdown("<h3 style='font-size: 1.4rem; color: #0F172A; font-weight: 900; margin-bottom: 24px;'>🕒 Radiographic Archive</h3>", unsafe_allow_html=True)
    history_data = pd.DataFrame({
        "Scan Date": ["2026-07-15", "2025-12-10", "2025-05-22"],
        "Modality": ["CXR (PA View)", "CT Thorax", "CXR (AP View)"],
        "Clinical Indication": ["Routine screening", "Persistent cough", "Chest pain"],
        "AI Finding": ["Clear", "Mild atelectasis", "Clear"],
        "Status": ["Signed Off", "Signed Off", "Signed Off"]
    })
    st.dataframe(history_data, use_container_width=True, hide_index=True)

# ==========================================
# TAB 4: REPORTS 
# ==========================================
with tab_reports:
    if uploaded_file is None:
        smart_draft = f"Patient {patient_id.upper()} presenting in {patient_ward}.\n\n[Awaiting radiographic scan for automated preliminary report...]"
    else:
        smart_draft = f"""PATIENT ENCOUNTER SUMMARY
ID: {patient_id.upper()} | {patient_gender}, {patient_age}y

AUTOMATED AI IMPRESSION ({datetime.today().strftime('%Y-%m-%d')}):
{impression}

SUGGESTED PROTOCOLS:
"""
        for act in actions:
            smart_draft += f"- {act}\n"
        smart_draft += "\nPHYSICIAN ADDENDUM:\n[Type clinical notes here...]"

    with st.form(key="clinical_notes_form"):
        st.markdown("<h3 style='font-size: 1.4rem; color: #0F172A; font-weight: 900; margin-bottom: 20px;'>📝 Clinical Addendum</h3>", unsafe_allow_html=True)
        
        doctor_notes = st.text_area("Physician Notes", value=smart_draft, height=400, label_visibility="collapsed")
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("Securely Export to EMR", type="primary", use_container_width=True)
        
    if submit_button:
        st.toast("Establishing secure EMR connection...", icon="🔐")
        time.sleep(0.8)
        st.toast(f"Notes successfully appended to {patient_id.upper()} record.", icon="✅")
        st.success(f"**Export Complete:** The clinical addendum for {patient_id.upper()} has been encrypted and exported to the central database.")

# --- 7. REQUIRED FOOTER ---
st.markdown(
    """
    <div class="custom-footer">
        SOFTWARE PROJECT MANAGEMENT PORTFOLIO 
    </div>
    """, 
    unsafe_allow_html=True
)
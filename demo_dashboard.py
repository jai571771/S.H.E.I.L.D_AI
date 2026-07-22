import streamlit as st
import time
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

# Add project root to path to ensure correct imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.streaming.telemetry_validator import TelemetryValidator
from src.streaming.buffer import TelemetryBuffer
from src.streaming.rolling_stats import RollingStatisticsEngine
from src.streaming.derived_features import DerivedFeatureEngine
from src.streaming.online_store import OnlineFeatureStore
from src.inference.engine import InferenceEngine
from src.utils import get_config

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION - Professional Minimalistic Dark Theme
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="S.H.I.E.L.D. - Scenario Demonstration Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching the design reference exactly
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800;900&display=swap');
    
    /* Spacing and Typography Token System */
    :root {
        /* Colors (Improved Contrast) */
        --bg-dark: #070D19;
        --card-bg: #0B132B;
        --border-color: rgba(255, 255, 255, 0.05);
        
        --text-primary: #F5F7FA;
        --text-secondary: #C7CFDA;
        --text-muted: #9AA4B2;
        
        --color-cyan: #38BDF8;
        --color-green: #10B981;
        --color-orange: #F97316;
        --color-red: #EF4444;
        --color-purple: #C084FC;

        /* 8px Spacing System */
        --space-xs: 8px;     /* Text elements */
        --space-sm: 16px;    /* Component spacing */
        --space-md: 24px;    /* Internal card padding / Card gap */
        --space-lg: 32px;    /* Section gaps */

        /* Typography Tokens */
        --font-size-heading: 34px;
        --font-size-section: 22px;
        --font-size-card: 18px;
        --font-size-body: 15px;
        --font-size-caption: 13px;
        
        /* Shadows & Radii */
        --radius-sm: 4px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --radius-xl: 16px;
        --shadow-card: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    /* Root App Background */
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: var(--font-size-body);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Layout Padding */
    .block-container {
        padding-top: var(--space-md);
        padding-bottom: var(--space-lg);
        padding-left: var(--space-lg);
        padding-right: var(--space-lg);
        max-width: 100% !important;
    }
    
    /* Scenario Selection Card Styling */
    .scenario-card {
        background: #111827;
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 16px 12px;
        text-align: center;
        min-height: 130px;
        transition: all 200ms ease;
        box-shadow: var(--shadow-card);
    }
    .scenario-card:hover {
        transform: translateY(-3px);
    }
    .scenario-card-active {
        border-color: var(--color-purple);
        background: rgba(192, 132, 252, 0.08);
        box-shadow: 0 0 8px rgba(192, 132, 252, 0.2);
    }
    .scenario-icon {
        font-size: 24px;
        margin-bottom: 8px;
    }
    .scenario-title {
        font-family: 'Outfit', sans-serif;
        font-size: var(--font-size-caption);
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .scenario-desc {
        font-size: 11px;
        color: var(--text-muted);
        line-height: 1.4;
        height: 38px;
        overflow: hidden;
    }
    
    /* Section Container Styling */
    .section-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-xl);
        padding: var(--space-md);
        margin-bottom: var(--space-lg);
        box-shadow: var(--shadow-card);
        transition: transform 200ms ease, box-shadow 200ms ease;
    }
    .section-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    
    .section-header {
        display: flex;
        align-items: center;
        margin-bottom: 14px; /* Reduced by 40% */
    }
    
    .section-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px; /* Reduced size */
        height: 26px; /* Reduced size */
        border: 1.5px solid var(--text-muted);
        border-radius: var(--radius-md);
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: var(--text-secondary);
        margin-right: 12px;
        font-size: var(--font-size-caption);
    }
    
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: var(--font-size-section);
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: 0.75px;
        text-transform: uppercase;
    }
    
    /* Section 1 CSS components */
    .s1-left-panel {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    .s1-center-panel {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        height: 100%;
        padding: var(--space-sm);
    }
    .s1-score-lbl {
        font-size: var(--font-size-caption);
        font-weight: 800;
        color: var(--text-muted);
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: var(--space-xs);
    }
    .s1-score-display {
        display: flex;
        align-items: baseline;
        gap: 4px;
        margin-bottom: var(--space-xs);
    }
    .s1-score-number {
        font-family: 'Outfit', sans-serif;
        font-size: 80px;
        font-weight: 900;
        line-height: 1;
    }
    .s1-score-max {
        font-family: 'Outfit', sans-serif;
        font-size: 26px;
        color: var(--text-muted);
        font-weight: 700;
    }
    .s1-risk-title {
        font-family: 'Outfit', sans-serif;
        font-size: var(--font-size-section);
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .s1-desc-text {
        font-size: var(--font-size-body);
        color: var(--text-secondary);
        line-height: 1.6;
        margin-top: var(--space-md);
        max-width: 400px;
    }
    
    .risk-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-sm);
    }
    .risk-kpi-card {
        display: flex;
        align-items: center;
        background: #111827;
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: var(--space-sm);
        height: 90px;
        box-shadow: var(--shadow-card);
        transition: transform 200ms ease;
    }
    .risk-kpi-card:hover {
        transform: translateY(-2px);
    }
    .risk-kpi-icon {
        font-size: 22px;
        margin-right: var(--space-sm);
        display: flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: var(--radius-md);
        background: rgba(255, 255, 255, 0.02);
    }
    .risk-kpi-text {
        display: flex;
        flex-direction: column;
    }
    .risk-kpi-label {
        font-size: 10px;
        color: var(--text-muted);
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .risk-kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 15px;
        font-weight: 800;
        color: var(--text-primary);
        margin-top: 2px;
    }
    .ai-summary-box {
        display: flex;
        align-items: flex-start;
        background: rgba(192, 132, 252, 0.03);
        border: 1px solid rgba(192, 132, 252, 0.12);
        border-radius: var(--radius-lg);
        padding: var(--space-md);
        margin-top: var(--space-md);
    }
    .ai-summary-icon {
        font-size: 22px;
        color: var(--color-purple);
        margin-right: var(--space-md);
        display: flex;
        align-items: center;
    }
    .ai-summary-content {
        display: flex;
        flex-direction: column;
    }
    .ai-summary-title {
        font-size: 11px;
        font-weight: 800;
        color: var(--color-purple);
        letter-spacing: 0.5px;
        margin-bottom: 4px;
        text-transform: uppercase;
    }
    .ai-summary-text {
        font-size: var(--font-size-body);
        color: var(--text-secondary);
        line-height: 1.5;
    }
    
    /* Section 2 CSS components */
    .health-overview-container {
        display: grid;
        grid-template-columns: repeat(8, 1fr);
        gap: var(--space-md);
    }
    @media (max-width: 1400px) {
        .health-overview-container {
            grid-template-columns: repeat(4, 1fr);
        }
    }
    @media (max-width: 768px) {
        .health-overview-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 480px) {
        .health-overview-container {
            grid-template-columns: 1fr;
        }
    }
    
    .health-overview-card {
        background: #111827;
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: var(--space-md) var(--space-xs);
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;
        height: 230px;
        box-shadow: var(--shadow-card);
        transition: transform 200ms ease;
    }
    .health-overview-card:hover {
        transform: translateY(-3px);
    }
    .health-overview-icon {
        font-size: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 24px;
        margin-bottom: var(--space-xs);
    }
    .health-overview-gauge {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: var(--space-xs);
    }
    .health-overview-label {
        font-size: var(--font-size-caption);
        font-weight: 700;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        height: 36px;
        line-height: 1.3;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .health-overview-status {
        font-size: var(--font-size-caption);
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    
    /* Section 3 CSS components */
    .ai-explanation-container {
        max-width: 700px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: var(--space-sm);
        text-align: center;
    }
    .ai-explanation-brain {
        font-size: 40px;
        color: var(--color-purple);
        display: flex;
        align-items: center;
        justify-content: center;
        width: 56px;
        height: 56px;
        margin-bottom: var(--space-xs);
    }
    .ai-explanation-heading {
        font-family: 'Outfit', sans-serif;
        font-size: var(--font-size-card);
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: var(--space-xs);
    }
    .ai-explanation-text {
        font-size: var(--font-size-body);
        color: var(--text-secondary);
        line-height: 1.7;
    }
    .ai-explanation-highlight {
        color: var(--color-orange);
        font-weight: 700;
    }
    
    /* Section 4 CSS components */
    .suggestion-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: var(--space-md);
    }
    @media (max-width: 1200px) {
        .suggestion-grid {
            grid-template-columns: repeat(4, 1fr);
        }
    }
    @media (max-width: 768px) {
        .suggestion-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 480px) {
        .suggestion-grid {
            grid-template-columns: 1fr;
        }
    }
    
    .suggestion-card {
        background: #111827;
        border: 1px solid var(--border-color);
        border-radius: var(--radius-xl);
        padding: var(--space-md);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 250px;
        box-shadow: var(--shadow-card);
        transition: transform 200ms ease;
    }
    .suggestion-card:hover {
        transform: translateY(-3px);
    }
    .suggestion-header {
        display: flex;
        align-items: center;
        gap: var(--space-xs);
        margin-bottom: var(--space-xs);
    }
    .suggestion-icon {
        font-size: 22px;
        display: flex;
        align-items: center;
    }
    .suggestion-title {
        font-family: 'Outfit', sans-serif;
        font-size: var(--font-size-body);
        font-weight: 700;
        color: var(--text-primary);
    }
    .suggestion-desc {
        font-size: var(--font-size-caption);
        color: var(--text-secondary);
        line-height: 1.5;
        margin-bottom: var(--space-xs);
        flex-grow: 1;
    }
    .suggestion-divider {
        border-top: 1px solid var(--border-color);
        margin: var(--space-xs) 0;
    }
    .suggestion-footer {
        display: flex;
        justify-content: space-between;
        gap: var(--space-md);
    }
    .suggestion-footer-item {
        display: flex;
        flex-direction: column;
    }
    .suggestion-footer-label {
        font-size: 10px;
        color: var(--text-muted);
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .suggestion-footer-val {
        font-size: var(--font-size-caption);
        font-weight: 800;
        margin-top: 2px;
    }
    
    /* Section 6 CSS components */
    .factor-row {
        display: grid;
        grid-template-columns: 32px 180px 1fr 140px 50px;
        align-items: center;
        gap: var(--space-sm);
        margin-bottom: var(--space-sm);
        background: rgba(255, 255, 255, 0.01);
        padding: 12px 16px;
        border-radius: var(--radius-md);
        border: 1px solid var(--border-color);
    }
    .factor-icon-box {
        font-size: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .factor-title {
        font-size: var(--font-size-body);
        font-weight: 700;
        color: var(--text-primary);
    }
    .factor-subtitle {
        font-size: var(--font-size-caption);
        color: var(--text-muted);
    }
    .factor-bar-container {
        display: flex;
        align-items: center;
    }
    .factor-bar-bg {
        width: 100%;
        height: 8px;
        background: #111827;
        border-radius: 4px;
        overflow: hidden;
    }
    .factor-bar-fill {
        height: 100%;
        border-radius: 4px;
    }
    .factor-value {
        text-align: right;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: var(--font-size-body);
    }
    
    /* Header Bar & Stream controls styling */
    .header-logo-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .header-logo-title {
        font-family: 'Outfit', sans-serif;
        font-size: var(--font-size-heading);
        font-weight: 800;
        color: var(--color-purple);
        letter-spacing: 1.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        font-size: var(--font-size-caption);
        color: var(--text-muted);
        font-weight: 500;
        margin-top: 4px;
    }
    
    .status-badge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--color-purple);
        font-size: var(--font-size-caption);
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: var(--space-xs);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: var(--color-purple);
        border-radius: 50%;
        box-shadow: 0 0 6px var(--color-purple);
    }
    
    .header-right-side {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        justify-content: center;
        text-align: right;
        gap: 4px;
        font-size: var(--font-size-caption);
        color: var(--text-secondary);
        font-weight: 600;
    }
    .header-right-status {
        color: var(--color-purple);
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    
    /* Footer Bar */
    .footer-bar {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-component);
        padding: 16px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: var(--font-size-caption);
        color: var(--text-secondary);
        margin-top: var(--space-lg);
    }
    .footer-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .footer-val {
        color: var(--text-primary);
        font-weight: 800;
    }
    
    /* Streamlit Button Text Styling */
    .stButton > button {
        font-size: var(--font-size-caption) !important;
        font-weight: 700 !important;
        border-radius: var(--radius-md) !important;
        padding: 6px 14px !important;
        height: 36px !important;
        background-color: #111827 !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        transition: all 200ms ease;
    }
    .stButton > button:hover {
        background-color: var(--color-purple) !important;
        color: var(--bg-dark) !important;
        border-color: var(--color-purple) !important;
        box-shadow: 0 0 12px rgba(192, 132, 252, 0.4) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------------------
# SCENARIOS DEFINITIONS & METADATA
# ------------------------------------------------------------------------------
SCENARIOS = [
    {
        "id": "s1",
        "name": "Normal Operation",
        "icon": "🟢",
        "desc": "All process variables remain within safe operating limits.",
        "file": "normal_operation.csv",
        "expected": "Low Risk",
        "explanation": "All process parameters are well within safe, standard operating limits. Cooling water flow, gas pressures, and temperature gradients are stable. Worker compliance and safety permits are up-to-date. The plant is operating under optimal conditions with zero hazards identified.",
        "recos": [
            ("Continue SCADA Monitoring", "All primary telemetry channels active.", "📡"),
            ("Maintain Cooling Line Throughput", "Verify auxiliary pumps operating on standby.", "💧"),
            ("Routine Refractory Inspection", "Ensure coke oven lining retains integrity.", "🔍"),
            ("Verify Industrial Safety Log", "Ensure compliance record saved to store.", "🛡️"),
            ("Normal Operation Baseline", "No risk factors triggered in coke oven or furnace.", "🟢")
        ]
    },
    {
        "id": "s2",
        "name": "Gas Pressure Rise",
        "icon": "🟡",
        "desc": "Blast Furnace gas pressure gradually increases above the recommended range.",
        "file": "gas_pressure_rise.csv",
        "expected": "Medium Risk",
        "explanation": "An elevated risk has been flagged due to a gradual rise in Blast Furnace top gas pressure and Coke Oven gas pressure. The pressure values are approaching or exceeding safety warning thresholds. Throttling is advised to stabilize the system before a critical pressure event develops.",
        "recos": [
            ("Inspect Gas Seal Valves", "Check for possible mechanical blockage or sticky seals.", "⚙️"),
            ("Throttle Blast Flow Rate", "Gently lower input blast volume to stabilize top pressure.", "💨"),
            ("Monitor Pressure Trends", "Analyze pressure gradients across 5-min intervals.", "📈"),
            ("Calibrate Flue Gas Sensors", "Verify zero-point and span calibration.", "🔍"),
            ("Restrict Direct Blast Floor Access", "Warn operators in Sector A of pressure fluctuations.", "🚧")
        ]
    },
    {
        "id": "s3",
        "name": "Cooling System Failure",
        "icon": "🟠",
        "desc": "Cooling water flow decreases while furnace temperature increases.",
        "file": "cooling_failure.csv",
        "expected": "High Risk",
        "explanation": "High risk predicted due to a severe furnace cooling failure. Telemetry indicates that Blast Furnace cooling water flow has dropped significantly below safety limits while top and water temperatures are rising rapidly. Immediate cooling circuit inspection and thermal adjustments are required.",
        "recos": [
            ("Increase Auxiliary Cooling Water Flow", "Open auxiliary valves and activate backup pump lines.", "💧"),
            ("Inspect Primary Cooling Pumps", "Check for motor tripping or cavitation in cooling circuit.", "⚙️"),
            ("Reduce Blast Temperature", "Gently drop blast furnace heat inputs to prevent over-heating.", "🔥"),
            ("Clear Vent Valve Accumulation", "Ensure steam vent lines remain free of blockages.", "💨"),
            ("Mobilize Sector B Safety Team", "Prepare emergency response crew for staging sector.", "🚨")
        ]
    },
    {
        "id": "s4",
        "name": "Compound Hazard",
        "icon": "🔴",
        "desc": "Simultaneous pressure rise, cooling decrease, temp surge, and gas leak.",
        "file": "compound_hazard.csv",
        "expected": "Critical Risk",
        "explanation": "CRITICAL ALERT: A compound hazard has been detected with multiple systems failing simultaneously. We observe rising gas pressures, declining cooling water flow, surging temperatures, elevated toxic CO concentrations, and abnormal equipment vibration. Emergency trip protocols should be initiated.",
        "recos": [
            ("Activate Emergency Safety Trip (ESD)", "Initiate automatic safety depressurization and blast cutoff.", "🚨"),
            ("Evacuate Sector A and B Floors", "Enforce emergency evacuation for all floor personnel.", "🚧"),
            ("Isolate Flammable Gas Lines", "Shut coke oven collector valves and isolate furnace lines.", "⚙️"),
            ("Engage Maximum Emergency Cooling", "Activate backup water spraying systems immediately.", "💧"),
            ("Establish Site Incident Command", "Initiate alarm siren and notify mill emergency response.", "📣")
        ]
    },
    {
        "id": "s5",
        "name": "Maintenance Mode",
        "icon": "🟣",
        "desc": "Plant is operating under scheduled maintenance with reduced production.",
        "file": "maintenance_mode.csv",
        "expected": "Low/Medium Risk",
        "explanation": "The system is operating under active maintenance. While some process rates are reduced, hot work permits are active and worker counts are elevated in hazard zones. Extra compliance checks are being enforced, and the overall plant risk remains controlled at low-to-medium.",
        "recos": [
            ("Audit Active Hot Work Permits", "Verify spark arrestors and fire extinguishers are on site.", "📋"),
            ("Enforce Continuous PPE Audits", "Confirm harness, safety goggles, and gas badges are worn.", "👷"),
            ("Perform Continuous Gas Testing", "Run hand-held CO/H2 sniffers in work sectors.", "🔍"),
            ("Establish Safety Watch Standby", "Assign spotters to monitor maintenance crew.", "👥"),
            ("Verify Safe Isolation Points", "Confirm double block and bleed valves are locked out.", "🔒")
        ]
    },
    {
        "id": "s6",
        "name": "Equipment Wear",
        "icon": "🟤",
        "desc": "Blower vibration gradually increases over time indicating mechanical wear.",
        "file": "equipment_wear.csv",
        "expected": "Medium Risk",
        "explanation": "A gradual machine wear pattern has been identified. Blower and pusher mechanical vibration levels are slowly drifting upward, indicating potential bearing wear or misalignment. Scheduling preventive maintenance is recommended before mechanical failure occurs.",
        "recos": [
            ("Inspect Blower Bearings", "Schedule ultrasonic and thermographic inspections.", "🔍"),
            ("Schedule Maintenance Inspection", "Book bearing replacement during next scheduled outage.", "🔧"),
            ("Monitor Vibration Gradients", "Track running root-mean-square (RMS) velocity.", "📈"),
            ("Audit Equipment Lubrication Logs", "Verify grease type and lubrication interval compliance.", "📋"),
            ("Check Blower Drive Alignment", "Audit coupling alignment at next machine stop.", "⚙️")
        ]
    },
    {
        "id": "s7",
        "name": "Emergency Shutdown",
        "icon": "⚫",
        "desc": "Critical alarms trigger controlled safety shutdown procedures.",
        "file": "emergency_shutdown.csv",
        "expected": "Critical Risk",
        "explanation": "EMERGENCY SHUTDOWN: Critical thresholds have been breached, triggering a controlled plant shutdown. Blast flow and blower current are being reduced to zero, and safety trip valves are active. Personnel must clear all production floors and follow emergency muster procedures.",
        "recos": [
            ("Isolate Furnace Gas Collectors", "Activate automated valves to isolate process vessels.", "🔒"),
            ("Verify Gas Bleeder Operation", "Confirm flare stack safety release valves are open.", "🔥"),
            ("Deploy Ventilation Fan Units", "Establish positive pressure air flow in electrical rooms.", "🌀"),
            ("Account for Floor Personnel", "Scan safety badge muster lists at designated points.", "👥"),
            ("Log System Trip Trigger sequence", "Ensure event log is exported for root-cause analysis.", "📝")
        ]
    }
]

# ------------------------------------------------------------------------------
# INITIALIZE STATE
# ------------------------------------------------------------------------------
if "demo_initialized" not in st.session_state:
    st.session_state.active_scenario = "Normal Operation"
    st.session_state.step_idx = 0
    st.session_state.is_playing = False
    st.session_state.scenario_results = {}
    st.session_state.demo_initialized = True

current_scenario_dict = next(sc for sc in SCENARIOS if sc["name"] == st.session_state.active_scenario)

# ------------------------------------------------------------------------------
# RUN INFERENCE ENGINE ON ALL SAMPLES OF SCENARIO FOR PLOTTING & INSTANT SCRUBBING
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_entire_scenario(file_name):
    import math as _math
    validator = TelemetryValidator()
    buffer = TelemetryBuffer(max_size=30)
    stats_engine = RollingStatisticsEngine(windows=[5, 15, 30])
    derived_engine = DerivedFeatureEngine()
    store = OnlineFeatureStore()
    inference_engine = InferenceEngine()
    
    csv_path = os.path.join("datasets", file_name)
    if not os.path.exists(csv_path):
        csv_path = "test_model.csv"
        
    df_raw = pd.read_csv(csv_path)
    raw_keys = validator.RAW_SENSORS.union(validator.CATEGORICAL_FIELDS.keys()).union(validator.BOOLEAN_FIELDS)
    results = []
    
    for idx, row in df_raw.iterrows():
        raw_sample = row.to_dict()
        raw_telemetry = {k: raw_sample[k] for k in raw_keys if k in raw_sample}
        raw_telemetry["Timestamp"] = raw_sample["Timestamp"]
        
        is_valid, _ = validator.validate(raw_telemetry)
        normalized = validator.normalize(raw_telemetry)
        normalized["Safety_Index"] = raw_sample.get("Safety_Index", 100.0)
        
        buffer.append(normalized)
        rolling_stats = stats_engine.update(buffer)
        derived_features = derived_engine.compute(normalized, rolling_stats)
        
        rolling_stats = {
            k: (0.0 if (not isinstance(v, (int, float)) or _math.isnan(v) or _math.isinf(v)) else float(v))
            for k, v in rolling_stats.items()
        }
        derived_features = {
            k: (0.0 if (not isinstance(v, (int, float)) or _math.isnan(v) or _math.isinf(v)) else float(v))
            for k, v in derived_features.items()
        }
        
        merged = {}
        merged.update(normalized)
        merged.update(rolling_stats)
        merged.update(derived_features)
        
        pred_res = inference_engine.predict_single(merged)
        
        results.append({
            "raw_sample": raw_sample,
            "normalized": normalized,
            "rolling_stats": rolling_stats,
            "derived_features": derived_features,
            "pred_res": pred_res
        })
        
    return results

# Load scenario data
with st.spinner("Running AI Real-Time Pipeline..."):
    scenario_data = compute_entire_scenario(current_scenario_dict["file"])

current_step_data = scenario_data[st.session_state.step_idx]
raw_sample = current_step_data["raw_sample"]
pred_res = current_step_data["pred_res"]
derived = current_step_data["derived_features"]
rolling_stats = current_step_data["rolling_stats"]

pred_class = pred_res.get("predicted_class", "Low")
confidence = float(pred_res.get("confidence", 0.98))
shap_list = pred_res.get("shap_contributions", [])

# Map risk level to health score & color scheme
RISK_CONFIG = {
    "Low": {
        "score_base": int(100 - (1.0 - confidence) * 20),
        "badge_title": "SAFE OPERATION",
        "badge_sub": "AI PREDICTS SAFE OPERATING CONDITION",
        "color": "#10B981",
        "bg_color": "rgba(16, 185, 129, 0.1)",
        "border_color": "rgba(16, 185, 129, 0.3)",
        "risk_level": "LOW",
        "risk_color": "#10B981",
        "explanation_banner": "Overall operational stability is high and all safety protocols are active.",
        "action_banner": "No immediate action required. Plant is operating safely."
    },
    "Medium": {
        "score_base": int(70 - (1.0 - confidence) * 20),
        "badge_title": "MEDIUM RISK WARNING",
        "badge_sub": "THERMAL / PRESSURE FLUCTUATION DETECTED",
        "color": "#F59E0B",
        "bg_color": "rgba(245, 158, 11, 0.1)",
        "border_color": "rgba(245, 158, 11, 0.3)",
        "risk_level": "MEDIUM",
        "risk_color": "#F59E0B",
        "explanation_banner": "Operational parameters require operator attention to stabilize thermal gradient.",
        "action_banner": "Operator action required: Adjust cooling flow and throttle pressure valve."
    },
    "High": {
        "score_base": int(45 - (1.0 - confidence) * 25),
        "badge_title": "HIGH RISK ELEVATED",
        "badge_sub": "RAPID HAZARD EXCURSION DETECTED",
        "color": "#F97316",
        "bg_color": "rgba(249, 115, 22, 0.12)",
        "border_color": "rgba(249, 115, 22, 0.4)",
        "risk_level": "HIGH",
        "risk_color": "#F97316",
        "explanation_banner": "HIGH RISK: Initiate emergency cooling and pressure relief procedures immediately.",
        "action_banner": "IMMEDIATE ACTION REQUIRED: Activate auxiliary cooling and evacuate Sector B."
    },
    "Critical": {
        "score_base": int(20 - (1.0 - confidence) * 15),
        "badge_title": "CRITICAL EMERGENCY ALERT",
        "badge_sub": "HAZARDOUS BREACH & SAFETY TRIP ACTIVE",
        "color": "#EF4444",
        "bg_color": "rgba(239, 68, 68, 0.15)",
        "border_color": "rgba(239, 68, 68, 0.5)",
        "risk_level": "CRITICAL",
        "risk_color": "#EF4444",
        "explanation_banner": "CRITICAL ALERT: Execute total plant emergency shutdown and site evacuation.",
        "action_banner": "CRITICAL HAZARD: Emergency trip active. Full site evacuation in progress."
    }
}

cfg = RISK_CONFIG.get(pred_class, RISK_CONFIG["Low"])
health_score = max(5, min(100, cfg["score_base"]))

# Build historical health scores array for scenario trend plot
all_health_scores = []
all_timestamps = []
for idx, res in enumerate(scenario_data):
    s_pred_class = res["pred_res"].get("predicted_class", "Low")
    s_conf = float(res["pred_res"].get("confidence", 0.98))
    s_cfg = RISK_CONFIG.get(s_pred_class, RISK_CONFIG["Low"])
    all_health_scores.append(max(5, min(100, s_cfg["score_base"])))
    all_timestamps.append(res["raw_sample"].get("Timestamp", str(idx)))

# Extract raw sensor values
bf_co_raw = float(raw_sample.get("BF_CO", 1.2) or 1.2)
bf_top_temp_raw = float(raw_sample.get("BF_Top_Temperature", 200.0) or 200.0)
bf_water_flow = float(raw_sample.get("BF_Cooling_Water_Flow", 150.0) or 150.0)
bf_water_temp = float(raw_sample.get("BF_Cooling_Water_Temperature", 35.0) or 35.0)
bf_press_raw = float(raw_sample.get("BF_Gas_Pressure", 2.0) or 2.0)
blower_vib_raw = float(raw_sample.get("BF_Blower_Vibration", 1.5) or 1.5)

co_co_raw = float(raw_sample.get("CO_CO", 1.1) or 1.1)
co_nh3_raw = float(raw_sample.get("CO_NH3", 0.1) or 0.1)
co_press_raw = float(raw_sample.get("CO_Gas_Pressure", 2.5) or 2.5)
co_oven_temp_raw = float(raw_sample.get("CO_Oven_Temperature", 1000.0) or 1000.0)
pusher_vib_raw = float(raw_sample.get("CO_Pusher_Vibration", 1.5) or 1.5)

ppe_comp = str(raw_sample.get("PPE_Compliance", True)).lower() in ("true", "1", "yes")
maint_active = str(raw_sample.get("Maintenance_Active", False)).lower() in ("true", "1", "yes")
gas_test_comp = str(raw_sample.get("Gas_Test_Completed", True)).lower() in ("true", "1", "yes")
worker_count = int(float(raw_sample.get("Worker_Count", 5) or 5))

# Compute Plant Health Scores
bf_health = max(10, min(100, int(100 - (max(0, 150 - bf_water_flow) * 0.3 + max(0, bf_top_temp_raw - 250) * 0.15 + max(0, bf_co_raw - 2.5) * 8.0))))
co_health = max(10, min(100, int(100 - (max(0, co_co_raw - 2.0) * 10.0 + max(0, co_nh3_raw - 0.3) * 30.0 + max(0, pusher_vib_raw - 2.0) * 8.0))))

thermal_balance_idx = max(10, min(100, int(100 - (max(0, bf_top_temp_raw - 250) * 0.15 + max(0, co_oven_temp_raw - 1100) * 0.1 + max(0, bf_water_temp - 45) * 0.8))))
gas_pressure_idx = max(10, min(100, int(100 - (max(0, bf_press_raw - 2.5) * 12.0 + max(0, co_press_raw - 3.0) * 10.0 + max(0, bf_co_raw - 2.0) * 5.0))))
process_stability_idx = max(10, min(100, int(100 - (max(0, blower_vib_raw - 2.0) * 15.0 + max(0, pusher_vib_raw - 2.0) * 12.0))))
safety_idx = max(10, min(100, int((100 if ppe_comp else 60) - (max(0, bf_co_raw - 2.5) * 8.0 + max(0, co_co_raw - 2.5) * 8.0))))
equipment_health_idx = max(10, min(100, int(100 - (blower_vib_raw * 7.0 + pusher_vib_raw * 7.0))))

energy_eff_val = f"{max(50, min(99, int(85 + (bf_water_flow - 150) * 0.1 - max(0, bf_top_temp_raw - 250) * 0.05)))}%"
maint_val = "SCHEDULED ACTIVE" if maint_active else "ON SCHEDULE"
op_stability_val = f"{process_stability_idx:.1f}%"
emission_level_val = "HIGH (ALERT)" if (bf_co_raw > 4.0 or co_co_raw > 4.0) else "MODERATE" if (bf_co_raw > 2.0 or co_co_raw > 2.0) else "LOW (SAFE)"
compliance_score_val = "100%" if (ppe_comp and gas_test_comp) else "75%" if ppe_comp else "50%"

# Dynamic indicators for overview section
env_health = max(10, min(100, int(100 - (bf_co_raw * 6.0 + co_co_raw * 6.0))))
safety_compliance = int(compliance_score_val.replace('%',''))
energy_eff = int(energy_eff_val.replace('%',''))
emission_health = max(10, min(100, int(100 - (max(0, bf_co_raw - 1.0) * 15.0 + max(0, co_co_raw - 1.0) * 15.0))))

# Helper to map risk text and descriptions
risk_level_str = cfg["risk_level"]
risk_color = cfg["color"]

# Dynamic Description Paragraph for Section 1
if pred_class == "Low":
    dynamic_desc = "System is stable and operating efficiently. All parameters are well within normal operating thresholds with zero safety violations."
elif pred_class == "Medium":
    dynamic_desc = "System is stable but showing signs of potential risk. Continuous monitoring and timely action recommended."
elif pred_class == "High":
    dynamic_desc = "Safety hazard thresholds exceeded. Elevated temperatures or gas leaks detected. Operator intervention required immediately."
else:
    dynamic_desc = "Critical hazard. Automated emergency shutdown valve sequences have been initiated. Evacuate sector floor immediately."

# Dynamic Summary for Section 1
summary_text = current_scenario_dict["explanation"]

# ------------------------------------------------------------------------------
# 1. HEADER BAR & REAL-TIME STREAMING CONTROL
# ------------------------------------------------------------------------------
logo_col, controls_col, status_col = st.columns([0.35, 0.35, 0.30])

with logo_col:
    st.markdown(
        f"""
        <div class="header-logo-container">
            <div class="header-logo-title">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="filter: drop-shadow(0px 0px 2px var(--color-purple));"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                S.H.I.E.L.D.
            </div>
            <div class="header-subtitle">
                Scenario Demonstration Simulator Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with controls_col:
    st.markdown(
        f"""
        <div class="status-badge-container">
            <div class="status-badge">
                <div class="status-dot"></div>
                <span>SIMULATION CONTROLLER</span>
            </div>
            <div style="font-size: var(--font-size-caption); color: var(--text-secondary); font-weight: 700; margin-top: 4px;">
                ACTIVE: <span style="color: var(--color-purple); font-weight: 800;">{st.session_state.active_scenario.upper()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with status_col:
    st.markdown(
        f"""
        <div class="header-right-side">
            <div class="header-right-status" style="color: var(--color-purple);">STATUS: SIMULATING</div>
            <div style="font-weight: 800; color: var(--text-primary); background: rgba(192, 132, 252, 0.1); border: 1px solid rgba(192, 132, 252, 0.3); padding: 2px 8px; border-radius: var(--radius-sm); font-size: 11px; margin-top: 2px;">
                STEP {st.session_state.step_idx + 1} / 50
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------------------
# SCENARIO SELECTION ROW
# ------------------------------------------------------------------------------
st.markdown("<div style='margin-top: 5px; margin-bottom: 5px; font-family: Outfit, sans-serif; font-size: 12px; font-weight: 800; color: #64748B; letter-spacing: 0.5px; text-transform: uppercase;'>SELECT PROCESS SCENARIO TO SIMULATE</div>", unsafe_allow_html=True)
sc_cols = st.columns(7)

for idx, sc in enumerate(SCENARIOS):
    with sc_cols[idx]:
        is_active = (st.session_state.active_scenario == sc["name"])
        active_class = "scenario-card-active" if is_active else ""
        
        st.markdown(
            f"""
            <div class="scenario-card {active_class}">
                <div class="scenario-icon">{sc['icon']}</div>
                <div class="scenario-title">{sc['name']}</div>
                <div class="scenario-desc">{sc['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.button(f"Simulate", key=f"sel_{sc['id']}", use_container_width=True):
            st.session_state.active_scenario = sc["name"]
            st.session_state.step_idx = 0
            st.session_state.is_playing = False
            st.rerun()

st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

# Timeline Scrubber and Controls
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([0.25, 0.55, 0.20])

with ctrl_col1:
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("⏮ First", use_container_width=True):
            st.session_state.step_idx = 0
            st.session_state.is_playing = False
            st.rerun()
    with btn_col2:
        if st.button("◀ Back", use_container_width=True):
            st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
            st.session_state.is_playing = False
            st.rerun()
    with btn_col3:
        if st.button("Next ▶", use_container_width=True):
            st.session_state.step_idx = min(49, st.session_state.step_idx + 1)
            st.session_state.is_playing = False
            st.rerun()

with ctrl_col2:
    selected_step = st.slider(
        "Simulation Timeline Step",
        min_value=0,
        max_value=49,
        value=st.session_state.step_idx,
        label_visibility="collapsed",
        key="timeline_scrubber"
    )
    if selected_step != st.session_state.step_idx:
        st.session_state.step_idx = selected_step
        st.session_state.is_playing = False
        st.rerun()

with ctrl_col3:
    play_label = "⏸ Pause Simulation" if st.session_state.is_playing else "▶ Play Simulation"
    if st.button(play_label, use_container_width=True):
        st.session_state.is_playing = not st.session_state.is_playing
        st.rerun()

st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: OVERALL RISK ANALYSIS
# ==============================================================================
st.markdown(
    """
    <div class="section-card">
        <div class="section-header">
            <div class="section-number">1</div>
            <div class="section-title">OVERALL RISK ANALYSIS</div>
        </div>
    """,
    unsafe_allow_html=True
)

shield_svg = f"""
<svg width="200" height="200" viewBox="0 0 200 200">
    <circle cx="100" cy="100" r="85" stroke="{risk_color}" stroke-dasharray="6 6" stroke-width="2" fill="none" opacity="0.3"/>
    <circle cx="100" cy="100" r="75" stroke="{risk_color}" stroke-width="4.5" fill="none" style="filter: drop-shadow(0px 0px 4px {risk_color}60);"/>
    <g transform="translate(60, 52) scale(0.4)">
       <path d="M100 0 L15 30 L15 110 C15 170 100 200 100 200 C100 200 185 170 185 110 L185 30 Z" fill="none" stroke="{risk_color}" stroke-width="12" style="filter: drop-shadow(0px 0px 3px {risk_color}40);"/>
       <path d="M60 100 L90 130 L145 70" fill="none" stroke="{risk_color}" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>
    </g>
</svg>
"""

risk_grid_html = f"""
<div class="risk-grid">
    <div class="risk-kpi-card">
        <div class="risk-kpi-icon" style="color: {risk_color};">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        </div>
        <div class="risk-kpi-text">
            <div class="risk-kpi-label">RISK LEVEL</div>
            <div class="risk-kpi-value" style="color: {risk_color};">{risk_level_str}</div>
        </div>
    </div>
    <div class="risk-kpi-card">
        <div class="risk-kpi-icon" style="color: var(--color-green);">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 6l-9.5 9.5-5-5L1 18"/></svg>
        </div>
        <div class="risk-kpi-text">
            <div class="risk-kpi-label">PREDICTION CONFIDENCE</div>
            <div class="risk-kpi-value">{confidence * 100:.2f}%</div>
        </div>
    </div>
    <div class="risk-kpi-card">
        <div class="risk-kpi-icon" style="color: var(--color-cyan);">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        </div>
        <div class="risk-kpi-text">
            <div class="risk-kpi-label">FORECAST HORIZON</div>
            <div class="risk-kpi-value">15 MINUTES</div>
        </div>
    </div>
    <div class="risk-kpi-card">
        <div class="risk-kpi-icon" style="color: var(--color-purple);">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
        </div>
        <div class="risk-kpi-text">
            <div class="risk-kpi-label">TIMESTAMP</div>
            <div class="risk-kpi-value">{raw_sample.get('Timestamp', 'N/A').split(' ')[1]}</div>
        </div>
    </div>
</div>
"""

col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
with col1:
    st.markdown(f'<div class="s1-left-panel">{shield_svg}</div>', unsafe_allow_html=True)
with col2:
    st.markdown(
        f"""
        <div class="s1-center-panel">
            <div class="s1-score-lbl">OVERALL HEALTH SCORE</div>
            <div class="s1-score-display">
                <span class="s1-score-number" style="color: {risk_color};">{health_score}</span>
                <span class="s1-score-max">/100</span>
            </div>
            <div class="s1-risk-title" style="color: {risk_color};">{risk_level_str} RISK</div>
            <div class="s1-desc-text">{dynamic_desc}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(f'<div class="s1-right">{risk_grid_html}</div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="margin-top: var(--space-md);">
        <div class="ai-summary-box">
            <div class="ai-summary-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-3.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2zM14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-3.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2z"/></svg>
            </div>
            <div class="ai-summary-content">
                <div class="ai-summary-title">AI SUMMARY</div>
                <div class="ai-summary-text">{summary_text}</div>
            </div>
        </div>
    </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# SECTION 2: PLANT HEALTH OVERVIEW
# ==============================================================================
st.markdown(
    """
    <div class="section-card">
        <div class="section-header">
            <div class="section-number">2</div>
            <div class="section-title">PLANT HEALTH OVERVIEW</div>
        </div>
    """,
    unsafe_allow_html=True
)

def make_circular_progress(label, pct, status_label, status_color, icon_svg):
    offset = 251.2 * (1 - pct / 100.0)
    return f"""
    <div class="health-overview-card">
        <div class="health-overview-icon" style="color: {status_color};">{icon_svg}</div>
        <div class="health-overview-gauge">
            <svg width="70" height="70" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" stroke="rgba(255,255,255,0.05)" stroke-width="8" fill="none"/>
                <circle cx="50" cy="50" r="40" stroke="{status_color}" stroke-width="8" fill="none"
                        stroke-dasharray="251.2" stroke-dashoffset="{offset}"
                        stroke-linecap="round" style="filter: drop-shadow(0px 0px 4px {status_color}); transform: rotate(-90deg); transform-origin: 50% 50%;"/>
                <text x="50" y="56" text-anchor="middle" fill="#F8FAFC" font-size="20" font-family="'Outfit', sans-serif" font-weight="900">{pct}%</text>
            </svg>
        </div>
        <div class="health-overview-label">{label}</div>
        <div class="health-overview-status" style="color: {status_color};">{status_label}</div>
    </div>
    """

# SVGs
tower_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 22h16M7 22l3-18h4l3 18M9 8h6M8 14h8"/></svg>'
factory_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 22h20M17 14h4v8h-4zM2 18h12v4H2zM14 6l3 4v8h-3zM7 10l3 4v8H7z"/></svg>'
leaf_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 22 2c0 5-2.07 9-7.8 10.2A7 7 0 0 1 11 20zM11 20v-5"/></svg>'
shield_check_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10zM9 11l2 2 4-4"/></svg>'
bolt_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>'
gear_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'
pulse_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>'
cloud_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg>'

c1_html = make_circular_progress("Blast Furnace Health", bf_health, "GOOD" if bf_health >= 80 else "FAIR" if bf_health >= 50 else "CRITICAL", "#10B981" if bf_health >= 80 else "#F59E0B" if bf_health >= 50 else "#EF4444", tower_svg)
c2_html = make_circular_progress("Coke Oven Plant Health", co_health, "GOOD" if co_health >= 80 else "FAIR" if co_health >= 50 else "CRITICAL", "#10B981" if co_health >= 80 else "#F59E0B" if co_health >= 50 else "#EF4444", factory_svg)
c3_html = make_circular_progress("Environmental Health", env_health, "GOOD" if env_health >= 80 else "FAIR" if env_health >= 60 else "POOR", "#38BDF8" if env_health >= 80 else "#F59E0B" if env_health >= 60 else "#EF4444", leaf_svg)
c4_html = make_circular_progress("Safety Compliance Score", safety_compliance, "EXCELLENT" if safety_compliance >= 90 else "GOOD" if safety_compliance >= 70 else "VIOLATION", "#A855F7" if safety_compliance >= 90 else "#38BDF8" if safety_compliance >= 70 else "#EF4444", shield_check_svg)
c5_html = make_circular_progress("Energy Efficiency", energy_eff, "GOOD" if energy_eff >= 80 else "FAIR" if energy_eff >= 60 else "POOR", "#10B981" if energy_eff >= 80 else "#F59E0B" if energy_eff >= 60 else "#EF4444", bolt_svg)
c6_html = make_circular_progress("Equipment Health", equipment_health_idx, "GOOD" if equipment_health_idx >= 80 else "FAIR" if equipment_health_idx >= 50 else "POOR", "#10B981" if equipment_health_idx >= 80 else "#F59E0B" if equipment_health_idx >= 50 else "#EF4444", gear_svg)
c7_html = make_circular_progress("Process Stability", process_stability_idx, "GOOD" if process_stability_idx >= 80 else "FAIR" if process_stability_idx >= 50 else "UNSTABLE", "#38BDF8" if process_stability_idx >= 80 else "#F59E0B" if process_stability_idx >= 50 else "#EF4444", pulse_svg)
c8_html = make_circular_progress("Emission Level Health", emission_health, "GOOD" if emission_health >= 85 else "FAIR" if emission_health >= 60 else "POOR", "#10B981" if emission_health >= 85 else "#F59E0B" if emission_health >= 60 else "#EF4444", cloud_svg)

st.markdown(
    f"""
    <div class="health-overview-container">
        {c1_html}
        {c2_html}
        {c3_html}
        {c4_html}
        {c5_html}
        {c6_html}
        {c7_html}
        {c8_html}
    </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# SECTION 3: AI EXPLANATION
# ==============================================================================
st.markdown(
    """
    <div class="section-card">
        <div class="section-header">
            <div class="section-number">3</div>
            <div class="section-title">AI EXPLANATION</div>
        </div>
    """,
    unsafe_allow_html=True
)

# Generate detailed text highlights depending on risk level
if pred_class == "Low":
    explain_detail = "The AI identified all process parameters, including temperatures, pressures, and flow rates, running well within optimal baselines."
else:
    reco_highlights = []
    if bf_press_raw > 2.2 or co_press_raw > 2.7:
        reco_highlights.append("<span class='ai-explanation-highlight'>increasing gas pressure</span>")
    if bf_top_temp_raw > 240 or co_oven_temp_raw > 1050:
        reco_highlights.append("<span class='ai-explanation-highlight'>elevated furnace temperature</span>")
    if bf_water_flow < 140:
        reco_highlights.append("<span class='ai-explanation-highlight'>reduced cooling water flow</span>")
    
    if not reco_highlights:
        reco_highlights = ["<span class='ai-explanation-highlight'>slight parameter fluctuations</span>"]
        
    explain_detail = f"The AI identified {', '.join(reco_highlights)} as the primary contributors to the reduction in plant health."

st.markdown(
    f"""
    <div class="ai-explanation-container">
        <div class="ai-explanation-brain" style="filter: drop-shadow(0 0 3px var(--color-purple)40);">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-3.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2zM14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-3.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2z"/></svg>
        </div>
        <div class="ai-explanation-heading">AI Operational Instability Assessment</div>
        <div class="ai-explanation-text">
            The overall health score is currently at <span style="font-weight: 800; color: {risk_color}">{health_score}/100</span>, indicating a <span style="font-weight: 800; color: {risk_color}">{risk_level_str} risk level</span>.<br/>
            {explain_detail}<br/>
            Although the system remains operational, these parameters indicate the early stages of operational instability.
            Timely corrective action is recommended to prevent potential incidents and maintain optimal performance.
        </div>
    </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# SECTION 4: SUGGESTED WAYS TO INCREASE SCORE
# ==============================================================================
st.markdown(
    """
    <div class="section-card">
        <div class="section-header">
            <div class="section-number">4</div>
            <div class="section-title">SUGGESTED WAYS TO INCREASE SCORE</div>
        </div>
    """,
    unsafe_allow_html=True
)

def make_suggestion_card(title, desc, priority, impact, color, icon_svg):
    return f"""
    <div class="suggestion-card" style="border-left: 4px solid {color};">
        <div class="suggestion-header">
            <div class="suggestion-icon" style="color: {color};">{icon_svg}</div>
            <div class="suggestion-title">{title}</div>
        </div>
        <div class="suggestion-desc">{desc}</div>
        <div class="suggestion-divider"></div>
        <div class="suggestion-footer">
            <div class="suggestion-footer-item">
                <div class="suggestion-footer-label">PRIORITY</div>
                <div class="suggestion-footer-val" style="color: {color};">{priority}</div>
            </div>
            <div class="suggestion-footer-item">
                <div class="suggestion-footer-label">IMPACT</div>
                <div class="suggestion-footer-val" style="color: {color};">{impact}</div>
            </div>
        </div>
    </div>
    """

# Icons
flame_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2s-1 2.5-1 4.5c0 2.2 1.8 4 4 4s4-1.8 4-4c0-2-1-4.5-1-4.5s-2 2-4 2s-2-2-2-2zm-3 5.5C9 5.5 8 4 8 4S5 7 5 10c0 3.9 3.1 7 7 7s7-3.1 7-7c0-2-1-3.5-1-3.5s-2.5 3-4.5 3s-4.5-4-4.5-5.5z"/></svg>'
water_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>'
thermometer_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/></svg>'
gear_svg_card = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'
cloud_svg_card = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg>'

# Suggested Cards Content
card_s1 = make_suggestion_card("Optimize Gas Pressure", "Reduce gas pressure to within optimal range (1.5 - 2.0 bar)", "HIGH", "HIGH", "#EF4444", flame_svg)
card_s2 = make_suggestion_card("Improve Cooling Water Flow", "Increase cooling water flow rate to 150 - 180 L/min", "HIGH", "HIGH", "#EF4444", water_svg)
card_s3 = make_suggestion_card("Monitor Temperature Closely", "Maintain furnace temperature below upper limit (≤ 1200 °C)", "MEDIUM", "MEDIUM", "#F59E0B", thermometer_svg)
card_s4 = make_suggestion_card("Regular Equipment Inspection", "Check critical equipment for wear and vibration issues", "MEDIUM", "MEDIUM", "#F59E0B", gear_svg_card)
card_s5 = make_suggestion_card("Maintain CO Levels", "Ensure CO concentration remains below 2.5% in top gas", "LOW", "LOW", "#10B981", cloud_svg_card)

st.markdown(
    f"""
    <div class="suggestion-grid">
        {card_s1}
        {card_s2}
        {card_s3}
        {card_s4}
        {card_s5}
    </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# SECTIONS 5 & 6: VOLATILITY (Left) & SHAP CONTRIBUTIONS (Right)
# ==============================================================================
bottom_left_col, bottom_right_col = st.columns([0.5, 0.5], gap="large")

# ================= SECTION 5: HEALTH SCORE VOLATILITY =================
with bottom_left_col:
    st.markdown(
        """
        <div class="section-card" style="height: 100%;">
            <div class="section-header" style="justify-content: space-between;">
                <div style="display: flex; align-items: center;">
                    <div class="section-number">5</div>
                    <div class="section-title">HEALTH SCORE VOLATILITY</div>
                </div>
                <div style="background: #111827; border-radius: 6px; padding: 2px; display: flex; gap: 4px; border: 1px solid var(--border-color);">
                    <span style="background: var(--color-purple); color: var(--bg-dark); padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 800;">1 DAY</span>
                    <span style="color: var(--text-muted); padding: 2px 8px; font-size: 10px; font-weight: 800;">1 WEEK</span>
                    <span style="color: var(--text-muted); padding: 2px 8px; font-size: 10px; font-weight: 800;">1 MONTH</span>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    # Plotly Trend Chart with threshold bands
    fig_line = go.Figure()
    
    # Grid lines
    fig_line.add_hline(y=75, line_width=1, line_dash="dash", line_color="rgba(16, 185, 129, 0.25)")
    fig_line.add_hline(y=50, line_width=1, line_dash="dash", line_color="rgba(245, 158, 11, 0.25)")
    fig_line.add_hline(y=25, line_width=1, line_dash="dash", line_color="rgba(239, 68, 68, 0.25)")
    
    # Safe/Warning annotations placed inside the chart area
    fig_line.add_annotation(x=0.02, y=87.5, text="SAFE", showarrow=False, font=dict(color="#10B981", size=10, family="Outfit", weight="bold"), xref="paper", yref="y", align="left")
    fig_line.add_annotation(x=0.02, y=62.5, text="MEDIUM", showarrow=False, font=dict(color="#F59E0B", size=10, family="Outfit", weight="bold"), xref="paper", yref="y", align="left")
    fig_line.add_annotation(x=0.02, y=37.5, text="HIGH RISK", showarrow=False, font=dict(color="#F97316", size=10, family="Outfit", weight="bold"), xref="paper", yref="y", align="left")
    fig_line.add_annotation(x=0.02, y=12.5, text="CRITICAL", showarrow=False, font=dict(color="#EF4444", size=10, family="Outfit", weight="bold"), xref="paper", yref="y", align="left")

    # Complete scenario dotted trajectory line
    fig_line.add_trace(go.Scatter(
        x=all_timestamps,
        y=all_health_scores,
        mode='lines',
        line=dict(color="rgba(100, 116, 139, 0.4)", width=2, dash="dot"),
        hoverinfo='none',
        showlegend=False
    ))

    # Active trajectory line up to scrubber
    fig_line.add_trace(go.Scatter(
        x=all_timestamps[:st.session_state.step_idx + 1],
        y=all_health_scores[:st.session_state.step_idx + 1],
        mode='lines+markers',
        line=dict(color=risk_color, width=3.5, shape='spline'),
        marker=dict(size=6, color="#070D19", line=dict(color=risk_color, width=2))
    ))

    # Vertical scrubber cursor line
    if st.session_state.step_idx < len(all_timestamps):
        scrub_x = all_timestamps[st.session_state.step_idx]
        fig_line.add_vline(x=scrub_x, line_width=1.5, line_dash="dash", line_color="#F8FAFC")
        
        # Scrubber value tooltip
        fig_line.add_annotation(
            x=scrub_x, y=health_score,
            text=f"{health_score}",
            showarrow=True,
            arrowhead=0,
            arrowsize=0.3,
            ax=0, ay=-25,
            bordercolor=risk_color,
            borderpad=4,
            bgcolor="#0B132B",
            font=dict(color=risk_color, size=11, family="Outfit", weight="bold")
        )

    fig_line.update_layout(
        margin=dict(l=15, r=20, t=15, b=25),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.03)", 
            tickfont=dict(color="var(--text-muted)", size=11),
            showgrid=True,
            linecolor="rgba(255,255,255,0.05)",
            nticks=6
        ),
        yaxis=dict(
            range=[0, 100], 
            gridcolor="rgba(255,255,255,0.03)", 
            tickfont=dict(color="var(--text-muted)", size=11),
            showgrid=True,
            linecolor="rgba(255,255,255,0.05)"
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig_line, use_container_width=True, key="plotly_volatility_line", config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ================= SECTION 6: MOST CONTRIBUTING FACTORS =================
with bottom_right_col:
    st.markdown(
        """
        <div class="section-card" style="height: 100%;">
            <div class="section-header" style="justify-content: space-between;">
                <div style="display: flex; align-items: center;">
                    <div class="section-number">6</div>
                    <div class="section-title">MOST CONTRIBUTING FACTORS TO HEALTH SCORE REDUCTION</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    FEATURE_MAP = {
        "BF_Gas_Pressure": ("Gas Pressure Fluctuation", "Increasing pressure beyond normal range", flame_svg),
        "BF_Top_Temperature": ("Furnace Temperature", "Approaching upper operational limit", thermometer_svg),
        "BF_Cooling_Water_Flow": ("Cooling Water Flow", "Flow rate below optimal level", water_svg),
        "BF_CO": ("CO Concentration", "Rising CO level in top gas", cloud_svg_card),
        "BF_Blower_Vibration": ("Equipment Vibration", "Slight increase in vibration intensity", gear_svg_card),
        "CO_CO": ("CO Concentration", "Rising CO level in top gas", cloud_svg_card),
        "CO_Gas_Pressure": ("Gas Pressure Fluctuation", "Increasing pressure beyond normal range", flame_svg),
        "CO_Oven_Temperature": ("Furnace Temperature", "Approaching upper operational limit", thermometer_svg),
        "CO_Pusher_Vibration": ("Equipment Vibration", "Slight increase in vibration intensity", gear_svg_card)
    }

    if shap_list:
        sorted_shaps = sorted(shap_list, key=lambda x: abs(x.get("contribution", 0.0)), reverse=True)[:5]
        
        for s in sorted_shaps:
            feat_id = s.get("feature", "Sensor Variance")
            contrib_val = float(s.get("contribution", 0.0))
            
            title, subtitle, icon = FEATURE_MAP.get(feat_id, (feat_id.replace("_", " "), "Feature variance index", gear_svg_card))
            
            points_impact = contrib_val * 100.0
            points_impact = max(-20.0, min(0.0, points_impact))
            
            pct_width = min(100, int((abs(points_impact) / 20.0) * 100))
            
            if points_impact <= -15:
                bar_color = "#EF4444"
            elif points_impact <= -10:
                bar_color = "#F97316"
            elif points_impact <= -5:
                bar_color = "#F59E0B"
            else:
                bar_color = "#10B981"
                
            st.markdown(
                f"""
                <div class="factor-row">
                    <div class="factor-icon-box" style="color: {bar_color};">{icon}</div>
                    <div class="factor-title">{title}</div>
                    <div class="factor-subtitle">{subtitle}</div>
                    <div class="factor-bar-container">
                        <div class="factor-bar-bg">
                            <div class="factor-bar-fill" style="width: {pct_width}%; background-color: {bar_color};"></div>
                        </div>
                    </div>
                    <div class="factor-value" style="color: {bar_color};">{points_impact:+.1f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown(
            """
            <div class="axis-row" style="display: flex; justify-content: flex-end; width: 100%; margin-top: 4px; padding-right: 0;">
                <div style="width: 190px; display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); font-weight: 700; padding-right: 50px;">
                    <span>-20</span>
                    <span>-15</span>
                    <span>-10</span>
                    <span>-5</span>
                    <span>0</span>
                </div>
            </div>
            <div style="text-align: center; font-size: 10px; color: var(--text-muted); font-weight: 800; margin-top: 8px; letter-spacing: 0.5px; text-transform: uppercase;">IMPACT ON HEALTH SCORE</div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div style="font-size: 13px; color: var(--text-muted); text-align: center; padding-top: 50px;">All features operating normally within plant baseline limits.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# FOOTER BAR
# ------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="footer-bar">
        <div class="footer-item">
            <span>🗄️</span>
            <span>DATA SOURCE</span>
            <span class="footer-val">{current_scenario_dict['file']}</span>
        </div>
        <div class="footer-item">
            <span>📋</span>
            <span>SCENARIO STATUS</span>
            <span class="footer-val" style="color: var(--color-purple);">{st.session_state.active_scenario} active</span>
        </div>
        <div class="footer-item">
            <span>⚙️</span>
            <span>PIPELINE ENGINE</span>
            <span class="footer-val" style="color: var(--color-green);">Active Stream Simulation</span>
        </div>
        <div class="footer-item">
            <span>💻</span>
            <span>MODEL STATUS</span>
            <span class="footer-val" style="color: var(--color-purple);">XGBoost v1.3.2 Loaded</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Playback loop handling
if st.session_state.is_playing:
    time.sleep(1.2)
    st.session_state.step_idx = (st.session_state.step_idx + 1) % 50
    st.rerun()


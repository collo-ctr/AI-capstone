# ui_app.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from modules.agent import IntelligentAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner import TreatmentPlanner

# Page configuration
st.set_page_config(
    page_title="Healthcare Diagnostic Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #1a5276, #2e86c1);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .diagnosis-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f0f2f6;
        border-left: 5px solid #2e86c1;
        margin: 1rem 0;
    }
    .critical { border-left-color: #e74c3c; background: #fdedec; }
    .high { border-left-color: #e67e22; background: #fef5e7; }
    .medium { border-left-color: #f1c40f; background: #fef9e7; }
    .low { border-left-color: #2ecc71; background: #eafaf1; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1rem;
        color: #7f8c8d;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background: #2e86c1;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: #1a5276;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
    st.session_state.agent = None
    st.session_state.history = []

def initialize_system():
    """Initialize the healthcare diagnostic system."""
    with st.spinner("Initializing AI modules... Please wait..."):
        agent = IntelligentAgent()
        
        # Register all modules
        modules = [
            ("Knowledge Base", MedicalKnowledgeBase()),
            ("Bayesian Network", SimpleBayesianDiagnostics()),
            ("ML Classifier", MLDiagnosticClassifier()),
            ("Neural Network", NeuralDiagnosticModel()),
            ("Fuzzy Logic", FuzzySeverityAssessor()),
            ("Treatment Planner", TreatmentPlanner())
        ]
        
        for name, module in modules:
            try:
                agent.register_module(name, module)
            except Exception as e:
                st.warning(f"Could not register {name}: {e}")
        
        st.session_state.agent = agent
        st.session_state.system_initialized = True
    
    st.success("✅ System initialized successfully!")

def diagnose_patient(symptoms, age, temperature, heart_rate, blood_pressure):
    """Diagnose a patient."""
    patient = PatientPercept(
        patient_id=f"P{len(st.session_state.history)+1:04d}",
        symptoms=symptoms,
        age=age,
        temperature=temperature,
        heart_rate=heart_rate,
        blood_pressure=blood_pressure
    )
    
    agent = st.session_state.agent
    agent.perceive(patient)
    agent.think()
    report = agent.act()
    
    # Add to history
    st.session_state.history.append({
        'patient': patient,
        'report': report,
        'timestamp': datetime.now()
    })
    
    return report, patient

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital.png", width=80)
    st.title("🏥 Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Dashboard", "🔬 New Diagnosis", "📊 History", "📈 Analytics"]
    )
    
    st.divider()
    st.caption("🔬 AI Healthcare Diagnostic System")
    st.caption(f"📅 {datetime.now().strftime('%B %d, %Y')}")

# Main content
if not st.session_state.system_initialized:
    st.markdown("""
        <div class="main-header">
            <h1>🏥 Intelligent Healthcare Diagnostic Assistant</h1>
            <p>Powered by 6 AI Modules • Ensemble Diagnosis • Treatment Planning</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        ### 🚀 System Ready to Launch
        
        This system uses:
        - 🤖 Intelligent Agent (Coordinator)
        - 📋 Knowledge Base (Rules)
        - 📊 Bayesian Network (Probability)
        - 🧠 ML Classifier (Pattern Recognition)
        - 🔬 Neural Network (Deep Learning)
        - 🌐 Fuzzy Logic (Severity Assessment)
        - 📝 Treatment Planner (Action Planning)
        """)
        
        if st.button("🚀 Initialize System", use_container_width=True):
            initialize_system()
            st.rerun()

else:
    # ========================
    # PAGE: DASHBOARD
    # ========================
    if page == "🏠 Dashboard":
        st.markdown("""
            <div class="main-header">
                <h1>🏥 AI Healthcare Dashboard</h1>
                <p>Real-time diagnostic intelligence</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        
        total_patients = len(st.session_state.history)
        diagnoses = [h['report']['diagnosis'] for h in st.session_state.history] if total_patients > 0 else []
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3>👤 Total Patients</h3>
                    <div class="value">{}</div>
                </div>
            """.format(total_patients), unsafe_allow_html=True)
        
        with col2:
            unique_diagnoses = len(set(diagnoses))
            st.markdown("""
                <div class="metric-card">
                    <h3>🦠 Unique Diagnoses</h3>
                    <div class="value">{}</div>
                </div>
            """.format(unique_diagnoses), unsafe_allow_html=True)
        
        with col3:
            if diagnoses:
                most_common = max(set(diagnoses), key=diagnoses.count)
            else:
                most_common = "N/A"
            st.markdown("""
                <div class="metric-card">
                    <h3>🔬 Most Common</h3>
                    <div class="value">{}</div>
                </div>
            """.format(most_common), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
                <div class="metric-card">
                    <h3>⚙️ AI Modules</h3>
                    <div class="value">6</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        if total_patients > 0:
            # Recent diagnoses
            st.subheader("📋 Recent Diagnoses")
            recent = st.session_state.history[-5:][::-1]
            
            for entry in recent:
                report = entry['report']
                patient = entry['patient']
                
                urgency_class = {
                    'CRITICAL': 'critical',
                    'HIGH': 'high',
                    'MEDIUM': 'medium',
                    'LOW': 'low'
                }.get(report['urgency'], '')
                
                st.markdown(f"""
                    <div class="diagnosis-box {urgency_class}">
                        <strong>Patient:</strong> {patient.patient_id} &nbsp;|&nbsp;
                        <strong>Diagnosis:</strong> {report['diagnosis'].upper()} &nbsp;|&nbsp;
                        <strong>Confidence:</strong> {report['confidence']:.1%} &nbsp;|&nbsp;
                        <strong>Urgency:</strong> {report['urgency']}
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📊 View Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Symptoms:**", ", ".join(patient.symptoms))
                        st.write("**Vitals:**", f"Temp: {patient.temperature}°C, HR: {patient.heart_rate} BPM")
                    with col2:
                        st.write("**Recommendations:**")
                        for rec in report['recommendations']:
                            st.write(f"- {rec}")
        else:
            st.info("No patients diagnosed yet. Go to 'New Diagnosis' to get started!")

    # ========================
    # PAGE: NEW DIAGNOSIS
    # ========================
    elif page == "🔬 New Diagnosis":
        st.markdown("""
            <div class="main-header">
                <h1>🔬 New Patient Diagnosis</h1>
                <p>Enter patient information for AI-powered diagnosis</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("👤 Patient Information")
            
            patient_id = st.text_input("Patient ID", value=f"P{len(st.session_state.history)+1:04d}", disabled=True)
            
            # Symptoms - use multiselect
            all_symptoms = [
                'fever', 'cough', 'fatigue', 'headache', 'body_ache',
                'runny_nose', 'sneezing', 'sore_throat', 'rash',
                'loss_of_smell', 'shortness_of_breath', 'chest_pain',
                'nausea', 'vomiting', 'diarrhea', 'joint_pain',
                'chills', 'swollen_lymph_nodes'
            ]
            
            symptoms = st.multiselect(
                "Select Symptoms",
                options=all_symptoms,
                help="Select all symptoms the patient is experiencing"
            )
            
            # Vitals
            col1a, col2a, col3a = st.columns(3)
            with col1a:
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
            with col2a:
                temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
            with col3a:
                heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=200, value=80)
            
            blood_pressure = st.text_input("Blood Pressure", value="120/80", help="Format: systolic/diastolic")
            
            st.divider()
            
            if st.button("🩺 Diagnose Patient", use_container_width=True):
                if not symptoms:
                    st.error("⚠️ Please select at least one symptom.")
                else:
                    with st.spinner("🤖 AI analyzing patient data..."):
                        report, patient = diagnose_patient(
                            symptoms=symptoms,
                            age=age,
                            temperature=temperature,
                            heart_rate=heart_rate,
                            blood_pressure=blood_pressure
                        )
                        st.success("✅ Diagnosis complete!")
                        st.session_state['last_report'] = report
                        st.session_state['last_patient'] = patient
                        st.rerun()
        
        with col2:
            st.subheader("📋 Diagnosis Result")
            
            if 'last_report' in st.session_state:
                report = st.session_state['last_report']
                patient = st.session_state['last_patient']
                
                urgency_color = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }.get(report['urgency'], '⚪')
                
                st.markdown(f"""
                    <div class="diagnosis-box">
                        <h3>{urgency_color} Diagnosis: {report['diagnosis'].upper()}</h3>
                        <p><strong>Confidence:</strong> {report['confidence']:.1%}</p>
                        <p><strong>Urgency:</strong> {report['urgency']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📌 Recommendations", expanded=True):
                    for rec in report['recommendations']:
                        st.write(f"• {rec}")
                
                with st.expander("🔬 Module Results"):
                    for module, result in report['all_module_results'].items():
                        if 'error' in result:
                            st.warning(f"⚠️ {module}: Error")
                        else:
                            if module == 'FuzzyLogic':
                                severity = result.get('severity_label', 'N/A')
                                score = result.get('severity_score', 0)
                                st.write(f"**{module}:** Severity = {severity} ({score:.1f}/100)")
                            elif module == 'TreatmentPlanner':
                                steps = result.get('total_steps', 0)
                                duration = result.get('total_duration', 'N/A')
                                st.write(f"**{module}:** {steps} steps, Duration = {duration}")
                            else:
                                diag = result.get('diagnosis', 'N/A')
                                conf = result.get('confidence', 0)
                                st.write(f"**{module}:** {diag} ({conf:.1%})")
            else:
                st.info("Enter patient information and click 'Diagnose Patient'")

    # ========================
    # PAGE: HISTORY
    # ========================
    elif page == "📊 History":
        st.markdown("""
            <div class="main-header">
                <h1>📊 Patient History</h1>
                <p>Complete record of all diagnoses</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.history:
            # Create DataFrame
            data = []
            for entry in st.session_state.history[::-1]:
                report = entry['report']
                patient = entry['patient']
                data.append({
                    'Patient ID': patient.patient_id,
                    'Diagnosis': report['diagnosis'].upper(),
                    'Confidence': f"{report['confidence']:.1%}",
                    'Urgency': report['urgency'],
                    'Symptoms': ', '.join(patient.symptoms),
                    'Temperature': f"{patient.temperature}°C",
                    'Heart Rate': f"{patient.heart_rate} BPM",
                    'Time': entry['timestamp'].strftime('%H:%M:%S')
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            
            # Export option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name=f"patient_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No patient history available yet.")

    # ========================
    # PAGE: ANALYTICS
    # ========================
    elif page == "📈 Analytics":
        st.markdown("""
            <div class="main-header">
                <h1>📈 Analytics Dashboard</h1>
                <p>Visual insights from patient data</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.history:
            # Diagnosis distribution
            diagnoses = [h['report']['diagnosis'].upper() for h in st.session_state.history]
            urgency_levels = [h['report']['urgency'] for h in st.session_state.history]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Diagnosis Distribution")
                fig = go.Figure(data=[go.Pie(
                    labels=list(set(diagnoses)),
                    values=[diagnoses.count(d) for d in set(diagnoses)],
                    hole=0.3
                )])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🚨 Urgency Levels")
                fig = go.Figure(data=[go.Bar(
                    x=list(set(urgency_levels)),
                    y=[urgency_levels.count(u) for u in set(urgency_levels)],
                    marker_color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
                )])
                fig.update_layout(height=400, xaxis_title="Urgency", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            
            # Confidence distribution
            st.subheader("📈 Confidence Distribution")
            confidences = [h['report']['confidence'] * 100 for h in st.session_state.history]
            fig = go.Figure(data=[go.Histogram(
                x=confidences,
                nbinsx=20,
                marker_color='#2e86c1'
            )])
            fig.update_layout(
                height=300,
                xaxis_title="Confidence (%)",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.divider()
            st.subheader("📊 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Patients", len(st.session_state.history))
            with col2:
                avg_confidence = sum(h['report']['confidence'] for h in st.session_state.history) / len(st.session_state.history)
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            with col3:
                critical_cases = sum(1 for h in st.session_state.history if h['report']['urgency'] == 'CRITICAL')
                st.metric("Critical Cases", critical_cases)
            with col4:
                avg_age = sum(h['patient'].age for h in st.session_state.history) / len(st.session_state.history)
                st.metric("Avg Age", f"{avg_age:.1f}")
        else:
            st.info("No data available for analytics. Diagnose some patients first!")

# Footer
st.divider()
st.caption("🏥 Dedan Kimathi University of Science and Technology • CCS 3101: Introduction to AI • Capstone Project")

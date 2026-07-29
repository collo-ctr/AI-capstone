# app.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime
from modules.agent import IntelligentAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner import TreatmentPlanner


class HealthcareDiagnosticSystem:
    """
    The complete Intelligent Healthcare Diagnostic Assistant.
    Integrates all 7 AI modules into one unified system.
    """
    
    def __init__(self):
        print("=" * 70)
        print("🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC ASSISTANT")
        print("=" * 70)
        print("Initializing system...")
        
        # Create the agent
        self.agent = IntelligentAgent()
        
        # Register all modules
        self._register_modules()
        
        # Track patients
        self.patient_history = []
        
        print("\n✅ System ready! Waiting for patients...")
        print("=" * 70)
    
    def _register_modules(self):
        """Register all AI modules with the agent."""
        print("\n📋 Registering AI Modules...")
        
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
                self.agent.register_module(name, module)
                print(f"  ✅ Registered: {name}")
            except Exception as e:
                print(f"  ❌ Failed to register {name}: {e}")
        
        print(f"\n✅ Total modules registered: {len(self.agent.modules)}")
    
    def diagnose_patient(self, patient_data: dict) -> dict:
        """
        Diagnose a patient using all AI modules.
        
        Args:
            patient_data: Dictionary with patient information:
                - patient_id: str
                - symptoms: list of str
                - age: int
                - temperature: float (Celsius)
                - heart_rate: int (BPM)
                - blood_pressure: str (optional)
        
        Returns:
            dict: Complete diagnosis report
        """
        # Create PatientPercept object
        patient = PatientPercept(
            patient_id=patient_data.get('patient_id', f"P{len(self.patient_history)+1:04d}"),
            symptoms=patient_data.get('symptoms', []),
            age=patient_data.get('age', 30),
            temperature=patient_data.get('temperature', 37.0),
            heart_rate=patient_data.get('heart_rate', 80),
            blood_pressure=patient_data.get('blood_pressure', "120/80")
        )
        
        print(f"\n👤 NEW PATIENT: {patient.patient_id}")
        print(f"   Symptoms: {', '.join(patient.symptoms)}")
        print(f"   Age: {patient.age}, Temp: {patient.temperature}°C, HR: {patient.heart_rate} BPM")
        print("-" * 70)
        
        # Process through agent
        try:
            self.agent.perceive(patient)
            self.agent.think()
            report = self.agent.act()
            
            # Add to history
            self.patient_history.append({
                'patient': patient,
                'report': report,
                'timestamp': datetime.now().isoformat()
            })
            
            return report
            
        except Exception as e:
            print(f"❌ Error diagnosing patient: {e}")
            return {
                'patient_id': patient.patient_id,
                'diagnosis': 'ERROR',
                'confidence': 0.0,
                'urgency': 'UNKNOWN',
                'error': str(e)
            }
    
    def print_report(self, report: dict):
        """Pretty print a diagnosis report."""
        print("\n" + "=" * 70)
        print("📋 FINAL DIAGNOSIS REPORT")
        print("=" * 70)
        print(f"Patient ID: {report['patient_id']}")
        print(f"Diagnosis: {report['diagnosis'].upper()}")
        print(f"Confidence: {report['confidence']:.2%}")
        print(f"Urgency: {report['urgency']}")
        print("-" * 70)
        
        print("\n📌 Recommendations:")
        for i, rec in enumerate(report.get('recommendations', []), 1):
            print(f"   {i}. {rec}")
        
        print("\n🔬 Module Results:")
        for module, result in report.get('all_module_results', {}).items():
            if 'error' in result:
                print(f"   ⚠️  {module}: ERROR")
            else:
                diag = result.get('diagnosis', 'N/A')
                conf = result.get('confidence', 0)
                if module == 'FuzzyLogic':
                    severity = result.get('severity_label', 'N/A')
                    score = result.get('severity_score', 0)
                    print(f"   ✅ {module}: Severity={severity} ({score:.1f}/100)")
                elif module == 'TreatmentPlanner':
                    steps = result.get('total_steps', 0)
                    duration = result.get('total_duration', 'N/A')
                    print(f"   ✅ {module}: {steps} steps, Duration={duration}")
                else:
                    print(f"   ✅ {module}: {diag} ({conf:.2%})")
        
        # Vitals
        vitals = report.get('vitals', {})
        if vitals:
            print("\n📊 Patient Vitals:")
            print(f"   Temperature: {vitals.get('temperature', 'N/A')}°C")
            print(f"   Heart Rate: {vitals.get('heart_rate', 'N/A')} BPM")
            print(f"   Blood Pressure: {vitals.get('blood_pressure', 'N/A')}")
        
        print("\n" + "=" * 70)
    
    def run_demo(self):
        """Run a demonstration with sample patients."""
        print("\n" + "=" * 70)
        print("🎯 RUNNING DEMO WITH SAMPLE PATIENTS")
        print("=" * 70)
        
        # Sample patients
        sample_patients = [
            {
                'patient_id': 'DEMO001',
                'symptoms': ['fever', 'cough', 'fatigue', 'loss_of_smell'],
                'age': 45,
                'temperature': 38.9,
                'heart_rate': 98,
                'blood_pressure': '130/85'
            },
            {
                'patient_id': 'DEMO002',
                'symptoms': ['runny_nose', 'sneezing', 'sore_throat'],
                'age': 28,
                'temperature': 37.5,
                'heart_rate': 75,
                'blood_pressure': '115/75'
            },
            {
                'patient_id': 'DEMO003',
                'symptoms': ['fever', 'rash', 'joint_pain', 'headache', 'fatigue'],
                'age': 32,
                'temperature': 39.2,
                'heart_rate': 105,
                'blood_pressure': '125/82'
            },
            {
                'patient_id': 'DEMO004',
                'symptoms': ['cough', 'shortness_of_breath', 'chest_pain', 'fever'],
                'age': 58,
                'temperature': 39.5,
                'heart_rate': 115,
                'blood_pressure': '145/90'
            }
        ]
        
        for i, patient_data in enumerate(sample_patients, 1):
            print(f"\n{'─' * 70}")
            print(f"PATIENT {i} OF {len(sample_patients)}")
            print('─' * 70)
            
            report = self.diagnose_patient(patient_data)
            self.print_report(report)
            
            if i < len(sample_patients):
                input("\nPress Enter to continue to next patient...")
    
    def interactive_mode(self):
        """Run in interactive mode for user input."""
        print("\n" + "=" * 70)
        print("💬 INTERACTIVE MODE")
        print("=" * 70)
        print("\nEnter patient details or type 'quit' to exit")
        
        patient_count = 0
        
        while True:
            print("\n" + "-" * 40)
            
            # Get patient ID
            patient_id = input("Patient ID (or 'quit'): ").strip()
            if patient_id.lower() == 'quit':
                break
            
            # Get symptoms
            symptoms_input = input("Symptoms (comma separated, e.g., fever,cough,fatigue): ").strip()
            symptoms = [s.strip().lower() for s in symptoms_input.split(',') if s.strip()]
            
            # Get age
            try:
                age = int(input("Age: ").strip())
            except:
                age = 30
            
            # Get temperature
            try:
                temp = float(input("Temperature (°C): ").strip())
            except:
                temp = 37.0
            
            # Get heart rate
            try:
                hr = int(input("Heart Rate (BPM): ").strip())
            except:
                hr = 80
            
            # Get blood pressure
            bp = input("Blood Pressure (e.g., 120/80): ").strip() or "120/80"
            
            patient_data = {
                'patient_id': patient_id,
                'symptoms': symptoms,
                'age': age,
                'temperature': temp,
                'heart_rate': hr,
                'blood_pressure': bp
            }
            
            report = self.diagnose_patient(patient_data)
            self.print_report(report)
            
            patient_count += 1
        
        print(f"\n✅ Processed {patient_count} patients in this session")
        print("Goodbye! 👋")


def main():
    """Main entry point for the application."""
    print("=" * 70)
    print("   DEDAN KIMATHI UNIVERSITY OF SCIENCE AND TECHNOLOGY")
    print("   CCS 3101: INTRODUCTION TO ARTIFICIAL INTELLIGENCE")
    print("   CAPSTONE PROJECT: INTELLIGENT HEALTHCARE DIAGNOSTIC ASSISTANT")
    print("=" * 70)
    
    # Create the system
    system = HealthcareDiagnosticSystem()
    
    # Ask user for mode
    print("\n" + "=" * 70)
    print("SELECT MODE:")
    print("  1. Run Demo (sample patients)")
    print("  2. Interactive Mode (enter your own patients)")
    print("  3. Quick Test (single patient)")
    print("=" * 70)
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        system.run_demo()
    
    elif choice == '2':
        system.interactive_mode()
    
    elif choice == '3':
        # Quick test
        print("\n" + "=" * 70)
        print("🔬 QUICK TEST")
        print("=" * 70)
        
        patient_data = {
            'patient_id': 'QUICK001',
            'symptoms': ['fever', 'cough', 'fatigue', 'loss_of_smell'],
            'age': 45,
            'temperature': 38.9,
            'heart_rate': 98
        }
        
        report = system.diagnose_patient(patient_data)
        system.print_report(report)
    
    else:
        print("Invalid choice. Running demo by default...")
        system.run_demo()
    
    print("\n✅ System execution complete!")


if __name__ == "__main__":
    main()

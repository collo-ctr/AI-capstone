# modules/agent.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatientPercept:
    """Patient data structure - the 'intake form' for each patient"""
    patient_id: str
    symptoms: List[str]
    age: int
    temperature: float  # Celsius
    heart_rate: int     # BPM
    blood_pressure: str = "120/80"
    
    def __post_init__(self):
        # Clean symptoms: convert to lowercase and replace spaces with underscores
        self.symptoms = [s.lower().replace(" ", "_") for s in self.symptoms]


class IntelligentAgent:
    """
    The brain and coordinator of the entire system.
    Implements a Model-Based, Goal-Based Agent using PEAS framework.
    """
    
    def __init__(self):
        # Internal state
        self.state = "IDLE"
        self.memory = {}  # Store patient history
        self.modules = {}  # Registered AI modules
        self.current_patient = None
        self.diagnosis_results = {}
        
        logger.info("IntelligentAgent initialized")
    
    def register_module(self, name: str, module):
        """
        Register an AI module (specialist) to consult during diagnosis.
        
        Args:
            name: Module identifier (e.g., "KnowledgeBase", "BayesianNet")
            module: The module instance with an analyze() method
        """
        if not hasattr(module, 'analyze'):
            raise ValueError(f"Module {name} must have an analyze() method")
        self.modules[name] = module
        logger.info(f"Registered module: {name}")
    
    def perceive(self, patient: PatientPercept):
        """
        Step 1: Receive patient data and store in memory.
        
        Args:
            patient: PatientPercept object containing symptoms and vitals
        """
        self.current_patient = patient
        self.state = "COLLECTING"
        
        # Store in memory
        self.memory[patient.patient_id] = {
            "patient": patient,
            "timestamp": datetime.now().isoformat(),
            "symptoms": patient.symptoms,
            "vitals": {
                "temperature": patient.temperature,
                "heart_rate": patient.heart_rate,
                "blood_pressure": patient.blood_pressure
            }
        }
        
        logger.info(f"Perceived patient {patient.patient_id} with {len(patient.symptoms)} symptoms")
        print(f"[collecting_symptoms] Perceived patient {patient.patient_id} with {len(patient.symptoms)} symptoms")
        
        self.state = "ANALYZING"
        return self.memory[patient.patient_id]
    
    def think(self):
        """
        Step 2: Consult all registered modules and collect their diagnoses.
        
        Returns:
            Dictionary of results from each module
        """
        if self.current_patient is None:
            raise ValueError("No patient perceived. Call perceive() first.")
        
        self.state = "THINKING"
        patient = self.current_patient
        results = {}
        
        logger.info(f"Consulting {len(self.modules)} modules for patient {patient.patient_id}")
        
        for module_name, module in self.modules.items():
            try:
                result = module.analyze(patient)
                results[module_name] = result
                logger.debug(f"Module {module_name}: {result}")
            except Exception as e:
                logger.error(f"Module {module_name} failed: {e}")
                results[module_name] = {"error": str(e), "diagnosis": "unknown", "confidence": 0.0}
        
        self.diagnosis_results = results
        self.state = "DECIDING"
        return results
    
    def act(self):
        """
        Step 3: Combine module opinions, determine urgency, and generate report.
        
        Returns:
            Final structured report with diagnosis, confidence, urgency, and recommendations
        """
        if not self.diagnosis_results:
            raise ValueError("No diagnosis results. Call think() first.")
        
        self.state = "ACTING"
        patient = self.current_patient
        
        # Collect all diagnoses and confidences
        diagnoses = []
        confidences = []
        
        for module_name, result in self.diagnosis_results.items():
            if "error" in result:
                continue
            diagnoses.append(result.get("diagnosis", "unknown"))
            confidences.append(result.get("confidence", 0.0))
        
        # Find the most common diagnosis (ensemble voting)
        from collections import Counter
        diagnosis_counter = Counter(diagnoses)
        top_diagnosis = diagnosis_counter.most_common(1)[0][0]
        
        # Average confidence for the top diagnosis
        top_confidences = [c for i, c in enumerate(confidences) if diagnoses[i] == top_diagnosis]
        avg_confidence = sum(top_confidences) / len(top_confidences) if top_confidences else 0.0
        
        # Determine urgency based on vitals
        urgency_level = self._determine_urgency(patient)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(top_diagnosis, urgency_level)
        
        # Build final report
        report = {
            "patient_id": patient.patient_id,
            "diagnosis": top_diagnosis,
            "confidence": round(avg_confidence, 3),
            "urgency": urgency_level,
            "recommendations": recommendations,
            "all_module_results": self.diagnosis_results,
            "vitals": {
                "temperature": patient.temperature,
                "heart_rate": patient.heart_rate,
                "blood_pressure": patient.blood_pressure
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self.state = "COMPLETE"
        logger.info(f"Report generated for patient {patient.patient_id}: {top_diagnosis} ({avg_confidence:.2%})")
        
        return report
    
    def _determine_urgency(self, patient: PatientPercept) -> str:
        """Determine urgency level based on vital signs."""
        urgency = "LOW"
        
        # Temperature urgency
        if patient.temperature >= 39.5:
            urgency = "CRITICAL"
        elif patient.temperature >= 38.5:
            urgency = "HIGH"
        elif patient.temperature >= 37.5:
            urgency = "MEDIUM"
        
        # Heart rate urgency (override)
        if patient.heart_rate >= 120:
            urgency = "CRITICAL"
        elif patient.heart_rate >= 100:
            if urgency == "LOW":
                urgency = "MEDIUM"
        
        return urgency
    
    def _generate_recommendations(self, diagnosis: str, urgency: str) -> List[str]:
        """Generate treatment recommendations based on diagnosis and urgency."""
        recommendations = []
        
        # General recommendations
        if urgency == "CRITICAL":
            recommendations.append("Immediate emergency care required")
            recommendations.append("Call emergency services")
        elif urgency == "HIGH":
            recommendations.append("See a doctor within 24 hours")
        
        # Diagnosis-specific recommendations
        if diagnosis == "covid19":
            recommendations.extend([
                "Isolate for 14 days",
                "PCR test recommended",
                "Monitor oxygen levels"
            ])
        elif diagnosis == "flu":
            recommendations.extend([
                "Rest and stay hydrated",
                "Over-the-counter fever reducers",
                "See doctor if symptoms worsen"
            ])
        elif diagnosis == "dengue":
            recommendations.extend([
                "Monitor for bleeding",
                "Hydration therapy recommended",
                "Seek immediate care for severe symptoms"
            ])
        elif diagnosis == "common_cold":
            recommendations.extend([
                "Rest and stay hydrated",
                "Over-the-counter cold remedies",
                "Symptoms should improve in 7-10 days"
            ])
        else:
            recommendations.append("Consult a healthcare professional")
        
        return recommendations
    
    def get_state(self) -> str:
        """Return the current agent state."""
        return self.state
    
    def reset(self):
        """Reset the agent for a new patient."""
        self.state = "IDLE"
        self.current_patient = None
        self.diagnosis_results = {}
        logger.info("Agent reset")


# Quick test to verify the agent works
if __name__ == "__main__":
    print("Testing IntelligentAgent...")
    
    # Create a mock module for testing
    class MockDiagnosticModule:
        def analyze(self, patient):
            return {
                "module": "MockModule",
                "diagnosis": "flu",
                "confidence": 0.85
            }
    
    agent = IntelligentAgent()
    
    # Register a mock module for testing
    agent.register_module("MockModule", MockDiagnosticModule())
    
    # Create a test patient
    patient = PatientPercept(
        patient_id="P001",
        symptoms=["fever", "cough", "fatigue", "loss of smell"],
        age=34,
        temperature=38.9,
        heart_rate=98
    )
    
    # Test the agent
    print("\nStep 1: Perceive patient")
    agent.perceive(patient)
    
    print("\nStep 2: Think (consult modules)")
    print("Consulting registered modules...")
    results = agent.think()
    print(f"Results from modules: {list(results.keys())}")
    
    print("\nStep 3: Act (generate report)")
    report = agent.act()
    print(f"\nFinal Report:")
    print(f"  Patient: {report['patient_id']}")
    print(f"  Diagnosis: {report['diagnosis']}")
    print(f"  Confidence: {report['confidence']:.2%}")
    print(f"  Urgency: {report['urgency']}")
    print(f"  Recommendations: {report['recommendations']}")
    
    print("\n✓ Agent test passed!")

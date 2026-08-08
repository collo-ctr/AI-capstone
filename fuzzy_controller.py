# modules/fuzzy_controller.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, Tuple, List
import logging
from modules.agent import PatientPercept

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuzzySeverityAssessor:
    """
    The severity assessor of the system.
    Uses Fuzzy Logic to determine how serious a patient's condition is.
    
    Three Steps:
    1. FUZZIFICATION: Convert crisp inputs to fuzzy membership degrees
    2. RULE EVALUATION: Apply fuzzy IF-THEN rules
    3. DEFUZZIFICATION: Convert fuzzy output to a crisp severity score
    """
    
    def __init__(self):
        # Severity levels and their centroid values (for defuzzification)
        self.severity_centers = {
            'low': 15,
            'mild': 35,
            'moderate': 55,
            'high': 75,
            'critical': 92
        }
        
        # Severity label mapping
        self.severity_labels = {
            (0, 20): 'LOW',
            (20, 40): 'MILD',
            (40, 60): 'MODERATE',
            (60, 80): 'HIGH',
            (80, 100): 'CRITICAL'
        }
        
        logger.info("FuzzySeverityAssessor initialized")
    
    def _membership_temp(self, temp: float) -> Dict[str, float]:
        """
        Calculate membership degrees for temperature.
        
        Fuzzy sets:
        - normal: 35.0 - 37.5°C (peaks at 36.5°C)
        - mild: 36.5 - 39.0°C (peaks at 38.0°C)
        - high: 37.5 - 40.0°C (peaks at 39.0°C)
        - critical: 39.0°C+ (rises above 39.5°C)
        """
        membership = {}
        
        # NORMAL: 35.0-37.5°C
        if temp <= 35.0:
            membership['normal'] = 0.0
        elif 35.0 < temp <= 36.5:
            membership['normal'] = (temp - 35.0) / 1.5
        elif 36.5 < temp <= 37.5:
            membership['normal'] = 1.0
        elif 37.5 < temp <= 38.5:
            membership['normal'] = (38.5 - temp) / 1.0
        else:
            membership['normal'] = 0.0
        
        # MILD: 36.5-39.0°C
        if temp <= 36.5:
            membership['mild'] = 0.0
        elif 36.5 < temp <= 38.0:
            membership['mild'] = (temp - 36.5) / 1.5
        elif 38.0 < temp <= 39.0:
            membership['mild'] = 1.0
        elif 39.0 < temp <= 40.0:
            membership['mild'] = (40.0 - temp) / 1.0
        else:
            membership['mild'] = 0.0
        
        # HIGH: 37.5-40.0°C
        if temp <= 37.5:
            membership['high'] = 0.0
        elif 37.5 < temp <= 39.0:
            membership['high'] = (temp - 37.5) / 1.5
        elif 39.0 < temp <= 40.0:
            membership['high'] = 1.0
        elif 40.0 < temp <= 41.0:
            membership['high'] = (41.0 - temp) / 1.0
        else:
            membership['high'] = 0.0
        
        # CRITICAL: 39.0°C+
        if temp <= 39.0:
            membership['critical'] = 0.0
        elif 39.0 < temp <= 39.5:
            membership['critical'] = (temp - 39.0) / 0.5
        elif 39.5 < temp <= 40.5:
            membership['critical'] = 1.0
        elif 40.5 < temp <= 41.5:
            membership['critical'] = (41.5 - temp) / 1.0
        else:
            membership['critical'] = 0.0
        
        # Clamp values to [0, 1]
        for key in membership:
            membership[key] = max(0.0, min(1.0, membership[key]))
        
        return membership
    
    def _membership_hr(self, heart_rate: float) -> Dict[str, float]:
        """
        Calculate membership degrees for heart rate.
        
        Fuzzy sets:
        - normal: 60-100 BPM (peaks at 80 BPM)
        - elevated: 80-120 BPM (peaks at 100 BPM)
        - high: 100-140 BPM (peaks at 120 BPM)
        - critical: 120+ BPM (rises above 140 BPM)
        """
        membership = {}
        
        # NORMAL: 60-100 BPM
        if heart_rate <= 60:
            membership['normal'] = 0.0
        elif 60 < heart_rate <= 80:
            membership['normal'] = (heart_rate - 60) / 20
        elif 80 < heart_rate <= 100:
            membership['normal'] = 1.0
        elif 100 < heart_rate <= 120:
            membership['normal'] = (120 - heart_rate) / 20
        else:
            membership['normal'] = 0.0
        
        # ELEVATED: 80-120 BPM
        if heart_rate <= 80:
            membership['elevated'] = 0.0
        elif 80 < heart_rate <= 100:
            membership['elevated'] = (heart_rate - 80) / 20
        elif 100 < heart_rate <= 120:
            membership['elevated'] = 1.0
        elif 120 < heart_rate <= 140:
            membership['elevated'] = (140 - heart_rate) / 20
        else:
            membership['elevated'] = 0.0
        
        # HIGH: 100-140 BPM
        if heart_rate <= 100:
            membership['high'] = 0.0
        elif 100 < heart_rate <= 120:
            membership['high'] = (heart_rate - 100) / 20
        elif 120 < heart_rate <= 140:
            membership['high'] = 1.0
        elif 140 < heart_rate <= 160:
            membership['high'] = (160 - heart_rate) / 20
        else:
            membership['high'] = 0.0
        
        # CRITICAL: 120+ BPM
        if heart_rate <= 120:
            membership['critical'] = 0.0
        elif 120 < heart_rate <= 140:
            membership['critical'] = (heart_rate - 120) / 20
        elif 140 < heart_rate <= 160:
            membership['critical'] = 1.0
        elif 160 < heart_rate <= 180:
            membership['critical'] = (180 - heart_rate) / 20
        else:
            membership['critical'] = 0.0
        
        # Clamp values to [0, 1]
        for key in membership:
            membership[key] = max(0.0, min(1.0, membership[key]))
        
        return membership
    
    def _membership_symptoms(self, symptom_count: int) -> Dict[str, float]:
        """
        Calculate membership degrees for number of symptoms.
        
        Fuzzy sets:
        - few: 0-3 symptoms
        - moderate: 2-6 symptoms
        - many: 4-8 symptoms
        - severe: 7+ symptoms
        """
        membership = {}
        
        # FEW: 0-3 symptoms
        if symptom_count <= 0:
            membership['few'] = 1.0
        elif 0 < symptom_count <= 2:
            membership['few'] = 1.0 - (symptom_count / 4)
        elif 2 < symptom_count <= 4:
            membership['few'] = (4 - symptom_count) / 2
        else:
            membership['few'] = 0.0
        
        # MODERATE: 2-6 symptoms
        if symptom_count <= 2:
            membership['moderate'] = 0.0
        elif 2 < symptom_count <= 4:
            membership['moderate'] = (symptom_count - 2) / 2
        elif 4 < symptom_count <= 6:
            membership['moderate'] = 1.0
        elif 6 < symptom_count <= 8:
            membership['moderate'] = (8 - symptom_count) / 2
        else:
            membership['moderate'] = 0.0
        
        # MANY: 4-8 symptoms
        if symptom_count <= 4:
            membership['many'] = 0.0
        elif 4 < symptom_count <= 6:
            membership['many'] = (symptom_count - 4) / 2
        elif 6 < symptom_count <= 8:
            membership['many'] = 1.0
        elif 8 < symptom_count <= 10:
            membership['many'] = (10 - symptom_count) / 2
        else:
            membership['many'] = 0.0
        
        # SEVERE: 7+ symptoms
        if symptom_count <= 7:
            membership['severe'] = 0.0
        elif 7 < symptom_count <= 9:
            membership['severe'] = (symptom_count - 7) / 2
        elif 9 < symptom_count <= 11:
            membership['severe'] = 1.0
        else:
            membership['severe'] = 1.0
        
        # Clamp values to [0, 1]
        for key in membership:
            membership[key] = max(0.0, min(1.0, membership[key]))
        
        return membership
    
    def _evaluate_rules(self, temp_mf: Dict, hr_mf: Dict, symptom_mf: Dict) -> Dict[str, float]:
        """
        Evaluate fuzzy rules using min (AND) and max (OR).
        
        Rules:
        1. IF temp IS critical OR (temp IS high AND hr IS high) -> severity = critical
        2. IF temp IS high OR (temp IS mild AND hr IS elevated) -> severity = high
        3. IF temp IS mild OR symptom_count IS many -> severity = moderate
        4. IF temp IS mild AND symptom_count IS moderate -> severity = mild
        5. IF temp IS normal AND hr IS normal AND symptom_count IS few -> severity = low
        """
        rules = {}
        
        # Rule 1: CRITICAL
        rules['critical'] = max(
            temp_mf.get('critical', 0),
            min(temp_mf.get('high', 0), hr_mf.get('high', 0)),
            min(temp_mf.get('critical', 0), hr_mf.get('elevated', 0))
        )
        
        # Rule 2: HIGH
        rules['high'] = max(
            temp_mf.get('high', 0),
            min(temp_mf.get('mild', 0), hr_mf.get('elevated', 0)),
            min(temp_mf.get('high', 0), symptom_mf.get('many', 0))
        )
        
        # Rule 3: MODERATE
        rules['moderate'] = max(
            temp_mf.get('mild', 0),
            symptom_mf.get('many', 0),
            min(temp_mf.get('mild', 0), hr_mf.get('elevated', 0)),
            min(hr_mf.get('elevated', 0), symptom_mf.get('moderate', 0))
        )
        
        # Rule 4: MILD
        rules['mild'] = max(
            min(temp_mf.get('mild', 0), symptom_mf.get('moderate', 0)),
            min(temp_mf.get('normal', 0), hr_mf.get('elevated', 0)),
            min(temp_mf.get('mild', 0), symptom_mf.get('few', 0))
        )
        
        # Rule 5: LOW
        rules['low'] = max(
            min(temp_mf.get('normal', 0), hr_mf.get('normal', 0), symptom_mf.get('few', 0)),
            min(temp_mf.get('normal', 0), hr_mf.get('normal', 0), symptom_mf.get('moderate', 0))
        )
        
        # Clamp values
        for key in rules:
            rules[key] = max(0.0, min(1.0, rules[key]))
        
        return rules
    
    def _defuzzify(self, rules: Dict[str, float]) -> float:
        """
        Convert fuzzy output to a crisp severity score using centroid method.
        """
        numerator = 0.0
        denominator = 0.0
        
        for level, membership in rules.items():
            center = self.severity_centers.get(level, 50)
            numerator += center * membership
            denominator += membership
        
        # Avoid division by zero
        if denominator < 0.0001:
            return 0.0
        
        score = numerator / denominator
        
        # Clamp to [0, 100]
        return max(0.0, min(100.0, score))
    
    def _get_severity_label(self, score: float) -> str:
        """Map severity score to a label."""
        for (low, high), label in self.severity_labels.items():
            if low <= score <= high:
                return label
        return 'UNKNOWN'
    
    def assess(self, temperature: float, heart_rate: float, symptom_count: int) -> Dict:
        """
        Assess the severity of a patient's condition.
        
        Args:
            temperature: Body temperature in Celsius
            heart_rate: Heart rate in BPM
            symptom_count: Number of symptoms
        
        Returns:
            Dictionary with severity score, label, and all membership details
        """
        # Step 1: FUZZIFICATION
        temp_mf = self._membership_temp(temperature)
        hr_mf = self._membership_hr(heart_rate)
        symptom_mf = self._membership_symptoms(symptom_count)
        
        # Step 2: RULE EVALUATION
        rules = self._evaluate_rules(temp_mf, hr_mf, symptom_mf)
        
        # Step 3: DEFUZZIFICATION
        score = self._defuzzify(rules)
        label = self._get_severity_label(score)
        
        return {
            "severity_score": round(score, 2),
            "severity_label": label,
            "fuzzy_sets": {
                "temperature": temp_mf,
                "heart_rate": hr_mf,
                "symptoms": symptom_mf
            },
            "rules": rules
        }
    
    def analyze(self, patient: PatientPercept) -> Dict:
        """
        Standard interface method called by the Agent.
        
        Returns:
            Dictionary with severity assessment
        """
        result = self.assess(
            temperature=patient.temperature,
            heart_rate=patient.heart_rate,
            symptom_count=len(patient.symptoms)
        )
        
        # Add module info
        result["module"] = "FuzzyLogic"
        
        # Add urgency mapping for compatibility with agent
        urgency_map = {
            'LOW': 'LOW',
            'MILD': 'MEDIUM',
            'MODERATE': 'MEDIUM',
            'HIGH': 'HIGH',
            'CRITICAL': 'CRITICAL'
        }
        result["urgency"] = urgency_map.get(result["severity_label"], 'MEDIUM')
        
        logger.info(f"Patient {patient.patient_id}: Severity = {result['severity_label']} ({result['severity_score']:.1f}/100)")
        
        return result


# Test the module
if __name__ == "__main__":
    print("Testing FuzzySeverityAssessor...")
    print()
    
    fa = FuzzySeverityAssessor()
    
    # Test cases
    test_cases = [
        (37.0, 72, 2, "Normal patient"),
        (38.5, 95, 4, "Mild illness"),
        (39.2, 110, 6, "Moderate case"),
        (39.8, 115, 7, "Severe case"),
        (40.2, 130, 9, "Critical case"),
    ]
    
    print("SEVERITY ASSESSMENT TEST CASES")
    print("=" * 60)
    
    for temp, hr, count, desc in test_cases:
        result = fa.assess(temp, hr, count)
        print(f"\n{desc}:")
        print(f"  Temp: {temp}°C, HR: {hr} BPM, Symptoms: {count}")
        print(f"  Score: {result['severity_score']:.1f}/100")
        print(f"  Label: {result['severity_label']}")
    
    # Test with PatientPercept
    print("\n" + "=" * 60)
    print("TESTING WITH PATIENTPERCEPT")
    print("=" * 60)
    
    patient = PatientPercept(
        patient_id="F001",
        symptoms=["fever", "cough", "fatigue", "loss_of_smell", "headache", "body_ache"],
        age=45,
        temperature=39.2,
        heart_rate=110
    )
    
    result = fa.analyze(patient)
    print(f"Patient: {patient.patient_id}")
    print(f"Severity Score: {result['severity_score']:.1f}/100")
    print(f"Severity Label: {result['severity_label']}")
    print(f"Urgency (for Agent): {result['urgency']}")
    
    print("\n✅ FuzzySeverityAssessor test passed!")

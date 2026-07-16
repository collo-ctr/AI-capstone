# modules/bayesian_net.py

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from typing import List, Dict, Tuple, Optional
import logging
from modules.agent import PatientPercept

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleBayesianDiagnostics:
    """
    The statistical probability calculator of the system.
    Uses Naive Bayes to reason under uncertainty.
    
    P(Disease | Symptoms) ∝ P(Disease) × P(Symptom1|Disease) × P(Symptom2|Disease) × ...
    """
    
    def __init__(self):
        # Prior probabilities - how common each disease is in the general population
        self.priors = {
            'flu': 0.15,           # 15% of sick patients have flu
            'covid19': 0.08,       # 8% have COVID-19
            'common_cold': 0.30,   # 30% have common cold
            'dengue': 0.05,        # 5% have dengue
            'strep_throat': 0.10,  # 10% have strep throat
            'allergy': 0.20,       # 20% have allergies
            'pneumonia': 0.07,     # 7% have pneumonia
            'bronchitis': 0.05     # 5% have bronchitis
        }
        
        # Likelihood tables - probability of each symptom given a disease
        # P(Symptom | Disease)
        self.likelihoods = {
            'flu': {
                'fever': 0.90,
                'cough': 0.85,
                'fatigue': 0.88,
                'headache': 0.75,
                'body_ache': 0.80,
                'runny_nose': 0.40,
                'sneezing': 0.30,
                'sore_throat': 0.60,
                'rash': 0.05,
                'loss_of_smell': 0.10,
                'shortness_of_breath': 0.15,
                'chest_pain': 0.10,
                'nausea': 0.35,
                'vomiting': 0.20,
                'diarrhea': 0.15,
                'joint_pain': 0.55,
                'chills': 0.70,
                'swollen_lymph_nodes': 0.20
            },
            'covid19': {
                'fever': 0.85,
                'cough': 0.80,
                'fatigue': 0.75,
                'headache': 0.50,
                'body_ache': 0.45,
                'runny_nose': 0.35,
                'sneezing': 0.20,
                'sore_throat': 0.40,
                'rash': 0.10,
                'loss_of_smell': 0.70,
                'shortness_of_breath': 0.45,
                'chest_pain': 0.30,
                'nausea': 0.25,
                'vomiting': 0.15,
                'diarrhea': 0.20,
                'joint_pain': 0.30,
                'chills': 0.55,
                'swollen_lymph_nodes': 0.15
            },
            'common_cold': {
                'fever': 0.30,
                'cough': 0.70,
                'fatigue': 0.40,
                'headache': 0.35,
                'body_ache': 0.25,
                'runny_nose': 0.85,
                'sneezing': 0.80,
                'sore_throat': 0.70,
                'rash': 0.05,
                'loss_of_smell': 0.10,
                'shortness_of_breath': 0.05,
                'chest_pain': 0.05,
                'nausea': 0.10,
                'vomiting': 0.05,
                'diarrhea': 0.05,
                'joint_pain': 0.10,
                'chills': 0.20,
                'swollen_lymph_nodes': 0.10
            },
            'dengue': {
                'fever': 0.98,
                'cough': 0.30,
                'fatigue': 0.85,
                'headache': 0.90,
                'body_ache': 0.85,
                'runny_nose': 0.15,
                'sneezing': 0.10,
                'sore_throat': 0.20,
                'rash': 0.75,
                'loss_of_smell': 0.05,
                'shortness_of_breath': 0.15,
                'chest_pain': 0.10,
                'nausea': 0.50,
                'vomiting': 0.40,
                'diarrhea': 0.20,
                'joint_pain': 0.85,
                'chills': 0.60,
                'swollen_lymph_nodes': 0.30
            },
            'strep_throat': {
                'fever': 0.70,
                'cough': 0.30,
                'fatigue': 0.50,
                'headache': 0.40,
                'body_ache': 0.35,
                'runny_nose': 0.20,
                'sneezing': 0.15,
                'sore_throat': 0.95,
                'rash': 0.10,
                'loss_of_smell': 0.05,
                'shortness_of_breath': 0.05,
                'chest_pain': 0.05,
                'nausea': 0.20,
                'vomiting': 0.15,
                'diarrhea': 0.05,
                'joint_pain': 0.15,
                'chills': 0.30,
                'swollen_lymph_nodes': 0.60
            },
            'allergy': {
                'fever': 0.05,
                'cough': 0.30,
                'fatigue': 0.20,
                'headache': 0.15,
                'body_ache': 0.05,
                'runny_nose': 0.90,
                'sneezing': 0.95,
                'sore_throat': 0.30,
                'rash': 0.25,
                'loss_of_smell': 0.05,
                'shortness_of_breath': 0.15,
                'chest_pain': 0.05,
                'nausea': 0.05,
                'vomiting': 0.05,
                'diarrhea': 0.05,
                'joint_pain': 0.05,
                'chills': 0.05,
                'swollen_lymph_nodes': 0.10
            },
            'pneumonia': {
                'fever': 0.90,
                'cough': 0.95,
                'fatigue': 0.80,
                'headache': 0.40,
                'body_ache': 0.45,
                'runny_nose': 0.20,
                'sneezing': 0.15,
                'sore_throat': 0.30,
                'rash': 0.05,
                'loss_of_smell': 0.10,
                'shortness_of_breath': 0.85,
                'chest_pain': 0.60,
                'nausea': 0.30,
                'vomiting': 0.20,
                'diarrhea': 0.15,
                'joint_pain': 0.25,
                'chills': 0.70,
                'swollen_lymph_nodes': 0.15
            },
            'bronchitis': {
                'fever': 0.50,
                'cough': 0.95,
                'fatigue': 0.60,
                'headache': 0.30,
                'body_ache': 0.35,
                'runny_nose': 0.40,
                'sneezing': 0.25,
                'sore_throat': 0.50,
                'rash': 0.05,
                'loss_of_smell': 0.05,
                'shortness_of_breath': 0.55,
                'chest_pain': 0.30,
                'nausea': 0.15,
                'vomiting': 0.10,
                'diarrhea': 0.05,
                'joint_pain': 0.15,
                'chills': 0.35,
                'swollen_lymph_nodes': 0.15
            }
        }
        
        # All possible symptoms (for feature vector creation)
        self.all_symptoms = [
            'fever', 'cough', 'fatigue', 'headache', 'body_ache',
            'runny_nose', 'sneezing', 'sore_throat', 'rash',
            'loss_of_smell', 'shortness_of_breath', 'chest_pain',
            'nausea', 'vomiting', 'diarrhea', 'joint_pain',
            'chills', 'swollen_lymph_nodes'
        ]
        
        # Disease list
        self.diseases = list(self.priors.keys())
        
        # Small epsilon to avoid log(0)
        self.epsilon = 0.001
        
        logger.info("SimpleBayesianDiagnostics initialized with 8 diseases and 18 symptoms")
    
    def clean_symptom(self, symptom: str) -> str:
        """Clean and normalize symptom names."""
        return symptom.lower().strip().replace(" ", "_")
    
    def compute_posterior(self, symptoms: List[str]) -> Dict[str, float]:
        """
        Compute posterior probabilities for all diseases given symptoms.
        
        Uses log space to avoid underflow:
        log(P(D|S)) ∝ log(P(D)) + Σ log(P(s|D))
        """
        # Clean symptoms
        clean_symptoms = [self.clean_symptom(s) for s in symptoms]
        
        # Initialize log scores with log(prior)
        log_scores = {}
        for disease in self.diseases:
            log_scores[disease] = math.log(self.priors[disease])
        
        # Add likelihoods for each symptom
        for symptom in clean_symptoms:
            for disease in self.diseases:
                # Get P(symptom | disease) or use epsilon if symptom not in likelihood table
                likelihood = self.likelihoods[disease].get(symptom, self.epsilon)
                # Clamp to avoid log(0)
                likelihood = max(likelihood, self.epsilon)
                log_scores[disease] += math.log(likelihood)
        
        # Convert from log space to probability space
        max_log = max(log_scores.values())
        
        # Convert to probabilities (softmax-style)
        exp_scores = {}
        total = 0.0
        for disease, log_score in log_scores.items():
            exp_score = math.exp(log_score - max_log)
            exp_scores[disease] = exp_score
            total += exp_score
        
        # Normalize to sum to 1.0
        posteriors = {}
        for disease, exp_score in exp_scores.items():
            posteriors[disease] = exp_score / total if total > 0 else 0.0
        
        return posteriors
    
    def analyze(self, patient: PatientPercept) -> Dict:
        """Standard interface method called by the Agent."""
        # Extract symptoms from patient
        symptoms = patient.symptoms.copy()
        
        # Add vital signs as symptoms
        if patient.temperature >= 38.0:
            symptoms.append("fever")
        if patient.temperature >= 39.0:
            symptoms.append("high_fever")
        if patient.heart_rate >= 100:
            symptoms.append("tachycardia")
        
        # Compute posterior probabilities
        posteriors = self.compute_posterior(symptoms)
        
        # Find top diagnosis
        sorted_diseases = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
        top_disease, top_confidence = sorted_diseases[0]
        
        # Get top 3 for detailed results
        top_3 = sorted_diseases[:3]
        
        logger.info(f"Patient {patient.patient_id}: Top diagnosis = {top_disease} ({top_confidence:.2%})")
        
        return {
            "module": "BayesianNet",
            "diagnosis": top_disease,
            "confidence": top_confidence,
            "all_probabilities": posteriors,
            "top_3": top_3,
            "symptoms_used": symptoms
        }
    
    def get_symptom_probability(self, disease: str, symptom: str) -> float:
        """Get the likelihood P(symptom | disease)."""
        disease = disease.lower().strip().replace(" ", "_")
        symptom = self.clean_symptom(symptom)
        return self.likelihoods.get(disease, {}).get(symptom, self.epsilon)
    
    def get_prior(self, disease: str) -> float:
        """Get the prior probability of a disease."""
        disease = disease.lower().strip().replace(" ", "_")
        return self.priors.get(disease, 0.0)
    
    def calculate_odds_ratio(self, disease1: str, disease2: str, symptoms: List[str]) -> float:
        """Calculate the odds ratio between two diseases given symptoms."""
        posteriors = self.compute_posterior(symptoms)
        return posteriors.get(disease1, 0.0) / (posteriors.get(disease2, 0.001))
    
    def print_diagnosis_report(self, patient: PatientPercept):
        """Pretty print a diagnosis report."""
        result = self.analyze(patient)
        
        print("=" * 60)
        print("BAYESIAN DIAGNOSIS REPORT")
        print("=" * 60)
        print(f"Patient ID: {patient.patient_id}")
        print(f"Symptoms: {', '.join(patient.symptoms)}")
        print(f"Temperature: {patient.temperature}°C")
        print(f"Heart Rate: {patient.heart_rate} BPM")
        print("-" * 60)
        print(f"TOP DIAGNOSIS: {result['diagnosis'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("-" * 60)
        print("All Disease Probabilities:")
        for disease, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 50)
            print(f"  {disease:15s}: {prob:6.2%} {bar}")
        print("=" * 60)


# Test the module
if __name__ == "__main__":
    print("Testing SimpleBayesianDiagnostics...")
    print()
    
    bn = SimpleBayesianDiagnostics()
    
    # Test case 1: COVID-19 symptoms
    print("--- Test 1: COVID-19 Symptoms ---")
    patient1 = PatientPercept(
        patient_id="B001",
        symptoms=["fever", "cough", "fatigue", "loss of smell", "headache"],
        age=45,
        temperature=38.9,
        heart_rate=98
    )
    
    result1 = bn.analyze(patient1)
    print(f"Patient: {patient1.patient_id}")
    print(f"Symptoms: {patient1.symptoms}")
    print(f"Top Diagnosis: {result1['diagnosis']}")
    print(f"Confidence: {result1['confidence']:.2%}")
    print(f"Top 3: {result1['top_3']}")
    print()
    
    # Test compute_posterior directly
    print("--- Test: Direct Posterior Computation ---")
    posteriors = bn.compute_posterior(["fever", "cough", "loss_of_smell"])
    print("Posteriors for [fever, cough, loss_of_smell]:")
    for disease, prob in sorted(posteriors.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {disease:15s}: {prob:6.2%}")
    print()
    
    # Test odds ratio
    print("--- Test: Odds Ratio ---")
    symptoms = ["fever", "cough", "loss_of_smell"]
    odds = bn.calculate_odds_ratio("covid19", "flu", symptoms)
    print(f"Odds ratio COVID-19 vs Flu given {symptoms}: {odds:.2f}")
    print()
    
    print("\n✓ SimpleBayesianDiagnostics test passed!")

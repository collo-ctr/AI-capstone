# modules/knowledge_base.py

from typing import List, Dict, Tuple, Optional, Set
import logging
from modules.agent import PatientPercept

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalKnowledgeBase:
    """
    The rule-book doctor of the system.
    Uses Forward Chaining and Backward Chaining for diagnosis.
    """
    
    def __init__(self):
        # Facts are stored as: {fact_name: confidence}
        self.facts: Dict[str, float] = {}
        
        # Rules are stored as: (conditions, conclusion, certainty_factor)
        self.rules: List[Tuple[List[str], str, float]] = []
        
        # Initialize with medical knowledge rules
        self._initialize_rules()
        
        logger.info("MedicalKnowledgeBase initialized")
    
    def _initialize_rules(self):
        """Initialize the medical knowledge base with rules."""
        # Rule format: (list_of_conditions, conclusion, certainty_factor)
        rules = [
            # COVID-19 rules
            (["fever", "cough", "loss_of_smell"], "covid19_suspected", 0.85),
            (["covid19_suspected", "fatigue"], "covid19_likely", 0.75),
            (["covid19_suspected", "shortness_of_breath"], "covid19_severe", 0.80),
            
            # Flu rules
            (["fever", "cough", "fatigue"], "flu_suspected", 0.75),
            (["flu_suspected", "body_ache"], "flu_likely", 0.70),
            
            # Dengue rules
            (["fever", "rash", "joint_pain"], "dengue_suspected", 0.80),
            (["dengue_suspected", "headache"], "dengue_likely", 0.65),
            
            # Common cold rules
            (["runny_nose", "sneezing", "mild_fever"], "cold_suspected", 0.70),
            
            # Severity rules
            (["covid19_likely", "high_fever"], "severe_condition", 0.85),
            (["flu_likely", "high_fever"], "severe_condition", 0.75),
        ]
        
        for conditions, conclusion, cf in rules:
            self.add_rule(conditions, conclusion, cf)
    
    def add_fact(self, fact: str, confidence: float = 1.0):
        """Add a fact to the knowledge base."""
        fact = fact.lower().strip().replace(" ", "_")
        self.facts[fact] = max(self.facts.get(fact, 0.0), confidence)
        logger.debug(f"Added fact: {fact} (CF={confidence:.2f})")
    
    def add_rule(self, conditions: List[str], conclusion: str, certainty_factor: float):
        """Add a rule to the knowledge base."""
        conditions = [c.lower().strip().replace(" ", "_") for c in conditions]
        conclusion = conclusion.lower().strip().replace(" ", "_")
        self.rules.append((conditions, conclusion, certainty_factor))
        logger.debug(f"Added rule: {conditions} → {conclusion} (CF={certainty_factor:.2f})")
    
    def clear_facts(self):
        """Clear all facts from the knowledge base."""
        self.facts = {}
        logger.debug("Facts cleared")
    
    def load_patient_symptoms(self, patient: PatientPercept):
        """
        Load patient symptoms as facts.
        Also processes vitals into facts.
        """
        self.clear_facts()
        
        # Add each symptom as a fact
        for symptom in patient.symptoms:
            cleaned = symptom.lower().strip().replace(" ", "_")
            self.add_fact(cleaned)
        
        # Add vital signs as facts
        if patient.temperature >= 39.5:
            self.add_fact("critical_fever")
        elif patient.temperature >= 38.5:
            self.add_fact("high_fever")
        elif patient.temperature >= 37.5:
            self.add_fact("mild_fever")
        
        if patient.heart_rate >= 120:
            self.add_fact("tachycardia")
        elif patient.heart_rate >= 100:
            self.add_fact("elevated_heart_rate")
        
        logger.info(f"Loaded patient {patient.patient_id} with {len(self.facts)} facts")
    
    def forward_chain(self, verbose: bool = False) -> Dict[str, float]:
        """
        Forward chaining: Start from symptoms, fire rules, reach diagnosis.
        
        Returns:
            Dictionary of inferred facts and their confidence
        """
        inferred = self.facts.copy()
        changed = True
        iteration = 0
        
        while changed:
            changed = False
            iteration += 1
            
            for conditions, conclusion, cf in self.rules:
                # Check if all conditions are met
                if all(cond in inferred for cond in conditions):
                    # Rule fires - add conclusion
                    if conclusion not in inferred or inferred[conclusion] < cf:
                        # Combine certainty factors
                        min_cond_cf = min(inferred[cond] for cond in conditions)
                        combined_cf = min_cond_cf * cf
                        
                        old_cf = inferred.get(conclusion, 0.0)
                        inferred[conclusion] = max(old_cf, combined_cf)
                        changed = True
                        
                        if verbose:
                            condition_str = " ∧ ".join(conditions)
                            print(f"Iter {iteration}: {condition_str} → {conclusion} (CF={combined_cf:.3f})")
        
        self.facts = inferred
        return inferred
    
    def backward_chain(self, goal: str, verbose: bool = False) -> Tuple[bool, float]:
        """
        Backward chaining: Start from a goal diagnosis, work backwards to see if symptoms support it.
        
        Args:
            goal: The goal to prove
            verbose: Whether to print reasoning steps
        
        Returns:
            Tuple of (goal_proved, confidence)
        """
        goal = goal.lower().strip().replace(" ", "_")
        
        # Check if goal is already a fact
        if goal in self.facts:
            return True, self.facts[goal]
        
        # Try to prove goal using rules
        for conditions, conclusion, cf in self.rules:
            if conclusion == goal:
                if verbose:
                    print(f"Trying to prove {goal} via: {conditions} → {goal}")
                
                # Recursively prove all conditions
                condition_confidences = []
                all_proved = True
                
                for condition in conditions:
                    proved, conf = self.backward_chain(condition)
                    if not proved:
                        all_proved = False
                        break
                    condition_confidences.append(conf)
                
                if all_proved:
                    # Combine condition confidences and rule CF
                    min_conf = min(condition_confidences) if condition_confidences else 1.0
                    combined_cf = min_conf * cf
                    return True, combined_cf
        
        return False, 0.0
    
    def analyze(self, patient: PatientPercept) -> Dict:
        """
        Standard interface method called by the Agent.
        
        Returns:
            Dictionary with diagnosis, confidence, and all inferred facts
        """
        # Load patient data
        self.load_patient_symptoms(patient)
        
        # Run forward chaining
        inferred = self.forward_chain(verbose=False)
        
        # Find the most likely diagnosis
        # Look for facts ending with _suspected, _likely, or _severe
        diagnosis_keywords = ["suspected", "likely", "severe", "condition"]
        potential_diagnoses = []
        
        for fact, cf in inferred.items():
            if any(keyword in fact for keyword in diagnosis_keywords):
                # Map to clean disease name
                disease = fact.replace("_suspected", "").replace("_likely", "").replace("_severe", "").replace("_condition", "")
                if disease in ["covid19", "flu", "dengue", "cold"]:
                    potential_diagnoses.append((disease, cf))
        
        # Sort by confidence
        potential_diagnoses.sort(key=lambda x: x[1], reverse=True)
        
        if potential_diagnoses:
            top_diagnosis, top_confidence = potential_diagnoses[0]
        else:
            top_diagnosis = "unknown"
            top_confidence = 0.0
        
        return {
            "module": "KnowledgeBase",
            "diagnosis": top_diagnosis,
            "confidence": top_confidence,
            "all_inferred": inferred,
            "facts": self.facts
        }


# Test the module
if __name__ == "__main__":
    print("Testing MedicalKnowledgeBase...")
    
    kb = MedicalKnowledgeBase()
    
    # Test case 1: COVID-19 symptoms
    print("\n--- Test 1: COVID-19 symptoms ---")
    patient1 = PatientPercept(
        patient_id="T001",
        symptoms=["fever", "cough", "fatigue", "loss of smell"],
        age=45,
        temperature=38.9,
        heart_rate=98
    )
    
    result = kb.analyze(patient1)
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"All inferred: {result['all_inferred']}")
    
    # Test 2: Backward chaining
    print("\n--- Test 2: Backward Chaining ---")
    proved, cf = kb.backward_chain("covid19_suspected", verbose=True)
    print(f"covid19_suspected: {proved}, Confidence: {cf:.3f}")
    
    # Test 3: Reset and test with cold symptoms
    print("\n--- Test 3: Cold symptoms ---")
    kb.clear_facts()
    kb.add_fact("runny_nose")
    kb.add_fact("sneezing")
    kb.add_fact("mild_fever")
    
    inferred = kb.forward_chain(verbose=True)
    print(f"Inferred: {inferred}")
    
    print("\n✓ MedicalKnowledgeBase test passed!")

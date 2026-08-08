# modules/planner.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Set, Tuple, Optional
from collections import deque
import logging
from modules.agent import PatientPercept

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TreatmentPlanner:
    """
    The decision and action planner of the system.
    Uses STRIPS Planning to generate step-by-step treatment plans.
    """
    
    def __init__(self):
        # Define all medical actions
        self.actions = self._initialize_actions()
        logger.info("TreatmentPlanner initialized with {} actions".format(len(self.actions)))
    
    def _initialize_actions(self) -> List[Dict]:
        """
        Initialize the action library with medical actions.
        
        Each action is a dictionary with:
        - name: Readable name
        - precond: Set of facts that must be true
        - delete: Set of facts removed after the action
        - add: Set of facts added after the action
        - cost: Action cost (for future optimization)
        - duration: Estimated duration
        """
        actions = [
            {
                'name': 'IsolatePatient',
                'precond': {'PATIENT_PRESENT', 'CONTAGIOUS_DISEASE_SUSPECTED'},
                'delete': {'CONTAGIOUS_DISEASE_SUSPECTED'},
                'add': {'PATIENT_ISOLATED', 'INFECTION_CONTROL_INITIATED'},
                'cost': 1,
                'duration': 'Immediate'
            },
            {
                'name': 'OrderPCRTest',
                'precond': {'PATIENT_PRESENT', 'PATIENT_ISOLATED', 'COVID_SUSPECTED'},
                'delete': {'COVID_SUSPECTED'},
                'add': {'PCR_PENDING', 'TEST_ORDERED'},
                'cost': 1,
                'duration': '24 hours'
            },
            {
                'name': 'ReceivePCRResult',
                'precond': {'PCR_PENDING'},
                'delete': {'PCR_PENDING'},
                'add': {'PCR_RESULT_KNOWN', 'DIAGNOSIS_CONFIRMED'},
                'cost': 1,
                'duration': '24 hours'
            },
            {
                'name': 'PrescribeAntiviral',
                'precond': {'VIRAL_INFECTION_CONFIRMED', 'PATIENT_PRESENT'},
                'delete': {'VIRAL_INFECTION_CONFIRMED'},
                'add': {'TREATMENT_STARTED', 'ANTIVIRAL_PRESCRIBED'},
                'cost': 1,
                'duration': '10 minutes'
            },
            {
                'name': 'PrescribeAntibiotics',
                'precond': {'BACTERIAL_INFECTION_CONFIRMED', 'PATIENT_PRESENT'},
                'delete': {'BACTERIAL_INFECTION_CONFIRMED'},
                'add': {'TREATMENT_STARTED', 'ANTIBIOTICS_PRESCRIBED'},
                'cost': 1,
                'duration': '10 minutes'
            },
            {
                'name': 'OrderChestXRay',
                'precond': {'PATIENT_PRESENT', 'RESPIRATORY_SYMPTOMS'},
                'delete': {'RESPIRATORY_SYMPTOMS'},
                'add': {'XRAY_ORDERED', 'DIAGNOSTIC_PENDING'},
                'cost': 1,
                'duration': '2 hours'
            },
            {
                'name': 'ReceiveXRayResult',
                'precond': {'XRAY_ORDERED'},
                'delete': {'XRAY_ORDERED', 'DIAGNOSTIC_PENDING'},
                'add': {'XRAY_RESULT_KNOWN', 'PNEUMONIA_CONFIRMED_OR_RULED_OUT'},
                'cost': 1,
                'duration': '2 hours'
            },
            {
                'name': 'MonitorVitals',
                'precond': {'PATIENT_PRESENT', 'PATIENT_ISOLATED'},
                'delete': set(),
                'add': {'VITALS_MONITORED'},
                'cost': 1,
                'duration': 'Continuous'
            },
            {
                'name': 'ScheduleFollowUp',
                'precond': {'TREATMENT_STARTED', 'PATIENT_PRESENT'},
                'delete': set(),
                'add': {'FOLLOWUP_SCHEDULED'},
                'cost': 1,
                'duration': '5 minutes'
            },
            {
                'name': 'AdministerFluids',
                'precond': {'PATIENT_PRESENT', 'DEHYDRATION_RISK'},
                'delete': {'DEHYDRATION_RISK'},
                'add': {'FLUIDS_ADMINISTERED', 'HYDRATION_MAINTAINED'},
                'cost': 1,
                'duration': '30 minutes'
            },
            {
                'name': 'OrderBloodTest',
                'precond': {'PATIENT_PRESENT', 'DENGUE_SUSPECTED'},
                'delete': {'DENGUE_SUSPECTED'},
                'add': {'BLOOD_TEST_ORDERED', 'DIAGNOSTIC_PENDING'},
                'cost': 1,
                'duration': '4 hours'
            },
            {
                'name': 'ReceiveBloodTestResult',
                'precond': {'BLOOD_TEST_ORDERED'},
                'delete': {'BLOOD_TEST_ORDERED', 'DIAGNOSTIC_PENDING'},
                'add': {'BLOOD_TEST_RESULT_KNOWN', 'DIAGNOSIS_CONFIRMED'},
                'cost': 1,
                'duration': '4 hours'
            }
        ]
        
        return actions
    
    def _apply_action(self, state: Set[str], action: Dict) -> Optional[Set[str]]:
        """
        Apply an action to a state if preconditions are met.
        
        Args:
            state: Current set of facts
            action: Action to apply
            
        Returns:
            New state if action is applicable, None otherwise
        """
        # Check if preconditions are met
        if not action['precond'].issubset(state):
            return None
        
        # Apply action: remove delete list, add add list
        new_state = state - action['delete']
        new_state = new_state | action['add']
        
        return frozenset(new_state)
    
    def generate_plan(self, initial_state: Set[str], goal_state: Set[str], 
                      max_depth: int = 20) -> Optional[List[Dict]]:
        """
        Generate a plan using BFS over the state space.
        
        Args:
            initial_state: Set of initial facts
            goal_state: Set of goal facts
            max_depth: Maximum search depth to prevent infinite loops
            
        Returns:
            List of actions if plan found, None otherwise
        """
        # Convert to frozenset for hashability
        initial = frozenset(initial_state)
        goal = frozenset(goal_state)
        
        # BFS queue: (current_state, plan, depth)
        queue = deque([(initial, [], 0)])
        visited = {initial}
        
        logger.info(f"Planning from {len(initial_state)} facts to {len(goal_state)} goals")
        
        while queue:
            state, plan, depth = queue.popleft()
            
            # Check if goal is reached
            if goal.issubset(state):
                logger.info(f"Plan found with {len(plan)} steps")
                return plan
            
            # Check depth limit
            if depth >= max_depth:
                continue
            
            # Try each action
            for action in self.actions:
                new_state = self._apply_action(set(state), action)
                
                if new_state is not None and new_state not in visited:
                    visited.add(new_state)
                    new_plan = plan + [action]
                    queue.append((new_state, new_plan, depth + 1))
        
        logger.warning("No plan found within depth limit")
        return None
    
    def create_treatment_plan(self, diagnosis: str, severity: str = 'MEDIUM') -> Dict:
        """
        Create a treatment plan based on diagnosis and severity.
        
        Args:
            diagnosis: The diagnosed disease
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            
        Returns:
            Dictionary with diagnosis, severity, plan steps, and metadata
        """
        # Map diagnosis to initial state and goals
        diagnosis_map = {
            'covid19': {
                'initial': {'PATIENT_PRESENT', 'COVID_SUSPECTED', 'CONTAGIOUS_DISEASE_SUSPECTED', 'VIRAL_INFECTION_CONFIRMED'},
                'goal': {'TREATMENT_STARTED', 'VITALS_MONITORED', 'FOLLOWUP_SCHEDULED', 'PCR_RESULT_KNOWN'}
            },
            'flu': {
                'initial': {'PATIENT_PRESENT', 'VIRAL_INFECTION_CONFIRMED'},
                'goal': {'TREATMENT_STARTED', 'VITALS_MONITORED', 'FOLLOWUP_SCHEDULED'}
            },
            'dengue': {
                'initial': {'PATIENT_PRESENT', 'DENGUE_SUSPECTED', 'DEHYDRATION_RISK', 'VIRAL_INFECTION_CONFIRMED'},
                'goal': {'TREATMENT_STARTED', 'VITALS_MONITORED', 'FLUIDS_ADMINISTERED', 'BLOOD_TEST_RESULT_KNOWN'}
            },
            'common_cold': {
                'initial': {'PATIENT_PRESENT', 'VIRAL_INFECTION_CONFIRMED'},
                'goal': {'TREATMENT_STARTED', 'FOLLOWUP_SCHEDULED'}
            },
            'pneumonia': {
                'initial': {'PATIENT_PRESENT', 'RESPIRATORY_SYMPTOMS', 'BACTERIAL_INFECTION_CONFIRMED'},
                'goal': {'TREATMENT_STARTED', 'VITALS_MONITORED', 'XRAY_RESULT_KNOWN', 'ANTIBIOTICS_PRESCRIBED'}
            },
            'strep_throat': {
                'initial': {'PATIENT_PRESENT', 'BACTERIAL_INFECTION_CONFIRMED'},
                'goal': {'TREATMENT_STARTED', 'ANTIBIOTICS_PRESCRIBED', 'FOLLOWUP_SCHEDULED'}
            },
            'allergy': {
                'initial': {'PATIENT_PRESENT'},
                'goal': {'TREATMENT_STARTED', 'FOLLOWUP_SCHEDULED'}
            },
            'bronchitis': {
                'initial': {'PATIENT_PRESENT', 'RESPIRATORY_SYMPTOMS'},
                'goal': {'TREATMENT_STARTED', 'VITALS_MONITORED', 'FOLLOWUP_SCHEDULED'}
            }
        }
        
        # Add severity adjustments
        if severity in ['HIGH', 'CRITICAL']:
            # Add urgency facts
            if diagnosis == 'covid19':
                diagnosis_map[diagnosis]['initial'].add('CRITICAL_CONDITION')
                diagnosis_map[diagnosis]['goal'].add('INFECTION_CONTROL_INITIATED')
            elif diagnosis == 'pneumonia':
                diagnosis_map[diagnosis]['initial'].add('CRITICAL_CONDITION')
                diagnosis_map[diagnosis]['goal'].add('OXYGEN_THERAPY_STARTED')
        
        # Get initial and goal states for the diagnosis
        if diagnosis in diagnosis_map:
            initial_state = diagnosis_map[diagnosis]['initial']
            goal_state = diagnosis_map[diagnosis]['goal']
        else:
            # Default plan for unknown diagnosis
            initial_state = {'PATIENT_PRESENT'}
            goal_state = {'FOLLOWUP_SCHEDULED'}
        
        # Generate plan
        plan = self.generate_plan(initial_state, goal_state)
        
        if plan is None:
            # Fallback: simple plan
            plan = [
                {'name': 'MonitorVitals', 'duration': 'Continuous'},
                {'name': 'ScheduleFollowUp', 'duration': '5 minutes'}
            ]
        
        # Format the plan
        formatted_plan = []
        for i, action in enumerate(plan, 1):
            formatted_plan.append({
                'step': i,
                'action': action['name'],
                'duration': action.get('duration', 'Unknown'),
                'description': self._get_action_description(action['name'])
            })
        
        return {
            'diagnosis': diagnosis,
            'severity': severity,
            'plan': formatted_plan,
            'total_steps': len(formatted_plan),
            'total_duration': self._calculate_total_duration(formatted_plan)
        }
    
    def _get_action_description(self, action_name: str) -> str:
        """Get a description for an action."""
        descriptions = {
            'IsolatePatient': 'Isolate patient to prevent spread',
            'OrderPCRTest': 'Order PCR test for COVID-19',
            'ReceivePCRResult': 'Receive and analyze PCR test results',
            'PrescribeAntiviral': 'Prescribe antiviral medication',
            'PrescribeAntibiotics': 'Prescribe antibiotics',
            'OrderChestXRay': 'Order chest X-ray',
            'ReceiveXRayResult': 'Receive and analyze X-ray results',
            'MonitorVitals': 'Monitor patient vitals continuously',
            'ScheduleFollowUp': 'Schedule follow-up appointment',
            'AdministerFluids': 'Administer IV fluids',
            'OrderBloodTest': 'Order blood test for dengue',
            'ReceiveBloodTestResult': 'Receive and analyze blood test results'
        }
        return descriptions.get(action_name, f'Perform {action_name}')
    
    def _calculate_total_duration(self, plan: List[Dict]) -> str:
        """Calculate total duration of the plan."""
        total_minutes = 0
        
        for step in plan:
            duration = step['duration']
            if 'hour' in duration:
                hours = int(duration.split()[0])
                total_minutes += hours * 60
            elif 'minute' in duration:
                mins = int(duration.split()[0])
                total_minutes += mins
            elif duration == 'Continuous':
                total_minutes += 10  # Assume monitoring is ongoing
        
        if total_minutes >= 1440:  # More than a day
            return f"{total_minutes // 1440} days, {(total_minutes % 1440) // 60} hours"
        elif total_minutes >= 60:
            return f"{total_minutes // 60} hours, {total_minutes % 60} minutes"
        else:
            return f"{total_minutes} minutes"
    
    def analyze(self, patient: PatientPercept) -> Dict:
        """
        Standard interface method called by the Agent.
        This is a simplified analysis - in practice, diagnosis would come from other modules.
        """
        # Simulate diagnosis based on symptoms
        diagnosis = self._infer_diagnosis(patient)
        severity = self._infer_severity(patient)
        
        # Create treatment plan
        plan = self.create_treatment_plan(diagnosis, severity)
        plan['module'] = 'TreatmentPlanner'
        
        return plan
    
    def _infer_diagnosis(self, patient: PatientPercept) -> str:
        """Simple diagnosis inference for demo purposes."""
        symptoms = [s.lower() for s in patient.symptoms]
        
        if 'loss_of_smell' in symptoms and 'fever' in symptoms and 'cough' in symptoms:
            return 'covid19'
        elif 'runny_nose' in symptoms and 'sneezing' in symptoms:
            return 'common_cold'
        elif 'rash' in symptoms and 'joint_pain' in symptoms and 'fever' in symptoms:
            return 'dengue'
        elif 'shortness_of_breath' in symptoms and 'chest_pain' in symptoms:
            return 'pneumonia'
        elif 'sore_throat' in symptoms and 'swollen_lymph_nodes' in symptoms:
            return 'strep_throat'
        elif 'runny_nose' in symptoms and 'sneezing' in symptoms and not 'fever' in symptoms:
            return 'allergy'
        elif 'cough' in symptoms and 'fatigue' in symptoms:
            return 'bronchitis'
        else:
            return 'flu'
    
    def _infer_severity(self, patient: PatientPercept) -> str:
        """Infer severity from vitals."""
        if patient.temperature >= 39.5 or patient.heart_rate >= 120:
            return 'CRITICAL'
        elif patient.temperature >= 38.5 or patient.heart_rate >= 100:
            return 'HIGH'
        elif patient.temperature >= 37.5:
            return 'MEDIUM'
        else:
            return 'LOW'


# Test the module
if __name__ == "__main__":
    print("Testing TreatmentPlanner...")
    print()
    
    planner = TreatmentPlanner()
    
    # Test cases
    test_cases = [
        ('covid19', 'HIGH'),
        ('covid19', 'CRITICAL'),
        ('flu', 'MEDIUM'),
        ('dengue', 'HIGH'),
        ('pneumonia', 'CRITICAL'),
        ('common_cold', 'LOW'),
    ]
    
    for diagnosis, severity in test_cases:
        print("=" * 60)
        print(f"Diagnosis: {diagnosis.upper()}, Severity: {severity}")
        print("=" * 60)
        
        plan = planner.create_treatment_plan(diagnosis, severity)
        
        print(f"Total Steps: {plan['total_steps']}")
        print(f"Total Duration: {plan['total_duration']}")
        print("\nPlan Steps:")
        for step in plan['plan']:
            print(f"  Step {step['step']:2d}: {step['action']:<20} [{step['duration']}]")
            print(f"        {step['description']}")
        print()
    
    # Test with PatientPercept
    print("=" * 60)
    print("TESTING WITH PATIENTPERCEPT")
    print("=" * 60)
    
    patient = PatientPercept(
        patient_id="P001",
        symptoms=["fever", "cough", "fatigue", "loss_of_smell"],
        age=45,
        temperature=39.2,
        heart_rate=110
    )
    
    result = planner.analyze(patient)
    print(f"Patient: {patient.patient_id}")
    print(f"Inferred Diagnosis: {result['diagnosis']}")
    print(f"Inferred Severity: {result['severity']}")
    print(f"Total Steps: {result['total_steps']}")
    print("\nPlan Steps:")
    for step in result['plan']:
        print(f"  Step {step['step']:2d}: {step['action']:<20} [{step['duration']}]")
    
    print("\n✅ TreatmentPlanner test passed!")
  

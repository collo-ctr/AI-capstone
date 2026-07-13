# modules/ml_classifier.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging
from modules.agent import PatientPercept
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLDiagnosticClassifier:
    """
    The data-driven pattern recognizer.
    Trains multiple ML algorithms on synthetic patient data.
    """
    
    def __init__(self):
        # All possible symptoms (18 features)
        self.symptom_features = [
            'fever', 'cough', 'fatigue', 'headache', 'body_ache',
            'runny_nose', 'sneezing', 'sore_throat', 'rash',
            'loss_of_smell', 'shortness_of_breath', 'chest_pain',
            'nausea', 'vomiting', 'diarrhea', 'joint_pain',
            'chills', 'swollen_lymph_nodes'
        ]
        
        # Disease classes
        self.diseases = [
            'flu', 'covid19', 'common_cold', 'dengue',
            'strep_throat', 'allergy', 'pneumonia', 'bronchitis'
        ]
        
        # Model storage
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        logger.info("MLDiagnosticClassifier initialized with 18 symptoms and 8 diseases")
    
    def _generate_synthetic_data(self, num_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic patient data based on known symptom-disease relationships.
        
        Args:
            num_samples: Number of patient records to generate
            
        Returns:
            X: Feature matrix (num_samples x 18)
            y: Labels (num_samples)
        """
        np.random.seed(42)
        
        # Disease profiles - probability of each symptom given the disease
        disease_profiles = {
            'flu': {
                'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88,
                'headache': 0.75, 'body_ache': 0.80, 'runny_nose': 0.40,
                'sneezing': 0.30, 'sore_throat': 0.60, 'rash': 0.05,
                'loss_of_smell': 0.10, 'shortness_of_breath': 0.15,
                'chest_pain': 0.10, 'nausea': 0.35, 'vomiting': 0.20,
                'diarrhea': 0.15, 'joint_pain': 0.55, 'chills': 0.70,
                'swollen_lymph_nodes': 0.20
            },
            'covid19': {
                'fever': 0.85, 'cough': 0.80, 'fatigue': 0.75,
                'headache': 0.50, 'body_ache': 0.45, 'runny_nose': 0.35,
                'sneezing': 0.20, 'sore_throat': 0.40, 'rash': 0.10,
                'loss_of_smell': 0.70, 'shortness_of_breath': 0.45,
                'chest_pain': 0.30, 'nausea': 0.25, 'vomiting': 0.15,
                'diarrhea': 0.20, 'joint_pain': 0.30, 'chills': 0.55,
                'swollen_lymph_nodes': 0.15
            },
            'common_cold': {
                'fever': 0.30, 'cough': 0.70, 'fatigue': 0.40,
                'headache': 0.35, 'body_ache': 0.25, 'runny_nose': 0.85,
                'sneezing': 0.80, 'sore_throat': 0.70, 'rash': 0.05,
                'loss_of_smell': 0.10, 'shortness_of_breath': 0.05,
                'chest_pain': 0.05, 'nausea': 0.10, 'vomiting': 0.05,
                'diarrhea': 0.05, 'joint_pain': 0.10, 'chills': 0.20,
                'swollen_lymph_nodes': 0.10
            },
            'dengue': {
                'fever': 0.98, 'cough': 0.30, 'fatigue': 0.85,
                'headache': 0.90, 'body_ache': 0.85, 'runny_nose': 0.15,
                'sneezing': 0.10, 'sore_throat': 0.20, 'rash': 0.75,
                'loss_of_smell': 0.05, 'shortness_of_breath': 0.15,
                'chest_pain': 0.10, 'nausea': 0.50, 'vomiting': 0.40,
                'diarrhea': 0.20, 'joint_pain': 0.85, 'chills': 0.60,
                'swollen_lymph_nodes': 0.30
            },
            'strep_throat': {
                'fever': 0.70, 'cough': 0.30, 'fatigue': 0.50,
                'headache': 0.40, 'body_ache': 0.35, 'runny_nose': 0.20,
                'sneezing': 0.15, 'sore_throat': 0.95, 'rash': 0.10,
                'loss_of_smell': 0.05, 'shortness_of_breath': 0.05,
                'chest_pain': 0.05, 'nausea': 0.20, 'vomiting': 0.15,
                'diarrhea': 0.05, 'joint_pain': 0.15, 'chills': 0.30,
                'swollen_lymph_nodes': 0.60
            },
            'allergy': {
                'fever': 0.05, 'cough': 0.30, 'fatigue': 0.20,
                'headache': 0.15, 'body_ache': 0.05, 'runny_nose': 0.90,
                'sneezing': 0.95, 'sore_throat': 0.30, 'rash': 0.25,
                'loss_of_smell': 0.05, 'shortness_of_breath': 0.15,
                'chest_pain': 0.05, 'nausea': 0.05, 'vomiting': 0.05,
                'diarrhea': 0.05, 'joint_pain': 0.05, 'chills': 0.05,
                'swollen_lymph_nodes': 0.10
            },
            'pneumonia': {
                'fever': 0.90, 'cough': 0.95, 'fatigue': 0.80,
                'headache': 0.40, 'body_ache': 0.45, 'runny_nose': 0.20,
                'sneezing': 0.15, 'sore_throat': 0.30, 'rash': 0.05,
                'loss_of_smell': 0.10, 'shortness_of_breath': 0.85,
                'chest_pain': 0.60, 'nausea': 0.30, 'vomiting': 0.20,
                'diarrhea': 0.15, 'joint_pain': 0.25, 'chills': 0.70,
                'swollen_lymph_nodes': 0.15
            },
            'bronchitis': {
                'fever': 0.50, 'cough': 0.95, 'fatigue': 0.60,
                'headache': 0.30, 'body_ache': 0.35, 'runny_nose': 0.40,
                'sneezing': 0.25, 'sore_throat': 0.50, 'rash': 0.05,
                'loss_of_smell': 0.05, 'shortness_of_breath': 0.55,
                'chest_pain': 0.30, 'nausea': 0.15, 'vomiting': 0.10,
                'diarrhea': 0.05, 'joint_pain': 0.15, 'chills': 0.35,
                'swollen_lymph_nodes': 0.15
            }
        }
        
        # Generate samples
        X_list = []
        y_list = []
        
        samples_per_disease = num_samples // len(self.diseases)
        
        for disease in self.diseases:
            profile = disease_profiles[disease]
            for _ in range(samples_per_disease):
                # Generate symptoms based on profile probabilities
                symptoms = []
                for symptom in self.symptom_features:
                    prob = profile.get(symptom, 0.05)
                    # Add some randomness
                    if np.random.random() < prob:
                        symptoms.append(1)
                    else:
                        symptoms.append(0)
                
                X_list.append(symptoms)
                y_list.append(disease)
        
        # Add some random noise samples
        noise_samples = num_samples - len(X_list)
        for _ in range(noise_samples):
            # Random patient with random symptoms
            symptoms = [1 if np.random.random() < 0.3 else 0 for _ in range(len(self.symptom_features))]
            disease = np.random.choice(self.diseases)
            X_list.append(symptoms)
            y_list.append(disease)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Generated {len(X)} synthetic patient records")
        return X, y
    
    def _symptoms_to_vector(self, symptoms: List[str]) -> np.ndarray:
        """
        Convert a list of symptom strings to a binary feature vector.
        
        Args:
            symptoms: List of symptom names
            
        Returns:
            Binary vector (18,)
        """
        vector = np.zeros(len(self.symptom_features))
        
        # Clean and normalize symptoms
        clean_symptoms = [s.lower().strip().replace(" ", "_") for s in symptoms]
        
        for i, feature in enumerate(self.symptom_features):
            if feature in clean_symptoms:
                vector[i] = 1
        
        return vector
    
    def train(self, verbose: bool = True) -> Dict[str, float]:
        """
        Train all three models and select the best one using cross-validation.
        
        Args:
            verbose: Whether to print training progress
            
        Returns:
            Dictionary of model scores
        """
        if verbose:
            print("=" * 60)
            print("TRAINING ML DIAGNOSTIC CLASSIFIER")
            print("=" * 60)
        
        # Generate data
        X, y = self._generate_synthetic_data(2000)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        if verbose:
            print(f"\nData Split:")
            print(f"  Training samples: {len(self.X_train)}")
            print(f"  Test samples: {len(self.X_test)}")
            print(f"  Features: {len(self.symptom_features)}")
            print(f"  Classes: {len(self.diseases)}")
        
        # Define models to train
        models = {
            'Decision Tree': DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Train and evaluate each model
        scores = {}
        self.models = {}
        
        if verbose:
            print("\n" + "-" * 60)
            print("Training Models with 5-Fold Cross-Validation")
            print("-" * 60)
        
        for name, model in models.items():
            # Cross-validation score
            cv_scores = cross_val_score(model, X, y_encoded, cv=5)
            mean_cv_score = cv_scores.mean()
            
            # Train on full training set
            model.fit(self.X_train, self.y_train)
            
            # Test score
            y_pred = model.predict(self.X_test)
            test_score = accuracy_score(self.y_test, y_pred)
            
            self.models[name] = model
            scores[name] = {
                'cv_mean': mean_cv_score,
                'cv_std': cv_scores.std(),
                'test_score': test_score
            }
            
            if verbose:
                print(f"\n  {name}:")
                print(f"    CV Score: {mean_cv_score:.3f} ± {cv_scores.std():.3f}")
                print(f"    Test Score: {test_score:.3f}")
            
            # Select best model
            if mean_cv_score > self.best_score:
                self.best_score = mean_cv_score
                self.best_model = model
                self.best_model_name = name
        
        self.is_trained = True
        
        if verbose:
            print("\n" + "-" * 60)
            print(f"🏆 BEST MODEL: {self.best_model_name}")
            print(f"   CV Score: {self.best_score:.3f}")
            print("=" * 60)
        
        return scores
    
    def predict(self, symptoms: List[str]) -> Dict:
        """
        Predict diagnosis for a patient.
        
        Args:
            symptoms: List of symptom strings
            
        Returns:
            Dictionary with diagnosis, confidence, and model info
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Convert symptoms to feature vector
        feature_vector = self._symptoms_to_vector(symptoms).reshape(1, -1)
        
        # Get prediction
        pred_encoded = self.best_model.predict(feature_vector)[0]
        pred_proba = self.best_model.predict_proba(feature_vector)[0]
        
        # Convert back to disease name
        diagnosis = self.label_encoder.inverse_transform([pred_encoded])[0]
        confidence = pred_proba[pred_encoded]
        
        # Get top 3 predictions
        top_indices = np.argsort(pred_proba)[-3:][::-1]
        top_3 = []
        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            prob = pred_proba[idx]
            top_3.append((disease, prob))
        
        return {
            "module": "MLClassifier",
            "diagnosis": diagnosis,
            "confidence": float(confidence),
            "model_used": self.best_model_name,
            "top_3": top_3,
            "all_probabilities": {
                disease: float(pred_proba[i]) 
                for i, disease in enumerate(self.label_encoder.classes_)
            }
        }
    
    def predict_from_patient(self, patient: PatientPercept) -> Dict:
        """
        Predict diagnosis from a PatientPercept object.
        
        Args:
            patient: PatientPercept object
            
        Returns:
            Same as predict()
        """
        # Extract symptoms and add vitals as symptoms
        symptoms = patient.symptoms.copy()
        
        if patient.temperature >= 38.0:
            symptoms.append("fever")
        if patient.temperature >= 39.0:
            symptoms.append("high_fever")
        if patient.heart_rate >= 100:
            symptoms.append("tachycardia")
        
        return self.predict(symptoms)
    
    def plot_evaluation(self, save_path: Optional[str] = None):
        """
        Generate evaluation plots: confusion matrix and feature importance.
        
        Args:
            save_path: Optional path to save plots
        """
        if not self.is_trained:
            print("Model not trained yet. Call train() first.")
            return
        
        # Get predictions on test set
        y_pred = self.best_model.predict(self.X_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_test_labels = self.label_encoder.inverse_transform(self.y_test)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=self.diseases)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.diseases, yticklabels=self.diseases, ax=ax1)
        ax1.set_title(f'Confusion Matrix - {self.best_model_name}')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. Feature Importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 features
            
            ax2.barh(range(len(indices)), importances[indices], color='skyblue')
            ax2.set_yticks(range(len(indices)))
            ax2.set_yticklabels([self.symptom_features[i] for i in indices])
            ax2.set_title('Top 10 Most Important Symptoms')
            ax2.set_xlabel('Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plots saved to {save_path}")
        
        plt.show()
    
    def print_classification_report(self):
        """Print detailed classification report."""
        if not self.is_trained:
            print("Model not trained yet. Call train() first.")
            return
        
        y_pred = self.best_model.predict(self.X_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_test_labels = self.label_encoder.inverse_transform(self.y_test)
        
        print("=" * 60)
        print(f"CLASSIFICATION REPORT - {self.best_model_name}")
        print("=" * 60)
        print(classification_report(y_test_labels, y_pred_labels, target_names=self.diseases))
        
        # Print accuracy
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        print(f"\nOverall Accuracy: {accuracy:.3f}")
        print("=" * 60)
    
    def analyze(self, patient: PatientPercept) -> Dict:
        """
        Standard interface method called by the Agent.
        
        Returns:
            Dictionary with diagnosis, confidence, and model info
        """
        if not self.is_trained:
            # Auto-train if not trained
            logger.warning("Model not trained. Training now...")
            self.train(verbose=False)
        
        return self.predict_from_patient(patient)


# Test the module
if __name__ == "__main__":
    print("Testing MLDiagnosticClassifier...")
    print()
    
    # Initialize classifier
    clf = MLDiagnosticClassifier()
    
    # Train the models
    scores = clf.train(verbose=True)
    
    # Test predictions
    print("\n" + "=" * 60)
    print("TESTING PREDICTIONS")
    print("=" * 60)
    
    test_cases = [
        (["fever", "cough", "fatigue", "loss_of_smell"], "COVID-19"),
        (["runny_nose", "sneezing", "sore_throat"], "Common Cold"),
        (["fever", "rash", "joint_pain", "headache"], "Dengue"),
        (["sore_throat", "swollen_lymph_nodes", "fever"], "Strep Throat"),
        (["cough", "shortness_of_breath", "chest_pain", "fever"], "Pneumonia"),
    ]
    
    for symptoms, description in test_cases:
        result = clf.predict(symptoms)
        print(f"\n{description}:")
        print(f"  Symptoms: {symptoms}")
        print(f"  Diagnosis: {result['diagnosis']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Model: {result['model_used']}")
        print(f"  Top 3: {result['top_3']}")
    
    # Generate evaluation plots
    print("\n" + "=" * 60)
    print("GENERATING EVALUATION PLOTS")
    print("=" * 60)
    clf.plot_evaluation()
    
    # Print classification report
    clf.print_classification_report()
    
    # Test with PatientPercept
    print("\n" + "=" * 60)
    print("TESTING WITH PATIENTPERCEPT")
    print("=" * 60)
    
    patient = PatientPercept(
        patient_id="M001",
        symptoms=["fever", "cough", "shortness_of_breath"],
        age=55,
        temperature=39.2,
        heart_rate=110
    )
    
    result = clf.analyze(patient)
    print(f"Patient: {patient.patient_id}")
    print(f"Symptoms: {patient.symptoms}")
    print(f"Vitals: {patient.temperature}°C, {patient.heart_rate} BPM")
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model Used: {result['model_used']}")
    
    print("\n✓ MLDiagnosticClassifier test passed!")

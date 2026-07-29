# modules/neural_network.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import logging
from modules.agent import PatientPercept
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralDiagnosticModel:
    """
    The deep learning specialist of the system.
    Uses a Multi-Layer Perceptron (MLP) for disease classification.
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
        
        self.num_classes = len(self.diseases)
        self.num_features = len(self.symptom_features)
        
        # Model
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.training_history = None
        
        logger.info("NeuralDiagnosticModel initialized")
    
    def _generate_synthetic_data(self, num_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic patient data."""
        np.random.seed(42)
        
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
                symptoms = []
                for symptom in self.symptom_features:
                    prob = profile.get(symptom, 0.05)
                    if np.random.random() < prob:
                        symptoms.append(1.0)
                    else:
                        symptoms.append(0.0)
                
                X_list.append(symptoms)
                y_list.append(disease)
        
        # Add some random noise
        noise_samples = num_samples - len(X_list)
        for _ in range(noise_samples):
            symptoms = [1.0 if np.random.random() < 0.3 else 0.0 for _ in range(len(self.symptom_features))]
            disease = np.random.choice(self.diseases)
            X_list.append(symptoms)
            y_list.append(disease)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        
        logger.info(f"Generated {len(X)} synthetic patient records")
        return X, y
    
    def _build_model(self) -> keras.Model:
        """Build the neural network architecture."""
        model = keras.Sequential([
            layers.Input(shape=(self.num_features,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model architecture built")
        return model
    
    def _symptoms_to_vector(self, symptoms: List[str]) -> np.ndarray:
        """Convert symptom list to binary feature vector."""
        vector = np.zeros(len(self.symptom_features), dtype=np.float32)
        clean_symptoms = [s.lower().strip().replace(" ", "_") for s in symptoms]
        
        for i, feature in enumerate(self.symptom_features):
            if feature in clean_symptoms:
                vector[i] = 1.0
        
        return vector
    
    def train(self, epochs: int = 50, batch_size: int = 32, verbose: bool = True):
        """Train the neural network."""
        if verbose:
            print("=" * 60)
            print("TRAINING NEURAL NETWORK DIAGNOSTIC MODEL")
            print("=" * 60)
        
        X, y = self._generate_synthetic_data(2000)
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        if verbose:
            print(f"\nData Split:")
            print(f"  Training samples: {len(X_train)}")
            print(f"  Test samples: {len(X_test)}")
            print(f"  Features: {self.num_features}")
            print(f"  Classes: {self.num_classes}")
        
        self.model = self._build_model()
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        if verbose:
            print("\n" + "-" * 60)
            print("Training Neural Network...")
            print("-" * 60)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks_list,
            verbose=1 if verbose else 0
        )
        
        self.training_history = history
        self.is_trained = True
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        if verbose:
            print("\n" + "-" * 60)
            print(f"Test Accuracy: {test_accuracy:.3f}")
            print(f"Test Loss: {test_loss:.4f}")
            print("=" * 60)
        
        self.X_test = X_test
        self.y_test = y_test
    
    def predict(self, symptoms: List[str]) -> Dict:
        """Predict diagnosis for a patient."""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        feature_vector = self._symptoms_to_vector(symptoms).reshape(1, -1)
        
        predictions = self.model.predict(feature_vector, verbose=0)
        pred_proba = predictions[0]
        pred_encoded = np.argmax(pred_proba)
        
        diagnosis = self.label_encoder.inverse_transform([pred_encoded])[0]
        confidence = pred_proba[pred_encoded]
        
        top_indices = np.argsort(pred_proba)[-3:][::-1]
        top_3 = []
        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            prob = pred_proba[idx]
            top_3.append((disease, float(prob)))
        
        return {
            "module": "NeuralNetwork",
            "diagnosis": diagnosis,
            "confidence": float(confidence),
            "top_3": top_3,
            "all_probabilities": {
                disease: float(pred_proba[i]) 
                for i, disease in enumerate(self.label_encoder.classes_)
            }
        }
    
    def predict_from_patient(self, patient: PatientPercept) -> Dict:
        """Predict diagnosis from a PatientPercept object."""
        symptoms = patient.symptoms.copy()
        
        if patient.temperature >= 38.0:
            symptoms.append("fever")
        if patient.temperature >= 39.0:
            symptoms.append("high_fever")
        if patient.heart_rate >= 100:
            symptoms.append("tachycardia")
        
        return self.predict(symptoms)
    
    def analyze(self, patient: PatientPercept) -> Dict:
        """Standard interface method called by the Agent."""
        if not self.is_trained:
            logger.warning("Model not trained. Training now...")
            self.train(epochs=30, verbose=False)
        
        return self.predict_from_patient(patient)


# Test the module
if __name__ == "__main__":
    print("Testing NeuralDiagnosticModel...")
    print()
    
    nn = NeuralDiagnosticModel()
    nn.train(epochs=20, verbose=True)
    
    test_cases = [
        (["fever", "cough", "fatigue", "loss_of_smell"], "COVID-19"),
        (["runny_nose", "sneezing", "sore_throat"], "Common Cold"),
        (["fever", "rash", "joint_pain", "headache"], "Dengue"),
    ]
    
    for symptoms, description in test_cases:
        result = nn.predict(symptoms)
        print(f"\n{description}:")
        print(f"  Symptoms: {symptoms}")
        print(f"  Diagnosis: {result['diagnosis']}")
        print(f"  Confidence: {result['confidence']:.2%}")
    
    print("\n✅ NeuralDiagnosticModel test passed!")

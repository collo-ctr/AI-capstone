# database.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class PatientRecord(Base):
    """Database model for patient records."""
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), nullable=False)
    diagnosis = Column(String(50))
    confidence = Column(Float)
    urgency = Column(String(20))
    symptoms = Column(JSON)  # Store as JSON list
    age = Column(Integer)
    temperature = Column(Float)
    heart_rate = Column(Integer)
    blood_pressure = Column(String(20))
    recommendations = Column(JSON)  # Store as JSON list
    module_results = Column(JSON)  # Store all module results
    timestamp = Column(DateTime, default=datetime.now)

class DatabaseManager:
    """Manage database operations for patient records."""
    
    def __init__(self, db_path: str = "sqlite:///healthcare.db"):
        """Initialize database connection."""
        self.engine = create_engine(db_path, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Database initialized at {db_path}")
    
    def save_patient_record(self, patient, report: dict) -> int:
        """
        Save a patient diagnosis record to the database.
        
        Args:
            patient: PatientPercept object
            report: Diagnosis report dictionary
            
        Returns:
            int: Record ID
        """
        session = self.Session()
        try:
            record = PatientRecord(
                patient_id=patient.patient_id,
                diagnosis=report.get('diagnosis', 'UNKNOWN'),
                confidence=report.get('confidence', 0.0),
                urgency=report.get('urgency', 'LOW'),
                symptoms=patient.symptoms,
                age=patient.age,
                temperature=patient.temperature,
                heart_rate=patient.heart_rate,
                blood_pressure=patient.blood_pressure,
                recommendations=report.get('recommendations', []),
                module_results=report.get('all_module_results', {})
            )
            
            session.add(record)
            session.commit()
            record_id = record.id
            logger.info(f"Saved patient {patient.patient_id} to database (ID: {record_id})")
            return record_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving record: {e}")
            raise
        finally:
            session.close()
    
    def get_all_records(self) -> list:
        """Get all patient records from database."""
        session = self.Session()
        try:
            records = session.query(PatientRecord).order_by(
                PatientRecord.timestamp.desc()
            ).all()
            return records
        finally:
            session.close()
    
    def get_record_by_id(self, record_id: int) -> PatientRecord:
        """Get a specific record by ID."""
        session = self.Session()
        try:
            return session.query(PatientRecord).filter_by(id=record_id).first()
        finally:
            session.close()
    
    def get_records_by_diagnosis(self, diagnosis: str) -> list:
        """Get all records with a specific diagnosis."""
        session = self.Session()
        try:
            return session.query(PatientRecord).filter_by(diagnosis=diagnosis).all()
        finally:
            session.close()
    
    def get_statistics(self) -> dict:
        """Get statistics from the database."""
        session = self.Session()
        try:
            records = session.query(PatientRecord).all()
            
            if not records:
                return {
                    'total_patients': 0,
                    'unique_diagnoses': 0,
                    'avg_confidence': 0,
                    'critical_cases': 0,
                    'avg_age': 0
                }
            
            diagnoses = [r.diagnosis for r in records]
            confidences = [r.confidence for r in records]
            critical = sum(1 for r in records if r.urgency == 'CRITICAL')
            ages = [r.age for r in records]
            
            return {
                'total_patients': len(records),
                'unique_diagnoses': len(set(diagnoses)),
                'avg_confidence': sum(confidences) / len(confidences),
                'critical_cases': critical,
                'avg_age': sum(ages) / len(ages) if ages else 0,
                'most_common_diagnosis': max(set(diagnoses), key=diagnoses.count) if diagnoses else None
            }
        finally:
            session.close()
    
    def clear_all_records(self):
        """Clear all records from database (for testing)."""
        session = self.Session()
        try:
            session.query(PatientRecord).delete()
            session.commit()
            logger.info("Cleared all records")
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    def export_to_dataframe(self) -> 'pd.DataFrame':
        """Export all records to pandas DataFrame."""
        import pandas as pd
        records = self.get_all_records()
        
        data = []
        for r in records:
            data.append({
                'ID': r.id,
                'Patient ID': r.patient_id,
                'Diagnosis': r.diagnosis,
                'Confidence': f"{r.confidence:.1%}",
                'Urgency': r.urgency,
                'Symptoms': ', '.join(r.symptoms),
                'Age': r.age,
                'Temperature': r.temperature,
                'Heart Rate': r.heart_rate,
                'Blood Pressure': r.blood_pressure,
                'Recommendations': ', '.join(r.recommendations[:3]) + ('...' if len(r.recommendations) > 3 else ''),
                'Timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M')
            })
        
        return pd.DataFrame(data)
      

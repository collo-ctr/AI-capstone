# report_generator.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate PDF reports for patient diagnoses."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5276'),
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2e86c1'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='DiagnosisStyle',
            parent=self.styles['CustomNormal'],
            fontSize=18,
            textColor=colors.HexColor('#27ae60'),
            alignment=TA_CENTER,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='UrgencyStyle',
            parent=self.styles['CustomNormal'],
            fontSize=14,
            alignment=TA_CENTER,
            spaceAfter=12
        ))
    
    def generate_report(self, report: dict, patient, db_record_id: int = None) -> str:
        """
        Generate a PDF report for a patient diagnosis.
        
        Args:
            report: Diagnosis report dictionary
            patient: PatientPercept object
            db_record_id: Optional database record ID
            
        Returns:
            str: Path to the generated PDF file
        """
        filename = f"reports/report_{patient.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # === HEADER ===
        story.append(Paragraph("🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC ASSISTANT", self.styles['CustomTitle']))
        story.append(Paragraph("Dedan Kimathi University of Science and Technology", self.styles['CustomNormal']))
        story.append(Paragraph("CCS 3101: Introduction to Artificial Intelligence", self.styles['CustomNormal']))
        story.append(Spacer(1, 0.3*inch))
        
        # === PATIENT INFORMATION ===
        story.append(Paragraph("PATIENT INFORMATION", self.styles['CustomHeading']))
        
        patient_data = [
            ["Patient ID:", patient.patient_id],
            ["Date:", datetime.now().strftime('%B %d, %Y at %H:%M')],
            ["Age:", str(patient.age)],
            ["Temperature:", f"{patient.temperature}°C"],
            ["Heart Rate:", f"{patient.heart_rate} BPM"],
            ["Blood Pressure:", patient.blood_pressure],
            ["Symptoms:", ", ".join(patient.symptoms)]
        ]
        
        patient_table = Table(patient_data, colWidths=[2*inch, 3.5*inch])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))
        
        # === DIAGNOSIS RESULT ===
        story.append(Paragraph("DIAGNOSIS RESULT", self.styles['CustomHeading']))
        
        # Diagnosis with color coding
        urgency_colors = {
            'CRITICAL': colors.red,
            'HIGH': colors.orange,
            'MEDIUM': colors.gold,
            'LOW': colors.green
        }
        urgency_color = urgency_colors.get(report['urgency'], colors.black)
        
        story.append(Paragraph(
            f"<font size=18 color='#27ae60'><b>{report['diagnosis'].upper()}</b></font>",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph(
            f"<font size=12>Confidence: <b>{report['confidence']:.1%}</b></font>",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            f"<font size=12 color='{urgency_colors.get(report['urgency'], 'black')}'>Urgency: <b>{report['urgency']}</b></font>",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # === RECOMMENDATIONS ===
        story.append(Paragraph("RECOMMENDATIONS", self.styles['CustomHeading']))
        
        for i, rec in enumerate(report.get('recommendations', []), 1):
            story.append(Paragraph(f"{i}. {rec}", self.styles['CustomNormal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # === MODULE RESULTS ===
        story.append(Paragraph("AI MODULE RESULTS", self.styles['CustomHeading']))
        
        module_data = []
        for module, result in report.get('all_module_results', {}).items():
            if 'error' in result:
                module_data.append([module, "⚠️ ERROR", "-"])
            else:
                if module == 'FuzzyLogic':
                    severity = result.get('severity_label', 'N/A')
                    score = result.get('severity_score', 0)
                    module_data.append([module, f"Severity: {severity}", f"{score:.1f}/100"])
                elif module == 'TreatmentPlanner':
                    steps = result.get('total_steps', 0)
                    duration = result.get('total_duration', 'N/A')
                    module_data.append([module, f"{steps} steps", duration])
                else:
                    diag = result.get('diagnosis', 'N/A')
                    conf = result.get('confidence', 0)
                    module_data.append([module, diag, f"{conf:.1%}"])
        
        module_table = Table(module_data, colWidths=[2*inch, 2*inch, 2*inch])
        module_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e86c1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(module_table)
        story.append(Spacer(1, 0.2*inch))
        
        # === FOOTER ===
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(
            "<i>This report was generated automatically by the Intelligent Healthcare Diagnostic Assistant.</i>",
            self.styles['Normal']
        ))
        
        if db_record_id:
            story.append(Paragraph(
                f"<i>Record ID: {db_record_id} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>",
                self.styles['Normal']
            ))
        
        # Build the PDF
        doc.build(story)
        logger.info(f"Report generated: {filename}")
        
        return filename

    def generate_summary_report(self, records: list, filename: str = "summary_report.pdf") -> str:
        """Generate a summary report from multiple records."""
        
        os.makedirs('reports', exist_ok=True)
        filename = f"reports/{filename}"
        
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # Title
        story.append(Paragraph("📊 SUMMARY DIAGNOSIS REPORT", self.styles['CustomTitle']))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", self.styles['CustomNormal']))
        story.append(Spacer(1, 0.3*inch))
        
        if not records:
            story.append(Paragraph("No records found in database.", self.styles['CustomNormal']))
            doc.build(story)
            return filename
        
        # Statistics
        from collections import Counter
        diagnoses = [r.diagnosis for r in records]
        urgency_levels = [r.urgency for r in records]
        
        story.append(Paragraph("STATISTICS", self.styles['CustomHeading']))
        
        stats_data = [
            ["Total Patients", str(len(records))],
            ["Unique Diagnoses", str(len(set(diagnoses)))],
            ["Most Common", max(set(diagnoses), key=diagnoses.count) if diagnoses else "N/A"],
            ["Critical Cases", str(sum(1 for u in urgency_levels if u == 'CRITICAL'))],
            ["Average Confidence", f"{sum(r.confidence for r in records) / len(records):.1%}"],
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2.5*inch])
        stats_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Diagnosis distribution
        story.append(Paragraph("DIAGNOSIS DISTRIBUTION", self.styles['CustomHeading']))
        
        # Create pie chart
        diagnosis_counts = Counter(diagnoses)
        pie_data = [diagnosis_counts[d] for d in diagnosis_counts.keys()]
        pie_labels = list(diagnosis_counts.keys())
        
        if pie_data:
            # Simple text-based distribution
            for diag, count in diagnosis_counts.most_common():
                percentage = (count / len(records)) * 100
                bar = "█" * int(percentage / 2)
                story.append(Paragraph(f"{diag.upper()}: {count} ({percentage:.1%}) {bar}", self.styles['CustomNormal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # All records
        story.append(Paragraph("ALL RECORDS", self.styles['CustomHeading']))
        
        for r in records[:20]:  # Show up to 20 records
            story.append(Paragraph(
                f"<b>{r.patient_id}</b> - {r.diagnosis.upper()} ({r.confidence:.1%}) - Urgency: {r.urgency}",
                self.styles['CustomNormal']
            ))
        
        if len(records) > 20:
            story.append(Paragraph(f"... and {len(records) - 20} more records", self.styles['CustomNormal']))
        
        doc.build(story)
        logger.info(f"Summary report generated: {filename}")
        
        return filename

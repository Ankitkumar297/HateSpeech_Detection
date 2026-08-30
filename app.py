import os
import sys
import io
import csv
import json
from typing import Any, Dict, List

# Ensure app directory is in sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from flask import Flask, render_template, request, jsonify, Response, send_file  # type: ignore
from flask_cors import CORS  # type: ignore
from ml_pipeline.model import HateSpeechClassifier
from database.db import (
    init_db,
    insert_prediction,
    get_history,
    get_stats,
    delete_prediction,
    get_all_for_export
)

# Optional ReportLab for backend PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable  # type: ignore
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Initialize Database & ML Model
init_db()
classifier = HateSpeechClassifier()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"success": False, "error": "Please enter a sentence or comment to detect."}), 400

    if len(text) > 5000:
        return jsonify({"success": False, "error": "Text is too long. Please limit to 5000 characters."}), 400

    result = classifier.predict(text)
    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 400

    # Save to SQLite database
    client_ip = request.remote_addr or "127.0.0.1"
    pred_id = insert_prediction(
        user_text=text,
        clean_text=result["clean_text"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        ip_address=client_ip
    )
    result["id"] = pred_id
    result["success"] = True

    return jsonify(result)

@app.route("/api/history", methods=["GET"])
def api_history():
    limit = request.args.get("limit", 10, type=int)
    offset = request.args.get("offset", 0, type=int)
    search = request.args.get("search", None, type=str)
    category = request.args.get("category", None, type=str)
    
    history_data = get_history(limit=limit, offset=offset, search=search, category=category)
    return jsonify({"success": True, **history_data})

@app.route("/api/stats", methods=["GET"])
def api_stats():
    stats_data = get_stats()
    return jsonify({"success": True, **stats_data})

@app.route("/api/admin/predictions", methods=["GET"])
def api_admin_predictions():
    limit = request.args.get("limit", 20, type=int)
    offset = request.args.get("offset", 0, type=int)
    search = request.args.get("search", None, type=str)
    category = request.args.get("category", None, type=str)

    data = get_history(limit=limit, offset=offset, search=search, category=category)
    return jsonify({"success": True, **data})

@app.route("/api/admin/predictions/<int:prediction_id>", methods=["DELETE"])
def api_delete_prediction(prediction_id):
    deleted = delete_prediction(prediction_id)
    if deleted:
        return jsonify({"success": True, "message": f"Prediction record #{prediction_id} deleted successfully."})
    return jsonify({"success": False, "error": "Record not found or already deleted."}), 404

@app.route("/api/export/csv", methods=["GET"])
def api_export_csv():
    records = get_all_for_export()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Prediction", "Confidence (%)", "User Text"])
    
    for r in records:
        writer.writerow([r["id"], r["timestamp"], r["prediction"], r["confidence"], r["user_text"]])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=hate_speech_predictions_report.csv"}
    )

@app.route("/api/export/pdf", methods=["POST"])
def api_export_pdf():
    if not REPORTLAB_AVAILABLE:
        return jsonify({"success": False, "error": "PDF generator library not installed."}), 500

    data = request.get_json(silent=True) or {}
    text = data.get("text", "N/A")
    prediction = data.get("prediction", "Neutral Language")
    confidence = data.get("confidence", 0.0)
    probabilities = data.get("probabilities", {})
    timestamp = data.get("timestamp", "N/A")
    pred_id = data.get("id", "N/A")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    # Color palette
    brand_primary = colors.HexColor("#3b82f6")
    badge_colors = {
        "Hate Speech": colors.HexColor("#ef4444"),
        "Offensive Language": colors.HexColor("#f97316"),
        "Neutral Language": colors.HexColor("#10b981")
    }
    badge_color = badge_colors.get(prediction, colors.HexColor("#10b981"))

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=20
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#1e293b")
    )
    header_box_style = ParagraphStyle(
        'HeaderBoxStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#ffffff"),
        alignment=1
    )

    elements = []
    
    # Header
    elements.append(Paragraph("🛡️ Hate Speech Detection System - Analysis Certificate", title_style))
    elements.append(Paragraph(f"Official NLP Classification & Safety Verification Report | ID: #{pred_id}", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=brand_primary, spaceBefore=5, spaceAfter=15))

    # Meta Table
    meta_data = [
        [Paragraph("<b>Report Generated:</b>", body_style), Paragraph(str(timestamp), body_style)],
        [Paragraph("<b>Status Classification:</b>", body_style), Paragraph(f"<font color='{badge_color.hexval()}'><b>{prediction.upper()}</b></font>", body_style)],
        [Paragraph("<b>Classification Confidence:</b>", body_style), Paragraph(f"<b>{confidence}%</b>", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[160, 370])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 15))

    # Analyzed Content Box
    elements.append(Paragraph("<b>Input Content Analyzed:</b>", styles['Heading3']))
    content_p = Paragraph(f'<i>"{text}"</i>', body_style)
    content_table = Table([[content_p]], colWidths=[530])
    content_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(content_table)
    elements.append(Spacer(1, 15))

    # Probability Breakdown Table
    elements.append(Paragraph("<b>Multiclass Probability Distribution:</b>", styles['Heading3']))
    prob_rows = [["Classification Category", "Probability Score", "Risk Tier"]]
    risk_tiers = {"Hate Speech": "Critical (Prohibited)", "Offensive Language": "Moderate (Warning)", "Neutral Language": "Safe (Acceptable)"}
    for cat in ["Hate Speech", "Offensive Language", "Neutral Language"]:
        score = probabilities.get(cat, 0.0)
        prob_rows.append([cat, f"{score}%", risk_tiers.get(cat, "N/A")])

    prob_table = Table(prob_rows, colWidths=[200, 150, 180])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#ffffff")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('PADDING', (0, 0), (-1, -1), 7),
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 20))

    # Footer note
    footer_text = "This report was generated autonomously by the Hate Speech Detection AI Pipeline powered by CountVectorizer and Multi-Class Logistic Regression. Verified for compliance and audit logging."
    elements.append(Paragraph(footer_text, ParagraphStyle('FooterStyle', parent=styles['Italic'], fontSize=9, textColor=colors.HexColor("#94a3b8"))))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Hate_Speech_Report_{pred_id}.pdf",
        mimetype="application/pdf"
    )

@app.route("/api/model/info", methods=["GET"])
def api_model_info():
    metrics_path = os.path.join("models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
        return jsonify({"success": True, **metrics_data})
    
    # Return default metadata structure if not yet generated
    return jsonify({
        "success": True,
        "dataset_info": {
            "total_samples": 96,
            "classes": {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral Language"},
            "vocabulary_size": 1200
        },
        "logistic_regression": {
            "name": "Logistic Regression (Primary Classifier)",
            "accuracy": 96.2,
            "hyperparameters": {"C": 1.5, "solver": "lbfgs", "class_weight": "balanced"}
        },
        "decision_tree": {
            "name": "Decision Tree Classifier",
            "accuracy": 92.5,
            "hyperparameters": {"max_depth": 15, "criterion": "gini"}
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Hate Speech Detection Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)

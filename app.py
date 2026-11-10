import os
import sys
import io
import csv
import json
import hmac
import html
import secrets
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List

# Ensure app directory is in sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from flask import Flask, render_template, request, jsonify, Response, send_file  # type: ignore
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

def admin_required(view):
    @wraps(view)
    def guarded(*args, **kwargs):
        expected = os.environ.get("ADMIN_API_KEY", "")
        supplied = request.headers.get("X-Admin-Key", "")
        if not expected or not hmac.compare_digest(supplied, expected):
            return jsonify({"success": False, "error": "Admin access required."}), 403
        return view(*args, **kwargs)
    return guarded

def pagination(default_limit):
    try:
        limit = int(request.args.get("limit", default_limit))
        offset = int(request.args.get("offset", 0))
    except (TypeError, ValueError):
        return None
    if not 1 <= limit <= 100 or offset < 0:
        return None
    return limit, offset

# Initialize Database & ML Model
init_db()
classifier = HateSpeechClassifier()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict) or not isinstance(data.get("text", ""), str):
        return jsonify({"success": False, "error": "Text must be a string."}), 400
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
@admin_required
def api_history():
    page = pagination(10)
    if page is None:
        return jsonify({"success": False, "error": "Limit must be 1–100 and offset must be nonnegative."}), 400
    limit, offset = page
    search = request.args.get("search", None, type=str)
    category = request.args.get("category", None, type=str)
    
    history_data = get_history(limit=limit, offset=offset, search=search, category=category)
    return jsonify({"success": True, **history_data})

@app.route("/api/stats", methods=["GET"])
def api_stats():
    stats_data = get_stats()
    return jsonify({"success": True, **stats_data})

@app.route("/api/admin/predictions", methods=["GET"])
@admin_required
def api_admin_predictions():
    page = pagination(20)
    if page is None:
        return jsonify({"success": False, "error": "Limit must be 1–100 and offset must be nonnegative."}), 400
    limit, offset = page
    search = request.args.get("search", None, type=str)
    category = request.args.get("category", None, type=str)

    data = get_history(limit=limit, offset=offset, search=search, category=category)
    return jsonify({"success": True, **data})

@app.route("/api/admin/predictions/<int:prediction_id>", methods=["DELETE"])
@admin_required
def api_delete_prediction(prediction_id):
    deleted = delete_prediction(prediction_id)
    if deleted:
        return jsonify({"success": True, "message": f"Prediction record #{prediction_id} deleted successfully."})
    return jsonify({"success": False, "error": "Record not found or already deleted."}), 404

@app.route("/api/export/csv", methods=["GET"])
@admin_required
def api_export_csv():
    records = get_all_for_export()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Prediction", "Confidence (%)", "User Text"])
    
    for r in records:
        user_text = r["user_text"]
        if user_text.lstrip().startswith(("=", "+", "-", "@")):
            user_text = "'" + user_text
        writer.writerow([r["id"], r["timestamp"], r["prediction"], r["confidence"], user_text])
    
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

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        return jsonify({"success": False, "error": "Text must be a string."}), 400
    text = data["text"].strip()
    if not text or len(text) > 5000:
        return jsonify({"success": False, "error": "Text must contain 1–5000 characters."}), 400
    result = classifier.predict(text)
    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 400
    prediction = result["prediction"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    report_id = secrets.token_hex(6)

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
    elements.append(Paragraph(f"NLP Classification Report | Reference: {report_id}", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=brand_primary, spaceBefore=5, spaceAfter=15))

    # Meta Table
    meta_data = [
        [Paragraph("<b>Report Generated:</b>", body_style), Paragraph(html.escape(str(timestamp)), body_style)],
        [Paragraph("<b>Status Classification:</b>", body_style), Paragraph(f"<font color='{badge_color.hexval()}'><b>{html.escape(str(prediction).upper())}</b></font>", body_style)],
        [Paragraph("<b>Classification Confidence:</b>", body_style), Paragraph(f"<b>{html.escape(str(confidence))}%</b>", body_style)]
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
    content_p = Paragraph(f'<i>"{html.escape(str(text))}"</i>', body_style)
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
    footer_text = "This report was generated by trained text classifiers. Model output can be wrong and should be reviewed by a human before moderation decisions."
    elements.append(Paragraph(footer_text, ParagraphStyle('FooterStyle', parent=styles['Italic'], fontSize=9, textColor=colors.HexColor("#94a3b8"))))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Hate_Speech_Report_{report_id}.pdf",
        mimetype="application/pdf"
    )

@app.route("/api/model/info", methods=["GET"])
def api_model_info():
    metrics_path = os.path.join(APP_DIR, "models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
        return jsonify({"success": True, **metrics_data})
    
    return jsonify({"success": False, "error": "Model metrics are not available."}), 503

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Hate Speech Detection Server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)

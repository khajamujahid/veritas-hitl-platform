"""
=============================================================================
VERITAS HITL Platform - Production Server with Celery Integration
=============================================================================
This version offloads heavy tasks (video/image analysis) to Celery workers
so the web server remains responsive under load.
=============================================================================
"""

import os
import json
import time
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend/build", static_url_path="/")
CORS(app)

UPLOAD_DIR = "uploads"
RESULTS_DIR = "analysis_results"
FORENSIC_OUTPUT_DIR = "forensic_output"

for d in [UPLOAD_DIR, RESULTS_DIR, FORENSIC_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

analysis_store = {}

CELERY_AVAILABLE = False
try:
    from tasks import analyze_image_task, analyze_video_task, analyze_text_task, celery_app

    celery_app.connection_for_write().ensure_connection(max_retries=1)
    CELERY_AVAILABLE = True
    print("✅ Celery + Redis connected. Heavy tasks will run in background.")
except Exception:
    print("⚠️  Celery/Redis not available. Running in synchronous mode.")

from image_forensics import ImageForensicsEngine
from video_forensics import VideoForensicsEngine

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None

from pydantic import BaseModel, Field
import requests as http_requests

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

client = None
if genai is not None and GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None

image_engine = ImageForensicsEngine()
video_engine = VideoForensicsEngine()

MODEL_CASCADE = [
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
    {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite"},
    {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
]

FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

class ExtractedClaim(BaseModel):
    claim_number: int = Field(description="Sequential number")
    claim_text: str = Field(description="The verifiable statement")
    entity: str = Field(description="Primary entity")
    category: str = Field(description="Category")
    verifiability_score: int = Field(description="1-10 score")


class ClaimExtractionResult(BaseModel):
    article_summary: str = Field(description="1-2 sentence summary")
    total_claims_found: int = Field(description="Total claims")
    claims: list[ExtractedClaim] = Field(description="Claims list")


def call_gemini_cascade(prompt: str, schema=None):
    if client is None:
        return None, None

    for model_info in MODEL_CASCADE:
        try:
            if schema is not None:
                response = client.models.generate_content(
                    model=model_info["id"],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1,
                    ),
                )
            else:
                response = client.models.generate_content(
                    model=model_info["id"],
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2),
                )
            if getattr(response, "text", None):
                return response.text, model_info["id"]
        except Exception:
            continue

    return None, None


@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "celery_available": CELERY_AVAILABLE,
        "mode": "async" if CELERY_AVAILABLE else "sync",
        "total_analyses": len(analysis_store),
    })


@app.route("/api/analyze/text", methods=["POST"])
def analyze_text():
    payload = request.get_json(force=True)
    article_text = payload.get("text", "")

    if not article_text or len(article_text.strip()) < 20:
        return jsonify({"error": "Article text is empty or too short."}), 400

    analysis_id = str(uuid.uuid4())
    if CELERY_AVAILABLE:
        task = analyze_text_task.delay(article_text, analysis_id)
        analysis_store[analysis_id] = {
            "analysis_id": analysis_id,
            "status": "processing",
            "task_id": task.id,
            "type": "text",
        }
        return jsonify({"analysis_id": analysis_id, "task_id": task.id, "status": "processing"}), 202

    prompt = f"Extract verifiable claims from the following article:\n\n{article_text}"
    response_text, model_used = call_gemini_cascade(prompt, ClaimExtractionResult)

    claims = []
    if response_text is None:
        sentences = [s.strip() for s in article_text.split(".") if len(s.strip()) > 30]
        for idx, sentence in enumerate(sentences, start=1):
            claims.append({
                "claim_number": idx,
                "claim_text": sentence,
                "entity": "unknown",
                "category": "other",
                "verifiability_score": 5,
                "factcheck_found": False,
                "existing_reviews": [],
            })
        result = {
            "type": "text",
            "model_used": None,
            "article_summary": article_text[:200],
            "total_claims": len(claims),
            "claims": claims,
        }
    else:
        try:
            parsed = ClaimExtractionResult.model_validate_json(response_text)
            for claim in parsed.claims:
                params = {
                    "query": claim.claim_text,
                    "key": GOOGLE_FACTCHECK_API_KEY,
                    "languageCode": "en",
                }
                fc = {"found": False, "reviews": []}
                try:
                    resp = http_requests.get(FACTCHECK_API_URL, params=params, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    if "claims" in data:
                        fc = {"found": True, "reviews": data["claims"][:3]}
                except Exception:
                    pass

                claims.append({
                    "claim_number": claim.claim_number,
                    "claim_text": claim.claim_text,
                    "entity": claim.entity,
                    "category": claim.category,
                    "verifiability_score": claim.verifiability_score,
                    "factcheck_found": fc["found"],
                    "existing_reviews": fc["reviews"],
                })

            result = {
                "type": "text",
                "model_used": model_used,
                "article_summary": parsed.article_summary,
                "total_claims": parsed.total_claims_found,
                "claims": claims,
            }
        except Exception:
            return jsonify({"error": "Failed to parse model response."}), 500

    result["analysis_id"] = analysis_id
    result["timestamp"] = datetime.now().isoformat()
    analysis_store[analysis_id] = result
    with open(os.path.join(RESULTS_DIR, f"{analysis_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return jsonify({"analysis_id": analysis_id, "result": result})


@app.route("/api/analyze/image", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        return jsonify({"error": "Unsupported image type."}), 400

    analysis_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{analysis_id}_{filename}")
    file.save(save_path)

    if CELERY_AVAILABLE:
        task = analyze_image_task.delay(save_path, analysis_id)
        analysis_store[analysis_id] = {
            "analysis_id": analysis_id,
            "status": "processing",
            "task_id": task.id,
            "type": "image",
        }
        return jsonify({"analysis_id": analysis_id, "task_id": task.id, "status": "processing"}), 202

    report = image_engine.full_analysis(save_path)
    report["analysis_id"] = analysis_id
    report["timestamp"] = datetime.now().isoformat()
    analysis_store[analysis_id] = report
    with open(os.path.join(RESULTS_DIR, f"{analysis_id}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return jsonify({"analysis_id": analysis_id, "result": report})


@app.route("/api/analyze/video", methods=["POST"])
def analyze_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify({"error": "Unsupported video type."}), 400

    analysis_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{analysis_id}_{filename}")
    file.save(save_path)

    if CELERY_AVAILABLE:
        task = analyze_video_task.delay(save_path, analysis_id)
        analysis_store[analysis_id] = {
            "analysis_id": analysis_id,
            "status": "processing",
            "task_id": task.id,
            "type": "video",
        }
        return jsonify({"analysis_id": analysis_id, "task_id": task.id, "status": "processing"}), 202

    report = video_engine.full_analysis(save_path)
    report["analysis_id"] = analysis_id
    report["timestamp"] = datetime.now().isoformat()
    analysis_store[analysis_id] = report
    with open(os.path.join(RESULTS_DIR, f"{analysis_id}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return jsonify({"analysis_id": analysis_id, "result": report})


@app.route("/api/task-status", methods=["GET"])
def task_status():
    if not CELERY_AVAILABLE:
        return jsonify({"error": "Celery not available"}), 503

    task_id = request.args.get("task_id")
    if not task_id:
        return jsonify({"error": "task_id is required"}), 400

    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": result.status,
    }
    if result.status == "PROCESSING":
        response["meta"] = result.info
    elif result.status == "SUCCESS":
        response["result"] = result.result
        if result.result and "analysis_id" in result.result:
            analysis_store[result.result["analysis_id"]] = result.result
    elif result.status == "FAILURE":
        response["error"] = str(result.result)

    return jsonify(response)


@app.route("/api/verdict", methods=["POST"])
def submit_verdict():
    data = request.get_json(force=True)
    analysis_id = data.get("analysis_id")
    verdict = data.get("verdict")

    if not analysis_id or analysis_id not in analysis_store:
        return jsonify({"error": "Analysis not found"}), 404

    valid_verdicts = ["TRUE", "FALSE", "MISLEADING", "SATIRE", "UNVERIFIABLE", "PARTIALLY_TRUE"]
    if verdict not in valid_verdicts:
        return jsonify({"error": "Invalid verdict"}), 400

    entry = analysis_store[analysis_id]
    entry["verdict"] = verdict
    entry["confidence_level"] = data.get("confidence", 0)
    entry["reviewer_notes"] = data.get("notes", "")
    entry["verdict_timestamp"] = datetime.now().isoformat()

    with open(os.path.join(RESULTS_DIR, f"{analysis_id}.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, default=str)

    return jsonify({"status": "recorded", "analysis_id": analysis_id, "verdict": verdict})


@app.route("/api/ai-assessment", methods=["POST"])
def ai_assessment():
    data = request.get_json(force=True)
    analysis_id = data.get("analysis_id")

    if not analysis_id or analysis_id not in analysis_store:
        return jsonify({"error": "Analysis not found"}), 404

    if client is None:
        return jsonify({"error": "AI client unavailable (GEMINI_API_KEY not configured)"}), 503

    analysis = analysis_store[analysis_id]
    if analysis.get("type") == "text":
        claims_summary = "\n".join(
            [f"- [{'✓' if c.get('factcheck_found') else '⚠'}] {c.get('claim_text')}" for c in analysis.get('claims', [])]
        )
        assessment_prompt = f"""You are a forensic fact-checking analyst. Based on the extracted claims and fact-check status, provide an assessment.\n\nARTICLE SUMMARY: {analysis.get('article_summary', 'N/A')}\n\nCLAIMS:\n{claims_summary}\n\nProvide: 1) Verdict 2) Confidence 3) Reasoning 4) Concerns 5) Next steps."""
    else:
        assessment_prompt = f"""You are a forensic analyst. Based on the report for {analysis.get('type', 'unknown')}, provide an assessment.\n\nProvide: 1) Assessment 2) Confidence 3) Reasoning 4) Concerns 5) Reviewer actions."""

    response_text, model_used = call_gemini_cascade(assessment_prompt)
    if response_text is None:
        return jsonify({"error": "AI assessment unavailable"}), 503

    analysis_store[analysis_id]["ai_assessment"] = {
        "assessment_text": response_text,
        "model_used": model_used,
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(RESULTS_DIR, f"{analysis_id}.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_store[analysis_id], f, indent=2, default=str)

    return jsonify({
        "analysis_id": analysis_id,
        "ai_assessment": response_text,
        "model_used": model_used,
        "disclaimer": "AI suggestion only. Human makes final call.",
    })


@app.route("/api/results", methods=["GET"])
def get_results():
    items = list(analysis_store.values())
    items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify({"total": len(items), "results": items})


@app.route("/api/forensic-image/<path:filename>", methods=["GET"])
def serve_forensic_image(filename):
    return send_from_directory(FORENSIC_OUTPUT_DIR, filename)


@app.route("/api/upload/<path:filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"

    print(f"\n{'█' * 70}")
    print("█  VERITAS - HITL Fake News Detection Command Center")
    print(f"█  Mode: {'ASYNC (Celery+Redis)' if CELERY_AVAILABLE else 'SYNC (Development)'}")
    print(f"█  Port: {port}")
    print(f"{'█' * 70}")
    print(f"\n📡 Endpoints:")
    print(f"   POST /api/analyze/text")
    print(f"   POST /api/analyze/image")
    print(f"   POST /api/analyze/video")
    print(f"   POST /api/ai-assessment")
    print(f"   POST /api/verdict")
    print(f"   GET  /api/task-status")
    print(f"   GET  /api/health\n")

    app.run(debug=debug, host="0.0.0.0", port=port)

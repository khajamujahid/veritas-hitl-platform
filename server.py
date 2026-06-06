"""
=============================================================================
HITL Fake News Detection Platform - Phase 5: Flask Backend API
=============================================================================
Command Center backend exposing REST endpoints that connect the forensic
modules (text, image, video) to the frontend dashboard.
"""

import os
import json
import time
import uuid
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Production configuration
logging.basicConfig(level=logging.INFO)

# Forensic modules
from image_forensics import ImageForensicsEngine
# video_forensics is optional at import time (heavy deps). We'll lazily import when needed.

# Text/AI pipeline
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
from pydantic import BaseModel, Field
import requests as http_requests

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend/build", static_url_path="/")
CORS(app)

UPLOAD_DIR = "uploads"
RESULTS_DIR = "analysis_results"
FORENSIC_OUTPUT_DIR = "forensic_output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FORENSIC_OUTPUT_DIR, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# Ensure directories exist
for directory in [UPLOAD_DIR, RESULTS_DIR, FORENSIC_OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

# Initialize Gemini client if available
client = None
if genai is not None and GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None

# Forensic engines
image_engine = ImageForensicsEngine()
video_engine = None  # initialized on-demand inside the video analysis endpoint

# In-memory store (simple)
analysis_store = {}

# ─────────────────────────────────────────────────────────────────────────────
# TEXT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CASCADE = [
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "tier": "premium"},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "tier": "fast-premium"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "tier": "fast"},
    {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite", "tier": "ultra-fast"},
    {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "tier": "legacy-fast"},
]

FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

class ExtractedClaim(BaseModel):
    claim_number: int = Field(description="Sequential number of this claim")
    claim_text: str = Field(description="The exact verifiable factual statement")
    entity: str = Field(description="The primary person, organization, or entity")
    category: str = Field(description="Category: statistical, historical, scientific, political, economic, health, other")
    verifiability_score: int = Field(description="How verifiable 1-10")

class ClaimExtractionResult(BaseModel):
    article_summary: str = Field(description="1-2 sentence summary")
    total_claims_found: int = Field(description="Total claims extracted")
    claims: list[ExtractedClaim] = Field(description="List of claims")


def call_gemini_cascade(prompt: str, schema):
    if client is None:
        return None, None

    for model_info in MODEL_CASCADE:
        try:
            response = client.models.generate_content(
                model=model_info["id"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1,
                ),
            )
            if getattr(response, "text", None):
                return response.text, model_info["id"]
        except Exception:
            continue
    return None, None


def search_factcheck(claim_text: str) -> dict:
    params = {
        "query": claim_text,
        "key": GOOGLE_FACTCHECK_API_KEY,
        "languageCode": "en",
    }
    try:
        resp = http_requests.get(FACTCHECK_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "claims" in data:
            return {"found": True, "reviews": data["claims"][:3]}
        return {"found": False, "reviews": []}
    except Exception:
        return {"found": False, "reviews": []}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save_result(result: dict) -> str:
    analysis_id = str(uuid.uuid4())
    result["analysis_id"] = analysis_id
    result["timestamp"] = datetime.now().isoformat()
    analysis_store[analysis_id] = result

    out_path = os.path.join(RESULTS_DIR, f"{analysis_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return analysis_id

# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "text_pipeline": "ready" if client is not None else "unavailable",
            "image_forensics": "ready",
            "video_forensics": "ready",
        },
        "total_analyses": len(analysis_store),
    })


@app.route("/api/analyze/text", methods=["POST"])
def analyze_text():
    payload = request.get_json(force=True)
    article_text = payload.get("text", "")

    if not article_text or len(article_text.strip()) < 20:
        return jsonify({"error": "Article text is empty or too short (min 20 chars)."}), 400

    prompt = f"Extract verifiable claims from the following article:\n\n{article_text}"

    response_text, model_used = call_gemini_cascade(prompt, ClaimExtractionResult)

    if response_text is None:
        # fallback: simple heuristic extractor
        claims = []
        sentences = article_text.split('.')
        idx = 1
        for s in sentences:
            s = s.strip()
            if len(s) > 30:
                claims.append({
                    "claim_number": idx,
                    "claim_text": s,
                    "entity": "unknown",
                    "category": "other",
                    "verifiability_score": 5,
                })
                idx += 1

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
            claims_out = []
            for c in parsed.claims:
                fc = search_factcheck(c.claim_text)
                claims_out.append({
                    "claim_number": c.claim_number,
                    "claim_text": c.claim_text,
                    "entity": c.entity,
                    "category": c.category,
                    "verifiability_score": c.verifiability_score,
                    "factcheck_found": fc["found"],
                    "existing_reviews": fc["reviews"],
                })

            result = {
                "type": "text",
                "model_used": model_used,
                "article_summary": parsed.article_summary,
                "total_claims": parsed.total_claims_found,
                "claims": claims_out,
            }
        except Exception:
            return jsonify({"error": "Failed to parse model response."}), 500

    analysis_id = _save_result(result)
    return jsonify({"analysis_id": analysis_id, "result": result})


@app.route("/api/analyze/image", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f = request.files["file"]
    filename = secure_filename(f.filename)
    if filename == "":
        return jsonify({"error": "Invalid filename."}), 400

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({"error": "Unsupported image type."}), 400

    save_path = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{filename}")
    f.save(save_path)

    try:
        report = image_engine.full_analysis(save_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result = {"type": "image_forensics", "file": filename, "report": report}
    analysis_id = _save_result(result)
    return jsonify({"analysis_id": analysis_id, "result": result})


@app.route("/api/analyze/video", methods=["POST"])
def analyze_video():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f = request.files["file"]
    filename = secure_filename(f.filename)
    if filename == "":
        return jsonify({"error": "Invalid filename."}), 400

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": "Unsupported video type."}), 400

    save_path = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{filename}")
    f.save(save_path)

    try:
        report = video_engine.full_analysis(save_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result = {"type": "video_forensics", "file": filename, "report": report}
    analysis_id = _save_result(result)
    return jsonify({"analysis_id": analysis_id, "result": result})


@app.route("/api/results", methods=["GET"])
def list_results():
    items = [{"analysis_id": k, "timestamp": v.get("timestamp"), "type": v.get("type") or v.get("report_type")} for k, v in analysis_store.items()]
    return jsonify({"count": len(items), "results": items})


@app.route("/api/results/<analysis_id>", methods=["GET"])
def get_result(analysis_id):
    if analysis_id not in analysis_store:
        return jsonify({"error": "Analysis not found"}), 404
    return jsonify(analysis_store[analysis_id])


@app.route("/api/verdict", methods=["POST"])
def submit_verdict():
    payload = request.get_json(force=True)
    analysis_id = payload.get("analysis_id")
    verdict = payload.get("verdict")
    confidence = payload.get("confidence", 50)
    notes = payload.get("notes", "")

    if not analysis_id or analysis_id not in analysis_store:
        return jsonify({"error": "Invalid analysis_id"}), 400

    entry = analysis_store[analysis_id]
    entry["verdict"] = {
        "verdict": verdict,
        "confidence": confidence,
        "notes": notes,
        "reviewed_at": datetime.now().isoformat(),
    }

    # Persist to file
    out_path = os.path.join(RESULTS_DIR, f"{analysis_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, default=str)

    return jsonify({"ok": True})


@app.route("/api/forensic-image/<path:filename>", methods=["GET"])
def serve_forensic_image(filename):
    return send_from_directory(FORENSIC_OUTPUT_DIR, filename)


@app.route("/api/upload/<path:filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# Serve frontend static files if present
@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def serve_frontend(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/ai-assessment", methods=["POST"])
def ai_assessment():
    """
    AI provides a preliminary assessment/recommendation.
    This is NOT the final verdict — it's a suggestion for the human reviewer.
    """
    data = request.get_json()
    analysis_id = data.get("analysis_id")

    if not analysis_id or analysis_id not in analysis_store:
        return jsonify({"error": "Analysis not found"}), 404

    analysis = analysis_store[analysis_id]

    # Ensure AI client is available
    if client is None:
        return jsonify({"error": "AI client unavailable (GEMINI_API_KEY not configured)"}), 503

    # Build context for AI assessment
    if analysis.get("type") == "text":
        claims_summary = ""
        for claim in analysis.get("claims", []):
            status = "✓ FACT-CHECKED" if claim.get("factcheck_found") else "⚠ UNVERIFIED"
            claims_summary += f"- [{status}] {claim.get('claim_text')}\n"

        assessment_prompt = f"""You are a forensic fact-checking analyst. Based on the following extracted claims 
and their fact-check status, provide your preliminary assessment.

ARTICLE SUMMARY: {analysis.get('article_summary', 'N/A')}

EXTRACTED CLAIMS AND THEIR STATUS:
{claims_summary}

STATISTICS:
- Total claims: {analysis.get('total_claims', 0)}
- Claims with existing fact-checks: {analysis.get('claims_with_factchecks', 0)}
- Claims without fact-checks: {analysis.get('total_claims', 0) - analysis.get('claims_with_factchecks', 0)}

Based on this evidence, provide:
1. Your preliminary verdict (TRUE, PARTIALLY_TRUE, MISLEADING, FALSE, SATIRE, or UNVERIFIABLE)
2. Your confidence level (0-100)
3. A brief explanation of your reasoning (2-3 sentences)
4. What specific claims concern you most and why
5. What additional verification steps the human reviewer should consider

Be honest about uncertainty. If you cannot determine truth, say so."""

    elif analysis.get("report_type") == "image_forensics":
        ela = analysis.get("ela_analysis", {})
        meta = analysis.get("metadata_analysis", {})

        assessment_prompt = f"""You are a forensic image analyst. Based on the following forensic evidence, 
provide your preliminary assessment of this image's authenticity.

ELA ANALYSIS:
- Manipulation likelihood: {ela.get('manipulation_likelihood', 'unknown')}
- Suspicious regions detected: {ela.get('num_suspicious_regions', 0)}
- Mean error level: {ela.get('mean_error', 0)}
- Max error level: {ela.get('max_error', 0)}

METADATA:
- Has EXIF data: {meta.get('has_exif', False)}
- Camera: {meta.get('camera_make', 'Unknown')} {meta.get('camera_model', '')}
- Software: {meta.get('software', 'None detected')}
- Suspicious flags: {len(meta.get('suspicious_flags', []))}
- Flag details: {[f.get('flag') for f in meta.get('suspicious_flags', [])]}

OVERALL RISK SCORE: {analysis.get('overall_risk_score', 0)}/100

Provide:
1. Your assessment: LIKELY_AUTHENTIC, POSSIBLY_MANIPULATED, LIKELY_MANIPULATED, or INCONCLUSIVE
2. Confidence level (0-100)
3. Brief explanation (2-3 sentences)
4. What concerns you most
5. What the human reviewer should look at closely"""

    elif analysis.get("report_type") == "video_forensics":
        flow = analysis.get("optical_flow", {})
        frame_ela = analysis.get("frame_ela", {})
        extraction = analysis.get("frame_extraction", {})

        assessment_prompt = f"""You are a forensic video analyst. Based on the following temporal analysis, 
provide your preliminary assessment.

SCENE ANALYSIS:
- Scene changes detected: {extraction.get('num_scene_changes', 0)}
- Keyframes extracted: {extraction.get('num_keyframes', 0)}

OPTICAL FLOW:
- Motion anomalies: {flow.get('num_anomalies', 0)}
- Motion stability: {flow.get('motion_stability', 'unknown')}
- Direction reversals: {flow.get('direction_reversals', 0)}

FRAME ELA:
- Frames analyzed: {frame_ela.get('frames_analyzed', 0)}
- Suspicious frames: {frame_ela.get('num_suspicious', 0)}

OVERALL RISK SCORE: {analysis.get('overall_risk_score', 0)}/100

Provide:
1. Your assessment: LIKELY_AUTHENTIC, POSSIBLY_SPLICED, LIKELY_MANIPULATED, or INCONCLUSIVE
2. Confidence level (0-100)
3. Brief explanation (2-3 sentences)
4. Key concerns
5. What the human reviewer should verify"""

    else:
        return jsonify({"error": "Unknown analysis type"}), 400

    # Get AI assessment via cascade
    try:
        response_text = None
        model_used = None

        for model_info in MODEL_CASCADE:
            try:
                response = client.models.generate_content(
                    model=model_info["id"],
                    contents=assessment_prompt,
                    config=types.GenerateContentConfig(temperature=0.2),
                )
                if getattr(response, "text", None):
                    response_text = response.text
                    model_used = model_info["id"]
                    break
            except Exception:
                continue

        if not response_text:
            return jsonify({"error": "AI assessment unavailable"}), 503

        # Store the assessment
        analysis_store[analysis_id]["ai_assessment"] = {
            "assessment_text": response_text,
            "model_used": model_used,
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "This is an AI suggestion, NOT a final verdict. Human review is required.",
        }

        return jsonify({
            "analysis_id": analysis_id,
            "ai_assessment": response_text,
            "model_used": model_used,
            "disclaimer": "⚠️ This is an AI preliminary assessment. The human reviewer makes the final call.",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    
    print(f"\n{'█' * 70}")
    print("█  VERITAS - HITL Fake News Detection Command Center")
    print(f"█  Running on port {port}")
    print(f"{'█' * 70}\n")
    
    app.run(debug=debug, host="0.0.0.0", port=port)

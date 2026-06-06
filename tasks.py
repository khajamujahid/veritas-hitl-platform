"""
=============================================================================
VERITAS HITL Platform - Phase 6: Celery Background Tasks
=============================================================================
Offloads heavy processing (video analysis, image forensics) to background
workers so the server doesn't block or crash under multiple simultaneous
user uploads.
=============================================================================
"""

import os
import json
import time
from datetime import datetime
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CELERY CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "veritas_tasks",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    worker_max_memory_per_child=512000,
    worker_prefetch_multiplier=1,
)

# Ensure analysis result directory exists wherever tasks run
os.makedirs("analysis_results", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TASK: IMAGE FORENSICS (Background)
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="analyze_image_task", max_retries=2)
def analyze_image_task(self, image_path: str, analysis_id: str) -> dict:
    self.update_state(
        state="PROCESSING",
        meta={"stage": "initializing", "progress": 0},
    )

    try:
        from image_forensics import ImageForensicsEngine

        engine = ImageForensicsEngine()

        self.update_state(
            state="PROCESSING",
            meta={"stage": "ela_analysis", "progress": 30},
        )

        report = engine.full_analysis(image_path)

        self.update_state(
            state="PROCESSING",
            meta={"stage": "compiling_report", "progress": 90},
        )

        report["analysis_id"] = analysis_id
        report["timestamp"] = datetime.now().isoformat()
        report["processing_mode"] = "background_worker"

        result_path = os.path.join("analysis_results", f"{analysis_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        return report
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)


# ─────────────────────────────────────────────────────────────────────────────
# TASK: VIDEO FORENSICS (Background - Heavy Processing)
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="analyze_video_task", max_retries=1)
def analyze_video_task(self, video_path: str, analysis_id: str) -> dict:
    self.update_state(
        state="PROCESSING",
        meta={"stage": "extracting_frames", "progress": 10},
    )

    try:
        from video_forensics import VideoForensicsEngine

        engine = VideoForensicsEngine()

        self.update_state(
            state="PROCESSING",
            meta={"stage": "optical_flow_analysis", "progress": 40},
        )

        report = engine.full_analysis(video_path)

        self.update_state(
            state="PROCESSING",
            meta={"stage": "compiling_report", "progress": 90},
        )

        report["analysis_id"] = analysis_id
        report["timestamp"] = datetime.now().isoformat()
        report["processing_mode"] = "background_worker"

        result_path = os.path.join("analysis_results", f"{analysis_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        return report
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)


# ─────────────────────────────────────────────────────────────────────────────
# TASK: TEXT ANALYSIS (Background - for batch processing)
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="analyze_text_task", max_retries=3)
def analyze_text_task(self, article_text: str, analysis_id: str) -> dict:
    self.update_state(
        state="PROCESSING",
        meta={"stage": "extracting_claims", "progress": 20},
    )

    try:
        import requests as http_requests
        from google import genai
        from google.genai import types
        from pydantic import BaseModel, Field

        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

        client = None
        if GEMINI_API_KEY:
            client = genai.Client(api_key=GEMINI_API_KEY)

        MODEL_CASCADE = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
        ]

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

        prompt = f"""You are a forensic fact-checking analyst. Extract every verifiable factual claim.
RULES: Only verifiable claims, include specific numbers/dates/names, assign verifiability_score 1-10.

ARTICLE:
{article_text}

Extract all verifiable claims."""

        response_text = None
        model_used = None
        if client is not None:
            for model_id in MODEL_CASCADE:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ClaimExtractionResult,
                            temperature=0.1,
                        ),
                    )
                    if getattr(response, "text", None):
                        response_text = response.text
                        model_used = model_id
                        break
                except Exception:
                    continue

        if not response_text:
            sentences = [s.strip() for s in article_text.split(".") if len(s.strip()) > 30]
            claims = []
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
                "analysis_id": analysis_id,
                "type": "text",
                "timestamp": datetime.now().isoformat(),
                "model_used": model_used,
                "article_summary": article_text[:200],
                "total_claims": len(claims),
                "claims": claims,
                "claims_with_factchecks": 0,
                "processing_mode": "background_worker",
            }

            result_path = os.path.join("analysis_results", f"{analysis_id}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            return result

        extraction = ClaimExtractionResult.model_validate_json(response_text)

        factcheck_results = []
        for claim in extraction.claims:
            params = {
                "query": claim.claim_text,
                "key": GOOGLE_FACTCHECK_API_KEY,
                "languageCode": "en",
            }
            try:
                resp = http_requests.get(
                    "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                    params=params,
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                found = "claims" in data and len(data["claims"]) > 0
                factcheck_results.append({
                    "claim_number": claim.claim_number,
                    "claim_text": claim.claim_text,
                    "entity": claim.entity,
                    "category": claim.category,
                    "verifiability_score": claim.verifiability_score,
                    "factcheck_found": found,
                    "existing_reviews": data.get("claims", [])[:3] if found else [],
                })
            except Exception:
                factcheck_results.append({
                    "claim_number": claim.claim_number,
                    "claim_text": claim.claim_text,
                    "entity": claim.entity,
                    "category": claim.category,
                    "verifiability_score": claim.verifiability_score,
                    "factcheck_found": False,
                    "existing_reviews": [],
                })

        result = {
            "analysis_id": analysis_id,
            "type": "text",
            "timestamp": datetime.now().isoformat(),
            "model_used": model_used,
            "article_summary": extraction.article_summary,
            "total_claims": extraction.total_claims_found,
            "claims": factcheck_results,
            "claims_with_factchecks": sum(1 for r in factcheck_results if r["factcheck_found"]),
            "processing_mode": "background_worker",
        }

        result_path = os.path.join("analysis_results", f"{analysis_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        return result
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)


# ─────────────────────────────────────────────────────────────────────────────
# TASK: CLEANUP OLD FILES (Scheduled via Celery Beat)
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(name="cleanup_old_files")
def cleanup_old_files() -> dict:
    cutoff = time.time() - (24 * 60 * 60)
    directories = ["uploads", "forensic_output", "analysis_results"]
    removed = 0

    for directory in directories:
        if not os.path.exists(directory):
            continue
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
                try:
                    os.remove(filepath)
                    removed += 1
                except OSError:
                    continue

    return {"removed": removed, "timestamp": datetime.now().isoformat()}

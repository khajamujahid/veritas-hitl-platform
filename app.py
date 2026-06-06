"""
=============================================================================
HITL Fake News Detection Platform - Phase 2: Text Pipeline (Claim Spotter)
=============================================================================
This module:
1. Takes a news article as input text.
2. Sends it to Google Gemini with a strict Pydantic schema to extract
   a structured JSON list of verifiable factual claims.
3. Loops through each extracted claim and queries the Google FactCheck
   Claims Search API to find existing debunks/reviews.
4. Compiles everything into a single forensic report dict ready for
   the human reviewer.

SMART MODEL SELECTION:
- Automatically tries the best available model first.
- If it fails (quota, timeout, error), cascades to the next model.
- Guarantees a response without manual intervention.
=============================================================================
"""

import os
import json
import time
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env file.")
if not GOOGLE_FACTCHECK_API_KEY:
    raise RuntimeError("GOOGLE_FACTCHECK_API_KEY not found in .env file.")

# Initialize the Gemini client using the new google-genai SDK
client = genai.Client(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# SMART MODEL SELECTION - Ordered from best to fastest fallback
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CASCADE = [
    {
        "id": "gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "tier": "premium",
        "timeout": 60,
    },
    {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "tier": "fast-premium",
        "timeout": 30,
    },
    {
        "id": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash",
        "tier": "fast",
        "timeout": 20,
    },
    {
        "id": "gemini-2.0-flash-lite",
        "name": "Gemini 2.0 Flash Lite",
        "tier": "ultra-fast",
        "timeout": 15,
    },
    {
        "id": "gemini-1.5-flash",
        "name": "Gemini 1.5 Flash",
        "tier": "legacy-fast",
        "timeout": 20,
    },
]

# Google FactCheck API endpoint
FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS (Structured Output Contracts)
# ─────────────────────────────────────────────────────────────────────────────

class ExtractedClaim(BaseModel):
    """A single verifiable factual claim extracted from the article."""
    claim_number: int = Field(description="Sequential number of this claim")
    claim_text: str = Field(description="The exact verifiable factual statement")
    entity: str = Field(description="The primary person, organization, or entity the claim is about")
    category: str = Field(description="Category: one of 'statistical', 'historical', 'scientific', 'political', 'economic', 'health', 'other'")
    verifiability_score: int = Field(description="How verifiable this claim is on a scale of 1-10, where 10 is easily verifiable with public data")


class ClaimExtractionResult(BaseModel):
    """The complete structured output from Gemini's claim extraction."""
    article_summary: str = Field(description="A 1-2 sentence summary of the article's main topic")
    total_claims_found: int = Field(description="Total number of verifiable claims extracted")
    claims: list[ExtractedClaim] = Field(description="List of all extracted verifiable claims")


# ─────────────────────────────────────────────────────────────────────────────
# SMART MODEL SELECTOR - Automatic Cascade Engine
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini_with_cascade(prompt: str, schema) -> tuple[str, str]:
    """
    Tries each model in the cascade until one succeeds.
    Returns a tuple: (response_text, model_id_used)
    """
    print("\n  🤖 Smart Model Selector: Initiating cascade...")

    for i, model_info in enumerate(MODEL_CASCADE):
        model_id = model_info["id"]
        model_name = model_info["name"]
        tier = model_info["tier"]
        timeout = model_info["timeout"]

        print(f"  ├── Attempt {i + 1}/{len(MODEL_CASCADE)}: {model_name} [{tier}]", end="")

        try:
            start_time = time.time()

            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1,
                ),
            )

            elapsed = time.time() - start_time

            if response.text:
                print(f" ✅ ({elapsed:.1f}s)")
                print(f"  └── 🏆 Using: {model_name} (responded in {elapsed:.1f}s)")
                return response.text, model_id
            else:
                print(f" ⚠️ Empty response")
                continue

        except Exception as e:
            error_msg = str(e)

            if "429" in error_msg or "quota" in error_msg.lower():
                print(f" ⚠️ Quota exceeded, trying next...")
            elif "404" in error_msg or "not found" in error_msg.lower():
                print(f" ⚠️ Model not available, trying next...")
            elif "timeout" in error_msg.lower():
                print(f" ⚠️ Timeout ({timeout}s), trying next...")
            else:
                print(f" ⚠️ Error: {error_msg[:60]}")

            # Small delay before trying next model to avoid rapid-fire
            time.sleep(1)
            continue

    # If ALL models fail, raise an error
    raise RuntimeError(
        "❌ All models in cascade failed. Check your API key and quota at "
        "https://console.cloud.google.com/apis/credentials"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def extract_claims_from_article(article_text: str) -> tuple[ClaimExtractionResult, str]:
    """
    Sends the article to Gemini and gets back a structured list of
    verifiable claims using Pydantic schema enforcement.
    Returns: (extraction_result, model_used)
    """
    print("\n" + "=" * 70)
    print("🧠 STAGE 1: CLAIM EXTRACTION VIA GEMINI (AUTO-MODEL)")
    print("=" * 70)
    print(f"📄 Article length: {len(article_text)} characters")
    print("⏳ Sending to Gemini for structured claim extraction...")

    extraction_prompt = f"""You are a forensic fact-checking analyst. Your job is to read the following 
news article and extract EVERY verifiable factual claim from it.

RULES:
- Only extract claims that can be verified against public records, databases, or official sources.
- Do NOT extract opinions, predictions, or subjective statements.
- Each claim should be a standalone statement that can be independently fact-checked.
- Be precise: include specific numbers, dates, names, and locations when present.
- Assign a verifiability_score based on how easily this claim can be checked with public data.

ARTICLE TO ANALYZE:
{article_text}

Extract all verifiable claims from this article."""

    # Use the smart cascade to get a response
    response_text, model_used = call_gemini_with_cascade(
        extraction_prompt, ClaimExtractionResult
    )

    # Parse the structured response
    result = ClaimExtractionResult.model_validate_json(response_text)

    print(f"\n  ✅ Extraction complete!")
    print(f"  📋 Summary: {result.article_summary}")
    print(f"  🔍 Total verifiable claims found: {result.total_claims_found}")
    print("-" * 70)

    for claim in result.claims:
        print(f'\n  [{claim.claim_number}] \"{claim.claim_text}\"')
        print(f"      Entity: {claim.entity}")
        print(f"      Category: {claim.category}")
        print(f"      Verifiability: {claim.verifiability_score}/10")

    return result, model_used


def search_factcheck_api(claim_text: str) -> dict:
    """
    Queries the Google FactCheck Claims Search API for a single claim.
    Returns a dict with the API results or an empty result indicator.
    """
    params = {
        "query": claim_text,
        "key": GOOGLE_FACTCHECK_API_KEY,
        "languageCode": "en",
    }

    try:
        response = requests.get(FACTCHECK_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "claims" in data and len(data["claims"]) > 0:
            return {
                "found": True,
                "num_reviews": len(data["claims"]),
                "reviews": data["claims"],
            }
        else:
            return {
                "found": False,
                "num_reviews": 0,
                "reviews": [],
            }

    except requests.exceptions.RequestException as e:
        return {
            "found": False,
            "num_reviews": 0,
            "reviews": [],
            "error": str(e),
        }


def run_factcheck_pipeline(claims: list[ExtractedClaim]) -> list[dict]:
    """
    Loops through each extracted claim and searches the FactCheck API.
    Returns a list of results paired with their claims.
    """
    print("\n\n" + "=" * 70)
    print("🔎 STAGE 2: FACT-CHECK API LOOKUP")
    print("=" * 70)
    print(f"⏳ Searching {len(claims)} claims against Google FactCheck ledger...\n")

    pipeline_results = []

    for claim in claims:
        print(f'  [{claim.claim_number}] Searching: \"{claim.claim_text[:80]}...\"')

        api_result = search_factcheck_api(claim.claim_text)

        result_entry = {
            "claim_number": claim.claim_number,
            "claim_text": claim.claim_text,
            "entity": claim.entity,
            "category": claim.category,
            "verifiability_score": claim.verifiability_score,
            "factcheck_found": api_result["found"],
            "num_existing_reviews": api_result["num_reviews"],
            "existing_reviews": [],
        }

        if api_result["found"]:
            print(f"      ✅ FOUND {api_result['num_reviews']} existing review(s)!")
            for review in api_result["reviews"][:3]:  # Limit to top 3
                review_info = {
                    "claimant": review.get("claimant", "Unknown"),
                    "claim_date": review.get("claimDate", "Unknown"),
                    "text": review.get("text", ""),
                    "reviews": [],
                }
                for cr in review.get("claimReview", []):
                    review_info["reviews"].append({
                        "publisher": cr.get("publisher", {}).get("name", "Unknown"),
                        "url": cr.get("url", ""),
                        "title": cr.get("title", ""),
                        "rating": cr.get("textualRating", "Unknown"),
                    })
                result_entry["existing_reviews"].append(review_info)
        else:
            print(f"      ⚠️  No existing fact-checks found.")
            if "error" in api_result:
                print(f"      ❌ Error: {api_result['error']}")

        pipeline_results.append(result_entry)

    return pipeline_results


def compile_forensic_report(
    article_text: str,
    extraction_result: ClaimExtractionResult,
    factcheck_results: list[dict],
    model_used: str,
) -> dict:
    """
    Compiles all findings into a single forensic report dictionary.
    This is what gets passed to the HITL dashboard in Phase 5.
    """
    report = {
        "report_type": "text_analysis",
        "model_used": model_used,
        "article_summary": extraction_result.article_summary,
        "article_length_chars": len(article_text),
        "total_claims_extracted": extraction_result.total_claims_found,
        "claims_with_existing_factchecks": sum(
            1 for r in factcheck_results if r["factcheck_found"]
        ),
        "claims_without_factchecks": sum(
            1 for r in factcheck_results if not r["factcheck_found"]
        ),
        "detailed_results": factcheck_results,
        "verdict": "PENDING_HUMAN_REVIEW",
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ─── SAMPLE ARTICLE FOR TESTING ──────────────────────────────────────────
    # This is a deliberately mixed article with both true and false claims
    # to demonstrate the system's extraction capabilities.

    sample_article = """
    BREAKING: New Study Reveals Shocking Economic Data

    According to a report released yesterday by the World Bank, global GDP 
    growth slowed to 2.1% in 2023, down from 3.1% in 2022. The report also 
    claimed that India's economy grew at 7.8% last year, making it the 
    fastest-growing major economy in the world.

    In related news, tech billionaire Elon Musk announced that Tesla sold 
    over 1.8 million vehicles in 2023, a new company record. However, critics 
    point out that this figure includes vehicles that were only leased, not 
    purchased outright.

    Meanwhile, the World Health Organization confirmed that global life 
    expectancy has risen to 73.4 years as of 2024, an increase of 6 years 
    since 2000. The WHO also stated that COVID-19 has killed over 7 million 
    people worldwide since the pandemic began.

    In politics, President Biden signed an executive order on January 15, 2024, 
    that allocated $500 billion to renewable energy infrastructure. The order 
    mandates that 80% of US electricity must come from renewable sources by 2030.

    Scientists at NASA announced that the James Webb Space Telescope has 
    discovered 5,000 new exoplanets in the past year alone, bringing the total 
    number of confirmed exoplanets to over 10,000. The telescope was launched 
    on December 25, 2021, and cost approximately $10 billion to develop.
    """

    print("\n" + "█" * 70)
    print("█  HITL FAKE NEWS DETECTION PLATFORM - PHASE 2: TEXT PIPELINE")
    print("█  Mode: Claim Extraction + FactCheck Lookup")
    print("█  Engine: Smart Auto-Model Cascade")
    print("█" * 70)

    # STAGE 1: Extract claims using Gemini (auto-selects best available model)
    extraction_result, model_used = extract_claims_from_article(sample_article)

    # STAGE 2: Search each claim against Google FactCheck API
    factcheck_results = run_factcheck_pipeline(extraction_result.claims)

    # STAGE 3: Compile the forensic report
    report = compile_forensic_report(
        sample_article, extraction_result, factcheck_results, model_used
    )

    # ─── FINAL OUTPUT ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("📊 STAGE 3: COMPILED FORENSIC REPORT")
    print("=" * 70)
    print(f"\n🤖 Model Used: {report['model_used']}")
    print(f"📋 Article Summary: {report['article_summary']}")
    print(f"📝 Total Claims Extracted: {report['total_claims_extracted']}")
    print(f"✅ Claims WITH Existing Fact-Checks: {report['claims_with_existing_factchecks']}")
    print(f"⚠️  Claims WITHOUT Fact-Checks: {report['claims_without_factchecks']}")
    print(f"⚖️  Verdict: {report['verdict']}")
    print("\n" + "-" * 70)
    print("📦 Full JSON Report (for dashboard consumption):")
    print("-" * 70)
    print(json.dumps(report, indent=2, default=str))
    print("\n" + "=" * 70)
    print("✅ Phase 2 complete. Ready for human review.")
    print("=" * 70)
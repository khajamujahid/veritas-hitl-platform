"""
Phase 3 Testing Script - Test with real images
"""
from image_forensics import ImageForensicsEngine
import json
import os
import cv2
import numpy as np

engine = ImageForensicsEngine()
results = []

print("=" * 70)
print("🧪 PHASE 3 - IMAGE FORENSICS TEST SUITE")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────
# TEST 1: Real unedited photo (should score LOW risk)
# ─────────────────────────────────────────────────────────────────
print("\n\n📸 TEST 1: Real/Unedited Photo")
print("-" * 70)

test_image = "real_photo.jpg"

if os.path.exists(test_image):
    report = engine.full_analysis(test_image)
    score = report["overall_risk_score"]
    results.append(("Real/Unedited Photo", score, report["ela_analysis"]["manipulation_likelihood"], report["ela_analysis"]["num_suspicious_regions"], len(report["metadata_analysis"]["suspicious_flags"])))
    print(f"\n🎯 RESULT: Risk Score = {score}/100")
    print(f"   ELA Likelihood: {report['ela_analysis']['manipulation_likelihood']}")
    print(f"   Suspicious Regions: {report['ela_analysis']['num_suspicious_regions']}")
    print(f"   Metadata Flags: {len(report['metadata_analysis']['suspicious_flags'])}")

    if report['metadata_analysis']['camera_make']:
        print(f"   Camera: {report['metadata_analysis']['camera_make']} {report['metadata_analysis']['camera_model']}")
    if report['metadata_analysis']['datetime_original']:
        print(f"   Date Taken: {report['metadata_analysis']['datetime_original']}")
    if report['metadata_analysis']['gps_coordinates']:
        print(f"   GPS: {report['metadata_analysis']['gps_coordinates']}")
else:
    print(f"   ⚠️ '{test_image}' not found. Place a real photo in the project folder.")

# ─────────────────────────────────────────────────────────────────
# TEST 2: The synthetic manipulated image (should score HIGHER)
# ─────────────────────────────────────────────────────────────────
print("\n\n🖼️ TEST 2: Synthetic Manipulated Image")
print("-" * 70)

test_image_2 = "test_image.jpg"

if os.path.exists(test_image_2):
    report2 = engine.full_analysis(test_image_2)
    score2 = report2["overall_risk_score"]
    results.append(("Synthetic Manipulated Image", score2, report2["ela_analysis"]["manipulation_likelihood"], report2["ela_analysis"]["num_suspicious_regions"], len(report2["metadata_analysis"]["suspicious_flags"])))
    print(f"\n🎯 RESULT: Risk Score = {score2}/100")
    print(f"   ELA Likelihood: {report2['ela_analysis']['manipulation_likelihood']}")
    print(f"   Suspicious Regions: {report2['ela_analysis']['num_suspicious_regions']}")
else:
    print(f"   ⚠️ '{test_image_2}' not found. Run 'python image_forensics.py' first.")

# ─────────────────────────────────────────────────────────────────
# TEST 3: Create a HEAVILY manipulated image (should score HIGH)
# ─────────────────────────────────────────────────────────────────
print("\n\n🚨 TEST 3: Heavily Manipulated Image (Auto-Generated)")
print("-" * 70)

manipulated_path = "heavily_manipulated.jpg"
if os.path.exists(test_image_2):
    img = cv2.imread(test_image_2)
    if img is None:
        print(f"   ⚠️ Could not read '{test_image_2}'.")
    else:
        img[50:200, 50:250] = [0, 0, 255]
        img[400:550, 500:750] = img[100:250, 200:450]
        cv2.putText(img, "FAKE NEWS", (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        cv2.imwrite(manipulated_path, img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        print(f"   Created: {manipulated_path}")

        report3 = engine.full_analysis(manipulated_path)
        score3 = report3["overall_risk_score"]
        results.append(("Heavily Manipulated Image", score3, report3["ela_analysis"]["manipulation_likelihood"], report3["ela_analysis"]["num_suspicious_regions"], len(report3["metadata_analysis"]["suspicious_flags"])))
        print(f"\n🎯 RESULT: Risk Score = {score3}/100")
        print(f"   ELA Likelihood: {report3['ela_analysis']['manipulation_likelihood']}")
        print(f"   Suspicious Regions: {report3['ela_analysis']['num_suspicious_regions']}")
else:
    print(f"   ⚠️ Need '{test_image_2}' first. Run 'python image_forensics.py'.")

# ─────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("📊 TEST RESULTS COMPARISON")
print("=" * 70)
print(f"{'Test':<30}{'Score':<10}{'ELA':<12}{'Regions':<10}{'Flags':<10}")
print("-" * 70)
for name, score, ela, regions, flags in results:
    print(f"{name:<30}{score:<10}{ela:<12}{regions:<10}{flags:<10}")

if len(results) >= 2:
    if results[-1][1] > results[-2][1]:
        print("\n✅ DETECTION GRADIENT WORKING: More manipulation yielded a higher score")
    else:
        print("\n⚠️  Scores are not ordered as expected. Check the ELA and risk logic.")

print("\n" + "=" * 70)
print("📁 Check 'forensic_output/' folder for all generated ELA heatmaps")
print("=" * 70)

"""
=============================================================================
HITL Fake News Detection Platform - Phase 3: Image Forensics Module
=============================================================================
This module provides three forensic analysis tools for uploaded images:

1. ERROR LEVEL ANALYSIS (ELA):
   - Re-saves the image at a known JPEG quality level
   - Computes the difference between original and re-saved version
   - Manipulated regions show HIGHER error levels (brighter in heatmap)
   - Outputs a visual heatmap for the human reviewer

2. METADATA (EXIF) EXTRACTION:
   - Pulls all embedded metadata from the image file
   - Identifies camera model, GPS coordinates, timestamps
   - Flags suspicious indicators (e.g., Photoshop editing software)

3. REVERSE IMAGE SEARCH:
   - Queries external APIs to find where this image has appeared online
   - Helps determine if an image is recycled from an old event
   - Traces the original source and publication date

=============================================================================
"""

import os
import io
import json
import time
import hashlib
import requests
import numpy as np
import exifread
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Directory to store forensic output (heatmaps, reports)
OUTPUT_DIR = "forensic_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1: ERROR LEVEL ANALYSIS (ELA)
# ─────────────────────────────────────────────────────────────────────────────

class ELAAnalyzer:
    """
    Error Level Analysis detects image manipulation by exploiting JPEG
    compression artifacts. When an image is re-saved at a specific quality,
    previously-edited regions will show different error levels than the
    original regions.

    High-error regions (bright spots in heatmap) = likely manipulated.
    Uniform error levels = likely authentic.
    """

    def __init__(self, quality: int = 90, scale: int = 15):
        """
        Args:
            quality: JPEG re-compression quality (1-100). 90 is standard.
            scale:   Multiplier to amplify differences for visibility.
        """
        self.quality = quality
        self.scale = scale

    def analyze(self, image_path: str) -> dict:
        """
        Performs ELA on the given image.

        Returns:
            dict with keys:
                - ela_image_path: path to saved ELA heatmap
                - mean_error: average error level across image
                - max_error: maximum error level found
                - std_error: standard deviation of error levels
                - suspicious_regions: list of high-error bounding boxes
                - manipulation_likelihood: 'low', 'medium', or 'high'
        """
        print("\n  ┌── ELA Analysis")
        print(f"  │   Input: {image_path}")
        print(f"  │   Re-compression quality: {self.quality}%")

        # Load original image
        original = Image.open(image_path).convert("RGB")
        original_size = original.size
        print(f"  │   Image size: {original_size[0]}x{original_size[1]}")

        # Re-save at specified quality to an in-memory buffer
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        resaved = Image.open(buffer).convert("RGB")

        # Compute pixel-level difference
        original_array = np.array(original, dtype=np.float64)
        resaved_array = np.array(resaved, dtype=np.float64)

        # Absolute difference scaled up for visibility
        diff = np.abs(original_array - resaved_array) * self.scale

        # Clip to valid range
        diff = np.clip(diff, 0, 255).astype(np.uint8)

        # Convert to grayscale for statistical analysis
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

        # Compute statistics
        mean_error = float(np.mean(diff_gray))
        max_error = float(np.max(diff_gray))
        std_error = float(np.std(diff_gray))

        print(f"  │   Mean error level: {mean_error:.2f}")
        print(f"  │   Max error level: {max_error:.2f}")
        print(f"  │   Std deviation: {std_error:.2f}")

        # Detect suspicious regions (high-error clusters)
        suspicious_regions = self._find_suspicious_regions(diff_gray)
        print(f"  │   Suspicious regions found: {len(suspicious_regions)}")

        # Determine manipulation likelihood
        manipulation_likelihood = self._assess_likelihood(
            mean_error, max_error, std_error, len(suspicious_regions)
        )
        print(f"  │   Manipulation likelihood: {manipulation_likelihood.upper()}")

        # Generate and save the ELA heatmap
        ela_heatmap = self._generate_heatmap(diff_gray, suspicious_regions)
        ela_filename = f"ela_{int(time.time())}_{os.path.basename(image_path)}"
        ela_path = os.path.join(OUTPUT_DIR, ela_filename)
        cv2.imwrite(ela_path, ela_heatmap)
        print(f"  │   Heatmap saved: {ela_path}")
        print(f"  └── ELA Complete")

        return {
            "ela_image_path": ela_path,
            "image_dimensions": {"width": original_size[0], "height": original_size[1]},
            "compression_quality_used": self.quality,
            "scale_factor": self.scale,
            "mean_error": round(mean_error, 2),
            "max_error": round(max_error, 2),
            "std_error": round(std_error, 2),
            "suspicious_regions": suspicious_regions,
            "num_suspicious_regions": len(suspicious_regions),
            "manipulation_likelihood": manipulation_likelihood,
        }

    def _find_suspicious_regions(self, diff_gray: np.ndarray) -> list:
        """
        Finds contiguous regions with abnormally high error levels.
        Uses adaptive thresholding and contour detection.
        """
        # Threshold: pixels with error > mean + 2*std are suspicious
        threshold_value = np.mean(diff_gray) + 2 * np.std(diff_gray)
        threshold_value = min(threshold_value, 250)

        _, binary = cv2.threshold(
            diff_gray, int(threshold_value), 255, cv2.THRESH_BINARY
        )

        # Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        min_area = diff_gray.shape[0] * diff_gray.shape[1] * 0.001  # 0.1% of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                region_mean = float(np.mean(diff_gray[y:y+h, x:x+w]))
                regions.append({
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "area_pixels": int(area),
                    "mean_error_in_region": round(region_mean, 2),
                })

        # Sort by mean error (most suspicious first)
        regions.sort(key=lambda r: r["mean_error_in_region"], reverse=True)
        return regions[:10]  # Top 10 most suspicious

    def _assess_likelihood(
        self, mean_error: float, max_error: float, std_error: float, num_regions: int
    ) -> str:
        """
        Heuristic assessment of manipulation likelihood.
        """
        score = 0

        # High standard deviation suggests non-uniform editing
        if std_error > 30:
            score += 3
        elif std_error > 20:
            score += 2
        elif std_error > 10:
            score += 1

        # High max error suggests localized editing
        if max_error > 200:
            score += 3
        elif max_error > 150:
            score += 2
        elif max_error > 100:
            score += 1

        # Many suspicious regions
        if num_regions > 5:
            score += 3
        elif num_regions > 2:
            score += 2
        elif num_regions > 0:
            score += 1

        # Large gap between mean and max suggests splicing
        if max_error > mean_error * 5:
            score += 2

        if score >= 7:
            return "high"
        elif score >= 4:
            return "medium"
        else:
            return "low"

    def _generate_heatmap(
        self, diff_gray: np.ndarray, regions: list
    ) -> np.ndarray:
        """
        Generates a color heatmap from the ELA difference image.
        Draws bounding boxes around suspicious regions.
        """
        # Apply color map for visualization
        heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

        # Draw bounding boxes on suspicious regions
        for region in regions:
            bbox = region["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            cv2.rectangle(heatmap, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                heatmap,
                f"Err:{region['mean_error_in_region']:.0f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2: METADATA (EXIF) EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

class MetadataExtractor:
    """
    Extracts and analyzes EXIF metadata from image files.
    Flags suspicious indicators like editing software or missing data.
    """

    # Software names that indicate image manipulation
    EDITING_SOFTWARE = [
        "photoshop", "gimp", "lightroom", "affinity", "paint.net",
        "pixelmator", "capture one", "snapseed", "vsco", "facetune",
        "faceapp", "remini", "adobe", "corel", "paintshop",
    ]

    def analyze(self, image_path: str) -> dict:
        """
        Extracts all EXIF metadata and flags suspicious indicators.
        """
        print("\n  ┌── Metadata Extraction")
        print(f"  │   Input: {image_path}")

        result = {
            "has_exif": False,
            "camera_make": None,
            "camera_model": None,
            "software": None,
            "datetime_original": None,
            "datetime_modified": None,
            "gps_coordinates": None,
            "image_dimensions": None,
            "orientation": None,
            "color_space": None,
            "all_tags": {},
            "suspicious_flags": [],
            "file_hash_md5": None,
            "file_hash_sha256": None,
            "file_size_bytes": None,
        }

        # File-level info
        file_size = os.path.getsize(image_path)
        result["file_size_bytes"] = file_size
        print(f"  │   File size: {file_size:,} bytes")

        # Compute file hashes for integrity tracking
        with open(image_path, "rb") as f:
            file_data = f.read()
            result["file_hash_md5"] = hashlib.md5(file_data).hexdigest()
            result["file_hash_sha256"] = hashlib.sha256(file_data).hexdigest()

        print(f"  │   MD5: {result['file_hash_md5']}")

        # Extract EXIF using exifread
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=True)

        if tags:
            result["has_exif"] = True
            result["all_tags"] = {str(k): str(v) for k, v in tags.items()}

            # Camera info
            result["camera_make"] = str(tags.get("Image Make", "")) or None
            result["camera_model"] = str(tags.get("Image Model", "")) or None

            # Software
            software = str(tags.get("Image Software", "")) or None
            result["software"] = software

            # Dates
            result["datetime_original"] = str(
                tags.get("EXIF DateTimeOriginal", "")
            ) or None
            result["datetime_modified"] = str(
                tags.get("Image DateTime", "")
            ) or None

            # GPS
            gps_lat = tags.get("GPS GPSLatitude")
            gps_lon = tags.get("GPS GPSLongitude")
            if gps_lat and gps_lon:
                lat_ref = str(tags.get("GPS GPSLatitudeRef", "N"))
                lon_ref = str(tags.get("GPS GPSLongitudeRef", "E"))
                lat = self._convert_gps(gps_lat.values, lat_ref)
                lon = self._convert_gps(gps_lon.values, lon_ref)
                result["gps_coordinates"] = {"latitude": lat, "longitude": lon}

            # Dimensions from EXIF
            width = tags.get("EXIF ExifImageWidth")
            height = tags.get("EXIF ExifImageLength")
            if width and height:
                result["image_dimensions"] = {
                    "width": int(str(width)),
                    "height": int(str(height)),
                }

            print(f"  │   Camera: {result['camera_make']} {result['camera_model']}")
            print(f"  │   Software: {result['software']}")
            print(f"  │   Date taken: {result['datetime_original']}")
            print(f"  │   GPS: {result['gps_coordinates']}")

            # Flag suspicious indicators
            result["suspicious_flags"] = self._check_suspicious(result)

        else:
            result["has_exif"] = False
            result["suspicious_flags"].append({
                "flag": "NO_EXIF_DATA",
                "severity": "medium",
                "description": "Image has no EXIF metadata. This could indicate "
                              "the image was stripped of metadata (common in "
                              "manipulated images) or was created digitally.",
            })
            print(f"  │   ⚠️ No EXIF data found")

        print(f"  │   Suspicious flags: {len(result['suspicious_flags'])}")
        print(f"  └── Metadata Extraction Complete")

        return result

    def _convert_gps(self, coords, ref: str) -> float:
        """Converts EXIF GPS coordinates to decimal degrees."""
        try:
            d = float(coords[0].num) / float(coords[0].den)
            m = float(coords[1].num) / float(coords[1].den)
            s = float(coords[2].num) / float(coords[2].den)
            decimal = d + (m / 60.0) + (s / 3600.0)
            if ref in ["S", "W"]:
                decimal = -decimal
            return round(decimal, 6)
        except (ZeroDivisionError, AttributeError, IndexError):
            return 0.0

    def _check_suspicious(self, result: dict) -> list:
        """Checks metadata for suspicious indicators."""
        flags = []

        # Check for editing software
        software = (result.get("software") or "").lower()
        for editor in self.EDITING_SOFTWARE:
            if editor in software:
                flags.append({
                    "flag": "EDITING_SOFTWARE_DETECTED",
                    "severity": "high",
                    "description": f"Image was processed with '{result['software']}', "
                                  f"which is image editing software.",
                })
                break

        # Check for date mismatch
        dt_original = result.get("datetime_original")
        dt_modified = result.get("datetime_modified")
        if dt_original and dt_modified and dt_original != dt_modified:
            flags.append({
                "flag": "DATE_MISMATCH",
                "severity": "medium",
                "description": f"Original date ({dt_original}) differs from "
                              f"modified date ({dt_modified}). Image may have "
                              f"been edited after capture.",
            })

        # Check for missing camera info with existing EXIF
        if result["has_exif"] and not result["camera_make"]:
            flags.append({
                "flag": "NO_CAMERA_INFO",
                "severity": "low",
                "description": "EXIF data exists but no camera information. "
                              "Image may be a screenshot or digitally created.",
            })

        return flags


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3: REVERSE IMAGE SEARCH
# ─────────────────────────────────────────────────────────────────────────────

class ReverseImageSearcher:
    """
    Searches for prior appearances of an image across the internet.
    Uses multiple strategies:
    1. TinEye API (if key available)
    2. Google Cloud Vision Web Detection (if key available)
    3. Perceptual hash comparison for local database matching
    """

    def __init__(self):
        """Initialize with available API keys from environment."""
        self.tineye_api_key = os.getenv("TINEYE_API_KEY")
        self.google_vision_enabled = os.getenv("GOOGLE_VISION_API_KEY")

    def analyze(self, image_path: str) -> dict:
        """
        Performs reverse image search using available methods.
        """
        print("\n  ┌── Reverse Image Search")
        print(f"  │   Input: {image_path}")

        result = {
            "perceptual_hash": None,
            "search_methods_used": [],
            "total_matches_found": 0,
            "matches": [],
            "oldest_appearance": None,
            "source_assessment": "unknown",
        }

        # Always compute perceptual hash (works offline)
        phash = self._compute_perceptual_hash(image_path)
        result["perceptual_hash"] = phash
        print(f"  │   Perceptual hash: {phash}")

        # Method 1: TinEye (if API key available)
        if self.tineye_api_key:
            print(f"  │   Searching TinEye...")
            tineye_results = self._search_tineye(image_path)
            result["search_methods_used"].append("tineye")
            if tineye_results:
                result["matches"].extend(tineye_results)
                print(f"  │   TinEye: {len(tineye_results)} matches found")
        else:
            print(f"  │   TinEye: Skipped (no API key)")
            result["search_methods_used"].append("tineye_skipped")

        # Method 2: Google Vision Web Detection (if available)
        if self.google_vision_enabled:
            print(f"  │   Searching Google Vision...")
            vision_results = self._search_google_vision(image_path)
            result["search_methods_used"].append("google_vision")
            if vision_results:
                result["matches"].extend(vision_results)
                print(f"  │   Google Vision: {len(vision_results)} matches found")
        else:
            print(f"  │   Google Vision: Skipped (no API key)")
            result["search_methods_used"].append("google_vision_skipped")

        # Method 3: Generate search URLs for manual verification
        search_urls = self._generate_manual_search_urls(image_path)
        result["manual_search_urls"] = search_urls
        result["search_methods_used"].append("manual_urls_generated")
        print(f"  │   Manual search URLs generated: {len(search_urls)}")

        # Compute totals
        result["total_matches_found"] = len(result["matches"])

        # Find oldest appearance
        if result["matches"]:
            dated_matches = [m for m in result["matches"] if m.get("date")]
            if dated_matches:
                dated_matches.sort(key=lambda m: m["date"])
                result["oldest_appearance"] = dated_matches[0]

        # Assess source
        result["source_assessment"] = self._assess_source(result)
        print(f"  │   Source assessment: {result['source_assessment']}")
        print(f"  └── Reverse Image Search Complete")

        return result

    def _compute_perceptual_hash(self, image_path: str) -> str:
        """
        Computes a perceptual hash (pHash) of the image.
        Similar images will have similar hashes regardless of resizing/compression.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "error_reading_image"

        # Resize to 32x32
        resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        # Apply DCT (Discrete Cosine Transform)
        dct = cv2.dct(np.float32(resized))

        # Use top-left 8x8 (low frequencies)
        dct_low = dct[:8, :8]

        # Compute median
        median = np.median(dct_low)

        # Generate hash: 1 if above median, 0 if below
        hash_bits = (dct_low > median).flatten()
        hash_str = "".join(["1" if b else "0" for b in hash_bits])

        # Convert to hex
        hash_hex = hex(int(hash_str, 2))[2:].zfill(16)
        return hash_hex

    def _search_tineye(self, image_path: str) -> list:
        """Searches TinEye API for matching images."""
        try:
            url = "https://api.tineye.com/rest/search/"
            with open(image_path, "rb") as f:
                files = {"image": f}
                headers = {"x-api-key": self.tineye_api_key}
                response = requests.post(
                    url, files=files, headers=headers, timeout=30
                )

            if response.status_code == 200:
                data = response.json()
                matches = []
                for match in data.get("result", {}).get("matches", [])[:10]:
                    matches.append({
                        "source": "tineye",
                        "url": match.get("backlinks", [{}])[0].get("url", ""),
                        "domain": match.get("domain", ""),
                        "date": match.get("crawl_date", ""),
                        "score": match.get("score", 0),
                    })
                return matches
            return []

        except Exception:
            return []

    def _search_google_vision(self, image_path: str) -> list:
        """Searches Google Cloud Vision Web Detection API."""
        try:
            url = f"https://vision.googleapis.com/v1/images:annotate?key={self.google_vision_enabled}"
            with open(image_path, "rb") as f:
                import base64
                image_content = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "requests": [{
                    "image": {"content": image_content},
                    "features": [{"type": "WEB_DETECTION", "maxResults": 10}],
                }]
            }

            response = requests.post(url, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                web_detection = (
                    data.get("responses", [{}])[0]
                    .get("webDetection", {})
                )
                matches = []

                for page in web_detection.get("pagesWithMatchingImages", [])[:10]:
                    matches.append({
                        "source": "google_vision",
                        "url": page.get("url", ""),
                        "page_title": page.get("pageTitle", ""),
                        "date": None,
                        "score": page.get("score", 0),
                    })
                return matches
            return []

        except Exception:
            return []

    def _generate_manual_search_urls(self, image_path: str) -> list:
        """
        Generates URLs that the human reviewer can click to manually
        perform reverse image searches on major engines.
        """
        # These URLs require the image to be uploaded manually by the reviewer
        # In Phase 5 (dashboard), we'll provide upload buttons
        return [
            {
                "engine": "Google Images",
                "url": "https://images.google.com/",
                "instructions": "Click camera icon, upload the image",
            },
            {
                "engine": "TinEye",
                "url": "https://tineye.com/",
                "instructions": "Upload the image or paste URL",
            },
            {
                "engine": "Yandex Images",
                "url": "https://yandex.com/images/",
                "instructions": "Click camera icon, upload the image",
            },
            {
                "engine": "Bing Visual Search",
                "url": "https://www.bing.com/visualsearch",
                "instructions": "Drag and drop the image",
            },
        ]

    def _assess_source(self, result: dict) -> str:
        """Assesses the likely source/authenticity based on search results."""
        total = result["total_matches_found"]

        if total == 0:
            return "no_prior_appearances_found"
        elif total <= 2:
            return "limited_appearances_possibly_original"
        elif total <= 5:
            return "moderate_appearances_verify_source"
        else:
            return "widely_circulated_check_original_date"


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FORENSICS ENGINE - Combines All Analyses
# ─────────────────────────────────────────────────────────────────────────────

class ImageForensicsEngine:
    """
    Master engine that orchestrates all image forensic analyses
    and compiles a unified report for the HITL dashboard.
    """

    def __init__(self):
        self.ela_analyzer = ELAAnalyzer(quality=90, scale=15)
        self.metadata_extractor = MetadataExtractor()
        self.reverse_searcher = ReverseImageSearcher()

    def full_analysis(self, image_path: str) -> dict:
        """
        Runs the complete forensic analysis pipeline on a single image.
        Returns a comprehensive report dict.
        """
        print("\n" + "=" * 70)
        print("🔬 IMAGE FORENSICS ENGINE - FULL ANALYSIS")
        print("=" * 70)
        print(f"📷 Target: {image_path}")

        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}

        start_time = time.time()

        # Run all three analyses
        ela_result = self.ela_analyzer.analyze(image_path)
        metadata_result = self.metadata_extractor.analyze(image_path)
        reverse_search_result = self.reverse_searcher.analyze(image_path)

        elapsed = time.time() - start_time

        # Compile unified report
        report = {
            "report_type": "image_forensics",
            "image_path": image_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_duration_seconds": round(elapsed, 2),
            "ela_analysis": ela_result,
            "metadata_analysis": metadata_result,
            "reverse_image_search": reverse_search_result,
            "overall_risk_score": self._compute_risk_score(
                ela_result, metadata_result, reverse_search_result
            ),
            "verdict": "PENDING_HUMAN_REVIEW",
        }

        # Print summary
        print("\n" + "-" * 70)
        print("📊 FORENSIC SUMMARY")
        print("-" * 70)
        print(f"  ELA Manipulation Likelihood: {ela_result['manipulation_likelihood'].upper()}")
        print(f"  Suspicious Regions: {ela_result['num_suspicious_regions']}")
        print(f"  Metadata Flags: {len(metadata_result['suspicious_flags'])}")
        print(f"  Reverse Search Matches: {reverse_search_result['total_matches_found']}")
        print(f"  Overall Risk Score: {report['overall_risk_score']}/100")
        print(f"  Analysis Time: {elapsed:.2f}s")
        print(f"  Verdict: PENDING_HUMAN_REVIEW")
        print("=" * 70)

        return report

    def _compute_risk_score(
        self, ela: dict, metadata: dict, reverse: dict
    ) -> int:
        """
        Computes an overall risk score (0-100) combining all analyses.
        Higher = more likely manipulated/suspicious.
        """
        score = 0

        # ELA contribution (0-40 points)
        ela_likelihood = ela.get("manipulation_likelihood", "low")
        if ela_likelihood == "high":
            score += 35
        elif ela_likelihood == "medium":
            score += 20
        else:
            score += 5

        # Add for suspicious regions
        num_regions = ela.get("num_suspicious_regions", 0)
        score += min(num_regions * 2, 10)

        # Metadata contribution (0-30 points)
        flags = metadata.get("suspicious_flags", [])
        for flag in flags:
            severity = flag.get("severity", "low")
            if severity == "high":
                score += 15
            elif severity == "medium":
                score += 8
            else:
                score += 3

        score = min(score, 70)  # Cap metadata+ELA at 70

        # Reverse search contribution (0-30 points)
        total_matches = reverse.get("total_matches_found", 0)
        if total_matches > 10:
            score += 25  # Widely recycled
        elif total_matches > 5:
            score += 15
        elif total_matches > 0:
            score += 5

        return min(score, 100)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION (For Testing)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█  HITL FAKE NEWS DETECTION - PHASE 3: IMAGE FORENSICS")
    print("█  Mode: Full Forensic Analysis")
    print("█" * 70)

    # ─── CREATE A TEST IMAGE ─────────────────────────────────────────────────
    # Since you may not have a test image ready, we'll generate one
    # that simulates a manipulated image for demonstration.

    test_image_path = "test_image.jpg"

    if not os.path.exists(test_image_path):
        print("\n⚠️  No test image found. Creating a synthetic test image...")
        print("   (In production, users will upload real images)\n")

        # Create a synthetic image with a "manipulated" region
        img = np.zeros((600, 800, 3), dtype=np.uint8)

        # Background: gradient (simulates natural photo)
        for y in range(600):
            for x in range(800):
                img[y, x] = [
                    int(100 + 50 * np.sin(x / 50)),
                    int(120 + 40 * np.sin(y / 40)),
                    int(80 + 60 * np.sin((x + y) / 60)),
                ]

        # Add a "spliced" region (uniform block - different compression)
        cv2.rectangle(img, (200, 150), (500, 400), (50, 100, 200), -1)
        cv2.putText(
            img, "SPLICED REGION", (220, 290),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        # Add some noise to make it realistic
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Save as JPEG (introduces compression artifacts)
        cv2.imwrite(test_image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"   ✅ Test image created: {test_image_path}")

    # ─── RUN FULL ANALYSIS ───────────────────────────────────────────────────
    engine = ImageForensicsEngine()
    report = engine.full_analysis(test_image_path)

    # ─── OUTPUT REPORT ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📦 FULL FORENSIC REPORT (JSON):")
    print("=" * 70)

    # Create a printable version (exclude huge data)
    printable_report = {k: v for k, v in report.items() if k != "metadata_analysis"}
    printable_report["metadata_analysis"] = {
        k: v for k, v in report["metadata_analysis"].items() if k != "all_tags"
    }

    print(json.dumps(printable_report, indent=2, default=str))

    print("\n" + "=" * 70)
    print("✅ Phase 3 complete. Image forensics engine operational.")
    print(f"📁 Check '{OUTPUT_DIR}/' folder for ELA heatmap output.")
    print("=" * 70)
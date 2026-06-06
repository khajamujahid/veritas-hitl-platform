"""
=============================================================================
HITL Fake NEWS DETECTION PLATFORM - Phase 4: Video Forensics Module
=============================================================================
This module provides temporal analysis tools for uploaded video files:

1. FRAME EXTRACTION & KEYFRAME DETECTION:
   - Extracts frames at configurable intervals
   - Uses scene-change detection (histogram comparison) to find keyframes
   - Identifies cuts, transitions, and potential splice points

2. OPTICAL FLOW ANALYSIS:
   - Computes dense optical flow between consecutive frames
   - Detects temporal inconsistencies (sudden motion changes)
   - Identifies jump cuts and spliced segments

3. PER-FRAME ELA:
   - Runs Error Level Analysis on selected keyframes
   - Detects individual doctored/composited frames

4. VIDEO METADATA & INTEGRITY:
   - Extracts codec, duration, resolution, bitrate
   - Checks for encoding inconsistencies
   - Analyzes audio track presence and sync

=============================================================================
"""

import importlib
import importlib.util
import os
import io
import json
import time
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
from image_forensics import ELAAnalyzer

VideoFileClip = None
if importlib.util.find_spec("moviepy") is not None:
    try:
        moviepy_editor = importlib.import_module("moviepy.editor")
        VideoFileClip = moviepy_editor.VideoFileClip
    except Exception:
        VideoFileClip = None


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "forensic_output"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "extracted_frames")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: FRAME EXTRACTION & KEYFRAME DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class FrameExtractor:
    """
    Extracts frames from video files using two strategies:
    1. Uniform sampling (every N frames)
    2. Scene-change detection (histogram difference threshold)
    """

    def __init__(self, sample_interval: int = 30, scene_threshold: float = 0.6):
        """
        Args:
            sample_interval: Extract every Nth frame for uniform sampling.
            scene_threshold: Histogram difference threshold for scene changes.
                           Lower = more sensitive (more keyframes detected).
                           0.6 is a good default.
        """
        self.sample_interval = sample_interval
        self.scene_threshold = scene_threshold

    def extract(self, video_path: str) -> dict:
        """
        Extracts keyframes from the video.
        Returns metadata and paths to extracted frame images.
        """
        print("\n  ┌── Frame Extraction & Keyframe Detection")
        print(f"  │   Input: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"error": f"Cannot open video: {video_path}"}

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

        print(f"  │   Resolution: {width}x{height}")
        print(f"  │   FPS: {fps:.2f}")
        print(f"  │   Total frames: {total_frames}")
        print(f"  │   Duration: {duration:.2f}s")
        print(f"  │   Codec: {codec_str}")

        # Extract keyframes using scene-change detection
        keyframes = []
        scene_changes = []
        prev_hist = None
        frame_index = 0

        # Create a subdirectory for this video's frames
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_frames_dir = os.path.join(FRAMES_DIR, video_name)
        os.makedirs(video_frames_dir, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Compute histogram for scene-change detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            is_scene_change = False

            if prev_hist is not None:
                # Compare histograms using correlation
                correlation = cv2.compareHist(
                    prev_hist, hist, cv2.HISTCMP_CORREL
                )

                # Low correlation = big visual change = scene change
                if correlation < self.scene_threshold:
                    is_scene_change = True
                    scene_changes.append({
                        "frame_index": frame_index,
                        "timestamp": frame_index / fps if fps > 0 else 0,
                        "correlation": round(float(correlation), 4),
                    })

            # Save frame if it's a scene change or at sample interval
            if is_scene_change or frame_index % self.sample_interval == 0:
                frame_filename = f"frame_{frame_index:06d}.jpg"
                frame_path = os.path.join(video_frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

                keyframes.append({
                    "frame_index": frame_index,
                    "timestamp": round(frame_index / fps, 3) if fps > 0 else 0,
                    "path": frame_path,
                    "is_scene_change": is_scene_change,
                    "type": "scene_change" if is_scene_change else "interval_sample",
                })

            prev_hist = hist
            frame_index += 1

        cap.release()

        print(f"  │   Keyframes extracted: {len(keyframes)}")
        print(f"  │   Scene changes detected: {len(scene_changes)}")
        print(f"  │   Frames saved to: {video_frames_dir}")
        print(f"  └── Frame Extraction Complete")

        return {
            "video_properties": {
                "path": video_path,
                "resolution": {"width": width, "height": height},
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_seconds": round(duration, 2),
                "codec": codec_str,
            },
            "keyframes": keyframes,
            "scene_changes": scene_changes,
            "num_keyframes": len(keyframes),
            "num_scene_changes": len(scene_changes),
            "frames_directory": video_frames_dir,
        }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: OPTICAL FLOW ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

class OpticalFlowAnalyzer:
    """
    Analyzes motion between consecutive frames using dense optical flow.
    Detects temporal inconsistencies that suggest video manipulation:
    - Sudden motion spikes (jump cuts or frame insertion)
    - Abnormal flow patterns (object pasting)
    - Flow direction reversals (reversed footage)
    """

    def __init__(self, sample_every: int = 5, anomaly_threshold: float = 2.5):
        """
        Args:
            sample_every: Analyze flow every N frames (for performance).
            anomaly_threshold: Standard deviations above mean to flag as anomaly.
        """
        self.sample_every = sample_every
        self.anomaly_threshold = anomaly_threshold

    def analyze(self, video_path: str) -> dict:
        """
        Computes optical flow across the video and identifies anomalies.
        """
        print("\n  ┌── Optical Flow Analysis")
        print(f"  │   Input: {video_path}")
        print(f"  │   Sampling every {self.sample_every} frames")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"error": f"Cannot open video: {video_path}"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        flow_magnitudes = []
        flow_directions = []
        frame_timestamps = []
        prev_gray = None
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % self.sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Resize for performance
                gray = cv2.resize(gray, (320, 240))

                if prev_gray is not None:
                    # Compute dense optical flow (Farneback method)
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray,
                        None,
                        pyr_scale=0.5,
                        levels=3,
                        winsize=15,
                        iterations=3,
                        poly_n=5,
                        poly_sigma=1.2,
                        flags=0,
                    )

                    # Compute magnitude and direction
                    magnitude, angle = cv2.cartToPolar(
                        flow[..., 0], flow[..., 1]
                    )

                    mean_mag = float(np.mean(magnitude))
                    mean_dir = float(np.mean(angle))

                    flow_magnitudes.append(mean_mag)
                    flow_directions.append(mean_dir)
                    frame_timestamps.append(
                        round(frame_index / fps, 3) if fps > 0 else 0
                    )

                prev_gray = gray

            frame_index += 1

        cap.release()

        if len(flow_magnitudes) < 3:
            print(f"  │   ⚠️ Not enough frames for flow analysis")
            print(f"  └── Optical Flow Complete (insufficient data)")
            return {
                "flow_data": [],
                "anomalies": [],
                "num_anomalies": 0,
                "average_motion": 0,
                "motion_stability": "insufficient_data",
            }

        # Convert to numpy for statistical analysis
        magnitudes = np.array(flow_magnitudes)
        directions = np.array(flow_directions)

        mean_motion = float(np.mean(magnitudes))
        std_motion = float(np.std(magnitudes))

        print(f"  │   Frames analyzed: {len(flow_magnitudes)}")
        print(f"  │   Average motion: {mean_motion:.4f}")
        print(f"  │   Motion std dev: {std_motion:.4f}")

        # Detect anomalies (motion spikes beyond threshold)
        anomalies = []
        for i, mag in enumerate(flow_magnitudes):
            if std_motion > 0:
                z_score = (mag - mean_motion) / std_motion
            else:
                z_score = 0

            if abs(z_score) > self.anomaly_threshold:
                anomalies.append({
                    "frame_timestamp": frame_timestamps[i],
                    "magnitude": round(mag, 4),
                    "z_score": round(z_score, 2),
                    "type": "motion_spike" if z_score > 0 else "motion_drop",
                    "severity": "high" if abs(z_score) > 4 else "medium",
                })

        # Check for direction reversals (potential reversed footage)
        direction_changes = 0
        for i in range(1, len(directions)):
            diff = abs(directions[i] - directions[i - 1])
            if diff > np.pi:  # More than 180 degrees
                direction_changes += 1

        # Assess overall motion stability
        if std_motion < mean_motion * 0.3:
            stability = "stable"
        elif std_motion < mean_motion * 0.7:
            stability = "moderate"
        else:
            stability = "unstable"

        print(f"  │   Anomalies detected: {len(anomalies)}")
        print(f"  │   Direction reversals: {direction_changes}")
        print(f"  │   Motion stability: {stability}")
        print(f"  └── Optical Flow Analysis Complete")

        # Build flow timeline data (for dashboard visualization)
        flow_timeline = []
        for i in range(len(flow_magnitudes)):
            flow_timeline.append({
                "timestamp": frame_timestamps[i],
                "magnitude": round(flow_magnitudes[i], 4),
                "direction": round(flow_directions[i], 4),
            })

        return {
            "flow_timeline": flow_timeline,
            "anomalies": anomalies,
            "num_anomalies": len(anomalies),
            "average_motion": round(mean_motion, 4),
            "motion_std_dev": round(std_motion, 4),
            "motion_stability": stability,
            "direction_reversals": direction_changes,
            "total_flow_samples": len(flow_magnitudes),
        }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: PER-FRAME ELA
# ─────────────────────────────────────────────────────────────────────────────

class FrameELAAnalyzer:
    """
    Runs Error Level Analysis on extracted keyframes to detect
    individual frames that may have been doctored or composited.
    """

    def __init__(self, max_frames: int = 10):
        """
        Args:
            max_frames: Maximum number of keyframes to analyze (for performance).
        """
        self.max_frames = max_frames
        self.ela = ELAAnalyzer(quality=90, scale=15)

    def analyze(self, keyframes: list) -> dict:
        """
        Runs ELA on selected keyframes.
        Prioritizes scene-change frames (more likely to be splice points).
        """
        print("\n  ┌── Per-Frame ELA Analysis")

        if not keyframes:
            print(f"  │   ⚠️ No keyframes to analyze")
            print(f"  └── Per-Frame ELA Complete")
            return {"frames_analyzed": 0, "results": [], "suspicious_frames": []}

        # Prioritize scene-change frames
        scene_frames = [kf for kf in keyframes if kf.get("is_scene_change")]
        interval_frames = [kf for kf in keyframes if not kf.get("is_scene_change")]

        # Select frames to analyze (scene changes first)
        selected = scene_frames[:self.max_frames]
        remaining_slots = self.max_frames - len(selected)
        if remaining_slots > 0:
            # Evenly space interval frames
            step = max(1, len(interval_frames) // remaining_slots)
            selected.extend(interval_frames[::step][:remaining_slots])

        print(f"  │   Keyframes available: {len(keyframes)}")
        print(f"  │   Frames selected for ELA: {len(selected)}")

        results = []
        suspicious_frames = []

        for i, kf in enumerate(selected):
            frame_path = kf["path"]
            if not os.path.exists(frame_path):
                continue

            print(f"  │   Analyzing frame {i + 1}/{len(selected)}: "
                  f"t={kf['timestamp']}s ({kf['type']})")

            ela_result = self.ela.analyze(frame_path)

            frame_result = {
                "frame_index": kf["frame_index"],
                "timestamp": kf["timestamp"],
                "type": kf["type"],
                "ela_mean_error": ela_result["mean_error"],
                "ela_max_error": ela_result["max_error"],
                "manipulation_likelihood": ela_result["manipulation_likelihood"],
                "suspicious_regions": ela_result["num_suspicious_regions"],
                "ela_heatmap_path": ela_result["ela_image_path"],
            }

            results.append(frame_result)

            if ela_result["manipulation_likelihood"] in ["medium", "high"]:
                suspicious_frames.append(frame_result)

        print(f"  │   Suspicious frames found: {len(suspicious_frames)}")
        print(f"  └── Per-Frame ELA Complete")

        return {
            "frames_analyzed": len(results),
            "results": results,
            "suspicious_frames": suspicious_frames,
            "num_suspicious": len(suspicious_frames),
        }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: VIDEO METADATA & INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

class VideoMetadataAnalyzer:
    """
    Extracts and analyzes video file metadata for integrity checks.
    """

    def analyze(self, video_path: str) -> dict:
        """
        Extracts comprehensive video metadata and flags inconsistencies.
        """
        print("\n  ┌── Video Metadata & Integrity Analysis")
        print(f"  │   Input: {video_path}")

        result = {
            "file_size_bytes": 0,
            "file_size_mb": 0,
            "container_format": "",
            "video_codec": "",
            "audio_present": False,
            "audio_codec": "",
            "creation_time": None,
            "modification_time": None,
            "suspicious_flags": [],
        }

        # File-level info
        file_size = os.path.getsize(video_path)
        result["file_size_bytes"] = file_size
        result["file_size_mb"] = round(file_size / (1024 * 1024), 2)

        # Determine container format from extension
        ext = os.path.splitext(video_path)[1].lower()
        format_map = {
            ".mp4": "MPEG-4",
            ".avi": "AVI",
            ".mov": "QuickTime",
            ".mkv": "Matroska",
            ".wmv": "Windows Media",
            ".flv": "Flash Video",
            ".webm": "WebM",
        }
        result["container_format"] = format_map.get(ext, f"Unknown ({ext})")

        # Use OpenCV to get video stream info
        cap = cv2.VideoCapture(video_path)

        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_str = "".join(
                [chr((codec_int >> 8 * i) & 0xFF) for i in range(4)]
            )

            result["video_codec"] = codec_str
            result["resolution"] = {"width": width, "height": height}
            result["fps"] = round(fps, 2)
            result["total_frames"] = frame_count
            result["duration_seconds"] = round(
                frame_count / fps if fps > 0 else 0, 2
            )

            # Compute expected vs actual bitrate
            duration = frame_count / fps if fps > 0 else 1
            actual_bitrate = (file_size * 8) / duration  # bits per second
            result["bitrate_bps"] = int(actual_bitrate)
            result["bitrate_mbps"] = round(actual_bitrate / 1_000_000, 2)

            # Expected bitrate for resolution
            pixel_count = width * height
            expected_bitrate_low = pixel_count * fps * 0.05  # Low quality
            expected_bitrate_high = pixel_count * fps * 0.3  # High quality

            if actual_bitrate < expected_bitrate_low * 0.5:
                result["suspicious_flags"].append({
                    "flag": "ABNORMALLY_LOW_BITRATE",
                    "severity": "medium",
                    "description": f"Bitrate ({result['bitrate_mbps']} Mbps) is "
                                  f"unusually low for {width}x{height}@{fps}fps. "
                                  f"Video may have been heavily re-encoded.",
                })

            cap.release()

        # Check for audio track using moviepy if available
        if VideoFileClip is not None:
            try:
                clip = VideoFileClip(video_path)
                if clip.audio is not None:
                    result["audio_present"] = True
                    result["audio_duration"] = round(clip.audio.duration, 2)
                    result["audio_fps"] = clip.audio.fps

                    # Check audio-video duration mismatch
                    video_duration = result.get("duration_seconds", 0)
                    audio_duration = clip.audio.duration
                    if abs(video_duration - audio_duration) > 1.0:
                        result["suspicious_flags"].append({
                            "flag": "AUDIO_VIDEO_DURATION_MISMATCH",
                            "severity": "high",
                            "description": f"Video duration ({video_duration}s) differs "
                                          f"from audio duration ({audio_duration:.2f}s) "
                                          f"by more than 1 second. Possible splice.",
                        })
                else:
                    result["audio_present"] = False
                clip.close()
            except Exception as e:
                result["audio_error"] = str(e)
        else:
            result["audio_present"] = None
            result["audio_note"] = "moviepy not available for audio analysis"

        # File timestamps
        try:
            stat = os.stat(video_path)
            result["creation_time"] = datetime.fromtimestamp(
                stat.st_ctime
            ).isoformat()
            result["modification_time"] = datetime.fromtimestamp(
                stat.st_mtime
            ).isoformat()
        except Exception:
            pass

        print(f"  │   File size: {result['file_size_mb']} MB")
        print(f"  │   Codec: {result.get('video_codec', 'unknown')}")
        print(f"  │   Audio present: {result['audio_present']}")
        print(f"  │   Bitrate: {result.get('bitrate_mbps', 'unknown')} Mbps")
        print(f"  │   Suspicious flags: {len(result['suspicious_flags'])}")
        print(f"  └── Metadata Analysis Complete")

        return result


# ─────────────────────────────────────────────────────────────────────────────
# MASTER VIDEO FORENSICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class VideoForensicsEngine:
    """
    Master engine that orchestrates all video forensic analyses
    and compiles a unified report for the HITL dashboard.
    """

    def __init__(self):
        self.frame_extractor = FrameExtractor(
            sample_interval=30, scene_threshold=0.6
        )
        self.optical_flow = OpticalFlowAnalyzer(
            sample_every=5, anomaly_threshold=2.5
        )
        self.frame_ela = FrameELAAnalyzer(max_frames=10)
        self.metadata_analyzer = VideoMetadataAnalyzer()

    def full_analysis(self, video_path: str) -> dict:
        """
        Runs the complete video forensic pipeline.
        Returns a comprehensive report dict.
        """
        print("\n" + "=" * 70)
        print("🎬 VIDEO FORENSICS ENGINE - FULL ANALYSIS")
        print("=" * 70)
        print(f"🎥 Target: {video_path}")

        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}

        start_time = time.time()

        # Step 1: Extract frames and detect scene changes
        extraction_result = self.frame_extractor.extract(video_path)

        if "error" in extraction_result:
            return extraction_result

        # Step 2: Optical flow analysis
        flow_result = self.optical_flow.analyze(video_path)

        # Step 3: Per-frame ELA on keyframes
        ela_result = self.frame_ela.analyze(extraction_result["keyframes"])

        # Step 4: Video metadata and integrity
        metadata_result = self.metadata_analyzer.analyze(video_path)

        elapsed = time.time() - start_time

        # Compile unified report
        report = {
            "report_type": "video_forensics",
            "video_path": video_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_duration_seconds": round(elapsed, 2),
            "video_properties": extraction_result["video_properties"],
            "frame_extraction": {
                "num_keyframes": extraction_result["num_keyframes"],
                "num_scene_changes": extraction_result["num_scene_changes"],
                "scene_changes": extraction_result["scene_changes"],
                "frames_directory": extraction_result["frames_directory"],
            },
            "optical_flow": flow_result,
            "frame_ela": ela_result,
            "metadata": metadata_result,
            "overall_risk_score": self._compute_risk_score(
                extraction_result, flow_result, ela_result, metadata_result
            ),
            "verdict": "PENDING_HUMAN_REVIEW",
        }

        # Print summary
        print("\n" + "-" * 70)
        print("📊 VIDEO FORENSIC SUMMARY")
        print("-" * 70)
        print(f"  Duration: {extraction_result['video_properties']['duration_seconds']}s")
        print(f"  Keyframes extracted: {extraction_result['num_keyframes']}")
        print(f"  Scene changes: {extraction_result['num_scene_changes']}")
        print(f"  Optical flow anomalies: {flow_result.get('num_anomalies', 0)}")
        print(f"  Suspicious frames (ELA): {ela_result.get('num_suspicious', 0)}")
        print(f"  Metadata flags: {len(metadata_result.get('suspicious_flags', []))}")
        print(f"  Overall Risk Score: {report['overall_risk_score']}/100")
        print(f"  Analysis Time: {elapsed:.2f}s")
        print(f"  Verdict: PENDING_HUMAN_REVIEW")
        print("=" * 70)

        return report

    def _compute_risk_score(
        self, extraction: dict, flow: dict, ela: dict, metadata: dict
    ) -> int:
        """
        Computes overall video risk score (0-100).
        """
        score = 0

        # Scene changes contribution (0-20)
        # Too many scene changes in short video = suspicious
        duration = extraction["video_properties"]["duration_seconds"]
        num_scenes = extraction["num_scene_changes"]
        if duration > 0:
            scenes_per_second = num_scenes / duration
            if scenes_per_second > 2:
                score += 20
            elif scenes_per_second > 1:
                score += 10
            elif scenes_per_second > 0.5:
                score += 5

        # Optical flow contribution (0-30)
        num_anomalies = flow.get("num_anomalies", 0)
        if num_anomalies > 5:
            score += 30
        elif num_anomalies > 3:
            score += 20
        elif num_anomalies > 1:
            score += 10
        elif num_anomalies > 0:
            score += 5

        # Direction reversals
        reversals = flow.get("direction_reversals", 0)
        if reversals > 3:
            score += 10

        # Frame ELA contribution (0-25)
        num_suspicious = ela.get("num_suspicious", 0)
        if num_suspicious > 3:
            score += 25
        elif num_suspicious > 1:
            score += 15
        elif num_suspicious > 0:
            score += 8

        # Metadata contribution (0-15)
        flags = metadata.get("suspicious_flags", [])
        for flag in flags:
            if flag.get("severity") == "high":
                score += 10
            elif flag.get("severity") == "medium":
                score += 5

        return min(score, 100)


# ─────────────────────────────────────────────────────────────────────────────
# TEST VIDEO GENERATOR (For testing without a real video)
# ─────────────────────────────────────────────────────────────────────────────

def create_test_video(output_path: str = "test_video.mp4") -> str:
    """
    Creates a synthetic test video with deliberate anomalies:
    - Normal scene (frames 0-60)
    - Sudden scene change / splice (frames 60-90)
    - Different content spliced in (frames 90-150)
    - Back to original-like content (frames 150-200)
    """
    print(f"\n⚠️  Creating synthetic test video: {output_path}")

    width, height, fps = 640, 480, 30
    total_frames = 200

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if i < 60:
            # Scene 1: Blue gradient with moving circle
            frame[:, :, 0] = 180  # Blue channel
            cx = int(100 + i * 3)
            cv2.circle(frame, (cx, 240), 40, (0, 255, 0), -1)
            cv2.putText(frame, "SCENE 1: Original", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        elif i < 90:
            # Scene 2: SPLICE - completely different (red, static)
            frame[:, :, 2] = 200  # Red channel
            cv2.rectangle(frame, (100, 100), (540, 380), (0, 255, 255), -1)
            cv2.putText(frame, "SCENE 2: SPLICED", (120, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        elif i < 150:
            # Scene 3: Green with different motion
            frame[:, :, 1] = 150  # Green channel
            cy = int(100 + (i - 90) * 4)
            cv2.circle(frame, (320, cy), 30, (255, 0, 255), -1)
            cv2.putText(frame, "SCENE 3: Different Source", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        else:
            # Scene 4: Back to blue-ish (but slightly different)
            frame[:, :, 0] = 160
            frame[:, :, 1] = 50
            cx = int(500 - (i - 150) * 3)
            cv2.circle(frame, (cx, 240), 40, (0, 255, 0), -1)
            cv2.putText(frame, "SCENE 4: Back to 'Original'", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add frame number timestamp
        cv2.putText(frame, f"Frame: {i}", (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(frame)

    out.release()
    print(f"   ✅ Test video created: {output_path}")
    print(f"   Duration: {total_frames / fps:.1f}s | {total_frames} frames | {fps} fps")
    print(f"   Contains 4 scenes with 3 deliberate splice points")

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█  HITL FAKE NEWS DETECTION - PHASE 4: VIDEO FORENSICS")
    print("█  Mode: Full Temporal Analysis")
    print("█" * 70)

    # Create or use test video
    test_video_path = "test_video.mp4"

    if not os.path.exists(test_video_path):
        create_test_video(test_video_path)

    # Run full analysis
    engine = VideoForensicsEngine()
    report = engine.full_analysis(test_video_path)

    # Output report
    print("\n" + "=" * 70)
    print("📦 VIDEO FORENSIC REPORT (JSON Summary):")
    print("=" * 70)

    # Create printable summary (exclude large data arrays)
    summary = {
        "report_type": report["report_type"],
        "video_path": report["video_path"],
        "analysis_duration": report["analysis_duration_seconds"],
        "video_properties": report["video_properties"],
        "frame_extraction": {
            "keyframes": report["frame_extraction"]["num_keyframes"],
            "scene_changes": report["frame_extraction"]["num_scene_changes"],
        },
        "optical_flow": {
            "anomalies": report["optical_flow"].get("num_anomalies", 0),
            "average_motion": report["optical_flow"].get("average_motion", 0),
            "stability": report["optical_flow"].get("motion_stability", "unknown"),
            "direction_reversals": report["optical_flow"].get("direction_reversals", 0),
        },
        "frame_ela": {
            "frames_analyzed": report["frame_ela"]["frames_analyzed"],
            "suspicious_frames": report["frame_ela"]["num_suspicious"],
        },
        "metadata_flags": len(report["metadata"].get("suspicious_flags", [])),
        "overall_risk_score": report["overall_risk_score"],
        "verdict": report["verdict"],
    }

    print(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 70)
    print("✅ Phase 4 complete. Video forensics engine operational.")
    print(f"📁 Extracted frames: forensic_output/extracted_frames/")
    print(f"📁 ELA heatmaps: forensic_output/")
    print("=" * 70)
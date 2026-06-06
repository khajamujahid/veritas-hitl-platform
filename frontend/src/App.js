import React, { useState } from "react";
import axios from "axios";
import {
  Upload,
  FileText,
  Image,
  Video,
  Eye,
  Target,
  AlertTriangle,
  CheckCircle,
  XCircle,
  TrendingUp,
  Search,
  RotateCcw,
} from "lucide-react";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000/api";

function App() {
  const [activeTab, setActiveTab] = useState("upload");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [verdictSubmitted, setVerdictSubmitted] = useState(false);

  return (
    <div className="min-h-screen bg-[#0a0e17] text-gray-100">
      <Header />
      <div className="max-w-7xl mx-auto p-6">
        <div className="grid gap-6 xl:grid-cols-[260px_1fr]">
          <aside className="space-y-4">
            <NavTabs activeTab={activeTab} setActiveTab={setActiveTab} />
          </aside>
          <main>
            {activeTab === "upload" && (
              <UploadPanel
                setAnalysisResult={setAnalysisResult}
                setActiveTab={setActiveTab}
                setLoading={setLoading}
                setLoadingMessage={setLoadingMessage}
              />
            )}
            {activeTab === "analysis" && analysisResult && (
              <AnalysisPanel result={analysisResult} verdictSubmitted={verdictSubmitted} />
            )}
            {activeTab === "verdict" && analysisResult && (
              <VerdictPanel
                result={analysisResult}
                setVerdictSubmitted={setVerdictSubmitted}
                setActiveTab={setActiveTab}
              />
            )}
            {loading && <LoadingOverlay message={loadingMessage} />}
          </main>
        </div>
      </div>
      <Footer />
    </div>
  );
}

function Header() {
  return (
    <header className="bg-gradient-to-r from-[#071827] via-[#0f1f38] to-[#071827] neon-border py-6">
      <div className="max-w-7xl mx-auto flex flex-col gap-3 px-6 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold neon-text">VERITAS</h1>
          <p className="text-sm text-gray-400">Human-in-the-Loop Forensic Intelligence</p>
        </div>
        <div className="flex flex-wrap gap-3 text-sm text-gray-300">
          <StatusBadge label="Backend" status="operational" />
          <StatusBadge label="Models" status="ready" />
        </div>
      </div>
    </header>
  );
}

function StatusBadge({ label, status }) {
  return (
    <div className="rounded-full border border-gray-800 bg-gray-900/70 px-3 py-1 text-xs uppercase tracking-[0.12em] text-gray-300">
      {label}: {status}
    </div>
  );
}

function NavTabs({ activeTab, setActiveTab }) {
  const tabs = [
    { id: "upload", label: "Upload & Analyze", icon: Upload },
    { id: "analysis", label: "Forensic Results", icon: Eye },
    { id: "verdict", label: "Issue Verdict", icon: Target },
  ];

  return (
    <div className="space-y-3">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => setActiveTab(tab.id)}
          className={`flex w-full items-center gap-2 rounded-2xl px-4 py-3 text-left text-sm font-medium transition-all ${
            activeTab === tab.id
              ? "bg-cyan-500/10 border border-cyan-500/50 text-cyan-200 shadow-lg shadow-cyan-500/10"
              : "border border-gray-800 bg-gray-900/70 text-gray-400 hover:border-gray-600 hover:text-white"
          }`}
        >
          <tab.icon className="h-4 w-4" />
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function UploadPanel({ setAnalysisResult, setActiveTab, setLoading, setLoadingMessage }) {
  const [articleText, setArticleText] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadType, setUploadType] = useState("text");

  const handleTextAnalysis = async () => {
    if (!articleText || articleText.length < 50) {
      alert("Please provide at least 50 characters of article text.");
      return;
    }

    setLoading(true);
    setLoadingMessage("Extracting claims and checking fact-check databases...");
    try {
      const res = await axios.post(`${API_BASE}/analyze/text`, { text: articleText });
      setAnalysisResult(res.data.result);
      setActiveTab("analysis");
    } catch (err) {
      alert("Text analysis failed: " + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleFileAnalyze = async () => {
    if (!selectedFile) {
      alert("Select a file first.");
      return;
    }
    const extension = selectedFile.name.split(".").pop().toLowerCase();
    const isImage = ["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff"].includes(extension);
    const isVideo = ["mp4", "mov", "avi", "mkv", "wmv"].includes(extension);
    if (!isImage && !isVideo) {
      alert("Unsupported file type. Upload an image or video.");
      return;
    }

    setLoading(true);
    setLoadingMessage(isImage ? "Running image forensics..." : "Running video forensics...");
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const endpoint = isImage ? "analyze/image" : "analyze/video";
      const res = await axios.post(`${API_BASE}/${endpoint}`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setAnalysisResult(res.data.result);
      setActiveTab("analysis");
    } catch (err) {
      alert("Upload failed: " + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel p-6">
      <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold">Upload & Analyze</h2>
          <p className="text-sm text-gray-400">Paste an article or upload an image/video for forensic review.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <TypeCard icon={FileText} label="Text" active={uploadType === "text"} onClick={() => setUploadType("text")} />
          <TypeCard icon={Image} label="Image" active={uploadType === "image"} onClick={() => setUploadType("image")} />
          <TypeCard icon={Video} label="Video" active={uploadType === "video"} onClick={() => setUploadType("video")} />
        </div>
      </div>

      {uploadType === "text" ? (
        <div className="space-y-4">
          <textarea
            value={articleText}
            onChange={(event) => setArticleText(event.target.value)}
            placeholder="Paste the full news article here. AI will extract every verifiable claim..."
            className="w-full resize-none rounded-2xl border border-gray-800 bg-gray-900/80 p-4 text-sm text-gray-100 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
            rows={10}
          />
          <div className="flex items-center justify-between gap-4">
            <div className="text-sm text-gray-400">{articleText.length} characters</div>
            <button onClick={handleTextAnalysis} className="rounded-2xl bg-cyan-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-cyan-500">
              Analyze Claims
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div
            className="group relative rounded-3xl border border-dashed border-gray-700 bg-gray-900/60 p-8 text-center transition hover:border-cyan-500 hover:bg-gray-900"
            onClick={() => document.getElementById("file-input").click()}
            onDrop={(event) => {
              event.preventDefault();
              if (event.dataTransfer.files.length) {
                setSelectedFile(event.dataTransfer.files[0]);
              }
            }}
            onDragOver={(event) => event.preventDefault()}
          >
            <input id="file-input" type="file" className="hidden" onChange={(event) => setSelectedFile(event.target.files[0])} />
            {selectedFile ? (
              <div className="space-y-2">
                <div className="text-sm text-gray-300">{selectedFile.name}</div>
                <div className="text-xs text-gray-500">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</div>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-lg font-semibold text-gray-100">Drop file here or click to browse</p>
                <p className="text-sm text-gray-400">
                  {uploadType === "image" ? "JPG, PNG, GIF, BMP, WebP" : "MP4, AVI, MOV, MKV"}
                </p>
              </div>
            )}
          </div>
          <button onClick={handleFileAnalyze} className="rounded-2xl bg-cyan-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-cyan-500">
            Run Forensics
          </button>
        </div>
      )}
    </div>
  );
}

function TypeCard({ icon: Icon, label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 rounded-2xl border px-4 py-3 text-sm transition ${
        active ? "border-cyan-500 bg-cyan-500/10 text-cyan-200" : "border-gray-800 bg-gray-900 text-gray-300 hover:border-gray-600"
      }`}
    >
      <Icon className="h-4 w-4" />
      <span>{label}</span>
    </button>
  );
}

function AnalysisPanel({ result, verdictSubmitted }) {
  const [aiAssessment, setAiAssessment] = useState(null);
  const [assessmentLoading, setAssessmentLoading] = useState(false);

  if (!result) {
    return (
      <div className="glass-panel p-6">
        <h2 className="text-xl font-semibold">Forensic Results</h2>
        <p className="mt-3 text-sm text-gray-400">Run an analysis first to view results here.</p>
      </div>
    );
  }

  const type = result.type || result.report_type;

  const requestAiAssessment = async () => {
    setAssessmentLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/ai-assessment`, { analysis_id: result.analysis_id });
      setAiAssessment(res.data);
    } catch (err) {
      alert("AI assessment failed: " + (err.response?.data?.error || err.message));
    } finally {
      setAssessmentLoading(false);
    }
  };

  return (
    <div className="glass-panel space-y-6 p-6">
      <div>
        <h2 className="text-xl font-semibold">Forensic Analysis Complete</h2>
        <p className="text-sm text-gray-400">ID: {result.analysis_id || "n/a"} | Duration: {result.duration_seconds || result.analysis_duration_seconds || 0}s</p>
      </div>

      {type === "text" && <TextAnalysisView result={result} />}
      {type === "image_forensics" && <ImageAnalysisView result={result.report || result} />}
      {type === "video_forensics" && <VideoAnalysisView result={result.report || result} />}

      {/* AI ASSESSMENT SECTION */}
      <div className="rounded-3xl border border-gray-800 bg-gray-950/40 p-4">
        <div className="flex items-center justify-between">
          <div className="font-semibold">AI Preliminary Assessment</div>
          {!aiAssessment && (
            <button onClick={requestAiAssessment} className="rounded-2xl bg-cyan-600 px-4 py-2 text-sm text-white">
              {assessmentLoading ? "Thinking..." : "Get AI Opinion"}
            </button>
          )}
        </div>

        {assessmentLoading && (
          <div className="mt-3 text-sm text-gray-300">AI is analyzing all evidence...</div>
        )}

        {aiAssessment && (
          <div className="mt-3 space-y-2 text-sm text-gray-200">
            <div>{aiAssessment.ai_assessment || aiAssessment.assessment_text}</div>
            <div className="text-xs text-gray-400">Model: {aiAssessment.model_used}</div>
            <div className="text-xs text-yellow-300">AI suggestion only — human verdict required</div>
          </div>
        )}

        {!aiAssessment && !assessmentLoading && (
          <div className="mt-3 text-sm text-gray-400">Click "Get AI Opinion" to have the AI analyze all forensic evidence and provide a preliminary assessment. You still make the final call.</div>
        )}
      </div>

      {/* VERDICT STATUS */}
      <div className="rounded-3xl border border-gray-800 bg-gray-950/40 p-4 text-sm text-gray-300">
        {verdictSubmitted ? "✅ Human verdict issued." : "⚠️ Awaiting human review. Go to \"Issue Verdict\" tab."}
      </div>
    </div>
  );
}

function TextAnalysisView({ result }) {
  return (
    <div className="space-y-4">
      <div className="rounded-3xl border border-gray-800 bg-gray-950/40 p-4">
        <h3 className="font-semibold">Article Summary</h3>
        <p className="mt-2 text-sm text-gray-300">{result.article_summary || "No summary available."}</p>
      </div>
      <div className="space-y-3">
        <h3 className="font-semibold">Extracted Claims ({result.total_claims || (result.claims?.length ?? 0)})</h3>
        {(result.claims || []).map((claim) => (
          <div key={claim.claim_number || claim.id} className="claim-card rounded-3xl border border-gray-800 bg-gray-950/40 p-4">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <div className="text-sm text-gray-400">#{claim.claim_number || "-"} • {claim.entity || claim.category || "Claim"}</div>
                <p className="mt-2 text-sm text-gray-100">{claim.claim_text}</p>
              </div>
              <div className="rounded-2xl border border-gray-800 bg-gray-900/70 px-3 py-2 text-xs uppercase tracking-[0.12em] text-gray-300">
                Verifiability: {claim.verifiability_score ?? "N/A"}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ImageAnalysisView({ result }) {
  const ela = result.ela_analysis || result.ela || {};
  const metadata = result.metadata_analysis || result.metadata || {};

  return (
    <div className="space-y-6">
      <div className="rounded-3xl border border-gray-800 bg-gray-950/40 p-4">
        <h3 className="font-semibold">ELA Analysis</h3>
        <p className="mt-2 text-sm text-gray-300">Evidence of recompression or edit anomalies is highlighted when available.</p>
        {ela.ela_image_path ? (
          <img src={ela.ela_image_path} alt="ELA" className="mt-4 w-full rounded-3xl border border-gray-800" />
        ) : (
          <p className="mt-4 text-sm text-gray-500">No ELA heatmap available.</p>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-3xl border border-gray-800 bg-gray-950/40 p-4">
          <h3 className="font-semibold">Metadata Summary</h3>
          <p className="mt-2 text-sm text-gray-300">{metadata.suspicious_flags?.length ? "Suspicious flags detected." : "No suspicious metadata flags found."}</p>
        </div>
        <div className="rounded-3xl border border-gray-800 bg-gray-950/40 p-4">
          <h3 className="font-semibold">Flags</h3>
          {metadata.suspicious_flags?.length ? (
            <ul className="mt-3 space-y-2 text-sm text-gray-300">
              {metadata.suspicious_flags.map((flag, index) => (
                <li key={index} className="rounded-2xl border border-gray-800 bg-gray-900/70 p-3">
                  <div className="font-semibold text-gray-100">{flag.flag}</div>
                  <div className="text-xs text-gray-400">{flag.description}</div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-sm text-gray-500">No metadata flags to display.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function VideoAnalysisView({ result }) {
  const props = result.video_properties || result; 
  const flow = result.optical_flow || {};

  return (
    <div className="space-y-6">
      <div className="rounded-3xl border border-gray-800 bg-gray-950/40 p-4">
        <h3 className="font-semibold">Video Properties</h3>
        <div className="mt-3 grid gap-3 sm:grid-cols-3">
          <StatCard label="Resolution" value={`${props.resolution?.width || "?"}x${props.resolution?.height || "?"}`} />
          <StatCard label="FPS" value={props.fps || "?"} />
          <StatCard label="Duration" value={`${props.duration_seconds || result.duration_seconds || 0}s`} />
        </div>
      </div>
      <div className="rounded-3xl border border-gray-800 bg-gray-950/40 p-4">
        <h3 className="font-semibold">Optical Flow</h3>
        <div className="mt-3 text-sm text-gray-300">Anomalies detected: {flow.num_anomalies ?? flow.anomalies?.length ?? 0}</div>
      </div>
    </div>
  );
}

function VerdictPanel({ result, setVerdictSubmitted, setActiveTab }) {
  const [selectedVerdict, setSelectedVerdict] = useState(null);
  const [confidence, setConfidence] = useState(60);
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const options = [
    { id: "TRUE", label: "True" },
    { id: "PARTIALLY_TRUE", label: "Partially True" },
    { id: "MISLEADING", label: "Misleading" },
    { id: "FALSE", label: "False" },
    { id: "SATIRE", label: "Satire" },
    { id: "UNVERIFIABLE", label: "Unverifiable" },
  ];

  const handleSubmit = async () => {
    if (!selectedVerdict) {
      alert("Select a verdict before submitting.");
      return;
    }
    setSubmitting(true);
    try {
      await axios.post(`${API_BASE}/verdict`, {
        analysis_id: result.analysis_id,
        verdict: selectedVerdict,
        confidence,
        notes,
      });
      setVerdictSubmitted(true);
      setActiveTab("analysis");
    } catch (err) {
      alert("Submit failed: " + (err.response?.data?.error || err.message));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="glass-panel p-6">
      <h2 className="text-lg font-semibold">Issue Human Verdict</h2>
      <p className="mt-2 text-sm text-gray-400">Review the forensic evidence and assign a final classification.</p>
      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        {options.map((option) => (
          <button
            key={option.id}
            type="button"
            onClick={() => setSelectedVerdict(option.id)}
            className={`rounded-3xl border px-4 py-4 text-left transition ${
              selectedVerdict === option.id
                ? "border-cyan-500 bg-cyan-500/10 text-cyan-100"
                : "border-gray-800 bg-gray-900/80 text-gray-300 hover:border-gray-600"
            }`}
          >
            <div className="font-semibold">{option.label}</div>
          </button>
        ))}
      </div>

      <div className="mt-5 space-y-4">
        <div>
          <label className="text-sm text-gray-300">Confidence: {confidence}%</label>
          <input
            type="range"
            min="0"
            max="100"
            value={confidence}
            onChange={(event) => setConfidence(Number(event.target.value))}
            className="w-full cursor-pointer appearance-none rounded-full bg-gray-800 accent-cyan-500"
          />
        </div>
        <div>
          <label className="text-sm text-gray-300">Reviewer Notes</label>
          <textarea
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            placeholder="Add context, reasoning, and observations..."
            className="mt-2 w-full rounded-3xl border border-gray-800 bg-gray-900/80 p-4 text-sm text-gray-100 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
            rows={5}
          />
        </div>
        <button
          onClick={handleSubmit}
          disabled={submitting}
          className="rounded-3xl bg-cyan-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {submitting ? "Submitting..." : "Submit Verdict"}
        </button>
      </div>
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="rounded-3xl border border-gray-800 bg-gray-950/40 p-4 text-sm text-gray-300">
      <div className="text-xs uppercase tracking-[0.18em] text-gray-500">{label}</div>
      <div className="mt-3 text-lg font-semibold text-white">{value}</div>
    </div>
  );
}

function LoadingOverlay({ message }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="rounded-3xl border border-cyan-500/20 bg-gray-950/95 px-8 py-6 text-center text-white shadow-2xl shadow-cyan-500/10">
        <div className="text-lg font-semibold">Analyzing...</div>
        <p className="mt-2 text-sm text-gray-300">{message}</p>
      </div>
    </div>
  );
}

function Footer() {
  return (
    <footer className="border-t border-gray-800 bg-[#07111f] py-5 text-center text-sm text-gray-500">
      VERITAS HITL Platform v1.0 • AI assists. Humans decide.
    </footer>
  );
}

export default App;

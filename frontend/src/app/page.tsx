"use client";

import Image from "next/image";
import { ChangeEvent, FormEvent, useEffect, useState } from "react";

type HealthStatus = {
  status: string;
  model_loaded: boolean;
  device: string;
  backbone: string;
  checkpoint_found: boolean;
};

type AnalysisResult = {
  request_id: string;
  filename: string;
  priority: string;
  top_finding: string;
  top_finding_confidence: number;
  flagged_findings: string[];
  class_scores: Record<string, number>;
  processing_time_ms: number;
  model_version: string;
  heatmap_url?: string | null;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/api/v1/health`);
        if (res.ok) {
          setHealth(await res.json());
        }
      } catch {
        setHealth({
          status: "offline",
          model_loaded: false,
          device: "cpu",
          backbone: "resnet34",
          checkpoint_found: false,
        });
      }
    };

    fetchHealth();
  }, []);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setError("");

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setPreviewUrl(null);
    }
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please select an X-ray image first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_URL}/api/v1/scan/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.detail || "Analysis failed.");
      }

      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const priorityTone = {
    HIGH: "bg-rose-100 text-rose-800 border-rose-200",
    MEDIUM: "bg-amber-100 text-amber-800 border-amber-200",
    LOW: "bg-emerald-100 text-emerald-800 border-emerald-200",
  } as const;

  return (
    <main className="min-h-screen bg-slate-100 text-slate-900">
      <div className="grid-pattern min-h-screen px-4 py-8 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <header className="mb-8 flex flex-col gap-4 rounded-3xl border border-slate-200 bg-white/80 p-5 shadow-sm backdrop-blur-sm sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-indigo-600">
                Medical AI triage demo
              </p>
              <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
                Radiography Anomaly Detection
              </h1>
            </div>

            <div className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-sm font-medium text-emerald-700">
              {health ? (health.status === "ok" ? "System online" : "Demo mode active") : "Checking service..."}
            </div>
          </header>

          <section className="mb-8 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="glass-panel rounded-3xl p-6 shadow-lg shadow-slate-200/60">
              <div className="mb-6 flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-slate-500">Quick upload</p>
                  <h2 className="mt-1 text-2xl font-bold text-slate-900">Analyze a scan</h2>
                </div>
                <span className="rounded-full bg-indigo-100 px-3 py-1 text-xs font-semibold uppercase tracking-[0.15em] text-indigo-700">
                  {health?.backbone ?? "resnet34"}
                </span>
              </div>

              <form onSubmit={handleSubmit} className="space-y-5">
                <label className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-indigo-300 bg-indigo-50/60 px-6 py-10 text-center transition hover:border-indigo-500 hover:bg-indigo-100/60">
                  <input type="file" accept=".png,.jpg,.jpeg,.dcm" onChange={handleFileChange} className="hidden" />
                  <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-white text-2xl shadow-sm">
                    📷
                  </div>
                  <span className="text-lg font-semibold text-slate-800">
                    {selectedFile ? selectedFile.name : "Choose an X-ray image"}
                  </span>
                  <span className="mt-2 text-sm text-slate-500">PNG, JPG, JPEG, or DICOM</span>
                </label>

                <div className="flex flex-col gap-3 sm:flex-row">
                  <button
                    type="submit"
                    disabled={loading || !selectedFile}
                    className="inline-flex flex-1 items-center justify-center rounded-xl bg-slate-900 px-5 py-3 text-base font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
                  >
                    {loading ? "Analyzing..." : "Analyze image"}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedFile(null);
                      setPreviewUrl(null);
                      setResult(null);
                      setError("");
                    }}
                    className="rounded-xl border border-slate-300 bg-white px-5 py-3 text-base font-semibold text-slate-700 transition hover:bg-slate-50"
                  >
                    Clear
                  </button>
                </div>
              </form>

              {error ? (
                <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                  {error}
                </div>
              ) : null}

              <div className="mt-6 grid gap-3 sm:grid-cols-3">
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Status</p>
                  <p className="mt-2 text-lg font-bold text-slate-900">{health?.status ?? "unknown"}</p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Model</p>
                  <p className="mt-2 text-lg font-bold text-slate-900">{health?.model_loaded ? "Loaded" : "Demo fallback"}</p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Device</p>
                  <p className="mt-2 text-lg font-bold text-slate-900">{health?.device ?? "cpu"}</p>
                </div>
              </div>
            </div>

            <div className="glass-panel rounded-3xl p-6 shadow-lg shadow-slate-200/60">
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">Overview</p>
              <h2 className="mt-2 text-2xl font-bold text-slate-900">Interpretation guide</h2>

              <div className="mt-6 space-y-4">
                <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
                  <p className="text-sm font-semibold text-emerald-800">Low priority</p>
                  <p className="mt-1 text-sm text-emerald-700">No immediate concern, normal screening band.</p>
                </div>
                <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4">
                  <p className="text-sm font-semibold text-amber-800">Medium priority</p>
                  <p className="mt-1 text-sm text-amber-700">Review carefully and consider follow-up analysis.</p>
                </div>
                <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4">
                  <p className="text-sm font-semibold text-rose-800">High priority</p>
                  <p className="mt-1 text-sm text-rose-700">Potentially urgent case requiring specialist review.</p>
                </div>
              </div>

              <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
                This tool is intended as an assistive screening aid. It does not replace clinical judgment.
              </div>
            </div>
          </section>

          <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-5 flex items-center justify-between gap-3">
              <h3 className="text-xl font-bold text-slate-900">Result panel</h3>
              {result ? (
                <span className={`rounded-full border px-3 py-1 text-sm font-semibold ${priorityTone[result.priority as keyof typeof priorityTone] ?? "bg-slate-100 text-slate-800 border-slate-200"}`}>
                  {result.priority}
                </span>
              ) : null}
            </div>

            {result ? (
              <div className="space-y-6">
                <div className="grid gap-4 md:grid-cols-4">
                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Top finding</p>
                    <p className="mt-2 text-lg font-bold capitalize text-slate-900">{result.top_finding}</p>
                  </div>
                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Confidence</p>
                    <p className="mt-2 text-lg font-bold text-slate-900">{(result.top_finding_confidence * 100).toFixed(1)}%</p>
                  </div>
                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Flagged findings</p>
                    <p className="mt-2 text-lg font-bold text-slate-900">{result.flagged_findings.length}</p>
                  </div>
                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Processing time</p>
                    <p className="mt-2 text-lg font-bold text-slate-900">{result.processing_time_ms.toFixed(0)} ms</p>
                  </div>
                </div>

                <div className="grid gap-6 xl:grid-cols-2">
                  <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
                    <div className="border-b border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-700">
                      Uploaded image
                    </div>
                    {previewUrl ? (
                      <Image src={previewUrl} alt="Uploaded radiograph" width={1200} height={800} unoptimized className="h-[350px] w-full object-contain bg-slate-50" />
                    ) : null}
                  </div>

                  <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
                    <div className="border-b border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-700">
                      Heatmap overlay
                    </div>
                    {result.heatmap_url ? (
                      <Image src={`${API_URL}${result.heatmap_url}`} alt="Model heatmap" width={1200} height={800} unoptimized className="h-[350px] w-full object-contain bg-slate-50" />
                    ) : (
                      <div className="flex h-[350px] items-center justify-center bg-slate-100 text-sm text-slate-500">
                        Heatmap unavailable in demo mode
                      </div>
                    )}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <p className="mb-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-500">Class scores</p>
                  <div className="space-y-2">
                    {Object.entries(result.class_scores || {}).map(([label, score]) => (
                      <div key={label}>
                        <div className="mb-1 flex items-center justify-between text-sm text-slate-700">
                          <span className="capitalize">{label}</span>
                          <span>{(score * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2.5 rounded-full bg-slate-200">
                          <div
                            className="h-2.5 rounded-full bg-gradient-to-r from-indigo-500 to-cyan-400"
                            style={{ width: `${Math.max(5, score * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex min-h-[260px] items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-slate-50 text-center text-slate-500">
                Submit an image to see anomaly results here.
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}

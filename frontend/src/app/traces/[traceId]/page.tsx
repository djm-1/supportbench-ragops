"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, FileText, Layers3, Gauge, Server, TestTube2 } from "lucide-react";
import { fetchTrace } from "@/lib/api";

type TracePayload = {
  trace_id: string;
  run_id?: number;
  question_id?: string;
  created_at?: string;
  payload: {
    question?: string;
    question_id?: string;
    reference_answer?: string;
    expected_doc?: string | null;
    expected_section?: string | null;
    expected_sources?: Array<{ document: string; section: string }>;
    question_type?: string;
    reference_facts?: string[];
    evaluation_notes?: string;
    model?: string;
    alpha?: number;
    retrieve_top_k?: number;
    rerank_top_n?: number;
    answer?: {
      answer: string;
      citations: { document: string; section: string; chunk_id: string }[];
      estimated_cost_usd: number;
      latency_ms: number;
    };
    retrieved_chunks?: {
      chunk_id: string;
      document: string;
      section: string;
      text: string;
      rerank_score: number;
    }[];
    metrics?: Record<string, any>;
  };
};

export default function TracePage({ params }: { params: { traceId: string } }) {
  const [trace, setTrace] = useState<TracePayload | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchTrace(params.traceId)
      .then(setTrace)
      .catch((err) => setError(err instanceof Error ? err.message : "Trace not found"));
  }, [params.traceId]);

  if (error) {
    return <TraceShell title="Trace unavailable">{error}</TraceShell>;
  }

  if (!trace) {
    return <TraceShell title="Loading trace">Loading...</TraceShell>;
  }

  const metrics = trace.payload.metrics ?? {};
  const runId = browserSearchParam("run_id") ?? String(trace.run_id ?? "");
  const returnView = browserSearchParam("return_view") ?? "tests";
  const backHref = runId ? `/?view=${returnView}&run_id=${runId}` : `/?view=${returnView}`;
  const judgeLabel = String(metrics.judge_label ?? "not_run");
  const judgeRationale = String(metrics.judge_rationale ?? "No answer-quality rationale was recorded for this result.");
  const missingFacts = Array.isArray(metrics.missing_facts) ? metrics.missing_facts : [];
  const contradictions = Array.isArray(metrics.contradictions) ? metrics.contradictions : [];
  const requiredSources = normalizeSources(
    metrics.required_expected_sources ?? trace.payload.expected_sources,
    trace.payload.expected_doc,
    trace.payload.expected_section
  );
  const retrievedExpectedSources = normalizeSources(metrics.retrieved_expected_sources);
  const gateFailures = Array.isArray(metrics.deterministic_gate_failures) ? metrics.deterministic_gate_failures : [];
  const failureCategory = String(metrics.failure_category ?? "passed");
  const resultLabel = traceResultLabel(metrics, failureCategory);
  const issueLabel = traceIssueLabel(metrics, failureCategory);
  const sourceRecall = Number(metrics.source_recall ?? (metrics.retrieval_hit ? 1 : 0));
  const citationScore = Number(metrics.citation_accuracy ?? 0);
  const latencyMs = Number(trace.payload.answer?.latency_ms ?? metrics.latency_ms ?? 0);
  const costUsd = Number(trace.payload.answer?.estimated_cost_usd ?? metrics.cost_usd ?? 0);
  const question = trace.payload.question ?? "Question snapshot unavailable for this legacy run.";

  return (
    <TraceShell title="Trace details" backHref={backHref}>
      <Panel title="Question" icon={FileText}>
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge tone="cyan">{humanizeQuestionType(String(trace.payload.question_type ?? "direct"))}</StatusBadge>
          <span className="text-xs text-slate-500">{trace.question_id ?? trace.payload.question_id ?? "-"}</span>
        </div>
        <p className="mt-3 whitespace-pre-wrap break-words text-lg leading-8 text-white">{question}</p>
      </Panel>

      <Panel title="Result summary" icon={Gauge}>
        <div className="grid gap-3 text-sm md:grid-cols-2 xl:grid-cols-4">
          <Row label="Result" value={humanizeResult(resultLabel)} />
          <Row label="Issue" value={humanizeIssue(issueLabel)} />
          <Row label="Model" value={trace.payload.model ?? "-"} />
          <Row label="RAG params" value={`a=${trace.payload.alpha ?? "-"}, k=${trace.payload.retrieve_top_k ?? "-"}, n=${trace.payload.rerank_top_n ?? "-"}`} />
          <Row label="Answer quality" value={humanizeAnswerCheck(judgeLabel)} />
          <Row label="Source found" value={requiredSources.length ? percent(sourceRecall) : "No expected source"} />
          <Row label="Citation" value={trace.payload.answer?.citations?.length ? percent(citationScore) : "None"} />
          <Row label="Latency" value={ms(latencyMs)} />
          <Row label="Cost" value={money(costUsd)} />
        </div>
      </Panel>

      <Panel title="Expected vs generated answer" icon={TestTube2}>
        <div className="grid gap-4 lg:grid-cols-2">
          <EvidenceBox title="Expected answer">{trace.payload.reference_answer ?? "Reference answer unavailable."}</EvidenceBox>
          <EvidenceBox title="Generated answer">{trace.payload.answer?.answer ?? "No answer"}</EvidenceBox>
        </div>
        <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.035] p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Answer quality rationale</p>
          <p className="mt-2 whitespace-pre-wrap break-words text-sm leading-7 text-slate-200">{judgeRationale}</p>
        </div>
        {(missingFacts.length > 0 || contradictions.length > 0) && (
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            {missingFacts.length > 0 && <IssueBox title="Missing facts" tone="amber" items={missingFacts} />}
            {contradictions.length > 0 && <IssueBox title="Contradictions" tone="rose" items={contradictions} />}
          </div>
        )}
      </Panel>

      <Panel title="Expected source evidence" icon={Layers3}>
        <div className="grid gap-4 md:grid-cols-2">
          <SourceList title="Required sources" sources={requiredSources} empty="No expected source was attached to this question." />
          <SourceList title="Retrieved expected sources" sources={retrievedExpectedSources} empty="No required source was retrieved." />
        </div>
      </Panel>

      <Panel title="Retrieved chunks" icon={Layers3}>
        <div className="grid gap-3">
          {(trace.payload.retrieved_chunks ?? []).map((chunk, index) => (
            <div key={chunk.chunk_id} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
              <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
                <span className="font-semibold text-white">
                  {index + 1}. {chunk.document} / {chunk.section}
                </span>
                <span className="rounded-full border border-white/10 bg-white/[0.04] px-2.5 py-1 text-xs text-slate-400">
                  rerank {chunk.rerank_score}
                </span>
              </div>
              <p className="mt-3 whitespace-pre-wrap break-words text-sm leading-7 text-slate-300">{chunk.text}</p>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Technical details" icon={Server}>
        <details className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
          <summary className="cursor-pointer text-sm font-semibold text-cyan-100">Show evaluator internals</summary>
          <div className="mt-4 grid gap-3 text-sm md:grid-cols-2">
            <Row label="Trace ID" value={trace.trace_id} />
            <Row label="Run ID" value={String(trace.run_id ?? "-")} />
            <Row label="Groundedness" value={formatMetricValue(metrics.groundedness)} />
            <Row label="MRR" value={formatMetricValue(metrics.mrr)} />
            <Row label="Check source" value={humanizeCheckSource(String(metrics.judge_source ?? "deterministic"))} />
            <Row label="Check model" value={humanizeJudgeModel(String(metrics.judge_model ?? "deterministic_fallback"))} />
            <Row label="Prompt version" value={String(metrics.judge_prompt_version ?? "unknown")} />
            <Row label="Failed gates" value={gateFailures.length ? gateFailures.map(humanize).join(", ") : "None"} />
          </div>
        </details>
      </Panel>
    </TraceShell>
  );
}

function TraceShell({ title, children, backHref = "/" }: { title: string; children: React.ReactNode; backHref?: string }) {
  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,#14213f_0,#080b13_34%,#05070d_100%)] text-slate-100">
      <div className="mx-auto max-w-6xl px-5 py-8 lg:px-8">
        <a href={backHref} className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-sm font-medium text-cyan-100 transition hover:border-cyan-300/40 hover:bg-cyan-300/10">
          <ArrowLeft className="h-4 w-4" />
          Back to dashboard
        </a>
        <p className="mt-7 text-xs font-semibold uppercase tracking-[0.22em] text-cyan-200">RAG trace drill-down</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white md:text-5xl">{title}</h1>
        <div className="mt-7 grid gap-5">{children}</div>
      </div>
    </main>
  );
}

function Panel({ title, icon: Icon, children }: { title: string; icon: typeof FileText; children: React.ReactNode }) {
  return (
    <section className="rounded-3xl border border-white/10 bg-[#0a0f1d]/82 p-5 shadow-2xl shadow-black/25 backdrop-blur-xl">
      <div className="flex items-center gap-3">
        <span className="grid h-9 w-9 place-items-center rounded-xl bg-cyan-300/10 text-cyan-100">
          <Icon className="h-4 w-4" />
        </span>
        <h2 className="text-lg font-semibold text-white">{title}</h2>
      </div>
      <div className="mt-5">{children}</div>
    </section>
  );
}

function EvidenceBox({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</p>
      <p className="mt-2 whitespace-pre-wrap break-words text-sm leading-7 text-slate-200">{children}</p>
    </div>
  );
}

function StatusBadge({ children, tone }: { children: React.ReactNode; tone: "emerald" | "rose" | "amber" | "cyan" }) {
  const tones = {
    emerald: "border-emerald-300/25 bg-emerald-300/10 text-emerald-100",
    rose: "border-rose-300/25 bg-rose-300/10 text-rose-100",
    amber: "border-amber-300/25 bg-amber-300/10 text-amber-100",
    cyan: "border-cyan-300/25 bg-cyan-300/10 text-cyan-100"
  };
  return (
    <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold ${tones[tone]}`}>
      {children}
    </span>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-w-0 items-start justify-between gap-4 rounded-xl border border-white/10 bg-white/[0.035] px-3 py-2">
      <dt className="shrink-0 capitalize text-slate-400">{label}</dt>
      <dd className="min-w-0 break-words text-right font-medium text-white">{value}</dd>
    </div>
  );
}

function percent(value: number) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return `${Math.round(value * 100)}%`;
}

function money(value: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return "$0";
  }
  if (value < 0.00001) {
    return "<$0.00001";
  }
  return `$${value.toFixed(5)}`;
}

function ms(value: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return "-";
  }
  return `${Math.round(value).toLocaleString()} ms`;
}

function IssueBox({ title, items, tone }: { title: string; items: string[]; tone: "amber" | "rose" }) {
  const toneClass = tone === "amber" ? "border-amber-300/20 bg-amber-300/10 text-amber-100" : "border-rose-300/20 bg-rose-300/10 text-rose-100";
  return (
    <div className={`rounded-2xl border p-4 ${toneClass}`}>
      <p className="text-xs font-semibold uppercase tracking-wide">{title}</p>
      <ul className="mt-2 list-disc space-y-1 pl-4 text-sm leading-6">
        {items.map((item) => (
          <li key={item} className="break-words">{item}</li>
        ))}
      </ul>
    </div>
  );
}

function SourceList({
  title,
  sources,
  empty
}: {
  title: string;
  sources: Array<{ document: string; section: string }>;
  empty: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</p>
      {sources.length ? (
        <div className="mt-3 space-y-2">
          {sources.map((source) => (
            <div
              key={`${source.document}-${source.section}`}
              className="break-words rounded-xl border border-white/10 bg-slate-950/40 px-3 py-2 text-sm leading-6 text-slate-200"
            >
              {source.document} / {source.section}
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-3 text-sm text-slate-500">{empty}</p>
      )}
    </div>
  );
}

function formatMetricKey(key: string) {
  return key.replaceAll("_", " ");
}

function formatMetricValue(value: any) {
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }
  return String(value ?? "-");
}

function humanizeAnswerCheck(value: string) {
  if (value === "judge_unavailable") {
    return "Check unavailable";
  }
  if (value === "not_run") {
    return "Not checked";
  }
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function humanize(value: string) {
  if (!value) {
    return "None";
  }
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function traceResultLabel(metrics: Record<string, any>, failureCategory: string): "right" | "partial" | "wrong" {
  const value = String(metrics.result_label ?? "");
  if (value === "right" || value === "partial" || value === "wrong") {
    return value;
  }
  if (failureCategory === "passed") {
    return "right";
  }
  if (["partial_answer", "partial_citation", "missing_citation", "needs_review"].includes(failureCategory)) {
    return "partial";
  }
  return "wrong";
}

function traceIssueLabel(metrics: Record<string, any>, failureCategory: string) {
  const value = String(metrics.issue_label ?? "");
  if (value) {
    return value;
  }
  if (failureCategory === "passed") {
    return "none";
  }
  if (failureCategory === "bad_retrieval") {
    return "source_not_found";
  }
  if (["missing_citation", "partial_citation", "wrong_citation", "unsupported_citation"].includes(failureCategory)) {
    return "citation_issue";
  }
  if (["partial_answer", "needs_review", "incorrect_refusal"].includes(failureCategory)) {
    return "incomplete_answer";
  }
  return "unsupported_answer";
}

function humanizeResult(result: string) {
  const labels: Record<string, string> = {
    right: "Right",
    partial: "Partial",
    wrong: "Wrong"
  };
  return labels[result] ?? humanize(result);
}

function humanizeIssue(issue: string) {
  const labels: Record<string, string> = {
    none: "None",
    source_not_found: "Source not found",
    citation_issue: "Citation issue",
    incomplete_answer: "Incomplete answer",
    unsupported_answer: "Unsupported answer",
    provider_error: "Provider error"
  };
  return labels[issue] ?? humanize(issue);
}

function humanizeCheckSource(value: string) {
  if (value === "llm") {
    return "LLM check";
  }
  if (value === "fallback") {
    return "Rule-based fallback";
  }
  return "Rule-based fallback";
}

function humanizeJudgeModel(value: string) {
  if (value === "deterministic_fallback") {
    return "Rule-based fallback";
  }
  return value;
}

function humanizeQuestionType(value: string) {
  const labels: Record<string, string> = {
    direct: "Direct",
    paraphrase: "Paraphrase",
    calculation: "Calculation",
    missing_data: "Missing data",
    cross_document: "Cross-doc"
  };
  return labels[value] ?? humanize(value);
}

function normalizeSources(
  sources?: Array<{ document?: string; section?: string }> | null,
  fallbackDoc?: string | null,
  fallbackSection?: string | null
) {
  const normalized = (sources ?? [])
    .filter((source) => source.document && source.section)
    .map((source) => ({ document: String(source.document), section: String(source.section) }));
  if (normalized.length) {
    return normalized;
  }
  if (fallbackDoc && fallbackSection) {
    return [{ document: fallbackDoc, section: fallbackSection }];
  }
  return [];
}

function browserSearchParam(key: string) {
  if (typeof window === "undefined") {
    return null;
  }
  return new URLSearchParams(window.location.search).get(key);
}

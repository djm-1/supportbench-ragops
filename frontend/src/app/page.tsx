"use client";

import { ReactNode, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Beaker,
  BookOpen,
  CheckCircle2,
  CircleDollarSign,
  ClipboardList,
  Download,
  FileText,
  Gauge,
  History as HistoryIcon,
  Info,
  Layers3,
  Play,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Target,
  TestTube2,
  XCircle,
  type LucideIcon
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import {
  ConfigLeaderboardRow,
  DashboardHistory,
  DashboardSummary,
  emptySummary,
  EvalQuestion,
  fetchDashboardHistory,
  fetchDashboardSummary,
  fetchEvalQuestions,
  fetchJob,
  fetchModels,
  fetchRetrievalStatus,
  fetchRunDetails,
  HistoryConfigPerformanceRow,
  HistoryModelPerformanceRow,
  JobProgress,
  ModelInfo,
  ModelLeaderboardRow,
  RetrievalStatus,
  RunDetails,
  RunResult,
  runExportUrl,
  startBenchmark,
  startIngestion
} from "@/lib/api";

type View = "decision" | "experiment" | "tests" | "history" | "methodology";
type TestStatusFilter = "all" | "right" | "partial" | "wrong";
type SourceFilter = "all" | "matched" | "missed";
type QuestionTypeFilter = "all" | string;

const defaultModelAliases = [
  "groq_llama_3_1_8b",
  "groq_llama_3_3_70b",
  "groq_gpt_oss_20b",
  "openai_primary",
  "gemini_flash"
];

const modelColors: Record<string, string> = {
  groq_llama_3_1_8b: "#34d399",
  groq_llama_3_3_70b: "#38bdf8",
  groq_gpt_oss_20b: "#fb923c",
  openai_primary: "#a78bfa",
  gemini_flash: "#fb7185"
};

const failureColors = ["#fb7185", "#fb923c", "#facc15", "#a78bfa", "#38bdf8", "#94a3b8"];
const alphaOptions = [0.25, 0.5, 0.75];
const retrieveKOptions = [5, 10, 20];
const rerankNOptions = [3, 5];

const navItems: Array<{ id: View; label: string; icon: LucideIcon; description: string }> = [
  { id: "decision", label: "Decision", icon: Sparkles, description: "What to ship" },
  { id: "experiment", label: "Experiment", icon: Beaker, description: "Run controls" },
  { id: "tests", label: "Test Cases", icon: ClipboardList, description: "Golden set proof" },
  { id: "history", label: "History", icon: HistoryIcon, description: "Averages over time" },
  { id: "methodology", label: "Methodology", icon: BookOpen, description: "How scoring works" }
];

function percent(value?: number) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function money(value?: number) {
  const amount = value ?? 0;
  if (amount === 0) {
    return "$0";
  }
  return amount < 0.0001 ? `$${amount.toFixed(5)}` : `$${amount.toFixed(4)}`;
}

function ms(value?: number) {
  return `${Math.round(value ?? 0).toLocaleString()} ms`;
}

function parseView(value: string | null): View | null {
  return value === "decision" || value === "experiment" || value === "tests" || value === "history" || value === "methodology"
    ? value
    : null;
}

function parseRunId(value: string | null): number | null {
  if (!value) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function browserSearchParam(key: string) {
  if (typeof window === "undefined") {
    return null;
  }
  return new URLSearchParams(window.location.search).get(key);
}

export default function DashboardPage() {
  const [summary, setSummary] = useState<DashboardSummary>(emptySummary);
  const [history, setHistory] = useState<DashboardHistory>({
    runs: [],
    model_performance: [],
    config_performance: []
  });
  const [retrievalStatus, setRetrievalStatus] = useState<RetrievalStatus | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>(defaultModelAliases);
  const [selectedAlphas, setSelectedAlphas] = useState<number[]>([0.25, 0.5, 0.75]);
  const [selectedRetrieveKs, setSelectedRetrieveKs] = useState<number[]>([10]);
  const [selectedRerankNs, setSelectedRerankNs] = useState<number[]>([3, 5]);
  const [questionLimit, setQuestionLimit] = useState(10);
  const [evalQuestions, setEvalQuestions] = useState<EvalQuestion[]>([]);
  const [selectedQuestionIds, setSelectedQuestionIds] = useState<string[]>([]);
  const [jobProgress, setJobProgress] = useState<JobProgress | null>(null);
  const [activeRunDetails, setActiveRunDetails] = useState<RunDetails | null>(null);
  const [status, setStatus] = useState("Loading benchmark state...");
  const [busyAction, setBusyAction] = useState<"ingest" | "benchmark" | null>(null);
  const [activeView, setActiveView] = useState<View>("decision");

  const isBusy = busyAction !== null;
  const modelMap = useMemo(() => Object.fromEntries(models.map((model) => [model.alias, model])), [models]);
  const effectiveQuestionCount = selectedQuestionIds.length || questionLimit;
  const runSize =
    selectedModels.length *
    selectedAlphas.length *
    selectedRetrieveKs.length *
    selectedRerankNs.length *
    effectiveQuestionCount;
  const selectedUnavailableModels = selectedModels
    .map((alias) => modelMap[alias])
    .filter((model): model is ModelInfo => Boolean(model && !model.is_available));

  const failureChartData = useMemo(() => {
    const byModel: Record<string, Record<string, number | string>> = {};
    for (const row of summary.failure_by_model) {
      byModel[row.model] ??= { model: displayModel(row.model, modelMap), alias: row.model };
      byModel[row.model][humanize(row.category)] = row.count;
    }
    return Object.values(byModel);
  }, [summary.failure_by_model, modelMap]);

  const failureCategories = useMemo(
    () => Array.from(new Set(summary.failure_by_model.map((row) => humanize(row.category)))),
    [summary.failure_by_model]
  );

  const qualityPoints = useMemo(
    () =>
      summary.quality_cost_points.slice(0, 60).map((point) => ({
        ...point,
        label: displayModel(point.model, modelMap)
      })),
    [summary.quality_cost_points, modelMap]
  );

  async function refreshAll() {
    try {
      const [dashboard, retrieval, modelCatalog, historyData, questions] = await Promise.all([
        fetchDashboardSummary(),
        fetchRetrievalStatus(),
        fetchModels(),
        fetchDashboardHistory(),
        fetchEvalQuestions()
      ]);
      setSummary(dashboard);
      setRetrievalStatus(retrieval);
      setModels(modelCatalog);
      setHistory(historyData);
      setEvalQuestions(questions);
      const requestedRunId = parseRunId(browserSearchParam("run_id"));
      const runIdToLoad = requestedRunId ?? dashboard.run_id;
      if (runIdToLoad) {
        fetchRunDetails(runIdToLoad)
          .then(setActiveRunDetails)
          .catch(() => setActiveRunDetails(null));
      } else {
        setActiveRunDetails(null);
      }
      if (modelCatalog.length && selectedModels.length === defaultModelAliases.length) {
        setSelectedModels(modelCatalog.map((model) => model.alias));
      }
      if (dashboard.run_id) {
        setStatus(`Loaded run ${dashboard.run_id}: ${dashboard.result_count ?? 0} stored results`);
      } else if (dashboard.dashboard_state === "empty") {
        setStatus("No completed benchmark yet. Run a benchmark to populate results.");
      } else {
        setStatus(`Dashboard state: ${dashboard.dashboard_state}`);
      }
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Backend unavailable");
    }
  }

  useEffect(() => {
    refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const requestedView = parseView(browserSearchParam("view"));
    if (requestedView) {
      setActiveView(requestedView);
    }
  }, []);

  async function handleIngest() {
    setBusyAction("ingest");
    setJobProgress(null);
    setActiveView("experiment");
    setStatus("Ingesting support docs...");
    try {
      const started = await startIngestion();
      const completed = await waitForJob(started, setJobProgress);
      const result = completed.result ?? {};
      setStatus(
        `Docs ingested: ${result.documents} documents, ${result.chunks} chunks. Run a benchmark to populate results.`
      );
      const [dashboard, retrieval, historyData, modelCatalog, questions] = await Promise.all([
        fetchDashboardSummary(),
        fetchRetrievalStatus(),
        fetchDashboardHistory(),
        fetchModels(),
        fetchEvalQuestions()
      ]);
      setSummary(dashboard);
      setRetrievalStatus(retrieval);
      setHistory(historyData);
      setModels(modelCatalog);
      setEvalQuestions(questions);
      if (dashboard.run_id) {
        setActiveRunDetails(await fetchRunDetails(dashboard.run_id).catch(() => null));
      } else {
        setActiveRunDetails(null);
      }
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Ingest failed");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleBenchmarkRun() {
    if (!selectedModels.length || !selectedAlphas.length || !selectedRetrieveKs.length || !selectedRerankNs.length) {
      setStatus("Select at least one model and one value for each RAG parameter.");
      return;
    }
    setBusyAction("benchmark");
    setJobProgress(null);
    setActiveView("experiment");
    setStatus(`Running ${runSize} model calls. Results will publish to the Decision tab when complete...`);
    try {
      const started = await startBenchmark({
        run_name: `benchmark-${new Date().toISOString()}`,
        question_limit: questionLimit,
        question_ids: selectedQuestionIds.length ? selectedQuestionIds : undefined,
        models: selectedModels,
        alphas: selectedAlphas,
        retrieve_top_k: selectedRetrieveKs,
        rerank_top_n: selectedRerankNs
      });
      const completed = await waitForJob(started, setJobProgress, async (job) => {
        if (job.run_id) {
          setActiveRunDetails(await fetchRunDetails(job.run_id).catch(() => null));
        }
      });
      const [activeSummary, historyData, runDetails] = await Promise.all([
        fetchDashboardSummary().catch(() => completed.result?.summary ?? emptySummary),
        fetchDashboardHistory(),
        completed.run_id ? fetchRunDetails(completed.run_id) : Promise.resolve(null)
      ]);
      setSummary(activeSummary);
      setHistory(historyData);
      setActiveRunDetails(runDetails);
      setActiveView("decision");
      setStatus(
        `Benchmark complete: ${completed.result?.total_results ?? completed.completed_items} results stored in run ${completed.run_id}`
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Benchmark failed");
      setHistory(await fetchDashboardHistory().catch(() => ({ runs: [], model_performance: [], config_performance: [] })));
    } finally {
      setBusyAction(null);
    }
  }

  function applyFullPreset() {
    setSelectedModels(models.length ? models.map((model) => model.alias) : defaultModelAliases);
    setSelectedAlphas([0.25, 0.5, 0.75]);
    setSelectedRetrieveKs([10]);
    setSelectedRerankNs([3, 5]);
    setQuestionLimit(10);
    setSelectedQuestionIds([]);
  }

  return (
    <AppShell activeView={activeView} setActiveView={setActiveView} status={status} retrievalStatus={retrievalStatus}>
      {activeView === "decision" && (
        <DecisionView
          summary={summary}
          modelMap={modelMap}
          qualityPoints={qualityPoints}
          failureChartData={failureChartData}
          failureCategories={failureCategories}
          activeRunDetails={activeRunDetails}
        />
      )}
      {activeView === "experiment" && (
        <ExperimentView
          models={models}
          modelMap={modelMap}
          selectedModels={selectedModels}
          setSelectedModels={setSelectedModels}
          selectedAlphas={selectedAlphas}
          setSelectedAlphas={setSelectedAlphas}
          selectedRetrieveKs={selectedRetrieveKs}
          setSelectedRetrieveKs={setSelectedRetrieveKs}
          selectedRerankNs={selectedRerankNs}
          setSelectedRerankNs={setSelectedRerankNs}
          questionLimit={questionLimit}
          setQuestionLimit={setQuestionLimit}
          evalQuestions={evalQuestions}
          selectedQuestionIds={selectedQuestionIds}
          setSelectedQuestionIds={setSelectedQuestionIds}
          jobProgress={jobProgress}
          isBusy={isBusy}
          busyAction={busyAction}
          runSize={runSize}
          selectedUnavailableModels={selectedUnavailableModels}
          retrievalStatus={retrievalStatus}
          onIngest={handleIngest}
          onBenchmark={handleBenchmarkRun}
          onPreset={applyFullPreset}
        />
      )}
      {activeView === "tests" && (
        <TestCasesPanel runDetails={activeRunDetails} questions={evalQuestions} modelMap={modelMap} />
      )}
      {activeView === "history" && <HistoryPanel history={history} modelMap={modelMap} />}
      {activeView === "methodology" && <MethodologyPanel retrievalStatus={retrievalStatus} />}
    </AppShell>
  );
}

function AppShell({
  activeView,
  setActiveView,
  status,
  retrievalStatus,
  children
}: {
  activeView: View;
  setActiveView: (view: View) => void;
  status: string;
  retrievalStatus: RetrievalStatus | null;
  children: ReactNode;
}) {
  return (
    <main className="min-h-screen overflow-x-hidden bg-[radial-gradient(circle_at_top_left,#14213f_0,#080b13_34%,#05070d_100%)] text-slate-100">
      <div className="lg:flex">
        <aside className="sticky top-0 z-30 max-w-full overflow-hidden border-b border-white/10 bg-[#070a12]/90 px-4 py-4 backdrop-blur-xl lg:h-screen lg:w-72 lg:overflow-visible lg:border-b-0 lg:border-r lg:px-5">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-cyan-300 to-emerald-300 text-slate-950 shadow-lg shadow-cyan-500/20">
              <Layers3 className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold text-white">SupportBench</p>
              <p className="text-xs text-slate-400">RAGOps evaluator</p>
            </div>
          </div>

          <nav className="no-scrollbar mt-5 flex max-w-full gap-2 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = item.id === activeView;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveView(item.id)}
                  className={`group flex min-w-max items-center gap-3 rounded-xl border px-3 py-3 text-left transition lg:min-w-0 ${
                    active
                      ? "border-cyan-300/40 bg-cyan-300/10 text-white shadow-lg shadow-cyan-950/40"
                      : "border-transparent text-slate-400 hover:border-white/10 hover:bg-white/[0.04] hover:text-slate-100"
                  }`}
                >
                  <Icon className={`h-4 w-4 ${active ? "text-cyan-200" : "text-slate-500 group-hover:text-slate-200"}`} />
                  <span>
                    <span className="block text-sm font-medium">{item.label}</span>
                    <span className="hidden text-xs text-slate-500 lg:block">{item.description}</span>
                  </span>
                </button>
              );
            })}
          </nav>

          <div className="mt-6 hidden rounded-2xl border border-white/10 bg-white/[0.035] p-4 lg:block">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Pipeline</p>
            <p className="mt-2 text-sm text-slate-200">
              {retrievalStatus?.dense_retrieval ?? "retrieval unknown"}
            </p>
            <p className="mt-1 text-xs leading-5 text-slate-500">
              {retrievalStatus?.use_pinecone
                ? `${retrievalStatus.pinecone_index}:${retrievalStatus.pinecone_namespace}`
                : "Local retrieval fallback"}
            </p>
            {retrievalStatus?.dataset && (
              <div className="mt-4 border-t border-white/10 pt-4">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Knowledge base</p>
                <p className="mt-2 text-sm text-slate-200">{retrievalStatus.dataset.active_dataset}</p>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  {retrievalStatus.dataset.faq_pair_count || retrievalStatus.dataset.support_doc_count} FAQ/doc items ·{" "}
                  {retrievalStatus.dataset.eval_question_count} eval questions
                </p>
                {retrievalStatus.dataset.generated_at && (
                  <p className="mt-1 text-xs text-slate-600">
                    Extracted {new Date(retrievalStatus.dataset.generated_at).toLocaleString()}
                  </p>
                )}
              </div>
            )}
          </div>
        </aside>

        <section className="min-w-0 flex-1">
          <header className="border-b border-white/10 bg-[#080b13]/70 backdrop-blur-xl">
            <div className="mx-auto max-w-7xl px-5 py-6 lg:px-8">
              <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-200">
                    Production RAGOps benchmark
                  </p>
                  <h1 className="mt-2 max-w-4xl text-3xl font-semibold tracking-tight text-white md:text-5xl">
                    Choose the model and RAG config worth shipping.
                  </h1>
                  <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
                    Compare answer quality, cost, latency, citations, retrieval evidence, and failure modes on the same
                    support knowledge base.
                  </p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/[0.035] px-4 py-3 text-sm text-slate-300 shadow-2xl shadow-black/30">
                  <div className="flex items-start gap-3">
                    <Activity className="mt-0.5 h-4 w-4 text-cyan-200" />
                    <p>{status}</p>
                  </div>
                </div>
              </div>
            </div>
          </header>
          <div className="mx-auto max-w-7xl px-5 py-6 lg:px-8">{children}</div>
        </section>
      </div>
    </main>
  );
}

function DecisionView({
  summary,
  modelMap,
  qualityPoints,
  failureChartData,
  failureCategories,
  activeRunDetails
}: {
  summary: DashboardSummary;
  modelMap: Record<string, ModelInfo>;
  qualityPoints: ConfigLeaderboardRow[];
  failureChartData: Record<string, number | string>[];
  failureCategories: string[];
  activeRunDetails: RunDetails | null;
}) {
  const hasResults = summary.config_leaderboard.length > 0;
  if (!hasResults) {
    return (
      <EmptyState
        icon={Target}
        title="No active benchmark decision yet"
        body="Run a benchmark from the Experiment tab. This screen will show the recommended setup, tradeoffs, and failed cases."
      />
    );
  }

  const exportHref = summary.run_id ? runExportUrl(summary.run_id) : undefined;
  const winningConfig = summary.config_leaderboard[0] ?? summary.best_config;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-sm text-slate-400">Run {summary.run_id}</p>
          <h2 className="text-2xl font-semibold tracking-tight text-white">Recommended setup</h2>
        </div>
        {exportHref && <ExportButton href={exportHref}>Export run CSV</ExportButton>}
      </div>

      {summary.summary_notice && (
        <div className={`rounded-2xl border px-4 py-3 text-sm ${
          summary.is_stale
            ? "border-amber-300/25 bg-amber-300/10 text-amber-100"
            : "border-cyan-300/20 bg-cyan-300/10 text-cyan-100"
        }`}>
          {summary.summary_notice}
        </div>
      )}

      <RecommendationCard summary={summary} modelMap={modelMap} />

      <section className="grid gap-4 xl:grid-cols-5">
        <MetricCard icon={ShieldCheck} label="Best overall model" value={displayModel(summary.best_model?.model, modelMap)} subvalue={percent(summary.best_model?.quality_score)} tone="emerald" />
        <MetricCard icon={CircleDollarSign} label="Cheapest acceptable" value={displayModel(summary.cheapest_acceptable_model?.model, modelMap)} subvalue={money(summary.cheapest_acceptable_model?.cost_per_query)} tone="cyan" />
        <MetricCard icon={Gauge} label="Fastest acceptable" value={displayModel(summary.fastest_acceptable_model?.model, modelMap)} subvalue={ms(summary.fastest_acceptable_model?.avg_latency_ms)} tone="amber" />
        <MetricCard icon={Layers3} label="Best RAG params" value={bestParamsLabel(winningConfig)} subvalue="alpha / k / n" tone="violet" />
        <MetricCard icon={AlertTriangle} label="Most common failure" value={humanize(summary.most_common_failure?.category ?? "none")} subvalue={`${summary.most_common_failure?.count ?? 0} cases`} tone="rose" />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <Panel title="Quality vs cost" subtitle="Each point is one model plus RAG parameter combination.">
          <QualityCostChart points={qualityPoints} modelMap={modelMap} />
        </Panel>
        <Panel title="Failure by model" subtitle="Shows which model failed and the dominant reason.">
          <FailureChart data={failureChartData} categories={failureCategories} />
        </Panel>
      </section>

      <section className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
        <Panel title="Model leaderboard" subtitle="One row per model, averaged across selected RAG configs.">
          <ModelLeaderboard rows={summary.model_leaderboard} modelMap={modelMap} />
        </Panel>
        <Panel title="Best RAG configs" subtitle="Repeated models mean a different alpha, retrieval depth, or rerank depth.">
          <ConfigLeaderboard rows={summary.config_leaderboard} modelMap={modelMap} />
        </Panel>
      </section>

      {activeRunDetails && <DecisionProofStrip runDetails={activeRunDetails} modelMap={modelMap} />}
    </div>
  );
}

function RecommendationCard({
  summary,
  modelMap
}: {
  summary: DashboardSummary;
  modelMap: Record<string, ModelInfo>;
}) {
  const config = summary.config_leaderboard[0] ?? summary.best_config;
  const model = displayModel(config?.model, modelMap);
  const quality = percent(config?.quality_score);
  const cost = money(config?.cost_per_query);
  const latency = ms(config?.avg_latency_ms);
  const failure = percent(config?.failure_rate);
  const resultCount = summary.result_count ?? 0;
  const confidence =
    resultCount >= 50 ? "High confidence" : resultCount >= 15 ? "Medium confidence" : "Low sample confidence";
  const cheapest = summary.cheapest_acceptable_model;
  const fastest = summary.fastest_acceptable_model;

  return (
    <section className="overflow-hidden rounded-3xl border border-cyan-300/20 bg-gradient-to-br from-cyan-300/14 via-white/[0.045] to-emerald-300/10 p-1 shadow-2xl shadow-cyan-950/30">
      <div className="rounded-[1.35rem] bg-[#090d18]/88 p-5 md:p-6">
        <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/25 bg-cyan-300/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-cyan-100">
              <Sparkles className="h-3.5 w-3.5" />
              Ship this config
            </div>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight text-white">
              {model} with {bestParamsLabel(config)}
            </h2>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
              This configuration ranks highest on the combined benchmark score while preserving citation and retrieval
              evidence for every answer.
            </p>
          </div>
          <div className="grid min-w-[260px] grid-cols-2 gap-3 text-sm">
            <MiniStat label="Quality" value={quality} />
            <MiniStat label="Cost/query" value={cost} />
            <MiniStat label="Latency" value={latency} />
            <MiniStat label="Failure risk" value={failure} />
          </div>
        </div>

        <div className="mt-6 grid gap-3 lg:grid-cols-4">
          <DecisionNote title="Why" icon={CheckCircle2}>
            Highest quality score in this active run, with groundedness {percent(config?.groundedness)} and citation
            accuracy {percent(config?.citation_accuracy)}.
          </DecisionNote>
          <DecisionNote title="Tradeoff" icon={CircleDollarSign}>
            {cheapest && cheapest.model !== config?.model
              ? `${displayModel(cheapest.model, modelMap)} is cheaper at ${money(cheapest.cost_per_query)}, but with ${percent(cheapest.quality_score)} quality.`
              : "The winning config is also cost-efficient among acceptable models for this run."}
          </DecisionNote>
          <DecisionNote title="Failure risk" icon={AlertTriangle}>
            Main observed risk is {humanize(summary.most_common_failure?.category ?? "none")}. Review failed test cases
            before production rollout.
          </DecisionNote>
          <DecisionNote title="Confidence" icon={Gauge}>
            {confidence} from {resultCount} stored test cases. Fastest acceptable: {displayModel(fastest?.model, modelMap)}.
          </DecisionNote>
        </div>
      </div>
    </section>
  );
}

function ExperimentView({
  models,
  modelMap,
  selectedModels,
  setSelectedModels,
  selectedAlphas,
  setSelectedAlphas,
  selectedRetrieveKs,
  setSelectedRetrieveKs,
  selectedRerankNs,
  setSelectedRerankNs,
  questionLimit,
  setQuestionLimit,
  evalQuestions,
  selectedQuestionIds,
  setSelectedQuestionIds,
  jobProgress,
  isBusy,
  busyAction,
  runSize,
  selectedUnavailableModels,
  retrievalStatus,
  onIngest,
  onBenchmark,
  onPreset
}: {
  models: ModelInfo[];
  modelMap: Record<string, ModelInfo>;
  selectedModels: string[];
  setSelectedModels: (values: string[]) => void;
  selectedAlphas: number[];
  setSelectedAlphas: (values: number[]) => void;
  selectedRetrieveKs: number[];
  setSelectedRetrieveKs: (values: number[]) => void;
  selectedRerankNs: number[];
  setSelectedRerankNs: (values: number[]) => void;
  questionLimit: number;
  setQuestionLimit: (value: number) => void;
  evalQuestions: EvalQuestion[];
  selectedQuestionIds: string[];
  setSelectedQuestionIds: (ids: string[]) => void;
  jobProgress: JobProgress | null;
  isBusy: boolean;
  busyAction: "ingest" | "benchmark" | null;
  runSize: number;
  selectedUnavailableModels: ModelInfo[];
  retrievalStatus: RetrievalStatus | null;
  onIngest: () => void;
  onBenchmark: () => void;
  onPreset: () => void;
}) {
  const modelCards = models.length ? models : defaultModelAliases.map(fallbackModel);

  return (
    <div className="space-y-6">
      <Panel
        title="Experiment controls"
        subtitle="Pick the model set, RAG parameters, and question scope for the next benchmark run."
        action={
          <div className="flex flex-wrap gap-2">
            <SecondaryButton onClick={onIngest} disabled={isBusy} icon={RefreshCw}>
              {busyAction === "ingest" ? "Ingesting" : "Ingest docs"}
            </SecondaryButton>
            <PrimaryButton onClick={onBenchmark} disabled={isBusy} icon={Play}>
              {busyAction === "benchmark" ? "Running" : "Run benchmark"}
            </PrimaryButton>
          </div>
        }
      >
        {jobProgress && <ProgressPanel progress={jobProgress} />}
        {retrievalStatus?.dataset && (
          <div className="mb-5 grid gap-3 rounded-2xl border border-white/10 bg-white/[0.035] p-4 md:grid-cols-4">
            <MiniStat label="Knowledge base" value={retrievalStatus.dataset.active_dataset} />
            <MiniStat label="FAQ pairs" value={`${retrievalStatus.dataset.faq_pair_count || retrievalStatus.dataset.support_doc_count}`} />
            <MiniStat label="Eval questions" value={`${retrievalStatus.dataset.eval_question_count}`} />
            <MiniStat
              label="Extracted"
              value={
                retrievalStatus.dataset.generated_at
                  ? new Date(retrievalStatus.dataset.generated_at).toLocaleDateString()
                : "Sample data"
              }
            />
            <p className="text-xs leading-5 text-slate-500 md:col-span-4">
              {retrievalStatus.dataset.is_generated
                ? "Generated source data is active. Refresh the extracted files, then run ingestion to update the retrieval index."
                : "Sample data is active. Generate the commerce support dataset, then run ingestion to switch the knowledge base."}
            </p>
          </div>
        )}

        <div className="grid gap-5 xl:grid-cols-[1.45fr_0.9fr]">
          <div>
            <ControlLabel>Models</ControlLabel>
            <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {modelCards.map((model) => (
                <button
                  key={model.alias}
                  onClick={() => setSelectedModels(toggleValue(selectedModels, model.alias))}
                  className={`rounded-2xl border p-4 text-left transition ${
                    selectedModels.includes(model.alias)
                      ? "border-cyan-300/50 bg-cyan-300/10 shadow-lg shadow-cyan-950/30"
                      : "border-white/10 bg-white/[0.035] hover:border-white/20 hover:bg-white/[0.06]"
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="font-medium text-white">{model.display_name}</p>
                      <p className="mt-1 text-xs text-slate-500">{model.provider} / {model.model}</p>
                    </div>
                    <StatusBadge tone={model.is_available ? "emerald" : "rose"}>
                      {model.is_available ? "ready" : "setup"}
                    </StatusBadge>
                  </div>
                  {!model.is_available && (
                    <p className="mt-3 text-xs leading-5 text-amber-200">{model.requires || model.setup_command}</p>
                  )}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-5 rounded-2xl border border-white/10 bg-[#070b13]/75 p-4">
            <div>
              <ControlLabel>Hybrid alpha</ControlLabel>
              <ChipGroup values={alphaOptions} selected={selectedAlphas} setSelected={setSelectedAlphas} />
            </div>
            <div>
              <ControlLabel>Retrieve top_k</ControlLabel>
              <ChipGroup values={retrieveKOptions} selected={selectedRetrieveKs} setSelected={setSelectedRetrieveKs} />
            </div>
            <div>
              <ControlLabel>Rerank top_n</ControlLabel>
              <ChipGroup values={rerankNOptions} selected={selectedRerankNs} setSelected={setSelectedRerankNs} />
            </div>
            <div>
              <ControlLabel>Question count {selectedQuestionIds.length ? "(explicit selection active)" : ""}</ControlLabel>
              <input
                type="number"
                min={1}
                max={100}
                value={questionLimit}
                disabled={selectedQuestionIds.length > 0}
                onChange={(event) => setQuestionLimit(Math.max(1, Math.min(100, Number(event.target.value) || 1)))}
                className="mt-2 w-28 rounded-xl border border-white/10 bg-slate-950 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-300 disabled:opacity-45"
              />
            </div>
            <div className="rounded-2xl border border-cyan-300/25 bg-cyan-300/10 p-4">
              <p className="text-sm font-semibold text-cyan-100">{runSize.toLocaleString()} model calls</p>
              <p className="mt-1 text-xs text-slate-400">models x alpha x retrieve_k x rerank_n x questions</p>
              {runSize > 100 && <p className="mt-3 text-xs text-amber-200">Large run. Reduce models or questions if API quota is tight.</p>}
              {selectedUnavailableModels.length > 0 && (
                <p className="mt-3 text-xs text-rose-200">
                  Setup needed: {selectedUnavailableModels.map((model) => model.display_name).join(", ")}
                </p>
              )}
            </div>
            <SecondaryButton onClick={onPreset} disabled={isBusy} icon={Sparkles}>Full comparison preset</SecondaryButton>
          </div>
        </div>
      </Panel>

      <QuestionPicker
        questions={evalQuestions}
        selectedQuestionIds={selectedQuestionIds}
        setSelectedQuestionIds={setSelectedQuestionIds}
        questionLimit={questionLimit}
        modelMap={modelMap}
      />
    </div>
  );
}

function QuestionPicker({
  questions,
  selectedQuestionIds,
  setSelectedQuestionIds,
  questionLimit
}: {
  questions: EvalQuestion[];
  selectedQuestionIds: string[];
  setSelectedQuestionIds: (ids: string[]) => void;
  questionLimit: number;
  modelMap: Record<string, ModelInfo>;
}) {
  const selectedCount = selectedQuestionIds.length || Math.min(questionLimit, questions.length);

  return (
    <Panel title="Question selection" subtitle="Choose the golden support questions to run. Explicit selections override the count.">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="text-sm text-slate-400">
          {selectedQuestionIds.length
            ? `${selectedQuestionIds.length} explicitly selected questions will run.`
            : `No explicit selection: the benchmark will use the first ${selectedCount} questions.`}
        </div>
        <div className="flex flex-wrap gap-2">
          <SmallButton onClick={() => setSelectedQuestionIds(questions.slice(0, 3).map((question) => question.id))}>First 3</SmallButton>
          <SmallButton onClick={() => setSelectedQuestionIds(questions.slice(0, 5).map((question) => question.id))}>First 5</SmallButton>
          <SmallButton onClick={() => setSelectedQuestionIds(questions.map((question) => question.id))}>All</SmallButton>
          <SmallButton onClick={() => setSelectedQuestionIds([])}>Use count</SmallButton>
        </div>
      </div>

      <div className="mt-5 max-h-[30rem] overflow-y-auto rounded-2xl border border-white/10 bg-[#050812]/70">
        {questions.length ? (
          <div className="divide-y divide-white/10">
            {questions.map((question) => {
              const checked = selectedQuestionIds.includes(question.id);
              return (
                <label key={question.id} className="flex cursor-pointer gap-3 px-4 py-4 text-sm transition hover:bg-white/[0.035]">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => setSelectedQuestionIds(toggleValue(selectedQuestionIds, question.id))}
                    className="mt-1 accent-cyan-300"
                  />
                  <span className="min-w-0">
                      <span className="font-semibold text-white">{question.id}</span>
                      <span className="ml-2 text-slate-300">{question.question}</span>
                      <span className="mt-2 flex flex-wrap gap-2 text-xs">
                      <span className="rounded-full bg-emerald-300/10 px-2 py-1 text-emerald-100">
                        {humanizeQuestionType(question.question_type)}
                      </span>
                      <span className="rounded-full bg-white/[0.06] px-2 py-1 text-slate-400">
                        {sourceListText(question.expected_sources, question.expected_doc, question.expected_section)}
                      </span>
                      {question.tags.map((tag) => (
                        <span key={tag} className="rounded-full bg-cyan-300/10 px-2 py-1 text-cyan-100">{tag}</span>
                      ))}
                    </span>
                  </span>
                </label>
              );
            })}
          </div>
        ) : (
          <EmptyInline>No question bank loaded yet.</EmptyInline>
        )}
      </div>
    </Panel>
  );
}

function TestCasesPanel({
  runDetails,
  questions,
  modelMap
}: {
  runDetails: RunDetails | null;
  questions: EvalQuestion[];
  modelMap: Record<string, ModelInfo>;
}) {
  const [statusFilter, setStatusFilter] = useState<TestStatusFilter>("all");
  const [modelFilter, setModelFilter] = useState("all");
  const [failureFilter, setFailureFilter] = useState("all");
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("all");
  const [questionTypeFilter, setQuestionTypeFilter] = useState<QuestionTypeFilter>("all");
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const questionMap = useMemo(() => Object.fromEntries(questions.map((question) => [question.id, question])), [questions]);
  const allRows = runDetails?.results ?? [];
  const models = Array.from(new Set(allRows.map((row) => row.model))).sort();
  const issues = Array.from(new Set(allRows.map((row) => rowIssueLabel(row)))).sort();
  const questionTypes = Array.from(new Set(allRows.map((row) => row.question_type || questionMap[row.question_id]?.question_type || "direct"))).sort();
  const rows = allRows.filter((row) => {
    const result = rowResultLabel(row);
    const expectedSources = expectedSourcesFor(row, questionMap[row.question_id]);
    const sourceKnown = expectedSources.length > 0;
    const sourceRecall = Number(row.source_recall ?? row.metrics?.source_recall ?? (row.expected_source_found ? 1 : 0));
    const sourceMissed = sourceKnown && sourceRecall < 1;
    const questionType = row.question_type || questionMap[row.question_id]?.question_type || "direct";
    return (
      (statusFilter === "all" || result === statusFilter) &&
      (modelFilter === "all" || row.model === modelFilter) &&
      (failureFilter === "all" || rowIssueLabel(row) === failureFilter) &&
      (sourceFilter === "all" || (sourceFilter === "matched" ? sourceKnown && sourceRecall >= 1 : sourceMissed)) &&
      (questionTypeFilter === "all" || questionType === questionTypeFilter)
    );
  });

  if (!runDetails) {
    return (
      <EmptyState
        icon={ClipboardList}
        title="No test case evidence yet"
        body="Run a benchmark from the Experiment tab. This view will show pass/fail, expected answers, generated answers, and retrieval evidence."
      />
    );
  }

  return (
    <div className="space-y-6">
      <Panel
        title="Test cases"
        subtitle={`Run ${runDetails.id} has ${runDetails.results.length} stored model/config/question test cases.`}
        action={<ExportButton href={runExportUrl(runDetails.id)}>Export CSV</ExportButton>}
      >
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          <FilterSelect label="Result" value={statusFilter} onChange={(value) => setStatusFilter(value as TestStatusFilter)} options={[["all", "All"], ["right", "Right"], ["partial", "Partial"], ["wrong", "Wrong"]]} />
          <FilterSelect label="Question type" value={questionTypeFilter} onChange={(value) => setQuestionTypeFilter(value)} options={[["all", "All types"], ...questionTypes.map((type) => [type, humanizeQuestionType(type)] as [string, string])]} />
          <FilterSelect label="Model" value={modelFilter} onChange={setModelFilter} options={[["all", "All models"], ...models.map((model) => [model, displayModel(model, modelMap)] as [string, string])]} />
          <FilterSelect label="Issue" value={failureFilter} onChange={setFailureFilter} options={[["all", "All issues"], ...issues.map((issue) => [issue, humanizeIssue(issue)] as [string, string])]} />
          <FilterSelect label="Source status" value={sourceFilter} onChange={(value) => setSourceFilter(value as SourceFilter)} options={[["all", "All"], ["matched", "Source found"], ["missed", "Source missing"]]} />
        </div>

        <div className="mt-5 overflow-hidden rounded-2xl border border-white/10 bg-[#050812]/70">
          <table className="w-full table-fixed text-left text-sm">
            <thead className="border-b border-white/10 bg-white/[0.025] text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="w-[110px] px-4 py-3">Result</th>
                <th className="px-4 py-3">Question</th>
                <th className="w-[220px] px-4 py-3">Model + RAG</th>
                <th className="w-[145px] px-4 py-3">Answer quality</th>
                <th className="w-[180px] px-4 py-3">Source evidence</th>
                <th className="w-[155px] px-4 py-3">Issue</th>
                <th className="w-[80px] px-4 py-3">Trace</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              {rows.map((row) => {
                const question = questionMap[row.question_id];
                const expanded = expandedId === row.id;
                const result = rowResultLabel(row);
                const citationScore = Number(row.metrics?.citation_accuracy ?? 0);
                const sourceRecall = Number(row.source_recall ?? row.metrics?.source_recall ?? (row.expected_source_found ? 1 : 0));
                const expectedSources = expectedSourcesFor(row, question);
                const questionType = row.question_type || question?.question_type || "direct";
                return (
                  <FragmentRow key={row.id}>
                    <tr className="align-top text-slate-200 hover:bg-white/[0.025]">
                      <td className="px-4 py-4">
                        <OutcomeBadge row={row} />
                      </td>
                      <td className="px-4 py-4">
                        <button onClick={() => setExpandedId(expanded ? null : row.id)} className="w-full text-left">
                          <span className="inline-flex rounded-full border border-white/10 bg-white/[0.04] px-2.5 py-1 text-xs text-slate-300">
                            {humanizeQuestionType(questionType)}
                          </span>
                          <span className="mt-2 block font-semibold leading-6 text-white">
                            {row.question ?? question?.question ?? "Question unavailable"}
                          </span>
                          <span className="mt-1 block text-xs text-slate-500">{row.question_id}</span>
                        </button>
                      </td>
                      <td className="px-4 py-4">
                        <div className="min-w-0">
                          <p className="truncate font-medium text-white" title={displayModel(row.model, modelMap)}>
                            <ModelDot model={row.model} />
                            {displayModel(row.model, modelMap)}
                          </p>
                          <p className="mt-1 text-xs text-slate-500">a={row.alpha}, k={row.retrieve_top_k}, n={row.rerank_top_n}</p>
                        </div>
                      </td>
                      <td className="px-4 py-4">
                        <AnswerCheckBadge label={row.judge_label || String(row.metrics?.judge_label ?? "not_run")} />
                        <p className="mt-2 font-semibold text-white">{percent(row.quality_score)}</p>
                      </td>
                      <td className="px-4 py-4">
                        <RetrievalBadge matched={sourceRecall >= 1} known={expectedSources.length > 0} />
                        {expectedSources.length > 0 && (
                          <p className="mt-1 text-xs text-slate-500">{Math.round(sourceRecall * 100)}% found</p>
                        )}
                        <CitationSupportBadge score={citationScore} shouldRefuse={row.should_refuse} />
                      </td>
                      <td className="px-4 py-4">
                        <span
                          className={
                            result === "right"
                              ? "text-slate-400"
                              : result === "partial"
                                ? "font-medium text-amber-100"
                                : "font-medium text-rose-100"
                          }
                        >
                          {humanizeIssue(rowIssueLabel(row))}
                        </span>
                      </td>
                      <td className="px-4 py-4">
                        <a className="font-medium text-cyan-200 hover:text-cyan-100" href={`/traces/${row.trace_id}?run_id=${runDetails.id}&return_view=tests`}>Open</a>
                      </td>
                    </tr>
                    {expanded && (
                      <tr>
                        <td colSpan={7} className="bg-cyan-300/[0.035] px-4 py-4">
                          <AnswerComparison row={row} question={question} />
                        </td>
                      </tr>
                    )}
                  </FragmentRow>
                );
              })}
            </tbody>
          </table>
          {!rows.length && <EmptyInline>No test cases match the current filters.</EmptyInline>}
        </div>
      </Panel>
    </div>
  );
}

function AnswerComparison({ row, question }: { row: RunResult; question?: EvalQuestion }) {
  const reference = row.reference_answer ?? question?.reference_answer ?? "Reference answer unavailable.";
  const generated = row.answer?.answer ?? "No generated answer.";
  const metrics = row.metrics ?? {};
  const retrieved = row.retrieved_chunks?.slice(0, 3) ?? [];
  const judgeLabel = row.judge_label || String(metrics.judge_label ?? "not_run");
  const judgeRationale = row.judge_rationale || String(metrics.judge_rationale ?? "");
  const missingFacts = (row.missing_facts ?? metrics.missing_facts ?? []) as string[];
  const contradictions = (row.contradictions ?? metrics.contradictions ?? []) as string[];
  const expectedSources = expectedSourcesFor(row, question);
  const retrievedExpectedSources = (metrics.retrieved_expected_sources ?? []) as Array<{ document: string; section: string }>;

  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_1fr_0.8fr]">
      <EvidenceCard title="Expected answer" icon={Target}>
        {reference}
      </EvidenceCard>
      <EvidenceCard title="Generated answer" icon={FileText}>
        {generated}
      </EvidenceCard>
      <div className="space-y-3">
        <EvidenceCard title="Answer quality check" icon={TestTube2}>
          <div className="flex flex-wrap items-center gap-2">
            <AnswerCheckBadge label={judgeLabel} />
          </div>
          <p className="mt-3 whitespace-pre-wrap break-words leading-6">{judgeRationale || "No answer-quality rationale was recorded for this result."}</p>
          {(missingFacts.length > 0 || contradictions.length > 0) && (
            <div className="mt-3 space-y-2 text-xs">
              {missingFacts.length > 0 && (
                <p className="text-amber-200">Missing facts: {missingFacts.join("; ")}</p>
              )}
              {contradictions.length > 0 && (
                <p className="text-rose-200">Contradictions: {contradictions.join("; ")}</p>
              )}
            </div>
          )}
        </EvidenceCard>
        <EvidenceCard title="Source evidence" icon={Layers3}>
          <SourceEvidenceList title="Required sources" sources={expectedSources} empty="No specific source was required." />
          <SourceEvidenceList title="Retrieved expected sources" sources={retrievedExpectedSources} empty="No required source was retrieved." />
          <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
            <MiniStat label="Source recall" value={percent(Number(row.source_recall ?? metrics.source_recall ?? 0))} />
            <MiniStat label="Citation matched" value={Boolean(row.citation_matched ?? metrics.citation_matched) ? "Yes" : "No"} />
          </div>
        </EvidenceCard>
        <EvidenceCard title="Why this failed or passed" icon={Info}>
          <p>{failureExplanation(rowIssueLabel(row))}</p>
        </EvidenceCard>
        <EvidenceCard title="Top retrieved chunks" icon={Layers3}>
          <div className="space-y-2">
            {retrieved.map((chunk, index) => (
              <div key={chunk.chunk_id ?? index} className="rounded-xl border border-white/10 bg-white/[0.035] p-2">
                <p className="text-xs font-semibold text-white">{index + 1}. {sourceLabel(chunk.document, chunk.section)}</p>
                <p className="mt-1 text-xs leading-5 text-slate-400">{String(chunk.text ?? "").slice(0, 220)}</p>
              </div>
            ))}
          </div>
        </EvidenceCard>
      </div>
    </div>
  );
}

function HistoryPanel({
  history,
  modelMap
}: {
  history: DashboardHistory;
  modelMap: Record<string, ModelInfo>;
}) {
  const [modelSort, setModelSort] = useState<HistorySort>({ key: "avg_quality", direction: "desc" });
  const [configSort, setConfigSort] = useState<HistorySort>({ key: "avg_quality", direction: "desc" });
  const [modelFilter, setModelFilter] = useState("");
  const [providerFilter, setProviderFilter] = useState("all");
  const [configModelFilter, setConfigModelFilter] = useState("all");
  const [alphaFilter, setAlphaFilter] = useState("all");
  const [retrieveKFilter, setRetrieveKFilter] = useState("all");
  const [rerankNFilter, setRerankNFilter] = useState("all");
  const providers = Array.from(new Set(history.model_performance.map((row) => row.provider))).sort();
  const configModels = Array.from(new Set(history.config_performance.map((row) => row.model))).sort();
  const alphaValues = Array.from(new Set(history.config_performance.map((row) => row.alpha))).sort();
  const retrieveKValues = Array.from(new Set(history.config_performance.map((row) => row.retrieve_top_k))).sort((a, b) => a - b);
  const rerankNValues = Array.from(new Set(history.config_performance.map((row) => row.rerank_top_n))).sort((a, b) => a - b);
  const chartData = history.runs.map((run) => ({
    ...run,
    label: `Run ${run.run_id}`,
    best_model_label: displayModel(run.best_model, modelMap)
  }));
  const modelRows = sortHistoryRows(
    history.model_performance.filter((row) => {
      const query = modelFilter.trim().toLowerCase();
      const label = displayModel(row.model, modelMap).toLowerCase();
      return (
        (!query || label.includes(query) || row.model.toLowerCase().includes(query)) &&
        (providerFilter === "all" || row.provider === providerFilter)
      );
    }),
    modelSort
  );
  const configRows = sortHistoryRows(
    history.config_performance.filter((row) => (
      (configModelFilter === "all" || row.model === configModelFilter) &&
      (alphaFilter === "all" || row.alpha === Number(alphaFilter)) &&
      (retrieveKFilter === "all" || row.retrieve_top_k === Number(retrieveKFilter)) &&
      (rerankNFilter === "all" || row.rerank_top_n === Number(rerankNFilter))
    )),
    configSort
  );

  return (
    <div className="space-y-6">
      <Panel title="Average model performance" subtitle="Aggregated across completed runs. This answers which model is consistently reliable.">
        <div className="grid gap-3 md:grid-cols-[1fr_220px]">
          <input
            value={modelFilter}
            onChange={(event) => setModelFilter(event.target.value)}
            placeholder="Filter model"
            className="rounded-xl border border-white/10 bg-slate-950 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-300"
          />
          <FilterSelect
            label="Provider"
            value={providerFilter}
            onChange={setProviderFilter}
            options={[["all", "All providers"], ...providers.map((provider) => [provider, provider] as [string, string])]}
          />
        </div>
        <div className="mt-5">
          {modelRows.length ? (
            <HistoryModelTable rows={modelRows} modelMap={modelMap} sort={modelSort} setSort={setModelSort} />
          ) : (
            <EmptyInline>No model rows match the current filters.</EmptyInline>
          )}
        </div>
      </Panel>

      <Panel title="Average RAG config performance" subtitle="Aggregated by model + alpha + retrieve_k + rerank_n.">
        <div className="grid gap-3 md:grid-cols-4">
          <FilterSelect label="Model" value={configModelFilter} onChange={setConfigModelFilter} options={[["all", "All models"], ...configModels.map((model) => [model, displayModel(model, modelMap)] as [string, string])]} />
          <FilterSelect label="Alpha" value={alphaFilter} onChange={setAlphaFilter} options={[["all", "All alpha"], ...alphaValues.map((alpha) => [String(alpha), `alpha ${alpha}`] as [string, string])]} />
          <FilterSelect label="K" value={retrieveKFilter} onChange={setRetrieveKFilter} options={[["all", "All k"], ...retrieveKValues.map((value) => [String(value), `k ${value}`] as [string, string])]} />
          <FilterSelect label="N" value={rerankNFilter} onChange={setRerankNFilter} options={[["all", "All n"], ...rerankNValues.map((value) => [String(value), `n ${value}`] as [string, string])]} />
        </div>
        <div className="mt-5">
          {configRows.length ? (
            <HistoryConfigTable rows={configRows} modelMap={modelMap} sort={configSort} setSort={setConfigSort} />
          ) : (
            <EmptyInline>No config rows match the current filters.</EmptyInline>
          )}
        </div>
      </Panel>

      <Panel title="Run log" subtitle="Audit trail of completed benchmark runs.">
        {chartData.length ? (
          <>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 12, right: 20, bottom: 8, left: 0 }}>
                  <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                  <XAxis dataKey="label" tick={{ fill: "#94a3b8", fontSize: 12 }} stroke="#475569" />
                  <YAxis
                    yAxisId="score"
                    domain={[0, 1]}
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`}
                    stroke="#475569"
                  />
                  <YAxis yAxisId="latency" orientation="right" tick={{ fill: "#94a3b8", fontSize: 12 }} stroke="#475569" />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend wrapperStyle={{ color: "#cbd5e1", fontSize: 12 }} />
                  <Line yAxisId="score" type="monotone" dataKey="best_quality" name="Best quality" stroke="#34d399" strokeWidth={2} dot={{ r: 3 }} />
                  <Line yAxisId="score" type="monotone" dataKey="failure_rate" name="Failure rate" stroke="#fb7185" strokeWidth={2} dot={{ r: 3 }} />
                  <Line yAxisId="latency" type="monotone" dataKey="avg_latency_ms" name="Latency ms" stroke="#38bdf8" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <RunLogTable rows={chartData.slice().reverse()} />
          </>
        ) : (
          <EmptyInline>No completed historical runs yet.</EmptyInline>
        )}
      </Panel>
    </div>
  );
}

function MethodologyPanel({ retrievalStatus }: { retrievalStatus: RetrievalStatus | null }) {
  return (
    <div className="space-y-6">
      <Panel title="Evaluation methodology" subtitle="Transparent scoring turns model outputs into auditable benchmark results.">
        <div className="grid gap-4 lg:grid-cols-2">
          <MethodCard title="Retrieval" icon={Layers3}>
            Same support docs, chunking strategy, Pinecone dense retrieval, BM25 sparse retrieval, hybrid alpha fusion,
            and lexical reranker are used across models. Only selected model and RAG parameters change per run.
          </MethodCard>
          <MethodCard title="Generation" icon={FileText}>
            Each model receives the same user question and retrieved context. Answers include citations and token/cost
            accounting where available.
          </MethodCard>
          <MethodCard title="Evaluation" icon={TestTube2}>
            Result is the overall gate across source evidence, citation support, refusal behavior, grounding, and answer
            quality. Answer quality is checked for factual equivalence, so matching wording is not required.
          </MethodCard>
          <MethodCard title="Decision" icon={Target}>
            The Decision tab combines quality, cost, latency, and failure behavior to recommend the config most suitable
            for production rollout.
          </MethodCard>
        </div>
      </Panel>

      <Panel title="Metric glossary" subtitle="How each dashboard metric is calculated.">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <GlossaryItem title="Answer correctness" value="Does the answer preserve the same facts as the golden answer, even if phrased differently?" />
          <GlossaryItem title="Answer quality" value="The factual-equivalence subcheck. It can pass while the overall result fails because source evidence or citations failed." />
          <GlossaryItem title="Source recall" value="For multi-source questions, the share of required sources found in retrieved chunks." />
          <GlossaryItem title="Groundedness" value="Is the answer supported by retrieved chunks?" />
          <GlossaryItem title="Citation accuracy" value="Do citations point to the expected document and section?" />
          <GlossaryItem title="Refusal correctness" value="Does the model refuse questions it should not answer?" />
          <GlossaryItem title="Quality score" value="Weighted benchmark score from the evaluation sub-metrics." />
          <GlossaryItem title="Cost/query" value="Estimated average model cost per answer." />
          <GlossaryItem title="Latency" value="Average end-to-end generation latency." />
          <GlossaryItem title="Failure rate" value="Share of cases that did not pass the evaluator." />
        </div>
      </Panel>

      <Panel title="Failure taxonomy" subtitle="Failure types make debugging actionable.">
        <div className="grid gap-3 md:grid-cols-2">
          <GlossaryItem title="Wrong answer" value="The generated answer misses or contradicts required facts from the reference answer." />
          <GlossaryItem title="Bad retrieval" value="The expected source was not retrieved, so generation started from weak evidence." />
          <GlossaryItem title="Ungrounded answer" value="The answer may be plausible but is not supported by retrieved context." />
          <GlossaryItem title="Missing citation" value="The answer may be correct, but it did not cite the required support source." />
          <GlossaryItem title="Wrong citation" value="The answer may be correct, but the citation points to an unrelated or unsupported source." />
          <GlossaryItem title="Partial citation" value="A cross-document answer cited only some of the required sources." />
          <GlossaryItem title="Bad refusal" value="The model refused when it should answer, or answered when it should refuse." />
        </div>
      </Panel>

      <Panel title="Mixed answer-quality strategy" subtitle="Use deterministic checks where exact facts are machine-verifiable, and LLM checks where wording changes.">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          <GlossaryItem title="Direct" value="Deterministic first. LLM only handles borderline wording." />
          <GlossaryItem title="Paraphrase" value="LLM factual-equivalence check because wording intentionally changes." />
          <GlossaryItem title="Calculation" value="Numeric/date facts are deterministic gates, with LLM rationale for the final answer." />
          <GlossaryItem title="Missing data" value="Deterministic refusal check. The model should not invent unsupported facts." />
          <GlossaryItem title="Cross-document" value="Requires all expected sources plus answer-quality checking across documents." />
        </div>
      </Panel>

      <Panel title="Current retrieval setup" subtitle="Runtime configuration for the active retrieval pipeline.">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <GlossaryItem title="Dense retrieval" value={retrievalStatus?.dense_retrieval ?? "unknown"} />
          <GlossaryItem title="Sparse retrieval" value={retrievalStatus?.bm25_retrieval ?? "unknown"} />
          <GlossaryItem title="Reranker" value={retrievalStatus?.reranker ?? "unknown"} />
          <GlossaryItem title="Embedding mode" value={retrievalStatus?.pinecone_embedding_mode ?? retrievalStatus?.embedding_model ?? "unknown"} />
          <GlossaryItem title="Active dataset" value={retrievalStatus?.dataset?.active_dataset ?? "unknown"} />
          <GlossaryItem title="Golden questions" value={`${retrievalStatus?.dataset?.eval_question_count ?? 0}`} />
        </div>
      </Panel>
    </div>
  );
}

function ProgressPanel({ progress }: { progress: JobProgress }) {
  const hasItemProgress = progress.total_items > 0;
  const ratio = hasItemProgress
    ? progress.completed_items / Math.max(1, progress.total_items)
    : progress.step_index / Math.max(1, progress.steps.length - 1);
  const width = `${Math.min(100, Math.max(0, Math.round(ratio * 100)))}%`;
  const isFailed = progress.status === "failed";
  const isComplete = progress.status === "completed";

  return (
    <section className="mb-5 rounded-2xl border border-cyan-300/20 bg-cyan-300/10 p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-sm font-semibold text-cyan-100">
            {progress.action === "ingest" ? "Ingestion progress" : "Benchmark progress"}
          </p>
          <p className="mt-1 text-sm text-slate-300">{isFailed ? progress.error : progress.detail}</p>
        </div>
        <StatusBadge tone={isFailed ? "rose" : isComplete ? "emerald" : "cyan"}>{progress.status}</StatusBadge>
      </div>
      <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-950">
        <div className={`h-full rounded-full ${isFailed ? "bg-rose-400" : "bg-cyan-300"}`} style={{ width }} />
      </div>
      <div className="mt-4 grid gap-2 text-xs text-slate-400 md:grid-cols-2 xl:grid-cols-5">
        {progress.steps.map((step, index) => (
          <div
            key={step}
            className={`rounded-xl border px-3 py-2 ${
              index <= progress.step_index ? "border-cyan-300/25 bg-cyan-300/10 text-cyan-100" : "border-white/10 bg-white/[0.025]"
            }`}
          >
            {index + 1}. {step}
          </div>
        ))}
      </div>
      {hasItemProgress && (
        <p className="mt-3 text-xs text-slate-300">
          {progress.completed_items.toLocaleString()} / {progress.total_items.toLocaleString()} test cases completed
        </p>
      )}
    </section>
  );
}

function QualityCostChart({ points, modelMap }: { points: ConfigLeaderboardRow[]; modelMap: Record<string, ModelInfo> }) {
  if (!points.length) {
    return <EmptyInline>No quality/cost points recorded.</EmptyInline>;
  }
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 22, right: 26, bottom: 34, left: 18 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="cost_per_query"
            name="Cost"
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            tickFormatter={(value) => money(Number(value))}
            stroke="#475569"
          />
          <YAxis
            type="number"
            dataKey="quality_score"
            name="Quality"
            domain={[0, 1]}
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`}
            stroke="#475569"
          />
          <Tooltip content={<QualityTooltip modelMap={modelMap} />} cursor={{ strokeDasharray: "3 3" }} />
          <Scatter data={points}>
            {points.map((row, index) => (
              <Cell key={`${row.model}-${row.alpha}-${index}`} fill={modelColors[row.model] ?? "#94a3b8"} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function FailureChart({ data, categories }: { data: Record<string, number | string>[]; categories: string[] }) {
  if (!data.length) {
    return <EmptyInline>No failures recorded for this run.</EmptyInline>;
  }
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 34, right: 12, bottom: 70, left: 0 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
          <XAxis
            dataKey="model"
            interval={0}
            angle={-20}
            textAnchor="end"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            stroke="#475569"
          />
          <YAxis allowDecimals={false} tick={{ fill: "#94a3b8", fontSize: 12 }} stroke="#475569" />
          <Tooltip contentStyle={tooltipStyle} />
          <Legend verticalAlign="top" align="center" wrapperStyle={{ color: "#cbd5e1", fontSize: 12, paddingBottom: 12 }} />
          {categories.map((category, index) => (
            <Bar key={category} dataKey={category} stackId="failures" fill={failureColors[index % failureColors.length]} radius={[4, 4, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ModelLeaderboard({ rows, modelMap }: { rows: ModelLeaderboardRow[]; modelMap: Record<string, ModelInfo> }) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-white/10 bg-[#050812]/70">
      <table className="min-w-full text-left text-sm">
        <thead className="border-b border-white/10 bg-white/[0.025] text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-4 py-3">Rank</th>
            <th className="px-4 py-3">Model</th>
            <th className="px-4 py-3">Provider</th>
            <th className="px-4 py-3">Quality</th>
            <th className="px-4 py-3">Cost</th>
            <th className="px-4 py-3">Latency</th>
            <th className="px-4 py-3">Failures</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/10">
          {rows.map((row, index) => (
            <tr key={row.model} className="text-slate-200">
              <td className="px-4 py-3 font-semibold text-white">{index + 1}</td>
              <td className="px-4 py-3 whitespace-nowrap"><ModelDot model={row.model} />{displayModel(row.model, modelMap)}</td>
              <td className="px-4 py-3 text-slate-400">{modelMap[row.model]?.provider ?? "Unknown"}</td>
              <td className="px-4 py-3 font-semibold text-white">{percent(row.quality_score)}</td>
              <td className="px-4 py-3">{money(row.cost_per_query)}</td>
              <td className="px-4 py-3">{ms(row.avg_latency_ms)}</td>
              <td className="px-4 py-3">{percent(row.failure_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ConfigLeaderboard({ rows, modelMap }: { rows: ConfigLeaderboardRow[]; modelMap: Record<string, ModelInfo> }) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-white/10 bg-[#050812]/70">
      <table className="min-w-full text-left text-sm">
        <thead className="border-b border-white/10 bg-white/[0.025] text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-4 py-3">Rank</th>
            <th className="px-4 py-3">Model</th>
            <th className="px-4 py-3">Alpha</th>
            <th className="px-4 py-3">K/N</th>
            <th className="px-4 py-3">Quality</th>
            <th className="px-4 py-3">Cost</th>
            <th className="px-4 py-3">Latency</th>
            <th className="px-4 py-3">Failures</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/10">
          {rows.slice(0, 15).map((row, index) => (
            <tr key={`${row.model}-${row.alpha}-${row.retrieve_top_k}-${row.rerank_top_n}`} className="text-slate-200">
              <td className="px-4 py-3 font-semibold text-white">{index + 1}</td>
              <td className="px-4 py-3 whitespace-nowrap"><ModelDot model={row.model} />{displayModel(row.model, modelMap)}</td>
              <td className="px-4 py-3">{row.alpha}</td>
              <td className="px-4 py-3">{row.retrieve_top_k}/{row.rerank_top_n}</td>
              <td className="px-4 py-3 font-semibold text-white">{percent(row.quality_score)}</td>
              <td className="px-4 py-3">{money(row.cost_per_query)}</td>
              <td className="px-4 py-3">{ms(row.avg_latency_ms)}</td>
              <td className="px-4 py-3">{percent(row.failure_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

type HistorySortKey =
  | "model"
  | "provider"
  | "run_count"
  | "result_count"
  | "avg_quality"
  | "avg_cost"
  | "avg_latency"
  | "failure_rate"
  | "answer_correctness"
  | "groundedness"
  | "citation_accuracy"
  | "refusal_correctness"
  | "alpha"
  | "retrieve_top_k"
  | "rerank_top_n";

type HistorySort = {
  key: HistorySortKey;
  direction: "asc" | "desc";
};

function HistoryModelTable({
  rows,
  modelMap,
  sort,
  setSort
}: {
  rows: HistoryModelPerformanceRow[];
  modelMap: Record<string, ModelInfo>;
  sort: HistorySort;
  setSort: (sort: HistorySort) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-white/10 bg-[#050812]/70">
      <table className="min-w-full text-left text-sm">
        <thead className="border-b border-white/10 bg-white/[0.025] text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <HistorySortHeader label="Model" sortKey="model" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Provider" sortKey="provider" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Runs" sortKey="run_count" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Results" sortKey="result_count" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Quality" sortKey="avg_quality" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Cost" sortKey="avg_cost" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Latency" sortKey="avg_latency" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Failures" sortKey="failure_rate" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Grounded" sortKey="groundedness" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Citations" sortKey="citation_accuracy" sort={sort} setSort={setSort} />
          </tr>
        </thead>
        <tbody className="divide-y divide-white/10">
          {rows.map((row) => (
            <tr key={row.model} className="text-slate-200">
              <td className="max-w-[240px] px-4 py-3"><span className="block truncate font-medium text-white" title={displayModel(row.model, modelMap)}><ModelDot model={row.model} />{displayModel(row.model, modelMap)}</span></td>
              <td className="px-4 py-3 text-slate-400">{row.provider}</td>
              <td className="px-4 py-3">{row.run_count}</td>
              <td className="px-4 py-3">{row.result_count}</td>
              <td className="px-4 py-3 font-semibold text-white">{percent(row.avg_quality)}</td>
              <td className="px-4 py-3">{money(row.avg_cost)}</td>
              <td className="px-4 py-3">{ms(row.avg_latency)}</td>
              <td className="px-4 py-3">{percent(row.failure_rate)}</td>
              <td className="px-4 py-3">{percent(row.groundedness)}</td>
              <td className="px-4 py-3">{percent(row.citation_accuracy)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HistoryConfigTable({
  rows,
  modelMap,
  sort,
  setSort
}: {
  rows: HistoryConfigPerformanceRow[];
  modelMap: Record<string, ModelInfo>;
  sort: HistorySort;
  setSort: (sort: HistorySort) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-white/10 bg-[#050812]/70">
      <table className="min-w-full text-left text-sm">
        <thead className="border-b border-white/10 bg-white/[0.025] text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <HistorySortHeader label="Model" sortKey="model" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Alpha" sortKey="alpha" sort={sort} setSort={setSort} />
            <HistorySortHeader label="K" sortKey="retrieve_top_k" sort={sort} setSort={setSort} />
            <HistorySortHeader label="N" sortKey="rerank_top_n" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Results" sortKey="result_count" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Quality" sortKey="avg_quality" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Cost" sortKey="avg_cost" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Latency" sortKey="avg_latency" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Failures" sortKey="failure_rate" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Answer" sortKey="answer_correctness" sort={sort} setSort={setSort} />
            <HistorySortHeader label="Grounded" sortKey="groundedness" sort={sort} setSort={setSort} />
          </tr>
        </thead>
        <tbody className="divide-y divide-white/10">
          {rows.map((row) => (
            <tr key={`${row.model}-${row.alpha}-${row.retrieve_top_k}-${row.rerank_top_n}`} className="text-slate-200">
              <td className="max-w-[240px] px-4 py-3"><span className="block truncate font-medium text-white" title={displayModel(row.model, modelMap)}><ModelDot model={row.model} />{displayModel(row.model, modelMap)}</span></td>
              <td className="px-4 py-3">{row.alpha}</td>
              <td className="px-4 py-3">{row.retrieve_top_k}</td>
              <td className="px-4 py-3">{row.rerank_top_n}</td>
              <td className="px-4 py-3">{row.result_count}</td>
              <td className="px-4 py-3 font-semibold text-white">{percent(row.avg_quality)}</td>
              <td className="px-4 py-3">{money(row.avg_cost)}</td>
              <td className="px-4 py-3">{ms(row.avg_latency)}</td>
              <td className="px-4 py-3">{percent(row.failure_rate)}</td>
              <td className="px-4 py-3">{percent(row.answer_correctness)}</td>
              <td className="px-4 py-3">{percent(row.groundedness)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HistorySortHeader({
  label,
  sortKey,
  sort,
  setSort
}: {
  label: string;
  sortKey: HistorySortKey;
  sort: HistorySort;
  setSort: (sort: HistorySort) => void;
}) {
  const active = sort.key === sortKey;
  return (
    <th className="px-4 py-3">
      <button
        onClick={() => setSort(nextHistorySort(sort, sortKey))}
        className={`whitespace-nowrap text-left font-semibold uppercase ${active ? "text-cyan-200" : "text-slate-500 hover:text-slate-200"}`}
      >
        {label}{active ? (sort.direction === "asc" ? " up" : " down") : ""}
      </button>
    </th>
  );
}

function RunLogTable({ rows }: { rows: Array<any> }) {
  return (
    <div className="mt-5 overflow-x-auto rounded-2xl border border-white/10 bg-[#050812]/70">
      <table className="min-w-full text-left text-sm">
        <thead className="border-b border-white/10 bg-white/[0.025] text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-4 py-3">Run</th>
            <th className="px-4 py-3">Best model</th>
            <th className="px-4 py-3">Best quality</th>
            <th className="px-4 py-3">Cost/query</th>
            <th className="px-4 py-3">Latency</th>
            <th className="px-4 py-3">Failures</th>
            <th className="px-4 py-3">Results</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/10">
          {rows.map((run) => (
            <tr key={run.run_id} className="text-slate-200">
              <td className="px-4 py-3 font-semibold text-white">Run {run.run_id}</td>
              <td className="px-4 py-3">{run.best_model_label}</td>
              <td className="px-4 py-3">{percent(run.best_quality)}</td>
              <td className="px-4 py-3">{money(run.cost_per_query)}</td>
              <td className="px-4 py-3">{ms(run.avg_latency_ms)}</td>
              <td className="px-4 py-3">{percent(run.failure_rate)}</td>
              <td className="px-4 py-3">{run.result_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DecisionProofStrip({ runDetails, modelMap }: { runDetails: RunDetails; modelMap: Record<string, ModelInfo> }) {
  const failures = runDetails.results.filter((row) => rowResultLabel(row) !== "right").slice(0, 3);
  const passes = runDetails.results.filter((row) => rowResultLabel(row) === "right").slice(0, 3);
  const examples = [...passes, ...failures].slice(0, 4);
  if (!examples.length) {
    return null;
  }
  return (
    <Panel title="Proof samples" subtitle="Sample evidence behind the active recommendation.">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {examples.map((row) => (
          <div key={row.id} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
            <div className="flex items-center justify-between gap-2">
              <StatusBadge tone={rowResultLabel(row) === "right" ? "emerald" : rowResultLabel(row) === "partial" ? "amber" : "rose"}>
                {humanizeResult(rowResultLabel(row))}
              </StatusBadge>
              <span className="text-xs text-slate-500">{percent(row.quality_score)}</span>
            </div>
            <p className="mt-3 text-sm font-medium text-white">{row.question_id}</p>
            <p className="mt-1 line-clamp-3 text-xs leading-5 text-slate-400">{row.question}</p>
            <p className="mt-3 text-xs text-slate-500">
              <ModelDot model={row.model} />
              {displayModel(row.model, modelMap)}
            </p>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function Panel({
  title,
  subtitle,
  action,
  children
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="rounded-3xl border border-white/10 bg-[#0a0f1d]/82 p-5 shadow-2xl shadow-black/25 backdrop-blur-xl md:p-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-white">{title}</h2>
          {subtitle && <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-400">{subtitle}</p>}
        </div>
        {action}
      </div>
      <div className="mt-5">{children}</div>
    </section>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  subvalue,
  tone
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  subvalue: string;
  tone: "cyan" | "emerald" | "amber" | "violet" | "rose";
}) {
  const tones = {
    cyan: "from-cyan-300/18 text-cyan-100",
    emerald: "from-emerald-300/18 text-emerald-100",
    amber: "from-amber-300/18 text-amber-100",
    violet: "from-violet-300/18 text-violet-100",
    rose: "from-rose-300/18 text-rose-100"
  };
  return (
    <div className={`rounded-3xl border border-white/10 bg-gradient-to-br ${tones[tone]} to-white/[0.025] p-4 shadow-xl shadow-black/20`}>
      <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-slate-400">
        <Icon className="h-4 w-4" />
        {label}
      </div>
      <p className="mt-3 min-h-12 text-xl font-semibold leading-tight text-white">{value}</p>
      <p className="mt-2 text-xs text-slate-400">{subvalue}</p>
    </div>
  );
}

function DecisionNote({ title, icon: Icon, children }: { title: string; icon: LucideIcon; children: ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
      <div className="flex items-center gap-2 text-sm font-semibold text-white">
        <Icon className="h-4 w-4 text-cyan-200" />
        {title}
      </div>
      <div className="mt-2 text-sm leading-6 text-slate-400">{children}</div>
    </div>
  );
}

function MethodCard({ title, icon: Icon, children }: { title: string; icon: LucideIcon; children: ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-5">
      <div className="flex items-center gap-3">
        <span className="grid h-9 w-9 place-items-center rounded-xl bg-cyan-300/10 text-cyan-100">
          <Icon className="h-4 w-4" />
        </span>
        <h3 className="font-semibold text-white">{title}</h3>
      </div>
      <p className="mt-3 text-sm leading-6 text-slate-400">{children}</p>
    </div>
  );
}

function EvidenceCard({ title, icon: Icon, children }: { title: string; icon: LucideIcon; children: ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-[#070b13]/90 p-4">
      <div className="flex items-center gap-2 text-sm font-semibold text-white">
        <Icon className="h-4 w-4 text-cyan-200" />
        {title}
      </div>
      <div className="mt-3 text-sm leading-6 text-slate-300">{children}</div>
    </div>
  );
}

function SourceEvidenceList({
  title,
  sources,
  empty
}: {
  title: string;
  sources: Array<{ document: string; section: string }>;
  empty: string;
}) {
  return (
    <div className="mt-3 first:mt-0">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</p>
      {sources.length ? (
        <div className="mt-2 space-y-1.5">
          {sources.map((source) => (
            <div
              key={`${source.document}-${source.section}`}
              className="break-words rounded-xl border border-white/10 bg-white/[0.035] px-3 py-2 text-xs leading-5 text-slate-300"
            >
              {source.document} / {source.section}
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-2 text-xs text-slate-500">{empty}</p>
      )}
    </div>
  );
}

function GlossaryItem({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
      <p className="text-sm font-semibold text-white">{title}</p>
      <p className="mt-2 text-sm leading-6 text-slate-400">{value}</p>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.045] p-3">
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 font-semibold text-white">{value}</p>
    </div>
  );
}

function StatusBadge({ tone, children }: { tone: "emerald" | "rose" | "cyan" | "amber"; children: ReactNode }) {
  const classes = {
    emerald: "border-emerald-300/25 bg-emerald-300/10 text-emerald-100",
    rose: "border-rose-300/25 bg-rose-300/10 text-rose-100",
    cyan: "border-cyan-300/25 bg-cyan-300/10 text-cyan-100",
    amber: "border-amber-300/25 bg-amber-300/10 text-amber-100"
  };
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${classes[tone]}`}>
      {children}
    </span>
  );
}

function AnswerCheckBadge({ label }: { label: string }) {
  if (label === "correct") {
    return <StatusBadge tone="emerald">Correct</StatusBadge>;
  }
  if (label === "partial") {
    return <StatusBadge tone="amber">Partial</StatusBadge>;
  }
  if (label === "incorrect") {
    return <StatusBadge tone="rose">Incorrect</StatusBadge>;
  }
  if (label === "judge_unavailable") {
    return <StatusBadge tone="amber">Check unavailable</StatusBadge>;
  }
  return <StatusBadge tone="cyan">{humanizeAnswerCheck(label || "not_run")}</StatusBadge>;
}

function OutcomeBadge({ row }: { row: RunResult }) {
  const result = rowResultLabel(row);
  if (result === "right") {
    return <StatusBadge tone="emerald">Right</StatusBadge>;
  }
  if (result === "partial") {
    return <StatusBadge tone="amber">Partial</StatusBadge>;
  }
  return <StatusBadge tone="rose">Wrong</StatusBadge>;
}

function PrimaryButton({
  onClick,
  disabled,
  icon: Icon,
  children
}: {
  onClick: () => void;
  disabled?: boolean;
  icon: LucideIcon;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-300 to-emerald-300 px-4 py-2 text-sm font-semibold text-slate-950 shadow-lg shadow-cyan-950/30 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45"
    >
      <Icon className="h-4 w-4" />
      {children}
    </button>
  );
}

function SecondaryButton({
  onClick,
  disabled,
  icon: Icon,
  children
}: {
  onClick: () => void;
  disabled?: boolean;
  icon: LucideIcon;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2 text-sm font-semibold text-slate-100 transition hover:border-cyan-300/40 hover:bg-cyan-300/10 disabled:cursor-not-allowed disabled:opacity-45"
    >
      <Icon className="h-4 w-4" />
      {children}
    </button>
  );
}

function ExportButton({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      href={href}
      download
      className="inline-flex items-center justify-center gap-2 rounded-xl border border-cyan-300/25 bg-cyan-300/10 px-4 py-2 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-300/15"
    >
      <Download className="h-4 w-4" />
      {children}
    </a>
  );
}

function SmallButton({ onClick, children }: { onClick: () => void; children: ReactNode }) {
  return (
    <button
      onClick={onClick}
      className="rounded-xl border border-white/10 bg-white/[0.035] px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-cyan-300/40 hover:bg-cyan-300/10"
    >
      {children}
    </button>
  );
}

function ControlLabel({ children }: { children: ReactNode }) {
  return <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{children}</p>;
}

function ChipGroup({
  values,
  selected,
  setSelected
}: {
  values: number[];
  selected: number[];
  setSelected: (values: number[]) => void;
}) {
  return (
    <div className="mt-2 flex flex-wrap gap-2">
      {values.map((value) => (
        <button
          key={value}
          onClick={() => setSelected(toggleValue(selected, value))}
          className={`rounded-xl border px-3 py-1.5 text-sm font-medium transition ${
            selected.includes(value)
              ? "border-cyan-300/45 bg-cyan-300/10 text-cyan-100"
              : "border-white/10 bg-white/[0.035] text-slate-300 hover:border-white/20"
          }`}
        >
          {value}
        </button>
      ))}
    </div>
  );
}

function FilterSelect({
  label,
  value,
  onChange,
  options
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<[string, string]>;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-xl border border-white/10 bg-slate-950 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-300"
      >
        {options.map(([optionValue, optionLabel]) => (
          <option key={optionValue} value={optionValue}>{optionLabel}</option>
        ))}
      </select>
    </label>
  );
}

function RetrievalBadge({ matched, known }: { matched: boolean; known: boolean }) {
  if (!known) {
    return <StatusBadge tone="amber">No expected source</StatusBadge>;
  }
  return <StatusBadge tone={matched ? "emerald" : "rose"}>{matched ? "Source found" : "Source missing"}</StatusBadge>;
}

function CitationSupportBadge({ score, shouldRefuse }: { score: number; shouldRefuse: boolean }) {
  if (shouldRefuse) {
    return (
      <StatusBadge tone={score >= 1 ? "emerald" : "rose"}>
        {score >= 1 ? "No citation expected" : "Unexpected citation"}
      </StatusBadge>
    );
  }
  if (score >= 1) {
    return <StatusBadge tone="emerald">Citation OK</StatusBadge>;
  }
  if (score > 0) {
    return <StatusBadge tone="amber">Partial citation</StatusBadge>;
  }
  return <StatusBadge tone="rose">Citation missing</StatusBadge>;
}

function EmptyState({ icon: Icon, title, body }: { icon: LucideIcon; title: string; body: string }) {
  return (
    <div className="rounded-3xl border border-dashed border-white/15 bg-white/[0.025] p-10 text-center">
      <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-cyan-300/10 text-cyan-100">
        <Icon className="h-6 w-6" />
      </div>
      <h2 className="mt-4 text-xl font-semibold text-white">{title}</h2>
      <p className="mx-auto mt-2 max-w-2xl text-sm leading-6 text-slate-400">{body}</p>
    </div>
  );
}

function EmptyInline({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-2xl border border-dashed border-white/15 bg-white/[0.025] p-6 text-sm text-slate-400">
      {children}
    </div>
  );
}

function QualityTooltip({ active, payload, modelMap }: any) {
  if (!active || !payload?.length) {
    return null;
  }
  const row = payload[0].payload as ConfigLeaderboardRow;
  return (
    <div className="rounded-2xl border border-white/10 bg-slate-950 p-3 text-xs text-slate-200 shadow-xl">
      <p className="font-semibold text-white">{displayModel(row.model, modelMap)}</p>
      <p className="text-slate-400">{modelMap[row.model]?.provider ?? "Provider"} / alpha {row.alpha}, k {row.retrieve_top_k}, n {row.rerank_top_n}</p>
      <p className="mt-2">Quality: {percent(row.quality_score)}</p>
      <p>Cost/query: {money(row.cost_per_query)}</p>
      <p>Latency: {ms(row.avg_latency_ms)}</p>
      <p>Failures: {percent(row.failure_rate)}</p>
    </div>
  );
}

function ModelDot({ model }: { model: string }) {
  return (
    <span
      className="mr-2 inline-block h-2.5 w-2.5 rounded-full align-middle"
      style={{ backgroundColor: modelColors[model] ?? "#94a3b8" }}
    />
  );
}

function FragmentRow({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

function nextHistorySort(current: HistorySort, key: HistorySortKey): HistorySort {
  if (current.key !== key) {
    return { key, direction: defaultSortDirection(key) };
  }
  return { key, direction: current.direction === "asc" ? "desc" : "asc" };
}

function defaultSortDirection(key: HistorySortKey): "asc" | "desc" {
  return key === "avg_cost" || key === "avg_latency" || key === "failure_rate" ? "asc" : "desc";
}

function sortHistoryRows<T extends HistoryModelPerformanceRow | HistoryConfigPerformanceRow>(rows: T[], sort: HistorySort): T[] {
  return [...rows].sort((a, b) => {
    const left = (a as Record<HistorySortKey, unknown>)[sort.key];
    const right = (b as Record<HistorySortKey, unknown>)[sort.key];
    const multiplier = sort.direction === "asc" ? 1 : -1;
    if (typeof left === "number" && typeof right === "number") {
      return (left - right) * multiplier;
    }
    return String(left ?? "").localeCompare(String(right ?? "")) * multiplier;
  });
}

async function waitForJob(
  started: JobProgress,
  setProgress: (progress: JobProgress) => void,
  onTick?: (progress: JobProgress) => Promise<void> | void
) {
  let latest = started;
  setProgress(latest);
  await onTick?.(latest);
  while (latest.status === "queued" || latest.status === "running") {
    await sleep(1200);
    latest = await fetchJob(latest.job_id);
    setProgress(latest);
    await onTick?.(latest);
  }
  if (latest.status === "failed") {
    throw new Error(latest.error || `${latest.action} failed`);
  }
  return latest;
}

function sleep(durationMs: number) {
  return new Promise((resolve) => window.setTimeout(resolve, durationMs));
}

function toggleValue<T>(values: T[], value: T): T[] {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

function fallbackModel(alias: string): ModelInfo {
  return {
    alias,
    display_name: humanize(alias),
    provider: "unknown",
    model: alias,
    is_available: true,
    requires: ""
  };
}

function displayModel(alias?: string | null, modelMap?: Record<string, ModelInfo>) {
  if (!alias) {
    return "No data";
  }
  return modelMap?.[alias]?.display_name ?? humanize(alias);
}

function bestParamsLabel(config?: ConfigLeaderboardRow | null) {
  if (!config) {
    return "No run";
  }
  return `a=${config.alpha}, k=${config.retrieve_top_k}, n=${config.rerank_top_n}`;
}

function sourceLabel(doc?: string | null, section?: string | null) {
  if (!doc && !section) {
    return "No expected source";
  }
  return `${doc ?? "unknown"} / ${section ?? "unknown"}`;
}

function expectedSourcesFor(row: RunResult, question?: EvalQuestion) {
  if (row.question_type === "unknown" || row.question?.includes("snapshot unavailable")) {
    return [];
  }
  const sources = row.expected_sources?.length ? row.expected_sources : question?.expected_sources ?? [];
  if (sources.length) {
    return sources;
  }
  if (row.expected_doc && row.expected_section) {
    return [{ document: row.expected_doc, section: row.expected_section }];
  }
  if (question?.expected_doc && question.expected_section) {
    return [{ document: question.expected_doc, section: question.expected_section }];
  }
  return [];
}

function sourceListText(
  sources?: Array<{ document: string; section: string }>,
  fallbackDoc?: string | null,
  fallbackSection?: string | null
) {
  const normalized = sources?.length ? sources : fallbackDoc && fallbackSection ? [{ document: fallbackDoc, section: fallbackSection }] : [];
  if (!normalized.length) {
    return "No expected source";
  }
  if (normalized.length === 1) {
    return sourceLabel(normalized[0].document, normalized[0].section);
  }
  return `${normalized.length} required sources`;
}

function rowResultLabel(row: RunResult): "right" | "partial" | "wrong" {
  const value = String(row.result_label ?? row.metrics?.result_label ?? "");
  if (value === "right" || value === "partial" || value === "wrong") {
    return value;
  }
  if (row.failure_category === "passed") {
    return "right";
  }
  if (["partial_answer", "partial_citation", "missing_citation", "needs_review"].includes(row.failure_category)) {
    return "partial";
  }
  return "wrong";
}

function rowIssueLabel(row: RunResult): string {
  const value = String(row.issue_label ?? row.metrics?.issue_label ?? "");
  if (value) {
    return value;
  }
  if (row.failure_category === "passed") {
    return "none";
  }
  if (row.failure_category === "bad_retrieval") {
    return "source_not_found";
  }
  if (["missing_citation", "partial_citation", "wrong_citation", "unsupported_citation"].includes(row.failure_category)) {
    return "citation_issue";
  }
  if (["partial_answer", "needs_review", "incorrect_refusal"].includes(row.failure_category)) {
    return "incomplete_answer";
  }
  return "unsupported_answer";
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

function humanizeResult(result: string) {
  const labels: Record<string, string> = {
    right: "Right",
    partial: "Partial",
    wrong: "Wrong"
  };
  return labels[result] ?? humanize(result);
}

function failureExplanation(issue: string) {
  switch (issue) {
    case "none":
      return "The answer is acceptable for this test case.";
    case "source_not_found":
      return "The retriever did not bring back the source needed to answer this question.";
    case "citation_issue":
      return "The answer has weak or wrong evidence: the citation is missing, partial, or points to the wrong source.";
    case "incomplete_answer":
      return "The answer has the core idea but misses an important detail, caveat, or required source.";
    case "unsupported_answer":
      return "The answer is not supported by the docs, contradicts a required fact, or answered when it should have refused.";
    case "provider_error":
      return "The answer-quality check could not complete because a provider/API check was unavailable.";
    default:
      return "Review the expected answer, generated answer, and retrieved chunks to diagnose this case.";
  }
}

function humanize(value: string) {
  if (!value || value === "none") {
    return "None";
  }
  return value
    .replace(/^groq_/, "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function humanizeAnswerCheck(value: string) {
  if (value === "judge_unavailable") {
    return "Check unavailable";
  }
  if (value === "not_run") {
    return "Not checked";
  }
  return humanize(value);
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

const tooltipStyle = {
  background: "#020617",
  border: "1px solid rgba(148, 163, 184, 0.25)",
  borderRadius: 14,
  color: "#e2e8f0",
  boxShadow: "0 18px 60px rgba(0,0,0,0.35)"
};

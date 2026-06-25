export type ConfigLeaderboardRow = {
  model: string;
  alpha: number;
  retrieve_top_k: number;
  rerank_top_n: number;
  quality_score: number;
  cost_per_query: number;
  avg_latency_ms: number;
  failure_rate: number;
  answer_correctness: number;
  groundedness: number;
  citation_accuracy: number;
  refusal_correctness: number;
};

export type ModelLeaderboardRow = {
  model: string;
  quality_score: number;
  cost_per_query: number;
  avg_latency_ms: number;
  failure_rate: number;
  best_alpha: number;
  best_retrieve_top_k: number;
  best_rerank_top_n: number;
};

export type QualityCostPoint = ConfigLeaderboardRow & {
  label: string;
  config_label: string;
};

export type FailureByModelRow = {
  model: string;
  category: string;
  count: number;
};

export type DashboardSummary = {
  run_id?: number;
  run_name?: string;
  dashboard_state: "empty" | "running" | "completed" | "failed";
  last_ingested_at?: string | null;
  summary_notice?: string | null;
  is_stale?: boolean;
  best_config: ConfigLeaderboardRow | null;
  best_model: ModelLeaderboardRow | null;
  cheapest_acceptable_model: ModelLeaderboardRow | null;
  fastest_acceptable_model: ModelLeaderboardRow | null;
  most_common_failure: { category: string; count: number } | null;
  leaderboard: ConfigLeaderboardRow[];
  config_leaderboard: ConfigLeaderboardRow[];
  model_leaderboard: ModelLeaderboardRow[];
  quality_cost_points: QualityCostPoint[];
  failure_breakdown: { category: string; count: number }[];
  failure_by_model: FailureByModelRow[];
  model_summary: Record<
    string,
    {
      quality_score: number;
      cost_per_query: number;
      avg_latency_ms: number;
      failure_rate?: number;
    }
  >;
  result_count?: number;
};

export type ModelInfo = {
  alias: string;
  display_name: string;
  provider: string;
  model: string;
  is_available: boolean;
  requires: string;
  setup_command?: string | null;
};

export type RetrievalStatus = {
  dense_retrieval: string;
  use_pinecone: boolean;
  pinecone_index: string;
  pinecone_namespace: string;
  pinecone_embedding_mode?: string;
  pinecone_embed_model?: string;
  pinecone_text_field?: string;
  embedding_model: string;
  embedding_dimension: number;
  bm25_retrieval: string;
  reranker: string;
  dataset?: {
    active_dataset: string;
    dataset_key: string;
    data_dir: string;
    is_generated: boolean;
    source_page_count: number;
    faq_pair_count: number;
    support_doc_count: number;
    eval_question_count: number;
    generated_at?: string | null;
    source_pages: Array<Record<string, any>>;
  };
};

export type EvalRunPayload = {
  run_name: string;
  question_limit: number;
  question_ids?: string[];
  models: string[];
  alphas: number[];
  retrieve_top_k: number[];
  rerank_top_n: number[];
};

export type EvalQuestion = {
  id: string;
  question_type: string;
  question: string;
  reference_answer: string;
  expected_doc?: string | null;
  expected_section?: string | null;
  expected_sources: Array<{ document: string; section: string }>;
  reference_facts: string[];
  evaluation_notes: string;
  tags: string[];
  should_refuse: boolean;
};

export type JobProgress = {
  job_id: string;
  action: "ingest" | "benchmark" | string;
  status: "queued" | "running" | "completed" | "failed";
  run_id?: number | null;
  steps: string[];
  step_index: number;
  current_step: string;
  completed_items: number;
  total_items: number;
  detail: string;
  error?: string | null;
  result?: any;
  started_at: string;
  updated_at: string;
};

export type RunResult = {
  id: number;
  question_id: string;
  question?: string | null;
  reference_answer?: string | null;
  expected_doc?: string | null;
  expected_section?: string | null;
  expected_sources: Array<{ document: string; section: string }>;
  question_type: string;
  reference_facts: string[];
  evaluation_notes: string;
  tags: string[];
  should_refuse: boolean;
  top_retrieved_doc?: string | null;
  top_retrieved_section?: string | null;
  expected_source_found: boolean;
  source_recall?: number;
  citation_matched?: boolean;
  judge_label?: string;
  judge_rationale?: string;
  judge_source?: string;
  missing_facts?: string[];
  contradictions?: string[];
  judge_model?: string;
  judge_prompt_version?: string;
  model: string;
  alpha: number;
  retrieve_top_k: number;
  rerank_top_n: number;
  quality_score: number;
  cost_usd: number;
  latency_ms: number;
  failure_category: string;
  result_label?: string;
  issue_label?: string;
  trace_id: string;
  metrics: Record<string, any>;
  answer: {
    answer?: string;
    citations?: Array<Record<string, string>>;
    estimated_cost_usd?: number;
    latency_ms?: number;
  };
  retrieved_chunks: Array<Record<string, any>>;
};

export type RunDetails = {
  id: number;
  run_name: string;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  config_grid: Record<string, any>;
  summary: Partial<DashboardSummary>;
  results: RunResult[];
};

export type HistoryRun = {
  run_id: number;
  run_name: string;
  started_at: string | null;
  completed_at: string | null;
  best_model?: string | null;
  best_quality: number;
  cost_per_query: number;
  avg_latency_ms: number;
  failure_rate: number;
  result_count: number;
};

export type HistoryModelPerformanceRow = {
  model: string;
  provider: string;
  run_count: number;
  result_count: number;
  avg_quality: number;
  avg_cost: number;
  avg_latency: number;
  failure_rate: number;
  answer_correctness: number;
  groundedness: number;
  citation_accuracy: number;
  refusal_correctness: number;
};

export type HistoryConfigPerformanceRow = HistoryModelPerformanceRow & {
  alpha: number;
  retrieve_top_k: number;
  rerank_top_n: number;
};

export type DashboardHistory = {
  runs: HistoryRun[];
  model_performance: HistoryModelPerformanceRow[];
  config_performance: HistoryConfigPerformanceRow[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function fetchDashboardSummary(): Promise<DashboardSummary> {
  const response = await fetch(`${API_BASE}/dashboard/summary`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readError(response, "Failed to fetch dashboard summary"));
  }
  return normalizeSummary(await response.json());
}

export async function fetchDashboardHistory(): Promise<DashboardHistory> {
  const response = await fetch(`${API_BASE}/dashboard/history`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readError(response, "Failed to fetch dashboard history"));
  }
  const data = await response.json();
  return {
    runs: data.runs ?? [],
    model_performance: data.model_performance ?? [],
    config_performance: data.config_performance ?? []
  };
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const response = await fetch(`${API_BASE}/models`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readError(response, "Failed to fetch model catalog"));
  }
  const data = await response.json();
  return data.models ?? [];
}

export async function fetchRetrievalStatus(): Promise<RetrievalStatus> {
  const response = await fetch(`${API_BASE}/retrieval/status`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readError(response, "Failed to fetch retrieval status"));
  }
  return response.json();
}

export async function ingestDocuments() {
  const response = await fetch(`${API_BASE}/documents/ingest`, { method: "POST" });
  if (!response.ok) {
    throw new Error(await readError(response, "Failed to ingest documents"));
  }
  return response.json();
}

export async function startIngestion(): Promise<JobProgress> {
  const response = await fetch(`${API_BASE}/documents/ingest/start`, { method: "POST" });
  if (!response.ok) {
    throw new Error(await readError(response, "Failed to start ingestion"));
  }
  return response.json();
}

export async function runBenchmark(payload: EvalRunPayload) {
  const response = await fetch(`${API_BASE}/eval/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(await readError(response, "Failed to run benchmark"));
  }
  const data = await response.json();
  data.summary = normalizeSummary(data.summary);
  return data;
}

export async function startBenchmark(payload: EvalRunPayload): Promise<JobProgress> {
  const response = await fetch(`${API_BASE}/eval/run/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(await readError(response, "Failed to start benchmark"));
  }
  return response.json();
}

export async function fetchJob(jobId: string): Promise<JobProgress> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readError(response, "Failed to fetch job status"));
  }
  return response.json();
}

export async function fetchEvalQuestions(): Promise<EvalQuestion[]> {
  const response = await fetch(`${API_BASE}/eval/questions`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readError(response, "Failed to fetch eval questions"));
  }
  return response.json();
}

export async function fetchRunDetails(runId: number): Promise<RunDetails> {
  const response = await fetch(`${API_BASE}/eval/runs/${runId}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readError(response, "Failed to fetch run details"));
  }
  return response.json();
}

export function runExportUrl(runId: number): string {
  return `${API_BASE}/eval/runs/${runId}/export.csv`;
}

export async function fetchTrace(traceId: string) {
  const response = await fetch(`${API_BASE}/traces/${traceId}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readError(response, "Failed to fetch trace"));
  }
  return response.json();
}

async function readError(response: Response, fallback: string) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") {
      return data.detail;
    }
  } catch {
    // Ignore non-JSON responses and use fallback.
  }
  return fallback;
}

function normalizeSummary(data: Partial<DashboardSummary>): DashboardSummary {
  return {
    dashboard_state: data.dashboard_state ?? "empty",
    last_ingested_at: data.last_ingested_at ?? null,
    summary_notice: data.summary_notice ?? null,
    is_stale: Boolean(data.is_stale),
    run_id: data.run_id,
    run_name: data.run_name,
    best_config: data.best_config ?? null,
    best_model: data.best_model ?? null,
    cheapest_acceptable_model: data.cheapest_acceptable_model ?? null,
    fastest_acceptable_model: data.fastest_acceptable_model ?? null,
    most_common_failure: data.most_common_failure ?? null,
    leaderboard: data.leaderboard ?? data.config_leaderboard ?? [],
    config_leaderboard: data.config_leaderboard ?? data.leaderboard ?? [],
    model_leaderboard: data.model_leaderboard ?? [],
    quality_cost_points: data.quality_cost_points ?? [],
    failure_breakdown: data.failure_breakdown ?? [],
    failure_by_model: data.failure_by_model ?? [],
    model_summary: data.model_summary ?? {},
    result_count: data.result_count ?? 0
  };
}

export const emptySummary: DashboardSummary = normalizeSummary({});

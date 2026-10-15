import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { getVercelOidcTokenSync } from "@vercel/oidc";
import type { Writer } from "@/lib/stream";

/**
 * Provider resolution.
 *
 * LangChain's ChatOpenAI speaks the OpenAI wire protocol, so it can be pointed at
 * either OpenAI directly or Vercel AI Gateway (OpenAI-compatible, multi-provider).
 * Auth order: AI_GATEWAY_API_KEY → OPENAI_API_KEY → Vercel OIDC token (automatic
 * when deployed on Vercel). With none of these the app runs in "demo mode": every
 * agent has a deterministic fallback so the full LangGraph pipeline still executes.
 */
const GATEWAY_URL = "https://ai-gateway.vercel.sh/v1";

type Provider = { kind: "gateway" | "openai" | "gateway-oidc"; apiKey: string; baseURL?: string };

// Circuit breaker: after an auth / quota failure, stop calling the provider for a while.
let disabledUntil = 0;
let lastError: string | null = null;

function resolveProvider(): Provider | null {
  if (process.env.LLM_DISABLED === "1" || Date.now() < disabledUntil) return null;
  if (process.env.AI_GATEWAY_API_KEY) {
    return { kind: "gateway", apiKey: process.env.AI_GATEWAY_API_KEY, baseURL: GATEWAY_URL };
  }
  if (process.env.OPENAI_API_KEY) {
    return { kind: "openai", apiKey: process.env.OPENAI_API_KEY };
  }
  try {
    const token = getVercelOidcTokenSync();
    if (token) return { kind: "gateway-oidc", apiKey: token, baseURL: GATEWAY_URL };
  } catch {
    // not running on Vercel / OIDC not configured
  }
  return null;
}

export function llmStatus() {
  const p = resolveProvider();
  return {
    mode: p ? ("live" as const) : ("demo" as const),
    provider: p?.kind ?? "none",
    model: p ? chatModelId(p) : "deterministic-fallback",
    lastError,
  };
}

function chatModelId(p: Provider) {
  const m = process.env.LLM_MODEL ?? "openai/gpt-5.4-mini";
  return p.kind === "openai" ? m.replace(/^openai\//, "") : m;
}

export function getChatModel(opts: { temperature?: number } = {}) {
  const p = resolveProvider();
  if (!p) return null;
  const model = chatModelId(p);
  // Reasoning-family models reject a custom temperature.
  const supportsTemperature = !/gpt-5|o\d/.test(model);
  return new ChatOpenAI({
    model,
    apiKey: p.apiKey,
    ...(supportsTemperature ? { temperature: opts.temperature ?? 0.2 } : {}),
    configuration: p.baseURL ? { baseURL: p.baseURL } : undefined,
    maxRetries: 1,
    timeout: 45_000,
  });
}

export function getRemoteEmbeddings() {
  const p = resolveProvider();
  if (!p) return null;
  return new OpenAIEmbeddings({
    model: p.kind === "openai" ? "text-embedding-3-small" : "openai/text-embedding-3-small",
    apiKey: p.apiKey,
    configuration: p.baseURL ? { baseURL: p.baseURL } : undefined,
  });
}

/**
 * Run an LLM-backed step, degrading to a deterministic implementation if no
 * model is configured or the call fails. Keeps every graph node total.
 */
export async function withLLM<T>(
  node: string,
  send: Writer,
  live: (llm: ChatOpenAI) => Promise<T>,
  fallback: () => T | Promise<T>,
  opts: { temperature?: number } = {},
): Promise<T> {
  const llm = getChatModel(opts);
  if (!llm) return fallback();
  try {
    return await live(llm);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    lastError = msg.slice(0, 200);
    if (/40[123]|unauthori[sz]ed|forbidden|credit|quota|billing|api key/i.test(msg)) disabledUntil = Date.now() + 10 * 60_000;
    send({ type: "log", node, message: `LLM call failed (${msg.slice(0, 120)}); used deterministic fallback` });
    return fallback();
  }
}

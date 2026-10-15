import type { AgentEvent } from "@/lib/stream";

/** POST JSON and invoke `onEvent` for every NDJSON line the LangGraph route streams back. */
export async function streamAgent(url: string, body: unknown, onEvent: (e: AgentEvent) => void, signal?: AbortSignal) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });
  if (!res.ok || !res.body) {
    const text = await res.text();
    let msg = text;
    try {
      const err = JSON.parse(text).error;
      msg = typeof err === "string" ? err : JSON.stringify(err);
    } catch {}
    throw new Error(res.status === 429 ? msg : `Request failed (${res.status}): ${msg}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) onEvent(JSON.parse(line) as AgentEvent);
    }
  }
}

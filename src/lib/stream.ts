/** Events emitted by LangGraph nodes through `config.writer` and forwarded to the browser as NDJSON. */
export type AgentEvent =
  | { type: "step"; node: string; status: "start" | "done" | "skip"; detail?: string; data?: unknown }
  | { type: "log"; node: string; message: string }
  | { type: "token"; node: string; text: string }
  | { type: "reset"; node: string }
  | { type: "result"; data: unknown }
  | { type: "interrupt"; data: unknown }
  | { type: "meta"; data: unknown }
  | { type: "error"; message: string };

export type Writer = (e: AgentEvent) => void;

export function emitter(config: { writer?: (chunk: unknown) => void }): Writer {
  return (e) => config.writer?.(e);
}

/** Run a producer that writes AgentEvents and stream them as newline-delimited JSON. */
export function ndjsonResponse(run: (send: Writer) => Promise<void>) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send: Writer = (e) => controller.enqueue(encoder.encode(JSON.stringify(e) + "\n"));
      try {
        await run(send);
      } catch (err) {
        send({ type: "error", message: err instanceof Error ? err.message : String(err) });
      } finally {
        controller.close();
      }
    },
  });
  return new Response(stream, {
    headers: { "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-cache, no-transform" },
  });
}

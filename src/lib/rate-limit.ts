/**
 * Best-effort sliding-window rate limit per client IP, held in instance memory.
 * Protects the LLM budget of a public demo; pair it with a hard spending cap on the provider account.
 * Swap for Upstash Redis (@upstash/ratelimit) if a shared, cross-instance limit is needed.
 */
const hits = new Map<string, number[]>();

export function rateLimit(req: Request, bucket: string, { limit, windowMs }: { limit: number; windowMs: number }) {
  if (process.env.RATE_LIMIT_DISABLED === "1") return null;
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || req.headers.get("x-real-ip") || "local";
  const key = `${bucket}:${ip}`;
  const now = Date.now();
  const recent = (hits.get(key) ?? []).filter((t) => now - t < windowMs);
  if (recent.length >= limit) {
    const retryAfter = Math.ceil((windowMs - (now - recent[0])) / 1000);
    return Response.json(
      { error: `Rate limit reached for this demo (${limit} runs per ${Math.round(windowMs / 60_000)} min). Try again in ${retryAfter}s.` },
      { status: 429, headers: { "Retry-After": String(retryAfter) } },
    );
  }
  recent.push(now);
  hits.set(key, recent);
  if (hits.size > 5000) for (const [k, v] of hits) if (now - v[v.length - 1] > windowMs) hits.delete(k);
  return null;
}

"use client";

export type NodeStatus = "pending" | "running" | "done" | "skipped" | "waiting";

export type GraphNode = { id: string; title: string; role: string; x: number; y: number };
export type GraphEdge = { from: string; to: string; kind?: "cycle" | "conditional"; label?: string; arc?: number };


/** Live view of a LangGraph state machine: nodes light up as `step` events stream in. */
export function AgentGraph({
  nodes,
  edges,
  status,
  badges = {},
  width,
  height,
  onSelect,
  selected,
  nodeWidth: W = 150,
  nodeHeight: H = 50,
  fontScale = 1,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  status: Record<string, NodeStatus>;
  badges?: Record<string, string>;
  width: number;
  height: number;
  onSelect?: (id: string) => void;
  selected?: string | null;
  nodeWidth?: number;
  nodeHeight?: number;
  fontScale?: number;
}) {
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Agent graph">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--text-muted)" />
        </marker>
        <marker id="arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" />
        </marker>
      </defs>
      {edges.map((e, i) => {
        const a = byId[e.from];
        const b = byId[e.to];
        const active = status[e.from] === "done" && !["pending", "skipped"].includes(status[e.to] ?? "pending");
        let d: string;
        if (e.kind === "cycle") {
          const top = Math.min(a.y, b.y) - H / 2 - (e.arc ?? 26);
          d = `M${a.x},${a.y - H / 2} C${a.x},${top} ${b.x},${top} ${b.x},${b.y - H / 2}`;
        } else if (b.x - W / 2 < a.x + W / 2 + 10) {
          // Target is not to the right (backward / stacked): connect vertically.
          const down = b.y > a.y;
          const y1 = a.y + (down ? H / 2 : -H / 2);
          const y2 = b.y + (down ? -H / 2 : H / 2);
          const my = (y1 + y2) / 2;
          d = `M${a.x},${y1} C${a.x},${my} ${b.x},${my} ${b.x},${y2}`;
        } else {
          const x1 = a.x + W / 2;
          const x2 = b.x - W / 2;
          const mx = (x1 + x2) / 2;
          d = `M${x1},${a.y} C${mx},${a.y} ${mx},${b.y} ${x2},${b.y}`;
        }
        return (
          <g key={i}>
            <path
              d={d}
              fill="none"
              stroke={active ? "var(--accent)" : "var(--border)"}
              strokeWidth={active ? 2 : 1.5}
              strokeDasharray={e.kind ? "5 4" : undefined}
              markerEnd={`url(#${active ? "arrow-active" : "arrow"})`}
            />
            {e.kind === "cycle" && (
              <text x={(a.x + b.x) / 2} y={Math.min(a.y, b.y) - H / 2 - (e.arc ?? 26) + 4} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
                {e.label ?? "revise (if issues)"}
              </text>
            )}
          </g>
        );
      })}
      {nodes.map((n) => {
        const s = status[n.id] ?? "pending";
        const stroke =
          s === "running" ? "var(--accent)" : s === "done" ? "var(--series-3)" : s === "waiting" ? "var(--status-warning)" : "var(--border)";
        return (
          <g
            key={n.id}
            transform={`translate(${n.x - W / 2},${n.y - H / 2})`}
            onClick={() => onSelect?.(n.id)}
            style={{ cursor: onSelect ? "pointer" : undefined }}
          >
            {badges[n.id] && (
              // Stacked cards hint that this node runs as several parallel workers (Send API).
              <>
                <rect x={6} y={6} width={W} height={H} rx={10} fill="var(--surface-1)" stroke="var(--border)" />
                <rect x={3} y={3} width={W} height={H} rx={10} fill="var(--surface-1)" stroke="var(--border)" />
              </>
            )}
            {(s === "running" || s === "waiting") && (
              <rect width={W} height={H} rx={10} fill="none" stroke={s === "waiting" ? "var(--status-warning)" : "var(--accent)"} strokeWidth={2} opacity={0.6}>
                <animate attributeName="opacity" values="0.8;0.1;0.8" dur="1.2s" repeatCount="indefinite" />
              </rect>
            )}
            <rect
              width={W}
              height={H}
              rx={10}
              fill={selected === n.id ? "var(--surface-2)" : "var(--surface-1)"}
              stroke={stroke}
              strokeWidth={s === "pending" || s === "skipped" ? 1 : 2}
              strokeDasharray={s === "skipped" ? "4 3" : undefined}
            />
            <text x={10} y={H / 2 - 4} fontSize={12 * fontScale} fontWeight={600} fill={s === "pending" || s === "skipped" ? "var(--text-muted)" : "var(--text-primary)"}>
              {n.title}
            </text>
            {badges[n.id] && (
              <text x={W} y={-6} textAnchor="end" fontSize={10 * fontScale} fontFamily="var(--font-geist-mono)" fill="var(--accent)">
                {badges[n.id]}
              </text>
            )}
            <text x={10} y={H / 2 + 12 * fontScale} fontSize={10 * fontScale} fill="var(--text-muted)">
              {n.role}
            </text>
            <g transform={`translate(${W - 18},${14})`}>
              {s === "done" && <path d="M0,4 L3,7 L8,1" fill="none" stroke="var(--series-3)" strokeWidth={2} strokeLinecap="round" />}
              {(s === "running" || s === "waiting") && (
                <circle cx={4} cy={4} r={4} fill={s === "waiting" ? "var(--status-warning)" : "var(--accent)"}>
                  <animate attributeName="r" values="2;4;2" dur="0.9s" repeatCount="indefinite" />
                </circle>
              )}
            </g>
          </g>
        );
      })}
    </svg>
  );
}

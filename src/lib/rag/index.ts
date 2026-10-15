import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import kb from "@/data/knowledge-base.json";
import { getRemoteEmbeddings } from "@/lib/llm";
import { LocalHashingEmbeddings, tokenize } from "./embeddings";
import { InMemoryCosineStore } from "./vectorstore";

export type KBEntry = {
  id: string;
  title: string;
  category: string;
  source: string;
  url: string;
  content: string;
};

export const KNOWLEDGE_BASE = kb as KBEntry[];

export type ChunkMeta = {
  chunkId: string;
  docId: string;
  title: string;
  category: string;
  source: string;
  url: string;
};

export type RetrievedChunk = {
  chunkId: string;
  docId: string;
  title: string;
  source: string;
  url: string;
  category: string;
  text: string;
  dense: number | null;
  bm25: number | null;
  denseRank: number | null;
  bm25Rank: number | null;
  rrf: number;
};

/** Okapi BM25 lexical index — complements dense retrieval on exact terms (e.g. "Lung-RADS 4B", "LIDC-IDRI"). */
class BM25 {
  private docs: string[][] = [];
  private df = new Map<string, number>();
  private avgLen = 0;
  constructor(
    texts: string[],
    private k1 = 1.4,
    private b = 0.75,
  ) {
    this.docs = texts.map(tokenize);
    this.avgLen = this.docs.reduce((s, d) => s + d.length, 0) / Math.max(1, this.docs.length);
    for (const d of this.docs) for (const t of new Set(d)) this.df.set(t, (this.df.get(t) ?? 0) + 1);
  }
  search(query: string, k: number): { index: number; score: number }[] {
    const q = tokenize(query);
    const N = this.docs.length;
    return this.docs
      .map((d, index) => {
        let score = 0;
        for (const t of q) {
          const f = d.filter((x) => x === t).length;
          if (!f) continue;
          const n = this.df.get(t) ?? 0;
          const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
          score += (idf * (f * (this.k1 + 1))) / (f + this.k1 * (1 - this.b + (this.b * d.length) / this.avgLen));
        }
        return { index, score };
      })
      .filter((r) => r.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }
}

type Index = {
  store: InMemoryCosineStore;
  bm25: BM25;
  chunks: Document<ChunkMeta>[];
  embeddingModel: string;
};

let indexPromise: Promise<Index> | null = null;

async function buildIndex(): Promise<Index> {
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 900, chunkOverlap: 150 });
  const chunks: Document<ChunkMeta>[] = [];
  for (const entry of KNOWLEDGE_BASE) {
    const parts = await splitter.splitText(entry.content);
    parts.forEach((p, i) =>
      chunks.push(
        new Document<ChunkMeta>({
          // Prefix the title so each chunk carries its own context.
          pageContent: `${entry.title}\n${p}`,
          metadata: {
            chunkId: `${entry.id}#${i}`,
            docId: entry.id,
            title: entry.title,
            category: entry.category,
            source: entry.source,
            url: entry.url,
          },
        }),
      ),
    );
  }

  let store: InMemoryCosineStore;
  let embeddingModel = "local-hashing-768d";
  const remote = getRemoteEmbeddings();
  try {
    if (!remote) throw new Error("no remote embeddings");
    store = await InMemoryCosineStore.fromDocuments(chunks, remote);
    embeddingModel = "text-embedding-3-small";
  } catch {
    store = await InMemoryCosineStore.fromDocuments(chunks, new LocalHashingEmbeddings());
  }
  return { store, bm25: new BM25(chunks.map((c) => c.pageContent)), chunks, embeddingModel };
}

export function getIndex() {
  indexPromise ??= buildIndex().catch((e) => {
    indexPromise = null;
    throw e;
  });
  return indexPromise;
}

/**
 * Hybrid retrieval: dense (cosine) + sparse (BM25) fused with Reciprocal Rank
 * Fusion (Cormack et al., 2009; k=60). Multiple query rewrites are fused too.
 */
export async function hybridSearch(queries: string[], k = 6, categories?: string[]): Promise<RetrievedChunk[]> {
  const { store, bm25, chunks } = await getIndex();
  const RRF_K = 60;
  const fused = new Map<string, RetrievedChunk>();
  const allowed = (c: Document<ChunkMeta>) => !categories?.length || categories.includes(c.metadata.category);

  const upsert = (doc: Document<ChunkMeta>) => {
    const m = doc.metadata;
    let r = fused.get(m.chunkId);
    if (!r) {
      r = {
        chunkId: m.chunkId,
        docId: m.docId,
        title: m.title,
        source: m.source,
        url: m.url,
        category: m.category,
        text: doc.pageContent,
        dense: null,
        bm25: null,
        denseRank: null,
        bm25Rank: null,
        rrf: 0,
      };
      fused.set(m.chunkId, r);
    }
    return r;
  };

  for (const q of queries) {
    const dense = await store.similaritySearchWithScore(q, 12, allowed as never);
    dense.forEach(([doc, score], rank) => {
      const r = upsert(doc as Document<ChunkMeta>);
      r.dense = Math.max(r.dense ?? -1, score);
      r.denseRank = Math.min(r.denseRank ?? Infinity, rank + 1);
      r.rrf += 1 / (RRF_K + rank + 1);
    });
    bm25
      .search(q, 30)
      .filter(({ index }) => allowed(chunks[index]))
      .slice(0, 12)
      .forEach(({ index, score }, rank) => {
        const r = upsert(chunks[index]);
        r.bm25 = Math.max(r.bm25 ?? 0, score);
        r.bm25Rank = Math.min(r.bm25Rank ?? Infinity, rank + 1);
        r.rrf += 1 / (RRF_K + rank + 1);
      });
  }
  return [...fused.values()].sort((a, b) => b.rrf - a.rrf).slice(0, k);
}

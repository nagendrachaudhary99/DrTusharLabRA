import { Embeddings } from "@langchain/core/embeddings";

/**
 * Offline, deterministic embeddings (feature hashing over unigrams, bigrams and
 * character trigrams, with a small radiology synonym map). Used when no remote
 * embedding provider is configured so RAG still works with zero setup.
 * It implements LangChain's `Embeddings` contract, so it is a drop-in swap for
 * OpenAIEmbeddings / any hosted model.
 */
const DIM = 768;

const SYNONYMS: Record<string, string> = {
  tumour: "tumor",
  tumors: "tumor",
  lesion: "nodule",
  lesions: "nodule",
  nodules: "nodule",
  mass: "nodule",
  ggo: "groundglass",
  "ground-glass": "groundglass",
  subsolid: "partsolid",
  "part-solid": "partsolid",
  synthetic: "synthesis",
  synthesized: "synthesis",
  generated: "synthesis",
  generative: "synthesis",
  diffusion: "diffusion",
  phantoms: "phantom",
  simulator: "simulation",
  simulated: "simulation",
  "in-silico": "insilico",
  silico: "insilico",
  vit: "virtualtrial",
  vits: "virtualtrial",
  scanners: "scanner",
  vendors: "vendor",
  kernels: "kernel",
  "low-dose": "lowdose",
  ldct: "lowdose",
};

const STOP = new Set(
  "a an the of and or to in on for with by is are was were be as at from that this these those it its into than then can may which what how why when does do using use used".split(
    " ",
  ),
);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\-\s.]/g, " ")
    .split(/\s+/)
    .map((t) => t.replace(/^[.\-]+|[.\-]+$/g, ""))
    .filter((t) => t.length > 1 && !STOP.has(t))
    .map((t) => SYNONYMS[t] ?? t);
}

function hash(s: string) {
  // FNV-1a 32-bit
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function embed(text: string): number[] {
  const v = new Float64Array(DIM);
  const toks = tokenize(text);
  const add = (feat: string, w: number) => {
    const h = hash(feat);
    v[h % DIM] += h & 0x80000000 ? -w : w;
  };
  toks.forEach((t, i) => {
    add(`u:${t}`, 1);
    if (i > 0) add(`b:${toks[i - 1]}_${t}`, 0.6);
    if (t.length > 4) for (let j = 0; j < t.length - 2; j++) add(`c:${t.slice(j, j + 3)}`, 0.15);
  });
  let norm = 0;
  for (let i = 0; i < DIM; i++) {
    v[i] = Math.sign(v[i]) * Math.log1p(Math.abs(v[i]));
    norm += v[i] * v[i];
  }
  norm = Math.sqrt(norm) || 1;
  return Array.from(v, (x) => x / norm);
}

export class LocalHashingEmbeddings extends Embeddings {
  constructor() {
    super({});
  }
  async embedDocuments(documents: string[]) {
    return documents.map(embed);
  }
  async embedQuery(document: string) {
    return embed(document);
  }
}

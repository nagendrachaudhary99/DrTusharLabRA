import { VectorStore } from "@langchain/core/vectorstores";
import { Document, type DocumentInterface } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";

type Filter = (doc: DocumentInterface) => boolean;

/**
 * Minimal in-memory cosine-similarity vector store implementing LangChain's
 * `VectorStore` abstract class. Swap for PGVector / Pinecone / Upstash Vector in
 * production without touching the retrieval graph.
 */
export class InMemoryCosineStore extends VectorStore {
  declare FilterType: Filter;
  private rows: { vector: number[]; doc: DocumentInterface }[] = [];

  constructor(embeddings: EmbeddingsInterface) {
    super(embeddings, {});
  }

  _vectorstoreType() {
    return "in-memory-cosine";
  }

  get size() {
    return this.rows.length;
  }

  async addVectors(vectors: number[][], documents: DocumentInterface[]) {
    vectors.forEach((vector, i) => this.rows.push({ vector, doc: documents[i] }));
  }

  async addDocuments(documents: DocumentInterface[]) {
    const vectors = await this.embeddings.embedDocuments(documents.map((d) => d.pageContent));
    await this.addVectors(vectors, documents);
  }

  async similaritySearchVectorWithScore(query: number[], k: number, filter?: Filter) {
    const qn = Math.sqrt(query.reduce((s, x) => s + x * x, 0)) || 1;
    return this.rows
      .filter((r) => (filter ? filter(r.doc) : true))
      .map(({ vector, doc }) => {
        let dot = 0;
        let vn = 0;
        for (let i = 0; i < vector.length; i++) {
          dot += vector[i] * query[i];
          vn += vector[i] * vector[i];
        }
        return [doc, dot / (qn * (Math.sqrt(vn) || 1))] as [DocumentInterface, number];
      })
      .sort((a, b) => b[1] - a[1])
      .slice(0, k);
  }

  static async fromDocuments(docs: Document[], embeddings: EmbeddingsInterface) {
    const store = new InMemoryCosineStore(embeddings);
    await store.addDocuments(docs);
    return store;
  }
}

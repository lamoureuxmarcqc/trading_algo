"use client";

import { FormEvent, useState } from "react";

type KnowledgeSource = {
  document_id: string;
  chunk_id: string;
  page?: number | null;
  excerpt: string;
  score: number;
};

type KnowledgeResponse = {
  sources: KnowledgeSource[];
};

const KNOWLEDGE_URL = "http://localhost:8002/api/v1/knowledge/query";

export default function KnowledgePage() {
  const [query, setQuery] = useState("");
  const [sources, setSources] = useState<KnowledgeSource[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      return;
    }

    setError(null);
    setIsLoading(true);
    try {
      const response = await fetch(KNOWLEDGE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmedQuery,
          tenant_id: "lamoureux",
          top_k: 5,
          classification_max: "CONFIDENTIAL"
        })
      });
      if (!response.ok) {
        throw new Error(`La recherche a echoue (${response.status}).`);
      }

      const payload = (await response.json()) as KnowledgeResponse;
      setSources(payload.sources ?? []);
    } catch (requestError) {
      setSources([]);
      setError(
        requestError instanceof Error
          ? requestError.message
          : "L'API Knowledge ne repond pas."
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-3xl space-y-6 p-6">
      <div>
        <p className="text-sm font-medium text-slate-400">Lamoureux AI Platform</p>
        <h1 className="mt-1 text-3xl font-semibold text-white">Recherche documentaire</h1>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-3">
        <input
          aria-label="Requete de recherche"
          className="min-w-0 flex-1 rounded-lg border border-line bg-black/30 px-4 py-3 text-white outline-none focus:border-accent"
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Posez une question sur vos documents"
          value={query}
        />
        <button
          className="rounded-lg bg-sand px-4 py-3 font-medium text-ink disabled:cursor-not-allowed disabled:opacity-50"
          disabled={isLoading || !query.trim()}
          type="submit"
        >
          {isLoading ? "Recherche..." : "Rechercher"}
        </button>
      </form>

      {error ? <p className="rounded-lg border border-loss/40 bg-loss/10 p-4 text-loss">{error}</p> : null}

      <section className="space-y-3">
        {sources.map((source) => (
          <article key={source.chunk_id} className="rounded-lg border border-line bg-black/20 p-4">
            <div className="flex justify-between gap-4 text-sm text-slate-400">
              <span>Page {source.page ?? "n/a"}</span>
              <span>Score {source.score.toFixed(2)}</span>
            </div>
            <p className="mt-3 text-slate-200">{source.excerpt.slice(0, 300)}</p>
          </article>
        ))}
      </section>
    </main>
  );
}

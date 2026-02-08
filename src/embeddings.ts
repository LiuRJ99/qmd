import OpenAI from "openai";

// Configuration - support OpenAI-compatible API endpoints
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
const EMBEDDING_MODEL =
  process.env.QMD_EMBEDDING_MODEL || "openai/text-embedding-3-small";

// Determine which API key to use (prioritizing OPENAI_API_KEY if available, otherwise OPENROUTER_API_KEY)
const API_KEY = OPENAI_API_KEY || OPENROUTER_API_KEY;
// Determine base URL based on environment
const BASE_URL = OPENAI_API_KEY ? OPENAI_BASE_URL : "https://openrouter.ai/api/v1";

// Validate API key
if (!API_KEY) {
  console.error(
    "Warning: No API key set (either OPENAI_API_KEY or OPENROUTER_API_KEY). Embeddings will not be generated."
  );
}

// OpenAI-compatible client (supporting OpenAI, OpenRouter, and custom endpoints)
const client = API_KEY
  ? new OpenAI({
      apiKey: API_KEY,
      baseURL: BASE_URL,
      ...(OPENROUTER_API_KEY && !OPENAI_API_KEY && {
        defaultHeaders: {
          "HTTP-Referer": "https://github.com/qmd",
          "X-Title": "QMD Knowledge Base",
        },
      }),
    })
  : null;

// Embedding dimensions by model
const MODEL_DIMENSIONS: Record<string, number> = {
  "openai/text-embedding-3-small": 1536,
  "openai/text-embedding-3-large": 3072,
  "cohere/embed-english-v3.0": 1024,
  "cohere/embed-multilingual-v3.0": 1024,
};

export function getEmbeddingDimensions(): number {
  return MODEL_DIMENSIONS[EMBEDDING_MODEL] || 1536;
}

export function isEmbeddingsEnabled(): boolean {
  return !!client;
}

/**
 * Generate embeddings for one or more texts
 * @param texts Array of text strings to embed
 * @returns Array of embedding vectors (number arrays)
 */
export async function embed(texts: string[]): Promise<number[][]> {
  if (!client) {
    throw new Error("OPENROUTER_API_KEY not configured");
  }

  if (texts.length === 0) {
    return [];
  }

  // OpenRouter/OpenAI has a limit on batch size
  const BATCH_SIZE = 100;
  const results: number[][] = [];

  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);

    try {
      const response = await client.embeddings.create({
        model: EMBEDDING_MODEL,
        input: batch,
      });

      // Sort by index to maintain order
      const sorted = response.data.sort((a, b) => a.index - b.index);
      results.push(...sorted.map((d) => d.embedding));
    } catch (error) {
      console.error(`Error generating embeddings for batch ${i}:`, error);
      throw error;
    }
  }

  return results;
}

/**
 * Generate embedding for a single text
 * @param text Text string to embed
 * @returns Embedding vector
 */
export async function embedSingle(text: string): Promise<number[]> {
  const [embedding] = await embed([text]);
  return embedding;
}

/**
 * Compute cosine similarity between two vectors
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(
      `Vector dimension mismatch: ${a.length} vs ${b.length}`
    );
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) return 0;

  return dot / denominator;
}

/**
 * Get current embedding model name
 */
export function getModelName(): string {
  return EMBEDDING_MODEL;
}

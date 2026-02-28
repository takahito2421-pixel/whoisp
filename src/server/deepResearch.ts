import JSON5 from "json5";
import {
  ThinkingLevel,
  type Candidate,
  type Content,
  type GenerateContentConfig,
  type GroundingChunk,
} from "@google/genai";
import type { GoogleGenAI } from "@google/genai/node";
import { fetchSubjectImages } from "./fetchSubjectImages";
import { getGeminiClient } from "./geminiClient";
import { TIMEOUT, RETRY } from "@/shared/constants";
import type {
  DeepResearchImage,
  DeepResearchPlan,
  DeepResearchPlanStep,
  DeepResearchServerEvent,
  Locale,
  SourceReference,
  StepFinding,
  StepResult,
} from "@/shared/deep-research-types";

type RunOptions = {
  locale: Locale;
  signal?: AbortSignal;
  emit(event: DeepResearchServerEvent): Promise<void> | void;
};

type SourceCandidate = {
  index: number;
  url: string;
  title: string;
  domain?: string;
};

type PlanPayload = {
  primaryGoal: string;
  rationale: string;
  steps: Array<{
    id: string;
    title: string;
    query: string;
    angle: string;
    deliverable: string;
  }>;
  expectedInsights: string[];
};

type EvidencePayload = {
  summary: string;
  findings: Array<{
    heading: string;
    insight: string;
    evidence: string;
    confidence?: string;
    sourceIds?: number[];
  }>;
};

const PLAN_SCHEMA = {
  type: "object",
  required: ["primaryGoal", "rationale", "steps", "expectedInsights"],
  properties: {
    primaryGoal: { type: "string" },
    rationale: { type: "string" },
    steps: {
      type: "array",
      minItems: 3,
      maxItems: 6,
      items: {
        type: "object",
        required: ["id", "title", "query", "angle", "deliverable"],
        properties: {
          id: { type: "string" },
          title: { type: "string" },
          query: { type: "string" },
          angle: { type: "string" },
          deliverable: { type: "string" },
        },
      },
    },
    expectedInsights: {
      type: "array",
      items: { type: "string" },
    },
  },
} as const;

const EVIDENCE_SCHEMA = {
  type: "object",
  required: ["summary", "findings"],
  properties: {
    summary: { type: "string" },
    findings: {
      type: "array",
      items: {
        type: "object",
        required: ["heading", "insight", "evidence"],
        properties: {
          heading: { type: "string" },
          insight: { type: "string" },
          evidence: { type: "string" },
          confidence: { type: "string" },
          sourceIds: {
            type: "array",
            items: { type: "integer" },
          },
        },
      },
    },
  },
} as const;

const PLAN_SYSTEM_PROMPT = `Plan 3-5 sequential web research steps with ids, clear intents, and concrete deliverables. Each step must cover a distinct angle that moves the investigation forward.`;

const EVIDENCE_SYSTEM_PROMPT = `For the assigned sub-query, run targeted web searches, capture only verifiable facts, and reply with structured findings plus citations. Prefer the newest trustworthy information and never invent sources.`;

const FINAL_SYSTEM_PROMPT = `Integrate the vetted findings into a concise report that cites sources, highlights tensions, and surfaces next questions while weighting newer information more heavily.`;

const THINKING_CONFIG = { thinkingLevel: ThinkingLevel.HIGH } as const;
const THINKING_ERROR_PATTERN = /Thinking level is not supported/i;
const THINKING_UNSUPPORTED_MODELS = new Set<string>();

const TODAY = new Date().toISOString().split('T')[0];

function localeDirective(locale: Locale, base: string): string {
  if (locale === "ja") {
    return `${base}\n応答は自然な日本語で書いてください。`;
  }
  return `${base}\nRespond in natural English.`;
}

function createContent(text: string): Content[] {
  return [
    {
      role: "user",
      parts: [{ text }],
    },
  ];
}

type AdaptiveThinkingRequest = {
  client: GoogleGenAI;
  model: string;
  contents: Content[];
  config: GenerateContentConfig;
  enableThinking?: boolean;
};

async function generateContentWithAdaptiveThinking({
  client,
  model,
  contents,
  config,
  enableThinking = true,
}: AdaptiveThinkingRequest) {
  if (!enableThinking || THINKING_UNSUPPORTED_MODELS.has(model)) {
    return client.models.generateContent({
      model,
      contents,
      config,
    });
  }

  try {
    return await client.models.generateContent({
      model,
      contents,
      config: {
        ...config,
        thinkingConfig: THINKING_CONFIG,
      },
    });
  } catch (error) {
    if (isThinkingUnsupportedError(error)) {
      THINKING_UNSUPPORTED_MODELS.add(model);
      console.warn(
        `[deepResearch] Thinking level unsupported for model "${model}". Retrying without thinkingConfig.`,
      );
      return client.models.generateContent({
        model,
        contents,
        config,
      });
    }
    throw error;
  }
}

function isThinkingUnsupportedError(error: unknown): boolean {
  if (!error) {
    return false;
  }
  const message =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : (error as { message?: string })?.message;
  return typeof message === "string" && THINKING_ERROR_PATTERN.test(message);
}
/*
 function parseJson<T>(raw: string, context: string): T {
  try {
    return JSON.parse(raw) as T;
  } catch (error) {
    try {
      return JSON5.parse(raw) as T;
    } catch {
      throw new Error(`Failed to parse ${context} JSON payload.`);
    }
  }
} 
*/
function parseJson<T>(raw: string, context: string): T {
  // 🔥 追加：改行をスペースに変換してからパース
  const normalized = (raw ?? "")
    .replace(/\r?\n/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();

  console.log(`[parseJson] ${context} raw (first 500 chars):`, normalized.slice(0, 500));

  try {
    return JSON.parse(normalized) as T;
  } catch (error) {
    console.error(`[parseJson] JSON.parse failed for ${context}:`, (error as Error).message);
    try {
      return JSON5.parse(normalized) as T;
    } catch (e2) {
      console.error(`[parseJson] JSON5.parse also failed for ${context}:`, (e2 as Error).message);
      console.error(`[parseJson] Full raw content:`, normalized);
      throw new Error(`Failed to parse ${context} JSON payload.`);
    }
  }
}

function extractJsonObject(raw: string, context: string): string {
  if (!raw || raw.trim().length === 0) {
    console.error(`[extractJsonObject] Empty response for ${context}`);
    throw new Error(`${context} response was empty.`);
  }

  // Remove markdown code fences if present
  let cleaned = raw.replace(/```json\s*/g, "").replace(/```\s*/g, "").trim();

  // Try to find JSON object
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");

  if (start === -1 || end === -1 || end <= start) {
    console.error(`[extractJsonObject] No JSON object found in ${context} response:`, cleaned.slice(0, 200));
    throw new Error(`${context} response did not contain a JSON object.`);
  }

  const extracted = cleaned.slice(start, end + 1);

 return extracted;

function extractCitations(candidate?: Candidate | null): SourceCandidate[] {
  const metadata = candidate?.groundingMetadata;
  if (!metadata?.groundingChunks || metadata.groundingChunks.length === 0) {
    return [];
  }

  const seen = new Map<string, SourceCandidate>();

  metadata.groundingChunks.forEach((chunk, index) => {
    const citation = normalizeChunk(chunk, index);
    if (!citation) return;
    const key = citation.url.toLowerCase();
    if (!seen.has(key)) {
      seen.set(key, citation);
    }
  });

  return Array.from(seen.values());
}

function normalizeChunk(chunk: GroundingChunk, index: number): SourceCandidate | null {
  if (chunk.web?.uri) {
    return {
      index,
      url: chunk.web.uri,
      title: chunk.web.title ?? chunk.web.uri,
      domain: chunk.web.domain,
    };
  }

  if (chunk.retrievedContext?.uri) {
    return {
      index,
      url: chunk.retrievedContext.uri,
      title: chunk.retrievedContext.title ?? chunk.retrievedContext.uri,
    };
  }

  return null;
}

function inferDomain(url: string | undefined): string | undefined {
  if (!url) return undefined;
  try {
    return new URL(url).hostname;
  } catch {
    return undefined;
  }
}

function normalizeUrl(url: string): string {
  try {
    const parsed = new URL(url);
    // Remove query parameters and hash for better deduplication
    // Keep protocol and path normalization
    let normalized = `${parsed.protocol}//${parsed.hostname}${parsed.pathname}`;
    // Remove trailing slash unless it's the root path
    if (normalized.endsWith('/') && normalized.length > parsed.protocol.length + parsed.hostname.length + 3) {
      normalized = normalized.slice(0, -1);
    }
    return normalized.toLowerCase();
  } catch {
    return url.toLowerCase();
  }
}

async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  errorMessage: string,
  signal?: AbortSignal,
): Promise<T> {
  // If signal is already aborted, reject immediately
  if (signal?.aborted) {
    throw new DOMException('Aborted', 'AbortError');
  }

  return Promise.race([
    promise,
    new Promise<T>((_, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error(errorMessage));
      }, timeoutMs);

      // Clean up timeout if signal is aborted
      signal?.addEventListener('abort', () => {
        clearTimeout(timeoutId);
        reject(new DOMException('Aborted', 'AbortError'));
      });
    }),
  ]);
}

export async function runDeepResearch(query: string, options: RunOptions): Promise<void> {
  const trimmed = query.trim();
  if (!trimmed) {
    throw new Error("Query cannot be empty.");
  }

  const { client, model } = getGeminiClient();
  const sourceRegistry = new Map<string, SourceReference>();
  const orderedSources: SourceReference[] = [];

  let subjectImages: DeepResearchImage[] = [];
  try {
    subjectImages = await fetchSubjectImages(trimmed, {
      locale: options.locale,
      signal: options.signal,
    });
  } catch (error) {
    if ((error as DOMException)?.name === "AbortError") {
      // If aborted during image fetch, continue without images
      subjectImages = [];
    } else {
      console.error("Subject image lookup failed", error);
    }
  }

  await options.emit({ type: "images", images: subjectImages });

  const registerSource = (candidate: SourceCandidate): SourceReference => {
    const key = normalizeUrl(candidate.url);
    const existing = sourceRegistry.get(key);
    if (existing) {
      return existing;
    }
    const id = String(sourceRegistry.size + 1);
    const reference: SourceReference = {
      id,
      url: candidate.url,
      title: candidate.title ?? candidate.url,
      domain: candidate.domain ?? inferDomain(candidate.url),
    };
    sourceRegistry.set(key, reference);
    orderedSources.push(reference);
    return reference;
  };

  let plan: DeepResearchPlan | null = null;
  let stepResults: StepResult[] = [];

  try {
    plan = await generatePlan(trimmed, options.locale, options.signal, client, model);
    await options.emit({ type: "plan", plan });

    // Execute steps in parallel for better performance
    const stepPromises = plan.steps.map(async (step) => {
      const result = await gatherEvidence(trimmed, step, options.locale, options.signal, client, model, registerSource);
      await options.emit({ type: "search", step: result });
      return result;
    });

    const results = await Promise.allSettled(stepPromises);
    stepResults = results
      .filter((r): r is PromiseFulfilledResult<StepResult> => r.status === 'fulfilled')
      .map((r) => r.value);

    const finalReport = await synthesizeReport(
      trimmed,
      plan,
      stepResults,
      orderedSources,
      options.locale,
      options.signal,
      client,
      model,
    );

    await options.emit({ type: "final", report: finalReport, sources: orderedSources });
  } catch (error) {
    if ((error as DOMException)?.name === "AbortError") {
      // Generate partial report with data collected so far
      if (stepResults.length > 0 && plan) {
        try {
          // Set timeout for partial report generation (5 seconds)
          const reportPromise = synthesizeReport(
            trimmed,
            plan,
            stepResults,
            orderedSources,
            options.locale,
            undefined, // Don't pass signal for partial report generation
            client,
            model,
          );

          const timeoutPromise = new Promise<string>((_, reject) => {
            setTimeout(() => reject(new Error("Partial report generation timeout")), TIMEOUT.PARTIAL_REPORT);
          });

          const partialReport = await Promise.race([reportPromise, timeoutPromise]);
          await options.emit({ type: "final", report: partialReport, sources: orderedSources });
        } catch (reportError) {
          console.error("Failed to generate partial report:", reportError);
          // Emit whatever we have
          await options.emit({
            type: "final",
            report: generateFallbackReport(trimmed, stepResults, options.locale),
            sources: orderedSources
          });
        }
      } else if (plan) {
        // If we have plan but no step results yet
        await options.emit({
          type: "final",
          report: generatePlanOnlyReport(trimmed, plan, options.locale),
          sources: orderedSources
        });
      }
    } else {
      throw error;
    }
  }
}

function generateFallbackReport(query: string, steps: StepResult[], locale: Locale): string {
  if (locale === "ja") {
    const findings = steps.map(step => {
      const findingsText = step.findings.map(f => `- **${f.heading}**: ${f.insight}`).join("\n");
      return `### ${step.title}\n\n${step.summary}\n\n${findingsText}`;
    }).join("\n\n");

    return `# ${query}\n\n## 調査結果（部分的）\n\n調査は途中で中止されましたが、以下の情報を収集しました。\n\n${findings}`;
  }

  const findings = steps.map(step => {
    const findingsText = step.findings.map(f => `- **${f.heading}**: ${f.insight}`).join("\n");
    return `### ${step.title}\n\n${step.summary}\n\n${findingsText}`;
  }).join("\n\n");

  return `# ${query}\n\n## Research Results (Partial)\n\nThe research was stopped, but the following information was collected.\n\n${findings}`;
}

function generatePlanOnlyReport(query: string, plan: DeepResearchPlan, locale: Locale): string {
  if (locale === "ja") {
    const steps = plan.steps.map(step => `- **${step.title}**: ${step.query}`).join("\n");
    return `# ${query}\n\n## 調査計画\n\n${plan.primaryGoal}\n\n${plan.rationale}\n\n### 計画されたステップ\n\n${steps}\n\n*調査は開始前に中止されました。*`;
  }

  const steps = plan.steps.map(step => `- **${step.title}**: ${step.query}`).join("\n");
  return `# ${query}\n\n## Research Plan\n\n${plan.primaryGoal}\n\n${plan.rationale}\n\n### Planned Steps\n\n${steps}\n\n*Research was stopped before gathering evidence.*`;
}

async function generatePlan(
  query: string,
  locale: Locale,
  signal: AbortSignal | undefined,
  client: GoogleGenAI,
  model: string,
): Promise<DeepResearchPlan> {
  const planPrompt = localeDirective(
    locale,
    `Today is ${TODAY}. Break the research question into 3-5 decisive web investigation steps. Each step should target a different angle or source type and build toward a synthesis. When planning, prioritize finding the most recent and up-to-date information available.`,
  ) +
    `\n\nResearch question: ${query}`;

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= RETRY.MAX_ATTEMPTS; attempt++) {
    try {
      const response = await withTimeout(
        generateContentWithAdaptiveThinking({
          client,
          model,
          contents: createContent(planPrompt),
          config: {
            ...(signal ? { abortSignal: signal } : {}),
            maxOutputTokens: 2048,
            temperature: 0,
            responseMimeType: "application/json",
            responseSchema: PLAN_SCHEMA,
            systemInstruction: {
              role: "system",
              parts: [{ text: PLAN_SYSTEM_PROMPT }],
            },
          },
        }),
        TIMEOUT.PLAN_GENERATION,
        'Plan generation timed out',
        signal,
      );

      const text = response.text;
      if (!text) {
        throw new Error("Plan generation returned an empty response.");
      }

      const jsonPayload = extractJsonObject(text, "plan");
　　　 const payload = parseJson<PlanPayload>(jsonPayload, "plan");
      const steps = (payload.steps ?? []).map((step, index) => ({
        id: (step.id || `S${index + 1}`).trim(),
        title: step.title.trim(),
        query: step.query.trim(),
        angle: step.angle.trim(),
        deliverable: step.deliverable.trim(),
      }));

      if (steps.length === 0) {
        throw new Error("Plan generation did not return any steps.");
      }

      return {
        primaryGoal: payload.primaryGoal?.trim() ?? query,
        rationale: payload.rationale?.trim() ?? "",
        steps,
        expectedInsights: payload.expectedInsights?.map((line) => line.trim()).filter(Boolean) ?? [],
      };
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      console.error(`Plan generation attempt ${attempt + 1} failed:`, lastError.message);

      if (attempt < RETRY.MAX_ATTEMPTS) {
        await new Promise((resolve) => setTimeout(resolve, RETRY.BASE_DELAY_MS * (attempt + 1)));
      }
    }
  }

  throw lastError ?? new Error("Plan generation failed after all retries");
}

async function gatherEvidence(
  query: string,
  step: DeepResearchPlanStep,
  locale: Locale,
  signal: AbortSignal | undefined,
  client: GoogleGenAI,
  model: string,
  registerSource: (candidate: SourceCandidate) => SourceReference,
): Promise<StepResult> {
  const userPrompt = localeDirective(
    locale,
    `You are executing step ${step.id} of a research mission that targets the question: "${query}". Focus strictly on this sub-query: "${step.query}". Use real-time web research to pull verifiable facts. Prioritize the most recent and up-to-date information available, giving more weight to newer sources. Return findings that advance the deliverable: ${step.deliverable}.`
  );

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= RETRY.MAX_ATTEMPTS; attempt++) {
    try {
      const response = await withTimeout(
        generateContentWithAdaptiveThinking({
          client,
          model,
          contents: createContent(userPrompt),
          config: {
            ...(signal ? { abortSignal: signal } : {}),
            topP: 0.9,
            maxOutputTokens: 2048,
            automaticFunctionCalling: {},
            tools: [{ googleSearch: {} }],
            systemInstruction: {
              role: "system",
              parts: [
                {
                  text:
                    `${EVIDENCE_SYSTEM_PROMPT}\nFormat your entire reply as valid JSON with the exact shape: {"summary": string, "findings": [{"heading": string, "insight": string, "evidence": string, "confidence"?: string, "sourceIds"?: number[]}]}.\nDo not include Markdown fences or any text before/after the JSON.\nUse sourceIds to reference 1-indexed citations emitted by the tool calls in the order they are returned.`,
                },
              ],
            },
          },
        }),
        TIMEOUT.EVIDENCE_GATHERING,
        `Evidence gathering timed out for step ${step.id}`,
        signal,
      );

      const text = response.text;
      if (!text || text.trim().length === 0) {
        console.error(`[gatherEvidence] Empty response for step ${step.id}`);
        throw new Error(`Evidence generation for step ${step.id} returned an empty response.`);
      }

      const jsonPayload = extractJsonObject(text, `evidence for step ${step.id}`);
      const payload = parseJson<EvidencePayload>(jsonPayload, `evidence for step ${step.id}`);
      const candidate = response.candidates?.[0];
      const citations = extractCitations(candidate);

      const indexToSource = new Map<number, SourceReference>();
      citations.forEach((citation) => {
        const ref = registerSource(citation);
        indexToSource.set(citation.index, ref);
      });

      const uniqueSources = Array.from(new Map(Array.from(indexToSource.values()).map((ref) => [ref.id, ref])).values());

      const findings: StepFinding[] = (payload.findings ?? []).map((finding) => {
        const sourceIds = Array.isArray(finding.sourceIds) ? finding.sourceIds : [];
        const references = sourceIds
          .map((id) => indexToSource.get(id - 1))
          .filter((ref): ref is SourceReference => Boolean(ref));

        return {
          heading: finding.heading.trim(),
          insight: finding.insight.trim(),
          evidence: finding.evidence.trim(),
          confidence: finding.confidence?.trim(),
          sources: Array.from(new Map(references.map((ref) => [ref.id, ref])).values()),
        };
      });

      return {
        stepId: step.id,
        title: step.title,
        summary: payload.summary?.trim() ?? "",
        queries: candidate?.groundingMetadata?.retrievalQueries ?? [],
        findings,
        sources: uniqueSources,
      };
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      console.error(`Attempt ${attempt + 1} failed for step ${step.id}:`, lastError.message);

      if (attempt < RETRY.MAX_ATTEMPTS) {
        await new Promise((resolve) => setTimeout(resolve, RETRY.BASE_DELAY_MS * (attempt + 1)));
      }
    }
  }

  // If all retries failed, return a minimal result instead of throwing
  console.error(`All retries failed for step ${step.id}. Returning minimal result.`);
  return {
    stepId: step.id,
    title: step.title,
    summary: `Evidence collection failed after ${RETRY.MAX_ATTEMPTS + 1} attempts: ${lastError?.message ?? "Unknown error"}`,
    queries: [],
    findings: [],
    sources: [],
  };
}

async function synthesizeReport(
  query: string,
  plan: DeepResearchPlan,
  steps: StepResult[],
  sources: SourceReference[],
  locale: Locale,
  signal: AbortSignal | undefined,
  client: GoogleGenAI,
  model: string,
): Promise<string> {
  const planOutline = plan.steps
    .map((step) => `- ${step.id}: ${step.title} — focus: ${step.angle}`)
    .join("\n");

  const findingsOutline = steps
    .map((step) => {
      const lines = step.findings.map((finding) => {
        const citations = finding.sources.map((source) => `[${source.id}]`).join(" ");
        return `  • ${finding.heading}: ${finding.insight} ${citations}`.trim();
      });
      return [`Step ${step.stepId} – ${step.title}:`, ...lines].join("\n");
    })
    .join("\n\n");

  const referencesCatalog = sources
    .map((source) => `[${source.id}] ${source.title}${source.domain ? ` — ${source.domain}` : ""} (${source.url})`)
    .join("\n");

  const userPrompt = localeDirective(
    locale,
    `Synthesize the DeepResearch investigation into a concise but thorough report. Cite sources with bracket numbers (e.g., [1]) that correspond to the provided catalog. Give priority and more weight to newer, more recent information when synthesizing findings.`,
  ) +
    `\n\nResearch question: ${query}\n\nPrimary goal: ${plan.primaryGoal}\nRationale: ${plan.rationale}\nExpected insights: ${plan.expectedInsights.join("; ")}\n\nPlan outline:\n${planOutline}\n\nFindings summary:\n${findingsOutline}\n\nSources catalog:\n${referencesCatalog}\n\nStructure the response with the following sections in order: Overview, Key Findings (bulleted), Contradictions, Open Questions, References. Ensure every factual claim is cited.`;

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= RETRY.MAX_ATTEMPTS; attempt++) {
    try {
      const response = await withTimeout(
        generateContentWithAdaptiveThinking({
          client,
          model,
          contents: createContent(userPrompt),
          config: {
            ...(signal ? { abortSignal: signal } : {}),
            topP: 0.9,
            maxOutputTokens: 2048,
            systemInstruction: {
              role: "system",
              parts: [{ text: FINAL_SYSTEM_PROMPT }],
            },
          },
        }),
        TIMEOUT.REPORT_SYNTHESIS,
        'Report synthesis timed out',
        signal,
      );

      return response.text ?? "";
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      console.error(`Report synthesis attempt ${attempt + 1} failed:`, lastError.message);

      if (attempt < RETRY.MAX_ATTEMPTS) {
        await new Promise((resolve) => setTimeout(resolve, RETRY.BASE_DELAY_MS * (attempt + 1)));
      }
    }
  }

  throw lastError ?? new Error("Report synthesis failed after all retries");
}

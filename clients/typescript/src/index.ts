// The General Forecast API client. Dependency-free: `fetch` from Node 20+ or
// the browser, the envelope every route carries, and typed errors.

import type {
  ApiError,
  Envelope,
  ErrorCode,
  Manifest,
  Model,
  ModelList,
  Predictor,
  PredictorList,
  Queries,
  ResourceKind,
  Result,
} from "./types.js";
import { ERROR_CODES, ROUTES, VERSION } from "./types.js";

export * from "./types.js";

export interface Options {
  /** an API key from `gf admin key create`; omit for an unauthenticated gf serve */
  apiKey?: string;
  /** the project every route is scoped to */
  project: string;
  /** `https://app.generalforecast.com` (the default) or a `gf serve` address */
  baseUrl?: string;
  /** replaces the global `fetch`, for tests and custom transports */
  fetch?: typeof fetch;
  /** makes request ids; defaults to a random `r-<32 hex>` */
  requestId?: () => string;
}

/** The ad hoc manifest without its rows: what `predict` adds `queries` and `context` to. */
export type ManifestFragment = Omit<Manifest, "context" | "queries"> & {
  context: Manifest["context"];
  queries: Manifest["queries"];
};

export interface PredictOptions {
  queries: Queries["queries"];
  context?: Queries["context"];
  asOf?: string;
}

/** An API error: the server's `error` object plus the request id from the envelope. */
export class GFError extends Error {
  readonly code: ErrorCode;
  readonly param: string | undefined;
  readonly hint: string | undefined;
  readonly limit: number | undefined;
  readonly actual: number | undefined;
  readonly requestId: string | undefined;
  readonly status: number;

  constructor(error: ApiError, status: number, requestId?: string) {
    super(error.message);
    this.name = "GFError";
    this.code = error.code;
    this.param = error.param;
    this.hint = error.hint;
    this.limit = error.limit;
    this.actual = error.actual;
    this.requestId = requestId;
    this.status = status;
  }

  /** The CLI's rendering: `error:` line, then `hint:` and the request id. */
  override toString(): string {
    const lines = [`error: ${this.message}`];
    if (this.hint !== undefined) lines.push(`  hint: ${this.hint}`);
    if (this.requestId !== undefined) lines.push(`  request ${this.requestId}`);
    return lines.join("\n");
  }
}

const DEFAULT_BASE_URL = "https://app.generalforecast.com";

function randomRequestId(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return "r-" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function isApiError(value: unknown): value is ApiError {
  if (typeof value !== "object" || value === null) return false;
  const error = value as Record<string, unknown>;
  return (
    typeof error.code === "string" &&
    (ERROR_CODES as readonly string[]).includes(error.code) &&
    typeof error.message === "string"
  );
}

type Reply<T> = Envelope & { result: T };

export class GeneralForecast {
  readonly project: string;
  readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly fetchImpl: typeof fetch;
  private readonly requestId: () => string;

  readonly predictors: Predictors;
  readonly models: Models;

  constructor(options: Options) {
    if (!options.project) throw new TypeError("project is required");
    this.project = options.project;
    this.baseUrl = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetch ?? globalThis.fetch;
    this.requestId = options.requestId ?? randomRequestId;
    this.predictors = new Predictors(this);
    this.models = new Models(this);
  }

  /** Score queries with an ad hoc predictor: a manifest fragment plus `context` and `queries` rows. */
  async predict(manifest: ManifestFragment): Promise<Result> {
    return this.call<Result>(
      ROUTES.predict,
      { project: this.project },
      { kind: "predictor", name: "ad-hoc" },
      manifest,
    );
  }

  /** `GET /healthz`: resolves when every served model is loaded and compiled, else throws GFError(ModelNotReady). */
  async health(): Promise<void> {
    return this.probe(ROUTES.health.path);
  }

  /** `GET /livez`: resolves while the server answers, whether or not its models are ready yet. */
  async live(): Promise<void> {
    return this.probe(ROUTES.live.path);
  }

  /** A probe outside the envelope: resolves on 200, else throws the error the server sent. */
  private async probe(path: string): Promise<void> {
    const response = await this.fetchImpl(this.baseUrl + path, { method: "GET" });
    if (response.status === 200) return;
    const body: unknown = await response.json().catch(() => undefined);
    const error = (body as { error?: unknown } | undefined)?.error;
    if (isApiError(error)) throw new GFError(error, response.status);
    throw new GFError(
      { code: "Internal", message: `GET ${path} answered ${response.status}` },
      response.status,
    );
  }

  /** Send one enveloped request and unwrap its reply. */
  async call<T>(
    route: { method: string; path: string },
    params: Record<string, string>,
    resource: { kind: ResourceKind; name: string },
    payload?: object,
  ): Promise<T> {
    const requestId = this.requestId();
    const path = route.path.replace(/\{([a-z_]+)\}/g, (_, key: string) => {
      const value = params[key];
      if (value === undefined) throw new TypeError(`route needs ${key}`);
      return encodeURIComponent(value);
    });
    const headers: Record<string, string> = {
      Accept: "application/json",
      "X-GF-Request-ID": requestId,
      "X-GF-Project": this.project,
    };
    if (this.apiKey !== undefined) headers.Authorization = `Bearer ${this.apiKey}`;
    const init: RequestInit = { method: route.method, headers };
    if (route.method !== "GET") {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify({
        api_version: VERSION,
        project: this.project,
        request_id: requestId,
        resource,
        payload: payload ?? {},
      });
    }
    const response = await this.fetchImpl(this.baseUrl + path, init);
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new GFError(
        { code: "Internal", message: `${route.method} ${path} answered ${response.status} without JSON` },
        response.status,
        requestId,
      );
    }
    const envelope = body as Partial<Envelope & { result?: T; error?: unknown }>;
    if (isApiError(envelope.error)) {
      throw new GFError(envelope.error, response.status, envelope.request_id ?? requestId);
    }
    if (
      response.status !== 200 ||
      envelope.api_version !== VERSION ||
      envelope.project !== this.project ||
      envelope.request_id !== requestId ||
      envelope.resource?.kind !== resource.kind ||
      envelope.resource?.name !== resource.name ||
      envelope.result === undefined
    ) {
      throw new GFError(
        { code: "Internal", message: `${route.method} ${path} answered with a reply that does not match the request` },
        response.status,
        requestId,
      );
    }
    return envelope.result;
  }
}

export class Predictors {
  constructor(private readonly client: GeneralForecast) {}

  /** Score queries with a declared predictor; `context` only when its manifest declares `context: {request}`. */
  async predict(name: string, options: PredictOptions): Promise<Result> {
    const payload: Queries = { queries: options.queries };
    if (options.context !== undefined) payload.context = options.context;
    if (options.asOf !== undefined) payload.as_of = options.asOf;
    return this.client.call<Result>(
      ROUTES["predictors.predict"],
      { project: this.client.project, name },
      { kind: "predictor", name },
      payload,
    );
  }

  async list(): Promise<Predictor[]> {
    const result = await this.client.call<PredictorList>(
      ROUTES["predictors.list"],
      { project: this.client.project },
      { kind: "predictor", name: "predictors" },
    );
    return result.predictors;
  }

  async get(name: string): Promise<Predictor> {
    return this.client.call<Predictor>(
      ROUTES["predictors.get"],
      { project: this.client.project, name },
      { kind: "predictor", name },
    );
  }
}

export class Models {
  constructor(private readonly client: GeneralForecast) {}

  async list(): Promise<Model[]> {
    const result = await this.client.call<ModelList>(
      ROUTES["models.list"],
      { project: this.client.project },
      { kind: "model", name: "models" },
    );
    return result.models;
  }

  /** One served model by its name without the `general-forecast/` prefix, such as `tabicl-v2-classifier`. */
  async get(name: string): Promise<Model> {
    return this.client.call<Model>(
      ROUTES["models.get"],
      { project: this.client.project, model: name },
      { kind: "model", name },
    );
  }
}

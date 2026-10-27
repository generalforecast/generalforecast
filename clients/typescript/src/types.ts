// The HTTP API's request and reply types, generated from
// api/openapi.yaml by api/generate.py. Do not edit.

export const VERSION = "gf/v1";

/** a resource name */
export type Name = string;

/**
 * a catalog model id, `general-forecast/<name>`, with an optional variant suffix
 * (`@4*2`) or digest pin (`@sha256:<digest>`)
 */
export type ModelId = string;

export type Digest = string;

export type Task = "classification" | "regression" | "forecast";

export type ResourceKind = "predictor" | "model" | "project" | "api-key" | "secret" | "context";

/** what the envelope is about; GET lists use the collection name */
export interface Resource {
  kind: ResourceKind;
  name: string;
}

/** the fields every request and reply carries */
export interface Envelope {
  api_version: "gf/v1";
  project: Name;
  request_id: string;
  resource: Resource;
}

/**
 * ValidationError 422, NotFound 404, Unauthorized 401, CapacityExceeded 422,
 * ModelNotReady 503, Internal 500, Conflict 409 (a compare-and-swap on a stale
 * generation)
 */
export const ERROR_CODES = ["ValidationError", "NotFound", "Unauthorized", "CapacityExceeded", "ModelNotReady", "Internal", "Conflict"] as const;
export type ErrorCode = (typeof ERROR_CODES)[number];

/**
 * `param` names the payload field (`target`, `inputs.plan`, `queries`); every limit
 * error carries `limit` and `actual` and names the model in `message`. The request id
 * is in the envelope.
 */
export interface ApiError {
  code: ErrorCode;
  message: string;
  reason?: string;
  param?: string;
  hint?: string;
  limit?: number;
  actual?: number;
}

export interface ErrorReply extends Envelope {
  error: ApiError;
}

/** one record; column names are keys, cells are scalars */
export type Row = Record<string, string | number | boolean | null>;

export type ColumnType = "number" | "category" | "text" | "boolean" | "date" | "timestamp";

/**
 * inputs by rule rather than by list: `exclude` removes columns from the inferred set,
 * `types` overrides inferred types; forecast predictors use `past` and `future` instead
 */
export interface Inputs {
  exclude?: string[];
  types?: Record<string, ColumnType>;
  past?: string[];
  future?: string[];
}

/**
 * which context rows a query sees; fractions below 1 are of the model's row cap,
 * integers are rows; `core: null` is the whole table
 */
export interface Selection {
  core?: number | null;
  neighbors?: number;
  kind?: "stratified" | "random";
  recent?: number | null;
}

/**
 * the ad hoc predictor: the gf.yaml fields, with `context` and `queries` as rows
 * instead of a context declaration; `name` and `resources` are file-only
 */
export interface Manifest {
  model: ModelId;
  target: string;
  id?: string;
  inputs?: string[] | Inputs;
  time?: string;
  series?: string[];
  frequency?: string | number;
  horizon?: number;
  quantiles?: number[];
  outputs?: Array<"label" | "value" | "probabilities" | "quantiles">;
  selection?: Selection;
  context: Row[];
  queries: Row[];
}

export interface ManifestRequest extends Envelope {
  payload: Manifest;
}

/**
 * the declared predictor payload; `context` only when the manifest declares `context:
 * {request: ...}`
 */
export interface Queries {
  queries: Row[];
  context?: Row[];
  as_of?: string;
}

export interface QueriesRequest extends Envelope {
  payload: Queries;
}

export type ContextKind = "snapshot" | "request" | "live";

/** the context a prediction actually used */
export interface ContextInfo {
  kind: ContextKind;
  rows: number;
  digest?: Digest;
  as_of?: string;
}

/**
 * the query's id under the manifest's id column name when it declares one (omitted when
 * it does not); `label` and `probabilities` for classification, `value` for regression,
 * `value` and `quantiles` for forecasts
 */
export interface Prediction {
  label?: string | number | boolean;
  value?: number;
  probabilities?: Record<string, number>;
  quantiles?: Record<string, number>;
  [key: string]: string | number | boolean | Record<string, number> | undefined;
}

export interface Result {
  model: ModelId;
  task: Task;
  context: ContextInfo;
  revision?: number;
  environment?: Name;
  predictions: Prediction[];
}

export interface ResultReply extends Envelope {
  result: Result;
}

/** the most a model accepts in one call */
export interface Limits {
  context_rows: number;
  queries: number;
  inputs: number;
}

/** a compiled program's shape */
export interface Shape {
  context_rows: number;
  queries: number;
  inputs: number;
  classes?: number;
}

export interface Input {
  name: string;
  type: ColumnType;
}

/** the context as gf.lock pins it */
export interface ContextLock {
  kind: ContextKind;
  rows?: number;
  query_rows?: number;
  digest?: Digest;
  live?: Name;
  max_rows?: number;
}

/** the selection as gf.lock pins it, in rows */
export interface ResolvedSelection {
  core: number | null;
  neighbors: number;
  kind: "stratified" | "random";
  recent?: number | null;
}

/** the resolved predictor gf check prints and gf.lock pins */
export interface Resolution {
  name: Name;
  model: ModelId;
  model_digest: Digest;
  task: Task;
  target: string;
  target_type: ColumnType;
  classes?: string[];
  id?: string;
  time?: string;
  series?: string[];
  inputs: Input[];
  context: ContextLock;
  selection: ResolvedSelection;
}

/** whether this server has the predictor's model loaded and compiled */
export interface Serving {
  loaded: boolean;
  compiled: boolean;
  warm_shapes: Shape[];
}

export type State = "pending" | "building" | "ready" | "failed" | "stopped";

/** a pointer to a revision; `generation` guards promote and rollback */
export interface Environment {
  predictor: Name;
  name: Name;
  revision: number | null;
  generation: number;
  runtime: Name;
  status: State;
}

export interface Predictor extends Resolution {
  limits: Limits;
  environments: Environment[];
  serving: Serving;
}

export interface PredictorReply extends Envelope {
  result: Predictor;
}

export interface PredictorList {
  predictors: Predictor[];
}

export interface PredictorListReply extends Envelope {
  result: PredictorList;
}

export interface Artifact {
  repository: string;
  revision: string;
  digest: Digest;
  license: string;
  digest_format?: string;
  files: Record<string, string>;
  [key: string]: unknown;
}

export interface Capability {
  name: string;
  state: "supported" | "unsupported";
  code?: string | null;
  reason?: string | null;
}

/**
 * the catalog's description of a served model: `model` is the resolved id with its
 * content digest, `limits` what one call may send here, `warm_shapes` the shapes
 * compiled at startup
 */
export interface Model {
  model: string;
  name: ModelId;
  task: Task;
  repository: string;
  revision: string;
  license: string;
  runtime: "cpu" | "gpu";
  artifact: Artifact;
  artifact_verification?: string;
  bounds: Record<string, number | number | null>;
  capabilities: Capability[];
  limits: Limits;
  warm_shapes: Shape[];
}

export interface ModelList {
  models: Model[];
}

export interface ModelReply extends Envelope {
  result: Model;
}

export interface ModelListReply extends Envelope {
  result: ModelList;
}

export interface Health {
  status: "ok";
}

export interface HealthError {
  error: ApiError;
}

export interface Empty {
}

export interface EmptyRequest extends Envelope {
  payload: Empty;
}

export interface Project {
  name: Name;
  organization: Name;
}

export interface ProjectReply extends Envelope {
  result: Project;
}

export interface ProjectList {
  projects: Project[];
  organizations: Name[];
}

export interface ProjectListReply extends Envelope {
  result: ProjectList;
}

export interface ProjectCreate {
  name: Name;
  organization: Name;
}

export interface ProjectCreateRequest extends Envelope {
  payload: ProjectCreate;
}

export type Permission = "predictors:read" | "predictors:write" | "predictors:predict" | "contexts:read" | "contexts:write" | "contexts:history" | "sources:read" | "sources:write" | "models:read" | "keys:read" | "keys:write" | "secrets:read" | "secrets:write";

/** `token` is present only in the create and rotate replies */
export interface Key {
  id: string;
  project: Name;
  name: Name;
  permissions: Permission[];
  created_at: number;
  expires_at: number;
  revoked_at: number | null;
  token?: string;
  replaces?: string;
}

export interface KeyReply extends Envelope {
  result: Key;
}

export interface KeyList {
  keys: Key[];
}

export interface KeyListReply extends Envelope {
  result: KeyList;
}

export interface KeyCreate {
  name: Name;
  permissions: Permission[];
  expires_in?: number;
}

export interface KeyCreateRequest extends Envelope {
  payload: KeyCreate;
}

export interface KeyRef {
  id: string;
}

export interface KeyRefRequest extends Envelope {
  payload: KeyRef;
}

export interface KeyRevoked {
  id: string;
  revoked: true;
}

export interface KeyRevokedReply extends Envelope {
  result: KeyRevoked;
}

/** metadata only; a value is never returned */
export interface Secret {
  project: Name;
  name: Name;
  version: number;
  created_at: number;
  updated_at: number;
}

export interface SecretReply extends Envelope {
  result: Secret;
}

export interface SecretList {
  secrets: Secret[];
}

export interface SecretListReply extends Envelope {
  result: SecretList;
}

export interface SecretCreate {
  name: Name;
  value: string;
}

export interface SecretCreateRequest extends Envelope {
  payload: SecretCreate;
}

export interface SecretRotate {
  name: Name;
  value: string;
  expected_version: number;
}

export interface SecretRotateRequest extends Envelope {
  payload: SecretRotate;
}

export interface SecretRef {
  name: Name;
}

export interface SecretRefRequest extends Envelope {
  payload: SecretRef;
}

export interface SecretDeleted {
  name: Name;
  deleted: true;
}

export interface SecretDeletedReply extends Envelope {
  result: SecretDeleted;
}

/** an immutable pushed predictor */
export interface Revision {
  predictor: Name;
  number: number;
  predictor_digest: Digest;
  model: ModelId;
  created_at: number;
  status: State;
}

export interface RevisionReply extends Envelope {
  result: Revision;
}

export interface RevisionList {
  revisions: Revision[];
}

export interface RevisionListReply extends Envelope {
  result: RevisionList;
}

export interface LockedModel {
  files: Record<string, Digest>;
}

/** gf.lock */
export interface Lock {
  lock_version: 1;
  predictor: Resolution;
  model: LockedModel;
}

export interface RevisionRegister {
  lock: Lock;
  snapshot: Digest | null;
}

export interface RevisionRegisterRequest extends Envelope {
  payload: RevisionRegister;
}

export interface RevisionDetails {
  revision: Revision;
  lock: Lock;
}

export interface RevisionDetailsReply extends Envelope {
  result: RevisionDetails;
}

export interface EnvironmentReply extends Envelope {
  result: Environment;
}

export interface EnvironmentList {
  environments: Environment[];
}

export interface EnvironmentListReply extends Envelope {
  result: EnvironmentList;
}

export interface EnvironmentCreate {
  name: Name;
  runtime: Name;
}

export interface EnvironmentCreateRequest extends Envelope {
  payload: EnvironmentCreate;
}

export interface Promote {
  revision: number;
  expected_generation: number;
}

export interface PromoteRequest extends Envelope {
  payload: Promote;
}

/**
 * the context an environment serves: `source` is `snapshot`, `request` or
 * `live/<context>`; the version fields and `freshness_lag_seconds` are set for live
 * contexts
 */
export interface ServingContext {
  source: string;
  version: number | string | null;
  rows: number;
  as_of: number | null;
  accepted_version: number | string | null;
  serving_version: number | string | null;
  freshness_lag_seconds: number | null;
  [key: string]: unknown;
}

/**
 * serving facts: `context` is the context the environment is serving now,
 * `accepted_version`/`serving_version` compare the newest accepted write of a live
 * context with the one being served, `fresh` says whether they agree (null without a
 * context)
 */
export interface Metrics {
  status: State;
  context: ServingContext | null;
  accepted_version: number | string | null;
  serving_version: number | string | null;
  fresh: boolean | null;
  requests: number;
  errors: number;
  mean_latency_seconds: number | null;
}

export interface MetricsReply extends Envelope {
  result: Metrics;
}

export interface LogEntry {
  at: number;
  code: string;
}

export interface Logs {
  logs: LogEntry[];
}

export interface LogsReply extends Envelope {
  result: Logs;
}

export interface PredictorDeleted {
  name: Name;
  deleted: true;
}

export interface PredictorDeletedReply extends Envelope {
  result: PredictorDeleted;
}

/** a hosted, appendable context */
export interface LiveContext {
  name: Name;
  inputs: string[];
  targets: Record<string, ColumnType>;
  max_rows: number;
  rows: number;
  labeled_rows: number;
  sequence: number;
  accepted_at: number | null;
}

export interface LiveContextReply extends Envelope {
  result: LiveContext;
}

export interface LiveContextList {
  contexts: LiveContext[];
}

export interface LiveContextListReply extends Envelope {
  result: LiveContextList;
}

export interface LiveContextCreate {
  name: Name;
  inputs: string[];
  targets: Record<string, ColumnType>;
  max_rows?: number;
}

export interface LiveContextCreateRequest extends Envelope {
  payload: LiveContextCreate;
}

export interface LiveContextAppend {
  write_id: string;
  rows: Row[];
}

export interface LiveContextAppendRequest extends Envelope {
  payload: LiveContextAppend;
}

export interface Label {
  id: string | number;
  value: string | number | boolean;
}

export interface LiveContextLabel {
  write_id: string;
  target: string;
  labels: Label[];
}

export interface LiveContextLabelRequest extends Envelope {
  payload: LiveContextLabel;
}

/** durable acceptance of an append or label write */
export interface Receipt {
  context: Name;
  write_id: string;
  operation: "append" | "label";
  rows: number;
  sequence: number;
  accepted_at: number;
  durably_accepted: boolean;
}

export interface ReceiptReply extends Envelope {
  result: Receipt;
}

export const ROUTES = {
  "predictors.predict": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/predict", request: "QueriesRequest", reply: "ResultReply" },
  "predict": { method: "POST", path: "/v1/projects/{project}/predict", request: "ManifestRequest", reply: "ResultReply" },
  "predictors.list": { method: "GET", path: "/v1/projects/{project}/predictors", request: "null", reply: "PredictorListReply" },
  "predictors.get": { method: "GET", path: "/v1/projects/{project}/predictors/{name}", request: "null", reply: "PredictorReply" },
  "models.list": { method: "GET", path: "/v1/projects/{project}/models", request: "null", reply: "ModelListReply" },
  "models.get": { method: "GET", path: "/v1/projects/{project}/models/{model}", request: "null", reply: "ModelReply" },
  "health": { method: "GET", path: "/healthz", request: "null", reply: "Health" },
  "ready": { method: "GET", path: "/readyz", request: "null", reply: "Health" },
  "live": { method: "GET", path: "/livez", request: "null", reply: "Health" },
  "projects.list": { method: "GET", path: "/v1/projects", request: "null", reply: "ProjectListReply" },
  "projects.create": { method: "POST", path: "/v1/projects", request: "ProjectCreateRequest", reply: "ProjectReply" },
  "keys.list": { method: "GET", path: "/v1/projects/{project}/keys", request: "null", reply: "KeyListReply" },
  "keys.create": { method: "POST", path: "/v1/projects/{project}/keys", request: "KeyCreateRequest", reply: "KeyReply" },
  "keys.revoke": { method: "POST", path: "/v1/projects/{project}/keys/{key}/revoke", request: "KeyRefRequest", reply: "KeyRevokedReply" },
  "keys.rotate": { method: "POST", path: "/v1/projects/{project}/keys/{key}/rotate", request: "KeyRefRequest", reply: "KeyReply" },
  "secrets.list": { method: "GET", path: "/v1/projects/{project}/secrets", request: "null", reply: "SecretListReply" },
  "secrets.create": { method: "POST", path: "/v1/projects/{project}/secrets", request: "SecretCreateRequest", reply: "SecretReply" },
  "secrets.get": { method: "GET", path: "/v1/projects/{project}/secrets/{secret}", request: "null", reply: "SecretReply" },
  "secrets.rotate": { method: "POST", path: "/v1/projects/{project}/secrets/{secret}/rotate", request: "SecretRotateRequest", reply: "SecretReply" },
  "secrets.delete": { method: "POST", path: "/v1/projects/{project}/secrets/{secret}/delete", request: "SecretRefRequest", reply: "SecretDeletedReply" },
  "revisions.list": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/revisions", request: "null", reply: "RevisionListReply" },
  "revisions.register": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/revisions", request: "RevisionRegisterRequest", reply: "RevisionReply" },
  "revisions.get": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/revisions/{number}", request: "null", reply: "RevisionDetailsReply" },
  "environments.list": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments", request: "null", reply: "EnvironmentListReply" },
  "environments.create": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/environments", request: "EnvironmentCreateRequest", reply: "EnvironmentReply" },
  "environments.get": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}", request: "null", reply: "EnvironmentReply" },
  "environments.promote": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/promote", request: "PromoteRequest", reply: "EnvironmentReply" },
  "environments.rollback": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/rollback", request: "PromoteRequest", reply: "EnvironmentReply" },
  "environments.metrics": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/metrics", request: "null", reply: "MetricsReply" },
  "environments.logs": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/logs", request: "null", reply: "LogsReply" },
  "predictors.delete": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/delete", request: "EmptyRequest", reply: "PredictorDeletedReply" },
  "contexts.list": { method: "GET", path: "/v1/projects/{project}/contexts", request: "null", reply: "LiveContextListReply" },
  "contexts.create": { method: "POST", path: "/v1/projects/{project}/contexts", request: "LiveContextCreateRequest", reply: "LiveContextReply" },
  "contexts.show": { method: "GET", path: "/v1/projects/{project}/contexts/{name}", request: "null", reply: "LiveContextReply" },
  "contexts.append": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/append", request: "LiveContextAppendRequest", reply: "ReceiptReply" },
  "contexts.label": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/label", request: "LiveContextLabelRequest", reply: "ReceiptReply" },
} as const;

export type RouteName = keyof typeof ROUTES;

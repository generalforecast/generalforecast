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

export type ResourceKind = "predictor" | "model" | "project" | "api-key" | "secret" | "context" | "source" | "workspace";

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

/**
 * a predictor by name, with its environments. Its resolution (the fields `Resolution`
 * names, `limits` and `serving`) is there once it resolves: always for a predictor gf
 * serve declares, and for a hosted one once it has a revision, from its latest.
 * `created_at` and `latest_revision` come from the hosted journal; a declared predictor
 * has neither.
 */
export interface Predictor {
  name: Name;
  created_at?: number;
  latest_revision?: number | null;
  environments: Environment[];
  model?: ModelId;
  model_digest?: Digest;
  task?: Task;
  target?: string;
  target_type?: ColumnType;
  classes?: string[];
  id?: string;
  time?: string;
  series?: string[];
  inputs?: Input[];
  context?: ContextLock;
  selection?: ResolvedSelection;
  limits?: Limits;
  serving?: Serving;
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
  state: "supported" | "unsupported" | "unconfigured";
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

export type Permission = "predictors:read" | "predictors:write" | "predictors:predict" | "contexts:read" | "contexts:write" | "contexts:history" | "sources:read" | "sources:write" | "models:read" | "keys:read" | "keys:write" | "secrets:read" | "secrets:write" | "members:read" | "members:write" | "sso:read" | "sso:write" | "audit:read" | "operators:read" | "operators:write" | "projects:read";

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
  id?: string;
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
  id?: string;
  types?: Record<string, ColumnType>;
  max_writes?: number;
  max_pending_labels?: number;
  pending_ttl_seconds?: number;
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

export interface PredictorCreate {
  name: Name;
}

export interface PredictorCreateRequest extends Envelope {
  payload: PredictorCreate;
}

export interface PredictorCreatedReply extends Envelope {
  result: PredictorCreate;
}

/** where a live context's rows come from, and how its columns map */
export interface ContextImport {
  context: Name;
  source: string;
  format: "csv" | "tsv" | "parquet" | "arrow" | "json" | "jsonl" | "table";
  credential_ref?: string | null;
  digest?: Digest | null;
  mapping?: string[][];
}

export interface ContextImportRequest extends Envelope {
  payload: ContextImport;
}

export type WriteOperation = "create_features" | "attach_labels" | "correct_features" | "correct_labels" | "resupply_labels" | "import_record" | "checkpoint";

/** one observation's components; corrections name the versions they replace */
export interface Observation {
  key: string;
  features?: Record<string, string | number | boolean | null>;
  labels?: Record<string, string | number | boolean>;
  event_at?: number;
  expected_feature_version?: number;
  expected_label_versions?: Record<string, number>;
}

/** the writes a client makes; imports and checkpoints are the server's */
export type MutationOperation = "create_features" | "attach_labels" | "correct_features" | "correct_labels" | "resupply_labels";

export interface Mutation {
  write_id: string;
  operation: MutationOperation;
  records: Observation[];
}

export interface MutationRequest extends Envelope {
  payload: Mutation;
}

export interface ComponentVersions {
  key: string;
  feature_version: number | null;
  label_versions: Record<string, number>;
}

/** a write's durable receipt, the same for every retry of its write id */
export interface AcceptedWrite {
  project: Name;
  context: Name;
  write_id: string;
  operation: WriteOperation;
  sequence: number;
  accepted_at: number;
  durably_accepted: boolean;
  changes: ComponentVersions[];
  retry_valid_until: number | null;
}

export interface AcceptedWriteReply extends Envelope {
  result: AcceptedWrite;
}

export interface ImportReceipt {
  context: Name;
  digest: Digest;
  rows: number;
  sequence: number;
  accepted_at: number;
  receipt: AcceptedWrite;
  mode: "explicit-refresh";
  continuous: false;
}

export interface ImportReceiptReply extends Envelope {
  result: ImportReceipt;
}

export interface ContextSnapshotQuery {
  target: string;
  knowledge_at?: number | null;
  event_before?: number | null;
}

export interface ContextSnapshotQueryRequest extends Envelope {
  payload: ContextSnapshotQuery;
}

export interface SnapshotRow {
  key: string;
  features: Record<string, string | number | boolean | null>;
  label: string | number | boolean;
  feature_version: number;
  label_version: number;
  feature_accepted_at: number;
  label_accepted_at: number;
  event_at: number | null;
}

export interface ContextSnapshot {
  project: Name;
  context: Name;
  target: string;
  features: string[];
  through_sequence: number;
  knowledge_at: number;
  event_before: number | null;
  rows: SnapshotRow[];
  source_wait_count: number;
  pending_processing_count: number;
  oldest_unprocessed_at: number | null;
}

export interface ContextSnapshotReply extends Envelope {
  result: ContextSnapshot;
}

/**
 * which columns of a pinned file are an observation's key, features and labels; empty
 * for JSONL already in that shape
 */
export interface SourceMapping {
  key?: string;
  features?: Record<string, string>;
  labels?: Record<string, string>;
  event_at?: string;
  label_types?: Record<string, "boolean" | "numeric">;
}

export interface SourcePinSpec {
  path: string;
  format: "jsonl" | "csv";
  mapping: SourceMapping;
}

export interface SourcePinRequest extends Envelope {
  payload: SourcePinSpec;
}

export interface SourcePin {
  source_id: Name;
  digest: Digest;
  size_bytes: number;
  format: "jsonl" | "csv";
  mapping: SourceMapping;
}

export interface SourcePinReply extends Envelope {
  result: SourcePin;
}

export interface SourceList {
  sources: SourcePin[];
}

export interface SourceListReply extends Envelope {
  result: SourceList;
}

export interface UploadReceipt {
  digest: Digest;
  rows: number;
}

export interface UploadReceiptReply extends Envelope {
  result: UploadReceipt;
}

/** a permission keys may hold; one that is not enforced authorizes nothing yet */
export interface PermissionGrant {
  name: Permission;
  description: string;
  enforced: boolean;
  roles: Array<"admin" | "editor" | "viewer" | "scorer">;
}

export interface KeyIdentity {
  id: string;
  name: Name;
  project: Name;
  expires_at: number;
}

export interface WorkspaceIdentity {
  id: string;
  display_name: string;
}

/** the credential's kind and id, with what the server knows of it */
export interface Principal {
  kind: string;
  id: string;
  [key: string]: unknown;
}

export interface AuthStatus {
  authenticated: true;
  principal: Principal;
  google: "configured" | "unconfigured";
  sign_in: Array<"google" | "microsoft" | "github">;
  secrets: "configured" | "unconfigured";
  capabilities: Capability[];
  permission_inventory: PermissionGrant[];
  bootstrap: "unconfigured" | "available" | "consumed";
  key?: KeyIdentity;
  email?: string | null;
  machine?: string | null;
  workspace?: WorkspaceIdentity | null;
}

export interface AuthStatusReply extends Envelope {
  result: AuthStatus;
}

export interface LoggedOut {
  revoked: string;
}

export interface LoggedOutReply extends Envelope {
  result: LoggedOut;
}

export interface Organization {
  name: Name;
  account: Name;
}

export interface OrganizationReply extends Envelope {
  result: Organization;
}

export interface OrganizationCreateRequest extends Envelope {
  payload: Organization;
}

export interface OrganizationList {
  organizations: Organization[];
}

export interface OrganizationListReply extends Envelope {
  result: OrganizationList;
}

/** the sign-in provider a membership's subject belongs to */
export type Issuer = "https://accounts.google.com" | "https://login.microsoftonline.com" | "https://github.com";

export interface Membership {
  issuer: Issuer;
  subject: string;
  scope_kind: "account" | "organization" | "project";
  scope_id: Name;
  role: "admin" | "editor" | "viewer" | "scorer";
}

export interface MembershipAddRequest extends Envelope {
  payload: Membership;
}

export interface MembershipRef {
  issuer: Issuer;
  subject: string;
  scope_kind: "account" | "organization" | "project";
  scope_id: Name;
}

export interface MembershipRemoveRequest extends Envelope {
  payload: MembershipRef;
}

export interface MembershipChanged {
  issuer: Issuer;
  subject: string;
  scope_kind: "account" | "organization" | "project";
  scope_id: Name;
  removed: boolean;
}

export interface MembershipChangedReply extends Envelope {
  result: MembershipChanged;
}

export interface MembershipList {
  memberships: Membership[];
}

export interface MembershipListReply extends Envelope {
  result: MembershipList;
}

export interface Sso {
  organization: Name;
  provider: "google";
  providers: Array<"google" | "microsoft" | "github">;
  enabled: boolean;
  configured: boolean;
  membership_required: true;
}

export interface SsoReply extends Envelope {
  result: Sso;
}

export interface SsoConfigure {
  enabled: boolean;
}

export interface SsoConfigureRequest extends Envelope {
  payload: SsoConfigure;
}

export interface AuditEvent {
  sequence: number;
  at: number;
  actor: string;
  project: Name;
  action: string;
  target: string;
  outcome: string;
}

export interface AuditLog {
  events: AuditEvent[];
}

export interface AuditLogReply extends Envelope {
  result: AuditLog;
}

export interface CliLoginStart {
  machine: string;
}

export interface CliLoginGrant {
  device_code: string;
  user_code: string;
  verification_uri: string;
  verification_uri_complete: string;
  expires_in: number;
  interval: number;
}

export interface CliLoginPoll {
  device_code: string;
}

export interface CliLoginToken {
  access_token: string;
  token_type: "bearer";
  key: KeyIdentity;
  email: string;
  workspace: WorkspaceIdentity;
  project: Name;
}

/**
 * why a CLI login cannot go on, outside the envelope as the login is. `reason` says
 * which: `authorization_pending` (`Unauthorized`, poll again), `slow_down`
 * (`ValidationError`, poll less often), `expired_token` and `access_denied`
 * (`Unauthorized`, start again) or `rate_limited` (`CapacityExceeded`).
 */
export interface LoginErrorReply {
  error: ApiError;
}

/** a workspace member's role; `admin` manages the workspace */
export type WorkspaceRole = "admin" | "editor" | "viewer" | "scorer";

export type JoinRequestId = string;

export type InvitationId = string;

export type WaitlistId = string;

/** a workspace's display name, without leading or trailing spaces */
export type WorkspaceName = string;

/**
 * the code `gf login` prints, `XXXX-XXXX` from BCDFGHJKLMNPQRSTVWXZ, taken in any case
 * and with or without the dash
 */
export type UserCode = string;

/** a workspace the caller belongs to: its role there and the projects it may use */
export interface WorkspaceSummary {
  workspace: Name;
  display_name: string;
  role: WorkspaceRole | null;
  admin: boolean;
  default_project: Name | null;
  projects: Name[];
}

export interface WorkspaceDetails extends WorkspaceSummary {
  organization: Name;
  created_at: number;
}

/** a request to join a workspace, as the person who asked sees it */
export interface JoinRequest {
  id: JoinRequestId;
  workspace: Name;
  display_name: string;
  state: "pending" | "approved" | "denied" | "cancelled";
  created_at: number;
  decided_at: number | null;
}

/** a request to join a workspace, as its admins see it */
export interface JoinRequestAdmin {
  id: JoinRequestId;
  workspace: Name;
  subject: string;
  email: string;
  name: string | null;
  domain: string;
  state: "pending" | "approved" | "denied" | "cancelled";
  role: WorkspaceRole | null;
  created_at: number;
  decided_at: number | null;
  decided_by: string | null;
}

/** the signed-in person's verified email, all null until they sign in again */
export interface OnboardingProfile {
  email: string | null;
  email_domain: string | null;
  display_name: string | null;
  personal_domain: boolean | null;
}

/** a workspace the person's email domain may ask to join */
export interface DiscoveredWorkspace {
  workspace: Name;
  display_name: string;
  domain: string;
  request: JoinRequest | null;
}

/**
 * where a signed-in person stands: their workspaces, those their domain may join, and
 * whether they may create one
 */
export interface Onboarding {
  state: "member" | "onboarding" | "reauthentication_required";
  profile: OnboardingProfile;
  workspaces: WorkspaceSummary[];
  discovered: DiscoveredWorkspace[];
  requests: JoinRequest[];
  can_create_workspace: boolean;
  signup: "open" | "waitlist" | "pending";
  workspace_name: string | null;
  sign_in_url: string;
}

export interface WorkspaceList {
  workspaces: WorkspaceSummary[];
}

export interface WorkspaceCreate {
  display_name: WorkspaceName;
}

/** `created` is false when the request id already made this workspace */
export interface WorkspaceCreated {
  workspace: WorkspaceSummary;
  organization: Name;
  created: boolean;
}

export interface WorkspaceRef {
  workspace: Name;
}

export interface WorkspaceRename {
  workspace: Name;
  display_name: WorkspaceName;
}

/**
 * `workspace` is set once the caller is a member, `request` while a join request is
 * pending or was denied lately
 */
export interface WorkspaceJoin {
  status: "already_member" | "pending" | "denied";
  workspace: WorkspaceSummary | null;
  request: JoinRequest | null;
}

/** one membership in the workspace, its organizations or projects */
export interface WorkspaceMember {
  issuer: string;
  subject: string;
  email: string | null;
  name: string | null;
  scope_kind: "account" | "organization" | "project";
  scope_id: string;
  role: WorkspaceRole;
}

export interface WorkspaceMemberList {
  members: WorkspaceMember[];
}

export interface MemberRoleChange {
  workspace: Name;
  issuer: string;
  subject: string;
  role: WorkspaceRole;
}

export interface MemberRole {
  issuer: string;
  subject: string;
  role: WorkspaceRole;
}

export interface MemberRoleChanged {
  member: MemberRole;
}

export interface MemberRemove {
  workspace: Name;
  issuer: string;
  subject: string;
}

export interface MemberGone {
  issuer: string;
  subject: string;
  removed: true;
}

export interface MemberRemoved {
  member: MemberGone;
}

/** an invitation to a workspace; `state` is `expired` once a pending one lapses */
export interface Invitation {
  id: InvitationId;
  workspace: Name;
  email: string;
  role: WorkspaceRole;
  state: "pending" | "expired" | "accepted" | "revoked";
  created_at: number;
  expires_at: number;
  created_by: string;
  accepted_at: number | null;
  revoked_at: number | null;
}

export interface InvitationList {
  invitations: Invitation[];
}

/** `role` is `viewer` and `expires_in` a week unless given */
export interface InvitationCreate {
  workspace: Name;
  email: string;
  role?: WorkspaceRole;
  expires_in?: number;
}

/**
 * `token` and `invitation_url` only when the server has no mailer and `delivery` is
 * `link`; `replaces` names a pending invitation to the same email it revoked
 */
export interface InvitationCreated {
  invitation: Invitation;
  delivery: "email" | "link";
  replaces: InvitationId | null;
  token?: string;
  invitation_url?: string;
}

export interface InvitationRevoke {
  workspace: Name;
  id: InvitationId;
}

export interface InvitationView {
  invitation: Invitation;
}

/** the token from the invitation link */
export interface InvitationToken {
  token: string;
}

export interface InvitationPreview {
  workspace: Name;
  display_name: string;
  role: WorkspaceRole;
  expires_at: number;
}

export interface InvitationAccepted {
  status: "joined" | "already_member";
  workspace: WorkspaceSummary;
}

export interface JoinRequestRef {
  id: JoinRequestId;
}

export interface JoinRequestView {
  request: JoinRequest;
}

export interface JoinRequestList {
  requests: JoinRequestAdmin[];
}

/** `role` is `viewer` unless given */
export interface JoinRequestApprove {
  workspace: Name;
  id: JoinRequestId;
  role?: WorkspaceRole;
}

export interface JoinRequestDeny {
  workspace: Name;
  id: JoinRequestId;
}

/** the account membership an approval granted, or the one the person already held */
export interface WorkspaceMembership {
  issuer: string;
  subject: string;
  scope_kind: "account";
  scope_id: Name;
  role: WorkspaceRole;
}

export interface JoinRequestApproved {
  request: JoinRequestAdmin;
  membership: WorkspaceMembership;
}

export interface JoinRequestDenied {
  request: JoinRequestAdmin;
}

/** an email domain whose people may ask to join the workspace */
export interface JoinDomain {
  workspace: Name;
  domain: string;
  enabled_by: string;
  enabled_at: number;
}

/**
 * `email_domain` is the caller's own company domain, the one it may enable; null for a
 * personal one
 */
export interface JoinDomainList {
  domains: JoinDomain[];
  email_domain: string | null;
}

/** enabling takes only the domain of the admin's own verified email */
export interface JoinDomainConfigure {
  workspace: Name;
  domain: string;
  enabled: boolean;
}

export interface JoinDomainConfigured {
  workspace: Name;
  domain: string;
  enabled: boolean;
}

/** `display_name` is the workspace the person means to create */
export interface WaitlistJoin {
  display_name?: WorkspaceName;
}

export interface WaitlistStatus {
  status: "approved" | "pending";
  message: string | null;
}

/**
 * a waitlist sign-up; approving one from a company domain approves the whole domain
 */
export interface WaitlistEntry {
  id: WaitlistId;
  email: string;
  display_name: string | null;
  state: "pending" | "approved";
  approves: "person" | "domain";
  domain: string;
  created_at: number;
  decided_at: number | null;
  workspace_name: string | null;
}

export interface WaitlistRef {
  id: WaitlistId;
}

export interface WaitlistEntryView {
  entry: WaitlistEntry;
}

export interface CliLoginCode {
  user_code: UserCode;
}

export interface CliLoginWorkspace {
  workspace: Name;
  display_name: string;
}

/**
 * what the sign-in page confirms: the machine asking and the workspaces with a project
 * the key may serve
 */
export interface CliLoginPreview {
  user_code: string;
  machine: string;
  expires_at: number;
  email: string;
  workspaces: CliLoginWorkspace[];
}

export interface CliLoginApprove {
  user_code: UserCode;
  workspace: Name;
}

export interface CliLoginGranted {
  id: Name;
  display_name: string;
}

export interface CliLoginApproved {
  user_code: string;
  state: "approved";
  workspace: CliLoginGranted;
}

export interface CliLoginDenied {
  user_code: string;
  state: "denied";
}

export interface CliLoginApproveRequest extends Envelope {
  payload: CliLoginApprove;
}

export interface CliLoginCodeRequest extends Envelope {
  payload: CliLoginCode;
}

export interface InvitationCreateRequest extends Envelope {
  payload: InvitationCreate;
}

export interface InvitationRevokeRequest extends Envelope {
  payload: InvitationRevoke;
}

export interface InvitationTokenRequest extends Envelope {
  payload: InvitationToken;
}

export interface JoinDomainConfigureRequest extends Envelope {
  payload: JoinDomainConfigure;
}

export interface JoinRequestApproveRequest extends Envelope {
  payload: JoinRequestApprove;
}

export interface JoinRequestDenyRequest extends Envelope {
  payload: JoinRequestDeny;
}

export interface JoinRequestRefRequest extends Envelope {
  payload: JoinRequestRef;
}

export interface MemberRemoveRequest extends Envelope {
  payload: MemberRemove;
}

export interface MemberRoleChangeRequest extends Envelope {
  payload: MemberRoleChange;
}

export interface WaitlistJoinRequest extends Envelope {
  payload: WaitlistJoin;
}

export interface WaitlistRefRequest extends Envelope {
  payload: WaitlistRef;
}

export interface WorkspaceCreateRequest extends Envelope {
  payload: WorkspaceCreate;
}

export interface WorkspaceRefRequest extends Envelope {
  payload: WorkspaceRef;
}

export interface WorkspaceRenameRequest extends Envelope {
  payload: WorkspaceRename;
}

export interface CliLoginApprovedReply extends Envelope {
  result: CliLoginApproved;
}

export interface CliLoginDeniedReply extends Envelope {
  result: CliLoginDenied;
}

export interface CliLoginPreviewReply extends Envelope {
  result: CliLoginPreview;
}

export interface InvitationAcceptedReply extends Envelope {
  result: InvitationAccepted;
}

export interface InvitationCreatedReply extends Envelope {
  result: InvitationCreated;
}

export interface InvitationListReply extends Envelope {
  result: InvitationList;
}

export interface InvitationPreviewReply extends Envelope {
  result: InvitationPreview;
}

export interface InvitationViewReply extends Envelope {
  result: InvitationView;
}

export interface JoinDomainConfiguredReply extends Envelope {
  result: JoinDomainConfigured;
}

export interface JoinDomainListReply extends Envelope {
  result: JoinDomainList;
}

export interface JoinRequestApprovedReply extends Envelope {
  result: JoinRequestApproved;
}

export interface JoinRequestDeniedReply extends Envelope {
  result: JoinRequestDenied;
}

export interface JoinRequestListReply extends Envelope {
  result: JoinRequestList;
}

export interface JoinRequestViewReply extends Envelope {
  result: JoinRequestView;
}

export interface MemberRemovedReply extends Envelope {
  result: MemberRemoved;
}

export interface MemberRoleChangedReply extends Envelope {
  result: MemberRoleChanged;
}

export interface OnboardingReply extends Envelope {
  result: Onboarding;
}

export interface WaitlistEntryViewReply extends Envelope {
  result: WaitlistEntryView;
}

export interface WaitlistStatusReply extends Envelope {
  result: WaitlistStatus;
}

export interface WorkspaceCreatedReply extends Envelope {
  result: WorkspaceCreated;
}

export interface WorkspaceDetailsReply extends Envelope {
  result: WorkspaceDetails;
}

export interface WorkspaceJoinReply extends Envelope {
  result: WorkspaceJoin;
}

export interface WorkspaceListReply extends Envelope {
  result: WorkspaceList;
}

export interface WorkspaceMemberListReply extends Envelope {
  result: WorkspaceMemberList;
}

export interface WorkspaceSummaryReply extends Envelope {
  result: WorkspaceSummary;
}

export const ROUTES = {
  "predictors.predict": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/predict", request: "QueriesRequest", reply: "ResultReply", body: "application/json" },
  "predict": { method: "POST", path: "/v1/projects/{project}/predict", request: "ManifestRequest", reply: "ResultReply", body: "application/json" },
  "predictors.list": { method: "GET", path: "/v1/projects/{project}/predictors", request: "null", reply: "PredictorListReply", body: null },
  "predictors.create": { method: "POST", path: "/v1/projects/{project}/predictors", request: "PredictorCreateRequest", reply: "PredictorCreatedReply", body: "application/json" },
  "predictors.get": { method: "GET", path: "/v1/projects/{project}/predictors/{name}", request: "null", reply: "PredictorReply", body: null },
  "models.list": { method: "GET", path: "/v1/projects/{project}/models", request: "null", reply: "ModelListReply", body: null },
  "models.get": { method: "GET", path: "/v1/projects/{project}/models/{model}", request: "null", reply: "ModelReply", body: null },
  "health": { method: "GET", path: "/healthz", request: "null", reply: "Health", body: null },
  "ready": { method: "GET", path: "/readyz", request: "null", reply: "Health", body: null },
  "live": { method: "GET", path: "/livez", request: "null", reply: "Health", body: null },
  "projects.list": { method: "GET", path: "/v1/projects", request: "null", reply: "ProjectListReply", body: null },
  "projects.create": { method: "POST", path: "/v1/projects", request: "ProjectCreateRequest", reply: "ProjectReply", body: "application/json" },
  "keys.list": { method: "GET", path: "/v1/projects/{project}/keys", request: "null", reply: "KeyListReply", body: null },
  "keys.create": { method: "POST", path: "/v1/projects/{project}/keys", request: "KeyCreateRequest", reply: "KeyReply", body: "application/json" },
  "keys.revoke": { method: "POST", path: "/v1/projects/{project}/keys/{key}/revoke", request: "KeyRefRequest", reply: "KeyRevokedReply", body: "application/json" },
  "keys.rotate": { method: "POST", path: "/v1/projects/{project}/keys/{key}/rotate", request: "KeyRefRequest", reply: "KeyReply", body: "application/json" },
  "secrets.list": { method: "GET", path: "/v1/projects/{project}/secrets", request: "null", reply: "SecretListReply", body: null },
  "secrets.create": { method: "POST", path: "/v1/projects/{project}/secrets", request: "SecretCreateRequest", reply: "SecretReply", body: "application/json" },
  "secrets.get": { method: "GET", path: "/v1/projects/{project}/secrets/{secret}", request: "null", reply: "SecretReply", body: null },
  "secrets.rotate": { method: "POST", path: "/v1/projects/{project}/secrets/{secret}/rotate", request: "SecretRotateRequest", reply: "SecretReply", body: "application/json" },
  "secrets.delete": { method: "POST", path: "/v1/projects/{project}/secrets/{secret}/delete", request: "SecretRefRequest", reply: "SecretDeletedReply", body: "application/json" },
  "revisions.list": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/revisions", request: "null", reply: "RevisionListReply", body: null },
  "revisions.register": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/revisions", request: "RevisionRegisterRequest", reply: "RevisionReply", body: "application/json" },
  "revisions.get": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/revisions/{number}", request: "null", reply: "RevisionDetailsReply", body: null },
  "environments.list": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments", request: "null", reply: "EnvironmentListReply", body: null },
  "environments.create": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/environments", request: "EnvironmentCreateRequest", reply: "EnvironmentReply", body: "application/json" },
  "environments.get": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}", request: "null", reply: "EnvironmentReply", body: null },
  "environments.promote": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/promote", request: "PromoteRequest", reply: "EnvironmentReply", body: "application/json" },
  "environments.rollback": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/rollback", request: "PromoteRequest", reply: "EnvironmentReply", body: "application/json" },
  "environments.metrics": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/metrics", request: "null", reply: "MetricsReply", body: null },
  "environments.logs": { method: "GET", path: "/v1/projects/{project}/predictors/{name}/environments/{environment}/logs", request: "null", reply: "LogsReply", body: null },
  "predictors.delete": { method: "POST", path: "/v1/projects/{project}/predictors/{name}/delete", request: "EmptyRequest", reply: "PredictorDeletedReply", body: "application/json" },
  "contexts.list": { method: "GET", path: "/v1/projects/{project}/contexts", request: "null", reply: "LiveContextListReply", body: null },
  "contexts.create": { method: "POST", path: "/v1/projects/{project}/contexts", request: "LiveContextCreateRequest", reply: "LiveContextReply", body: "application/json" },
  "contexts.show": { method: "GET", path: "/v1/projects/{project}/contexts/{name}", request: "null", reply: "LiveContextReply", body: null },
  "contexts.append": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/append", request: "LiveContextAppendRequest", reply: "ReceiptReply", body: "application/json" },
  "contexts.label": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/label", request: "LiveContextLabelRequest", reply: "ReceiptReply", body: "application/json" },
  "contexts.imports": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/imports", request: "ContextImportRequest", reply: "ImportReceiptReply", body: "application/json" },
  "contexts.refresh": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/refresh", request: "EmptyRequest", reply: "ImportReceiptReply", body: "application/json" },
  "contexts.mutate": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/mutations", request: "MutationRequest", reply: "AcceptedWriteReply", body: "application/json" },
  "contexts.snapshot": { method: "POST", path: "/v1/projects/{project}/contexts/{name}/snapshot", request: "ContextSnapshotQueryRequest", reply: "ContextSnapshotReply", body: "application/json" },
  "sources.list": { method: "GET", path: "/v1/projects/{project}/sources", request: "null", reply: "SourceListReply", body: null },
  "sources.pin": { method: "POST", path: "/v1/projects/{project}/sources", request: "SourcePinRequest", reply: "SourcePinReply", body: "application/json" },
  "sources.get": { method: "GET", path: "/v1/projects/{project}/sources/{source}", request: "null", reply: "SourcePinReply", body: null },
  "snapshots.upload": { method: "POST", path: "/v1/projects/{project}/snapshots", request: "null", reply: "UploadReceiptReply", body: "application/vnd.apache.arrow.stream" },
  "auth.status": { method: "GET", path: "/v1/auth/status", request: "null", reply: "AuthStatusReply", body: null },
  "auth.logout": { method: "POST", path: "/v1/auth/logout", request: "EmptyRequest", reply: "LoggedOutReply", body: "application/json" },
  "workspace.get": { method: "GET", path: "/v1/workspace", request: "null", reply: "ProjectListReply", body: null },
  "organizations.list": { method: "GET", path: "/v1/organizations", request: "null", reply: "OrganizationListReply", body: null },
  "organizations.create": { method: "POST", path: "/v1/organizations", request: "OrganizationCreateRequest", reply: "OrganizationReply", body: "application/json" },
  "memberships.list": { method: "GET", path: "/v1/projects/{project}/memberships", request: "null", reply: "MembershipListReply", body: null },
  "memberships.add": { method: "POST", path: "/v1/projects/{project}/memberships", request: "MembershipAddRequest", reply: "MembershipChangedReply", body: "application/json" },
  "memberships.remove": { method: "POST", path: "/v1/projects/{project}/memberships/remove", request: "MembershipRemoveRequest", reply: "MembershipChangedReply", body: "application/json" },
  "sso.get": { method: "GET", path: "/v1/projects/{project}/sso", request: "null", reply: "SsoReply", body: null },
  "sso.configure": { method: "POST", path: "/v1/projects/{project}/sso", request: "SsoConfigureRequest", reply: "SsoReply", body: "application/json" },
  "audit.list": { method: "GET", path: "/v1/projects/{project}/audit", request: "null", reply: "AuditLogReply", body: null },
  "logins.start": { method: "POST", path: "/v1/cli-logins", request: "CliLoginStart", reply: "CliLoginGrant", body: "application/json" },
  "logins.token": { method: "POST", path: "/v1/cli-logins/token", request: "CliLoginPoll", reply: "CliLoginToken", body: "application/json" },
  "onboarding.get": { method: "GET", path: "/v1/onboarding", request: "null", reply: "OnboardingReply", body: null },
  "workspaces.list": { method: "GET", path: "/v1/workspaces", request: "null", reply: "WorkspaceListReply", body: null },
  "workspaces.create": { method: "POST", path: "/v1/workspaces", request: "WorkspaceCreateRequest", reply: "WorkspaceCreatedReply", body: "application/json" },
  "workspaces.get": { method: "GET", path: "/v1/workspaces/{workspace}", request: "null", reply: "WorkspaceDetailsReply", body: null },
  "workspaces.update": { method: "POST", path: "/v1/workspaces/{workspace}", request: "WorkspaceRenameRequest", reply: "WorkspaceSummaryReply", body: "application/json" },
  "workspaces.join": { method: "POST", path: "/v1/workspaces/{workspace}/join", request: "WorkspaceRefRequest", reply: "WorkspaceJoinReply", body: "application/json" },
  "workspaces.members": { method: "GET", path: "/v1/workspaces/{workspace}/members", request: "null", reply: "WorkspaceMemberListReply", body: null },
  "members.update": { method: "POST", path: "/v1/workspaces/{workspace}/members/role", request: "MemberRoleChangeRequest", reply: "MemberRoleChangedReply", body: "application/json" },
  "members.remove": { method: "POST", path: "/v1/workspaces/{workspace}/members/remove", request: "MemberRemoveRequest", reply: "MemberRemovedReply", body: "application/json" },
  "invitations.list": { method: "GET", path: "/v1/workspaces/{workspace}/invitations", request: "null", reply: "InvitationListReply", body: null },
  "invitations.create": { method: "POST", path: "/v1/workspaces/{workspace}/invitations", request: "InvitationCreateRequest", reply: "InvitationCreatedReply", body: "application/json" },
  "invitations.revoke": { method: "POST", path: "/v1/workspaces/{workspace}/invitations/{id}/revoke", request: "InvitationRevokeRequest", reply: "InvitationViewReply", body: "application/json" },
  "requests.list": { method: "GET", path: "/v1/workspaces/{workspace}/join-requests", request: "null", reply: "JoinRequestListReply", body: null },
  "requests.approve": { method: "POST", path: "/v1/workspaces/{workspace}/join-requests/{id}/approve", request: "JoinRequestApproveRequest", reply: "JoinRequestApprovedReply", body: "application/json" },
  "requests.deny": { method: "POST", path: "/v1/workspaces/{workspace}/join-requests/{id}/deny", request: "JoinRequestDenyRequest", reply: "JoinRequestDeniedReply", body: "application/json" },
  "domains.list": { method: "GET", path: "/v1/workspaces/{workspace}/domains", request: "null", reply: "JoinDomainListReply", body: null },
  "domains.configure": { method: "POST", path: "/v1/workspaces/{workspace}/domains", request: "JoinDomainConfigureRequest", reply: "JoinDomainConfiguredReply", body: "application/json" },
  "invitations.preview": { method: "POST", path: "/v1/invitations/preview", request: "InvitationTokenRequest", reply: "InvitationPreviewReply", body: "application/json" },
  "invitations.accept": { method: "POST", path: "/v1/invitations/accept", request: "InvitationTokenRequest", reply: "InvitationAcceptedReply", body: "application/json" },
  "requests.cancel": { method: "POST", path: "/v1/join-requests/{id}/cancel", request: "JoinRequestRefRequest", reply: "JoinRequestViewReply", body: "application/json" },
  "waitlist.join": { method: "POST", path: "/v1/waitlist", request: "WaitlistJoinRequest", reply: "WaitlistStatusReply", body: "application/json" },
  "waitlist.get": { method: "GET", path: "/v1/waitlist/{id}", request: "null", reply: "WaitlistEntryViewReply", body: null },
  "waitlist.approve": { method: "POST", path: "/v1/waitlist/{id}/approve", request: "WaitlistRefRequest", reply: "WaitlistEntryViewReply", body: "application/json" },
  "device.preview": { method: "POST", path: "/v1/cli-logins/preview", request: "CliLoginCodeRequest", reply: "CliLoginPreviewReply", body: "application/json" },
  "device.approve": { method: "POST", path: "/v1/cli-logins/approve", request: "CliLoginApproveRequest", reply: "CliLoginApprovedReply", body: "application/json" },
  "device.deny": { method: "POST", path: "/v1/cli-logins/deny", request: "CliLoginCodeRequest", reply: "CliLoginDeniedReply", body: "application/json" },
} as const;

export type RouteName = keyof typeof ROUTES;

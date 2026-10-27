// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

// The client against a mock that replays the examples of api/openapi.yaml.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { GeneralForecast, GFError, ROUTES } from "../dist/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const specText = readFileSync(join(here, "..", "..", "..", "api", "openapi.yaml"), "utf8");

// A tiny reader for the examples: every example value is one JSON-compatible
// YAML block under `examples: <name>: value:`. YAML's flow syntax is JSON's
// superset for what the spec uses, so each block is converted line by line.
function examples(spec) {
  const found = [];
  const lines = spec.split("\n");
  let path = null;
  let method = null;
  let status = null;
  let section = null;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const route = /^  (\/\S+):\s*$/.exec(line);
    if (route) { path = route[1]; continue; }
    const verb = /^    (get|post):\s*$/.exec(line);
    if (verb) { method = verb[1].toUpperCase(); status = null; continue; }
    if (/^      requestBody:/.test(line)) { section = "request"; status = null; continue; }
    if (/^      responses:/.test(line)) { section = "response"; continue; }
    const code = /^        "(\d{3})":\s*$/.exec(line);
    if (code && section === "response") { status = Number(code[1]); continue; }
    const name = /^(\s+)([a-z-]+):\s*$/.exec(line);
    // An example may carry `x-gf-*` annotations before its value.
    let next = i + 1;
    while (name && lines[next] && /^\s+x-gf-[a-z-]+:/.test(lines[next])) next++;
    const inline = name && lines[next] ? /^\s+value: (\{.*\})\s*$/.exec(lines[next]) : null;
    if (name && inline && path) {
      found.push({ path, method, section, status, name: name[2], value: flow(inline[1]) });
      continue;
    }
    if (name && lines[next] && /^\s+value:\s*$/.test(lines[next]) && path && path.startsWith("/")) {
      const indent = name[1].length;
      const block = [];
      for (let j = next + 1; j < lines.length; j++) {
        const current = lines[j];
        if (current.trim() === "" ) break;
        const currentIndent = current.length - current.trimStart().length;
        if (currentIndent <= indent + 2) break;
        block.push(current.slice(indent + 4));
      }
      found.push({ path, method, section, status, name: name[2], value: yamlBlock(block) });
    }
  }
  return found;
}

function yamlBlock(lines) {
  // Convert an indented block-style YAML object (with flow collections on
  // single lines) to a value. Supports what the spec's examples use.
  const items = [];
  let i = 0;
  function parse(indent) {
    const isList = lines[i] !== undefined && lines[i].trimStart().startsWith("- ");
    const out = isList ? [] : {};
    while (i < lines.length) {
      const line = lines[i];
      const lineIndent = line.length - line.trimStart().length;
      if (lineIndent < indent) break;
      if (lineIndent > indent) throw new Error(`unexpected indent in ${line}`);
      const text = line.trim();
      if (isList) {
        const rest = text.slice(2);
        if (/^[A-Za-z_"][^:{}[\]]*:(\s|$)/.test(rest)) {
          lines[i] = " ".repeat(lineIndent + 2) + rest;
          out.push(parse(lineIndent + 2));
        } else {
          i++;
          out.push(scalarOrFlow(rest));
        }
        continue;
      }
      const colon = text.indexOf(":");
      const key = unquote(text.slice(0, colon));
      const rest = text.slice(colon + 1).trim();
      i++;
      if (rest === "") {
        const nextIndent = lines[i] !== undefined ? lines[i].length - lines[i].trimStart().length : indent;
        out[key] = parse(nextIndent);
      } else {
        out[key] = scalarOrFlow(rest);
      }
    }
    return out;
  }
  return parse(lines[0].length - lines[0].trimStart().length);
}

function unquote(text) {
  return /^".*"$/.test(text) ? JSON.parse(text) : text;
}

function scalarOrFlow(text) {
  if (text.startsWith("{") || text.startsWith("[")) return flow(text);
  return scalar(text);
}

function scalar(text) {
  if (text === "null") return null;
  if (text === "true") return true;
  if (text === "false") return false;
  if (/^".*"$/.test(text)) return JSON.parse(text);
  if (/^-?\d+(\.\d+)?$/.test(text)) return Number(text);
  return text;
}

function flow(text) {
  // Tokenize a single-line flow collection.
  let pos = 0;
  function skip() { while (text[pos] === " ") pos++; }
  function value() {
    skip();
    if (text[pos] === "{") {
      pos++;
      const out = {};
      skip();
      if (text[pos] === "}") { pos++; return out; }
      for (;;) {
        skip();
        const key = token(true);
        skip();
        if (text[pos] !== ":") throw new Error(`expected : in ${text}`);
        pos++;
        out[key] = value();
        skip();
        if (text[pos] === ",") { pos++; continue; }
        if (text[pos] === "}") { pos++; return out; }
        throw new Error(`bad flow mapping in ${text}`);
      }
    }
    if (text[pos] === "[") {
      pos++;
      const out = [];
      skip();
      if (text[pos] === "]") { pos++; return out; }
      for (;;) {
        out.push(value());
        skip();
        if (text[pos] === ",") { pos++; continue; }
        if (text[pos] === "]") { pos++; return out; }
        throw new Error(`bad flow sequence in ${text}`);
      }
    }
    return scalar(token(false));
  }
  function token(isKey) {
    if (text[pos] === '"') {
      const end = text.indexOf('"', pos + 1);
      const raw = text.slice(pos, end + 1);
      pos = end + 1;
      return isKey ? JSON.parse(raw) : raw;
    }
    let end = pos;
    while (end < text.length && !",}]".includes(text[end]) && !(isKey && text[end] === ":")) end++;
    const raw = text.slice(pos, end).trim();
    pos = end;
    return raw;
  }
  return value();
}

const EXAMPLES = examples(specText);

function paired(path, method) {
  const requests = EXAMPLES.filter((e) => e.path === path && e.method === method && e.section === "request");
  const responses = EXAMPLES.filter((e) => e.path === path && e.method === method && e.section === "response");
  return requests.map((request) => {
    const response = responses.find((r) => r.name === request.name);
    assert.ok(response, `${path} ${request.name} has a matching response example`);
    return { request, response };
  });
}

/** A mock `fetch` that answers from the spec's examples, matched on path and payload. */
function mockFetch(log = []) {
  return async (url, init) => {
    const { pathname } = new URL(url);
    const method = init?.method ?? "GET";
    log.push({ pathname, method, init });
    const template = Object.values(ROUTES).find((route) => {
      if (route.method !== method) return false;
      const pattern = new RegExp("^" + route.path.replace(/\{[a-z_]+\}/g, "[^/]+") + "$");
      return pattern.test(pathname);
    });
    if (!template) return new Response(JSON.stringify({ error: { code: "NotFound", message: "route not found" } }), { status: 404 });
    const candidates = EXAMPLES.filter((e) => e.path === template.path && e.method === method);
    let chosen;
    let envelope;
    if (method === "GET") {
      const responses = candidates.filter((e) => e.section === "response");
      chosen = responses.find((r) => r.status === 200);
      if (pathname.endsWith("/nothing")) chosen = responses.find((r) => r.status === 404) ?? chosen;
      if (["/healthz", "/readyz", "/livez"].includes(template.path)) {
        return new Response(JSON.stringify(chosen.value), { status: chosen.status });
      }
      envelope = {
        api_version: "gf/v1",
        project: pathname.split("/")[3],
        request_id: init.headers["X-GF-Request-ID"],
        resource: chosen.value.resource,
      };
    } else {
      const sent = JSON.parse(init.body);
      const request = candidates.find(
        (e) => e.section === "request" && JSON.stringify(e.value.payload) === JSON.stringify(sent.payload),
      );
      assert.ok(request, `the mock knows the payload sent to ${pathname}`);
      chosen = candidates.find((e) => e.section === "response" && e.name === request.name);
      envelope = {
        api_version: sent.api_version,
        project: sent.project,
        request_id: sent.request_id,
        resource: sent.resource,
      };
    }
    const body = { ...envelope };
    if (chosen.value.result !== undefined) body.result = chosen.value.result;
    if (chosen.value.error !== undefined) body.error = chosen.value.error;
    return new Response(JSON.stringify(body), {
      status: chosen.status,
      headers: { "Content-Type": "application/json" },
    });
  };
}

function client(log) {
  return new GeneralForecast({
    apiKey: "gf_test",
    project: "acme",
    baseUrl: "http://mock.local",
    fetch: mockFetch(log),
  });
}

test("the spec has request examples for every POST route and a reply for each", () => {
  for (const [name, route] of Object.entries(ROUTES)) {
    // An Arrow upload's body is a stream, not an example.
    if (route.method !== "POST" || route.body !== "application/json") continue;
    assert.ok(paired(route.path, "POST").length > 0, `${name} has examples`);
  }
});

test("predictors.predict sends the envelope and unwraps the result", async () => {
  const log = [];
  const gf = client(log);
  const example = paired(ROUTES["predictors.predict"].path, "POST").find((p) => p.request.name === "scored");
  const result = await gf.predictors.predict("churn", { queries: example.request.value.payload.queries });
  assert.deepEqual(result, example.response.value.result);
  const sent = JSON.parse(log[0].init.body);
  assert.equal(sent.api_version, "gf/v1");
  assert.equal(sent.project, "acme");
  assert.deepEqual(sent.resource, { kind: "predictor", name: "churn" });
  assert.match(sent.request_id, /^r-[0-9a-f]{32}$/);
  assert.equal(log[0].init.headers.Authorization, "Bearer gf_test");
  assert.equal(log[0].pathname, "/v1/projects/acme/predictors/churn/predict");
});

test("a declared field is a typed 422 with param and hint", async () => {
  const gf = client();
  const example = paired(ROUTES["predictors.predict"].path, "POST").find((p) => p.request.name === "declared");
  await assert.rejects(
    // The client cannot send `target`; replay the raw payload through `call`.
    gf.call(ROUTES["predictors.predict"], { project: "acme", name: "churn" }, { kind: "predictor", name: "churn" }, example.request.value.payload),
    (error) => {
      assert.ok(error instanceof GFError);
      assert.equal(error.code, "ValidationError");
      assert.equal(error.param, "target");
      assert.equal(error.status, 422);
      assert.equal(error.hint, example.response.value.error.hint);
      assert.match(error.requestId, /^r-/);
      assert.match(String(error), /^error: declared by the predictor\n  hint: /);
      return true;
    },
  );
});

test("an unknown predictor is NotFound", async () => {
  const gf = client();
  const example = paired(ROUTES["predictors.predict"].path, "POST").find((p) => p.request.name === "missing");
  await assert.rejects(
    gf.predictors.predict("nothing", { queries: example.request.value.payload.queries }),
    (error) => error instanceof GFError && error.code === "NotFound" && error.status === 404,
  );
});

test("predict sends the manifest fragment and gets a result", async () => {
  const gf = client();
  for (const { request, response } of paired(ROUTES.predict.path, "POST")) {
    if (response.status === 200) {
      const result = await gf.predict(request.value.payload);
      assert.deepEqual(result, response.value.result);
    } else {
      await assert.rejects(gf.predict(request.value.payload), (error) => {
        assert.ok(error instanceof GFError, `${request.name} is a GFError`);
        assert.equal(error.code, response.value.error.code);
        assert.equal(error.param, response.value.error.param);
        return true;
      });
    }
  }
});

test("a capacity error carries limit and actual", async () => {
  const gf = new GeneralForecast({
    project: "acme",
    baseUrl: "http://mock.local",
    fetch: async (_url, init) => {
      const sent = JSON.parse(init.body);
      const example = EXAMPLES.find((e) => e.path === ROUTES.predict.path && e.name === "capacity");
      return new Response(
        JSON.stringify({
          api_version: sent.api_version,
          project: sent.project,
          request_id: sent.request_id,
          resource: sent.resource,
          error: example.value.error,
        }),
        { status: 422 },
      );
    },
  });
  const rows = Array.from({ length: 1000 }, (_, i) => ({ spend: i }));
  await assert.rejects(
    gf.predict({ model: "general-forecast/tabicl-v2-classifier", target: "churned", context: [{ spend: 1, churned: 0 }], queries: rows }),
    (error) => error instanceof GFError && error.code === "CapacityExceeded" && error.limit === 100 && error.actual === 1000,
  );
});

test("predictors.list, predictors.get, models.list and models.get unwrap their results", async () => {
  const gf = client();
  const predictors = await gf.predictors.list();
  assert.equal(predictors[0].name, "churn");
  assert.deepEqual(predictors[0].limits, { context_rows: 100000, queries: 100, inputs: 32 });
  const churn = await gf.predictors.get("churn");
  assert.equal(churn.serving.compiled, true);
  assert.equal(churn.environments.length, 2);
  await assert.rejects(gf.predictors.get("nothing"), (error) => error instanceof GFError && error.code === "NotFound");
  const models = await gf.models.list();
  // The catalog lists each model by its pinned id.
  assert.match(models[0].model, /^general-forecast\/tabicl-v2-classifier@sha256:[0-9a-f]{64}$/);
  assert.equal(models.length, 1);
  assert.equal(models[0].task, "classification");
  const model = await gf.models.get("tabicl-v2-classifier");
  assert.equal(model.name, "general-forecast/tabicl-v2-classifier");
  await assert.rejects(gf.models.get("nothing"), (error) => error instanceof GFError && error.code === "NotFound");
});

test("live resolves while the server answers, and the default server is the spec's", async () => {
  const log = [];
  await client(log).live();
  assert.deepEqual(log.map((call) => call.pathname), ["/livez"]);
  const asked = [];
  const hosted = new GeneralForecast({
    project: "acme",
    fetch: async (url) => {
      asked.push(url);
      return new Response(JSON.stringify({ status: "ok" }), { status: 200 });
    },
  });
  await hosted.health();
  const hostedServer = specText.match(/servers:\n\s+- url: (\S+)/)[1];
  assert.deepEqual(asked, [`${hostedServer}/healthz`]);
  assert.equal(hosted.baseUrl, "https://app.generalforecast.com");
});

test("health resolves on 200 and throws ModelNotReady on 503", async () => {
  await client().health();
  const loading = new GeneralForecast({
    project: "acme",
    baseUrl: "http://mock.local",
    fetch: async () => new Response(JSON.stringify(EXAMPLES.find((e) => e.path === "/healthz" && e.status === 503).value), { status: 503 }),
  });
  await assert.rejects(loading.health(), (error) => error instanceof GFError && error.code === "ModelNotReady");
});

test("a reply that does not echo the request is refused", async () => {
  const gf = new GeneralForecast({
    project: "acme",
    baseUrl: "http://mock.local",
    fetch: async () =>
      new Response(
        JSON.stringify({ api_version: "gf/v1", project: "other", request_id: "r-x", resource: { kind: "predictor", name: "churn" }, result: {} }),
        { status: 200 },
      ),
  });
  await assert.rejects(gf.predictors.predict("churn", { queries: [{ x: 1 }] }), (error) => error instanceof GFError && error.code === "Internal");
});

test("contexts list, show, append and label through the envelope", async () => {
  const log = [];
  const gf = client(log);
  const append = paired(ROUTES["contexts.append"].path, "POST")[0];
  const label = paired(ROUTES["contexts.label"].path, "POST")[0];
  const name = append.request.value.resource.name;
  const contexts = await gf.contexts.list();
  assert.ok(contexts.length > 0);
  assert.equal((await gf.contexts.get(name)).name, name);
  const appended = await gf.contexts.append(name, {
    writeId: append.request.value.payload.write_id,
    rows: append.request.value.payload.rows,
  });
  assert.deepEqual(appended, append.response.value.result);
  const labeled = await gf.contexts.label(name, {
    writeId: label.request.value.payload.write_id,
    target: label.request.value.payload.target,
    labels: label.request.value.payload.labels,
  });
  assert.deepEqual(labeled, label.response.value.result);
  assert.deepEqual(
    log.map((call) => `${call.method} ${call.pathname}`),
    [
      "GET /v1/projects/acme/contexts",
      `GET /v1/projects/acme/contexts/${name}`,
      `POST /v1/projects/acme/contexts/${name}/append`,
      `POST /v1/projects/acme/contexts/${name}/label`,
    ],
  );
});

test("predictors.predict names the environment in the query", async () => {
  const asked = [];
  const mock = mockFetch();
  const gf = new GeneralForecast({
    project: "acme",
    baseUrl: "http://mock.local",
    fetch: async (url, init) => {
      asked.push(new URL(url).search);
      return mock(url, init);
    },
  });
  const example = paired(ROUTES["predictors.predict"].path, "POST").find((p) => p.request.name === "scored");
  await gf.predictors.predict("churn", { queries: example.request.value.payload.queries, env: "staging" });
  assert.deepEqual(asked, ["?env=staging"]);
});

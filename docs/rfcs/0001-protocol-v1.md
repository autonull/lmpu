# RFC 0001 — LMPU Protocol v1

- **Status:** Accepted for v0.1.0
- **Date:** 2026-11-03
- **Normative schema:** `src/lmpu/schema/protocol-v1.schema.json`
- **Trace schema:** `src/lmpu/schema/trace-v1.schema.json`

## 1. Scope

This RFC freezes the wire contract between a host language model and the LMPU
control plane. It covers the JSON-RPC envelope, session and job lifecycle,
sleep/wake, buffers, events, errors, and the trace format. It does not specify
model internals, scheduler policy, or processor implementations.

The Pydantic models in `lmpu.protocol.messages` and the packaged JSON Schema are
normative. If this document and the schema disagree, the schema wins.

## 2. Transport

- Unix domain socket, length-prefixed JSON-RPC 2.0 frames (`u32` big-endian).
- One connection may carry many concurrent requests by `id`.
- UTF-8 JSON; unknown fields are accepted and ignored unless a model forbids
  them.
- The server owns ordering within a session. The client owns request ids.

## 3. Envelopes

Request:
```json
{"jsonrpc":"2.0","id":"1","method":"submit","params":{}}
```

Success response:
```json
{"jsonrpc":"2.0","id":"1","result":{"job_id":"job_01","status":"completed","seq":0}}
```

Error response:
```json
{"jsonrpc":"2.0","id":"1","error":{"code":"not_found","message":"job not found","retriable":false}}
```

Notifications omit `id`. The protocol version is exchanged in `hello`; the
current value is `lmpu/1`.

## 4. Session lifecycle

1. Client sends `hello` with `{"protocol":"lmpu/1"}`.
2. Server returns `HelloResult` with `session_id`, `protocol`, `schema_hash`,
   `resume_status`, and `server_time`.
3. All subsequent calls carry the session implicitly; `wake` may carry an
   explicit `session_id`.
4. `close` releases grants, buffers, and pending jobs.

Sessions are isolated: buffers and jobs belong to exactly one session. A closed
session cannot be resumed; clients must `hello` again.

## 5. Job lifecycle

Internal states: `created`, `pending`, `running`, `waiting_input`,
`waiting_event`, `cancel_requested`, `completed`, `failed`, `cancelled`.
Model-facing states collapse waiting states to `sleeping`.

- `submit` accepts one op or a `batch` of items. It returns `job_id`, `status`,
  `queue_position`, and `seq` immediately in `async` mode.
- `poll` returns progress and queue position.
- `wait` blocks until a terminal state or timeout. The result is summary-only;
  large values stay in buffers.
- `cancel` requests cancellation; the terminal event follows.

A job MUST end in exactly one terminal state. Failed jobs carry an `RPCError`
with a stable `ErrorCode`; `retriable` states whether resubmission is allowed.

## 6. Sleep and wake

A host may park while a job runs:

```json
{"method":"sleep","params":{"until":{"all":["job_01"]},"level":"S1"}}
```

- `S1` — host performs no decode forwards while parked.
- `S2`/`S3` — reserved for optional deeper parking; not required by v1.
- `wake` returns `woken` or `no_active_sleep`.
- The server MUST NOT inject events into a parked host except through an
  explicit `wake` or sleep-condition completion.
- The trace records `park_enter` and `wake` with job ids and level.

## 7. Buffers and context protection

Buffers are named typed regions. `buffer_import` may inline `data` or reference
a `path`. Reads are bounded:

```json
{"method":"read","params":{"buffer":"buf_01","max_inline_bytes":1024}}
```

If the requested slice exceeds the inline limit or the read is a full view of a
large buffer, the server returns a `BufferView` with `data: null` and a
`derived_buffer` id. Only metadata enters model context. The canonical bound is
**≤64 context tokens for a 16 MiB result** (A6, C3).

Buffers are reference-counted. Releasing a grant decrements one reference; the
reaper frees a buffer only when its count reaches zero and its TTL expires.

## 8. Events

Events have `event_id`, monotonic `seq`, `session`, optional `job_id`, `type`,
`ts_monotonic`, `ts_wall`, and typed `data`. `EventKind` values are frozen:
`JobSubmitted`, `JobStarted`, `Progress`, `JobCompleted`, `JobFailed`,
`JobCancelled`, `NeedsMoreInput`, `Sleeping`, `Woken`, `BufferCreated`,
`BufferUpdated`, `ResourceWarning`, `ControlError`, `ResultTooLarge`,
`Heartbeat`.

`Progress` and `Heartbeat` may be sampled. A sampled trace is not a contiguous
timeline; `lmpu trace --gaps` detects gaps.

## 9. Trace format

One JSON object per line. The schema freezes the required fields for:

- `session_start` / `session_end`
- `event_*` records
- `host_forward`
- `host_metrics` (`lmpu.host_metrics.v1`)
- `control_parsed`, `job_state`, `park_enter`, `wake`, `inject`
- `buffer_*` access records

Records may add fields; existing fields and discriminators are frozen.
Sensitive keys are redacted before persistence.

## 10. Errors and retry

`RPCError.code` is one of the frozen `ErrorCode` values. Clients retry only
when `retriable` is true and the method is idempotent. `submit` is not
idempotent unless the client supplies `job_id`.

## 11. Versioning

- Protocol version: `lmpu/1`. Breaking changes require `lmpu/2`.
- Schema hash: `HelloResult.schema_hash` identifies the frozen schema.
- Additive fields are allowed. Removing or retyping a field is breaking.
- Trace schema version is embedded in each `host_metrics` record.

## 12. Security and limits

- No network listener by default; Unix socket only.
- Worker sandbox and resource limits are enforced by the runtime.
- Payloads are size-limited; large results use buffers.
- Trace payloads are redacted and truncated before persistence.

## 13. Examples

Examples that validate against the schema live in
`tests/unit/test_schema.py`. The RFC is intentionally example-light to avoid
maintaining a parallel contract.

# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| < 0.1 | No |

## Reporting a vulnerability

Report suspected vulnerabilities privately to the maintainers. Do not open a
public issue containing exploit details, credentials, or user data. Include the
affected version, a minimal reproduction, and the impact.

## Security model

- **Transport:** Unix domain socket only. The runtime does not listen on TCP.
  Socket permissions are controlled by the deployment, not by LMPU.
- **Sessions:** jobs and buffers are scoped to one session. The control plane
  validates session ownership before every buffer operation.
- **Sandbox:** Python jobs execute in an isolated worker process. Worker crash
  recovery replaces the process; a crashed job is failed, not silently retried.
- **Limits:** buffer bytes, outstanding jobs, sleep duration, and job timeouts
  are enforced by quotas. Large results use buffers and bounded reads.
- **Context protection:** a 16 MiB result is represented by derived-buffer
  metadata; no full payload is inlined into model context.
- **Tracing:** sensitive keys (`authorization`, `password`, `token`, `secret`,
  `env`, and related names) are redacted before persistence. Oversized strings
  are truncated.
- **Plugins:** generated plugins run with the same worker privileges as built-in
  processors. Review third-party plugin code before deployment.

## Out of scope

- Network-exposed deployments without a reverse proxy and TLS.
- Host model prompt-injection resistance; LMPU bounds context and execution but
  does not validate model intent.
- Multi-tenant isolation across operating-system users.

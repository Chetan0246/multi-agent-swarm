# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub: **Security → Report a
vulnerability** on this repository. Don't open a public issue. We aim to reply within a few
days and to fix confirmed issues promptly.

## Scope and threat model

- **Prompt injection from web pages.** Fetched content is untrusted. Defences are layered:
  - a data-only prompt rule and delimiters;
  - pattern detection, which excludes affected findings;
  - quote checking against retrieved text;
  - report sanitising (no images, no HTML, links only to cited sources);
  - least-privilege tools (search and read only).

  A bypass that gets instructions acted on, or data sent anywhere, is a vulnerability.
- **The web server.** By default it binds to `127.0.0.1`. If you expose it, set
  `SWARM_API_TOKEN`, or anyone who can reach the port can spend your API credits. Server-side
  caps (`max_usd_per_job`) always apply.
- **Secrets.** Keys are read from the environment (`ANTHROPIC_API_KEY`, `TINYFISH_API_KEY`) and
  are never written to logs, reports or caches.
- **Local files.** `--docs` reads only the paths you pass on the command line. The web API
  doesn't accept file paths.

## Supported versions

Security fixes go into the latest release.

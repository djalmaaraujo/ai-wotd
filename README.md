# AI Word Of The Day

A public hub that surfaces the single AI term being hyped each day, built
entirely on GitHub. Every day a scheduled Action ingests a curated list of
RSS feeds, newsletters, and X accounts, runs a deterministic trending score
over the corpus, picks a word, has Claude write a short context blurb, and
commits both a Parquet-backed dataset and a minimal static site back to the
repo.

- **Site**: served by GitHub Pages from `/docs`.
- **Data**: Parquet + JSON under `/data`, queryable via DuckDB or the
  `wotd.data` Python library.
- **Sources**: `sources.yml`. Propose changes with a PR.

## How the word is chosen

The pipeline is deterministic and reproducible:

1. For each source, pull the feed and extract article / tweet text.
2. Tokenize (unigrams, bigrams, trigrams), drop English + news-generic
   stopwords, keep hyphenated compounds and curated AI phrases
   (`"context window"`, `"mcp"`, etc.).
3. Score each term with `log(1+tf_today) * (tf_today / avg_tf_baseline)
   * (0.5 + df_today/doc_count)`, where the baseline is the last 30 days.
   Boost allowlisted terms, demote terms that appear every day.
4. The top-scoring term is the word of the day.
5. Claude writes a ~150-word daily summary and a 2–3 sentence
   "why-it-trended" blurb. This step is optional and additive — the pick
   itself never depends on the LLM.

## Data policy

This repo only redistributes derivatives:

- Article metadata: title, canonical URL, author, publish date.
- Per-article token / term statistics.
- An attributed snippet of ≤280 characters per article (tweets are ≤280
  by construction and reproduced in full with attribution).

Full HTML / text is **never committed**. It lives only in the GitHub
Actions cache for the duration of a run and is discarded. Every snippet
on the site links back to the origin.

Code is MIT. Derivative data is offered under CC-BY-4.0 with the caveat
that individual article titles and snippets remain the property of their
original authors.

### Opt-out / takedown

Authors or publishers who do not want their content parsed can:
- Open an issue titled `takedown: <feed URL>`, or
- Open a PR removing their entry from `sources.yml`.

Requests are honored within 7 days. The fetcher also respects
`robots.txt` and `X-Robots-Tag: noindex`.

## Propose a source

Edit `sources.yml` and open a PR. Each entry has a `type` discriminator:

```yaml
- id: your-source
  type: rss | newsletter | twitter
  name: "Human-readable name"
  feed: "https://.../rss"       # rss/newsletter
  # or
  nitter_feed: "https://nitter.net/handle/rss"  # twitter (default)
  handle: "handle"              # twitter
  added: "YYYY-MM-DD"
```

The `validate-sources` Action enforces the JSON schema at
`schemas/sources.schema.json` and pings each feed URL on every PR.

## Querying the corpus

Two paths, same data:

**DuckDB CLI:**

```bash
duckdb -c "SELECT word, date FROM read_parquet('data/parquet/wotd.parquet') \
  ORDER BY date DESC LIMIT 10"
```

**Python:**

```python
from wotd.data import articles, terms, wotd, trending

trending(n=10, since="7d")          # top terms of the last week
articles(source="openai-blog")      # every article from a source
terms(term="mcp")                   # the whole history of a single term
wotd()                              # today's word + LLM blurb
```

Under the hood each helper opens a short-lived DuckDB connection against
`data/parquet/{articles,terms,wotd}.parquet` and returns a Polars
DataFrame.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Full pipeline against sources.yml
wotd run

# Browse the site
python -m http.server -d docs 8000
# → http://localhost:8000
```

Useful environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `WOTD_BASELINE_DAYS` | `30` | Baseline window for the trending score. |
| `WOTD_MAX_ARTICLES_PER_SOURCE` | `50` | Safety cap on the first fetch. |
| `WOTD_USER_AGENT` | `ai-wotd/1.0 ...` | HTTP User-Agent. |
| `WOTD_FULLTEXT_CACHE_DIR` | `.cache/wotd/fulltext` | Where full text is cached per run. |
| `ANTHROPIC_API_KEY` | _(unset)_ | Enables the LLM blurb step; unset → skipped. |
| `WOTD_LLM_MODEL` | `claude-sonnet-4-5` | Model for the blurb. |
| `WOTD_LINKFOLLOW_MAX_PER_ISSUE` | `10` | Cap on one-hop links per newsletter issue. |
| `WOTD_NITTER_INSTANCES` | _(builtin list)_ | Comma-separated Nitter hosts. |
| `X_BEARER_TOKEN` | _(unset)_ | Only needed for sources with `x_api: true`. |

## GitHub Pages

The site is served from `/docs` on `main`.

1. **Settings → Pages → Build and deployment → Source: Deploy from a branch.**
2. **Branch: `main`, folder: `/docs`.** Save.
3. The site resolves at `https://djalmaaraujo.github.io/ai-wotd/`.

The templates render all in-site links with a configurable base path so
project-page URLs (`/ai-wotd/...`) and custom domains both work:

| Variable | Default | When to change |
|---|---|---|
| `WOTD_SITE_BASE` | `/ai-wotd` | Set to `""` when serving from a custom domain at the root. |
| `WOTD_SITE_URL` | `https://djalmaaraujo.github.io/ai-wotd` | Only used in `feed.xml` to build absolute permalinks. |

`docs/.nojekyll` is written on every render so GitHub Pages serves every
file verbatim (no Jekyll filtering of underscore-prefixed paths).

## Repository layout

```
sources.yml                     # PR-able sources
schemas/sources.schema.json     # validated on every PR
src/wotd/                       # pipeline + data API
  fetch.py linkfollow.py corpus.py terms.py wotd.py llm.py
  export.py data.py site.py cli.py
  sources/{rss,newsletter,twitter}.py
data/
  articles/<date>/*.json        # derivative-only article records
  stats/<date>.json             # per-day term statistics
  wotd/<date>.json              # chosen word + LLM blurb
  parquet/{articles,terms,wotd}.parquet
docs/                           # static site (GitHub Pages)
templates/                      # Jinja + CSS
tests/                          # pytest + fixtures
.github/workflows/              # daily.yml, reprocess.yml, validate-sources.yml
```

## License

- Code: [MIT](LICENSE).
- Derivative data (Parquet + JSON): CC-BY-4.0.

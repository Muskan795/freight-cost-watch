# Freight Cost Watch

Freight Cost Watch identifies route-weeks where freight cost per tonne-km rises abnormally and determines whether a reviewed context note supports the increase. If no note passes verification, the result is marked **Unexplained**.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Groq API key in a `.env` file in the project directory:

```text
GROQ_API_KEY=your_groq_api_key
```

Run the extraction step once:

```bash
python extract_notes.py
```

Review `note_facts.json` manually. Correct unsupported or inaccurate facts before marking them as reviewed. Never set `needs_review=false` unless the facts have actually been verified.

Run the deterministic pipeline:

```bash
python run.py
```

This generates `output.csv` and `audit_candidates.csv`.

Run the tests and labelled evaluation:

```bash
python eval_harness.py
```

Check reproducibility:

```bash
python repro_check.py
```

The extraction step requires a Groq API key. The deterministic analysis pipeline does not require an LLM API call.

## How It Works

### 1. Weekly Cost Calculation

In `costs.py`, shipment records are grouped by route and Monday-to-Sunday week.

The weekly cost per tonne-km is calculated using pooled totals:

`cost_per_tonne_km = sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)`

This avoids taking an unweighted average of individual shipment ratios.

### 2. Baselines

- **Own-history baseline:** The mean of up to eight previous observed weeks for the same route, excluding the current week. Missing weeks are not filled with artificial values. The first observed week has no historical baseline.
- **Similar-route baseline:** The mean cost per tonne-km of other routes with the same `route_type` during the same week. The route being evaluated is excluded from its own peer average.

### 3. Anomaly Flagging

A route-week is flagged when its cost is at least 10% above either available baseline.

The threshold is a configurable assumption in `config.py`, not a statistically calibrated threshold. It is intended to identify potentially unusual increases for further investigation.

`REQUIRE_RISE_VS_OWN_HISTORY = False` means exceeding either baseline is sufficient for flagging. Setting it to `True` adds a requirement that the route must also be above its own historical baseline.

### 4. Context-Note Retrieval and Verification

In `notes.py`, TF-IDF vector similarity retrieves candidate notes. A deterministic verification gate then checks whether a note:

- Has reviewed structured facts.
- Concerns routes represented in the dataset.
- Applies to the route, its route type, or all routes.
- Explicitly supports an increase in costs.
- Has a supported date window overlapping the route-week.

A broad-scope note can explain a rise relative to a route's own history, but cannot explain why that route is more expensive than comparable routes when the same event affects those peers too.

If no candidate passes verification, the result remains unexplained. Retrieval similarity alone never determines the verdict.

### 5. Explanation and Audit Trail

Explanations are assembled from a template and quote the selected source note. The pipeline does not freely generate explanations with an LLM.

`audit_candidates.csv` records candidate note retrieval scores and verification outcomes, including rejection reasons.

## Where AI Is Used

The LLM is used by `extract_notes.py` to extract structured facts from context notes. The configured model is `openai/gpt-oss-20b` through Groq, with temperature set to 0.

The extraction script records token usage returned by the API and estimates cost using the prices configured in `config.py`. Extracted facts are drafts and must be reviewed before use.

The main `run.py` pipeline makes no LLM calls. It uses TF-IDF retrieval and deterministic verification rules, making its output reproducible without model-generated verdicts.

## Assumptions and Trade-offs

- The historical baseline uses up to eight previous observed weeks, with no padding for missing weeks.
- The 10% anomaly threshold is a configurable starting assumption and has not been calibrated against a large labelled dataset.
- Open-ended events are limited by the configured persistence window. A note must not be assigned an unsupported end date.
- Notes about recovery, no cost impact, or routes outside the dataset cannot justify an increase.
- A note is accepted only when its scope, effect, date window, and review status satisfy the verification rules.
- Unexplained means that no available reviewed note passed the checks; it does not prove that no real-world cause exists.

## Evaluation

The project includes six unit tests covering:

- Pooled cost calculation.
- Monday-to-Sunday weekly grouping.
- Historical baselines without look-ahead.
- Peer averages excluding the current route.
- Threshold flagging.
- Context-note verification and rejection conditions.

The six unit tests passed.

A manually labelled evaluation set containing four route-week examples achieved:

- **Verdict accuracy: 100% (4/4).**
- **Note-ID accuracy: 100% (4/4).**

This is a small initial sanity check, not a representative benchmark. The accuracy results apply only to these four examples. A larger independently reviewed set is needed for a stronger performance estimate.

The two route-weeks currently justified by the pipeline are associated with note N001. They should be checked against the original note text and date window before final submission.

## Reproducibility

Three consecutive runs of `repro_check.py` produced byte-identical `output.csv` files.

Verified SHA-256:

`88a13cb32004996c688a3855ffe5b64e0aecd06d691562653e304ab9b4b692e5`

This confirms deterministic output for the tested project state and inputs.

## Token and Cost Log

The existing `token_log.json` records the following extraction usage:

- Model: `openai/gpt-oss-20b` via Groq.
- Temperature: 0.
- Recorded API calls: 10.
- Input tokens: 4,096.
- Output tokens: 1,120.
- Logged estimated cost: $0.00097.

The cost is an estimate based on the rates configured in `config.py`; verify those rates against the applicable provider pricing before treating this figure as an accurate cost estimate. The log records estimated model usage cost, not necessarily the amount billed. `run.py` makes no LLM calls.

## Security

Keep API keys in `.env`. Do not commit `.env` or expose API keys in source files, logs, screenshots, or the submission package. Include a `.env.example` with a placeholder value if needed.
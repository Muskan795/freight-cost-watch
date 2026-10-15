"""All tunable settings in one place. Nothing here is random."""

# ---- Files -------------------------------------------------------------
SHIPMENTS_FILE = "shipment_records.csv"
NOTES_FILE = "context_notes.csv"
SAMPLE_FILE = "sample_output_format_v2.csv"   # header is used as the output contract
NOTE_FACTS_FILE = "note_facts.json"        # reviewed, structured facts per note
OUTPUT_FILE = "output.csv"
AUDIT_FILE = "audit_candidates.csv"        # why each candidate note was accepted/rejected
TOKEN_LOG_FILE = "token_log.json"

# ---- Cost logic --------------------------------------------------------
HISTORY_WEEKS = 8                 # trailing window, strictly before current week
ANOMALY_THRESHOLD_PCT = 10.0      # assumption: flag a rise of 10% or more
REQUIRE_RISE_VS_OWN_HISTORY = False
# False -> flag if EITHER baseline is exceeded by the threshold (brief's wording).
# True  -> additionally require the cost to be above the route's own history,
#          so a route that is always dearer than its peers is not flagged forever.

# ---- Notes / retrieval -------------------------------------------------
RETRIEVAL_TOP_K = 8               # candidates shortlisted by retrieval
MAX_PERSISTENT_WEEKS = 8          # open-ended notes (e.g. fuel hike) explain a rise
                                  # for at most this long: after 8 weeks the rolling
                                  # baseline has absorbed the step change anyway.

# ---- LLM (used ONLY by extract_notes.py, once per note) ----------------
LLM_MODEL ="openai/gpt-oss-20b"   # open-weights model on Groq's free tier
LLM_TEMPERATURE = 0
LLM_SEED = 0
# Fill from the provider's published pricing page (USD per 1M tokens).
PRICE_IN_PER_M_USD = 0.10
PRICE_OUT_PER_M_USD = 0.50

# ---- Output formatting (check against sample_output_format.csv!) --------
FLAGGED_TRUE = "True"
FLAGGED_FALSE = "False"
DECIMALS = 2

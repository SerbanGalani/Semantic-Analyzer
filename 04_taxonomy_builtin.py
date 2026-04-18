"""
===============================================================================
AISA - AI Semantic Analyzer
04_taxonomy_builtin.py - Built-in taxonomy implementation (v1.2.1)
===============================================================================

Concrete implementation of TaxonomyProvider using the hardcoded dual taxonomy.

This module is a LIVE module - false positive patterns and keywords will be
updated as new FPs are discovered during analysis runs.

When updating:
    1. Modify keywords / patterns / fp_patterns in this file
    2. Bump TAXONOMY_VERSION in version.py (e.g. 1.0.0 → 1.0.1)
    3. Document the change in the CHANGELOG below

Structure:
    - FALSE_POSITIVE_PATTERNS   dict of fp_category → list of regex strings
    - AMBIGUOUS_PRODUCT_VENDORS dict: ambiguous name → required vendor context
    - AI_APPLICATIONS           dict of A1-A8 category definitions
    - AI_TECHNOLOGIES           dict of B1-B8 category definitions
    - BuiltinTaxonomy           TaxonomyProvider implementation
    - TAXONOMY                  Module-level singleton (import this)
    - PATTERN_CACHE             Module-level compiled cache (import this)

CHANGELOG:
    v1.2.1 (2026-03) - FP additions from ChatGPT Deep Thinking labeled dataset
        (18,647 rows, Fortune 500 corpus 2020-2024)

        N) LLMS → "Loan Lifecycle Management System" in Indian/Asian bank reports
           (28 FP mislabeled as "LLM" by ChatGPT). Fast-path pattern added +
           Mistral added to AMBIGUOUS_PRODUCT_VENDORS.
        O) Acronym guards: ML (luxury/auto context), RPA (ESG audit context),
           DL (telecom downlink), NLP catch-all in non-tech contexts.
        P) model_banking_context: "model serving/deployment/inference" in credit
           risk / Basel IRB sections of bank annual reports (7 FP).
        Q) recommendation_non_ai: HR/wellness/ESG "recommendation system/engine"
           without ML/personalization signal (4 FP).
        R) frontier_model_export: "frontier model" in semiconductor export control
           sections — not AI model references.
        S) mistral_non_ai: "Mistral" as Mediterranean wind / French construction
           (Bouygues group). Vendor gate also updated with Mistral AI entry.

    v1.2.0 (2026-03) - FP reduction based on manual review of 9,528 false positives
        (51.1% FP rate observed in Fortune 500 corpus, v1.0.x)

        E) "intelligent" catch-all tightened: explicit patterns added for
           Chinese corporate buzzwords (energy, nuclear, port, mining, city,
           manufacturing) that dominate APAC filings. Prior catch-all caught
           only ~60% of real FPs. New patterns cover top-frequency contexts.
        F) "automation" catch-all replaced with explicit supply-chain /
           proxy-statement / financial-table contexts (these were 1,484 FPs).
        G) LoRA: added as ambiguous product (AMBIGUOUS_PRODUCT_VENDORS) — it is
           a person name (Lora Ho, TSMC) and substring in "Lightsource SPV" lists.
           Requires AI/fine-tuning vendor context to be accepted.
        H) "edge computing" telecom-only patterns added — 70 FPs from China Mobile,
           China Telecom where edge = 5G MEC, not AI edge inference.
        I) "chatbot"/"virtual assistant" boilerplate guard — when mentioned in
           HR/enrollment lists, multi-channel contact lists, or timeline blocks
           without explicit AI implementation language, treat as FP.
        J) "Sora" added as ambiguous product — substring in financial tables
           (bond issuance lists, SPV subsidiary names). Requires OpenAI/video
           generation context.
        K) "fraud detection" / "predictive maintenance" / "predictive analytics"
           boilerplate guard — these terms appear in risk-factor disclosures and
           executive bio sections without actual AI implementation evidence.
        L) "knowledge graph" / "foundation models" telecom/hardware guard —
           Chinese telco companies use these in network architecture contexts
           unrelated to ML.
        M) _AI_CONTEXT_RE strengthened: added vendor names that increase precision
           for catch-all bypass (Huawei AI, Alibaba AI, Baidu AI, etc.)

    v1.0.1 (2026-02) - Taxonomy precision improvements
        A) Vendor gate for ambiguous product names (Gemini, Claude, Copilot,
           Titan, Nova, Llama) — accepted as AI only when vendor context present
        B) robot* FP catch-all removed; replaced with explicit industrial patterns
           _AI_CONTEXT_RE extended: autonomous, self-driving, SLAM, sensor fusion,
           drone, path planning, reinforcement learning
        C) fine-tune + finance context FP patterns added (capital structure,
           liquidity, debt, dividend, interest rate, buyback)
        D) B8_General_AI reduced: predictive/forecasting → B1, neural/deep → B2,
           NER/sentiment → B3, LLM/prompt/RAG → B4, SageMaker/Vertex → B7

    v1.0.0 (2026-02) - AISA initial release
        Ported from ai_taxonomy_v7.py (v7.1.1 + v7.2.0 FP fixes)
        Merged ai_products_v1.py product detection into classify()
        Removed all Romanian comments, full English docstrings
        Integrated with TaxonomyProvider protocol (03_taxonomy_base.py)

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import importlib
import re
import sys
import os
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from version import TAXONOMY_VERSION
_m3 = importlib.import_module("03_taxonomy_base")
TaxonomyProvider        = _m3.TaxonomyProvider
CategoryInfo            = _m3.CategoryInfo
FPResult                = _m3.FPResult
ClassificationResult    = _m3.ClassificationResult
CompiledPatternCache    = _m3.CompiledPatternCache


# ============================================================================
# AMBIGUOUS PRODUCT VENDOR GATE  (improvement A)
#
# These product names are common in non-AI contexts (pharma trials, mythology,
# military, astronomy). They are accepted as AI references ONLY when at least
# one vendor / strong-context keyword is found in the same window.
#
# Structure: { product_name_lower: [vendor_or_context_keywords] }
#
# Release years: used to block references that predate the product launch.
# A "Gemini" in a 2020 annual report is NOT Google Gemini (launched Dec 2023).
# ============================================================================

AMBIGUOUS_PRODUCT_VENDORS: Dict[str, Dict] = {
    "gemini": {
        "vendors": [
            "google", "deepmind", "vertex", "workspace", "bard",
            "google ai", "alphabet",
        ],
        "context": ["llm", "model", "chatbot", "multimodal", "generative"],
        "release_year": 2023,   # Google Gemini announced Dec 2023
    },
    "claude": {
        "vendors": ["anthropic"],
        "context": ["llm", "model", "api", "assistant", "chatbot", "ai model"],
        "release_year": 2023,   # Claude 1 public — March 2023
    },
    "copilot": {
        "vendors": ["microsoft", "github", "azure", "365", "m365"],
        "context": ["ide", "vs code", "code completion", "llm", "ai assistant"],
        "release_year": 2021,   # GitHub Copilot technical preview Jun 2021
    },
    "titan": {
        "vendors": ["aws", "amazon", "bedrock"],
        "context": ["llm", "model", "embedding", "foundation model"],
        "release_year": 2023,   # Amazon Titan announced Nov 2023
    },
    "nova": {
        "vendors": ["aws", "amazon", "bedrock"],
        "context": ["llm", "model", "multimodal", "foundation model"],
        "release_year": 2024,   # Amazon Nova announced Dec 2024
    },
    "llama": {
        "vendors": ["meta", "facebook", "hugging face"],
        "context": ["llm", "model", "open source", "fine-tuning", "language model"],
        "release_year": 2023,   # LLaMA 1 — Feb 2023
    },
    "mistral": {
        "vendors": ["mistral ai", "mistral"],
        "context": ["llm", "model", "open source", "language model"],
        "release_year": 2023,   # Mistral 7B — Sep 2023
    },
    "falcon": {
        "vendors": ["tii", "technology innovation institute", "hugging face"],
        "context": ["llm", "model", "open source", "language model"],
        "release_year": 2023,   # Falcon — May 2023
    },
    # v1.2.0 additions
    "lora": {
        "vendors": ["hugging face", "microsoft", "meta", "google"],
        "context": ["fine-tun", "llm", "language model", "adapter", "rank", "weight", "peft", "neural", "training"],
        "release_year": 2021,   # LoRA paper — Oct 2021
        # NOTE: "Lora" is a common person name (Lora Ho at TSMC, etc.)
        # Also appears as substring in "Lightsource SPV" and similar entity lists
    },
    "sora": {
        "vendors": ["openai"],
        "context": ["video", "generation", "text-to-video", "model", "diffusion", "openai"],
        "release_year": 2024,   # OpenAI Sora — Feb 2024
        # NOTE: appears as substring in financial tables (bond issuance, SPV lists)
    },
    # v1.2.1 additions
    "mistral": {
        "vendors": ["mistral ai", "mistral", "hugging face"],
        "context": ["llm", "model", "language model", "7b", "8x7b", "mixtral", "open source", "instruct"],
        "release_year": 2023,   # Mistral 7B — Sep 2023
        # NOTE: "Mistral" is a Mediterranean wind and appears in French construction/
        # energy company reports (Bouygues group, offshore wind projects)
    },
}

# Pre-compiled vendor/context regex per ambiguous product
_VENDOR_GATE_RE: Dict[str, re.Pattern] = {}
for _product, _data in AMBIGUOUS_PRODUCT_VENDORS.items():
    _terms = _data["vendors"] + _data["context"]
    _VENDOR_GATE_RE[_product] = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in _terms) + r")\b",
        re.IGNORECASE,
    )


# ============================================================================
# FALSE POSITIVE PATTERNS
# Live section - update when new FP categories are discovered.
# Each key is a descriptive FP category name.
# Each value is a list of regex pattern strings (compiled at module load).
# ============================================================================

FALSE_POSITIVE_PATTERNS: Dict[str, List[str]] = {

    # -----------------------------------------------------------------------
    # EMBEDDING - 99.7% false positives (ESG / values / risk context)
    # -----------------------------------------------------------------------
    "embedding_non_ai": [
        r"embedding\s+(?:ESG|sustainability|environmental|social|governance)",
        r"embedding\s+(?:value|values|culture|ethic|ethics|principle|principles)",
        r"embedding\s+(?:risk|risks|risk\s+management|compliance)",
        r"embedding\s+(?:in|into|within|across)\s+(?:the|our|their)",
        r"embedding\s+(?:practice|practices|standard|standards|policy|policies)",
        r"embedding\s+(?:diversity|inclusion|safety|security)",
        r"(?:deeply|fully|firmly|strongly)\s+embedding",
        r"embedding\s+(?:throughout|across)\s+(?:the|our)",
    ],

    # -----------------------------------------------------------------------
    # TRANSFORMERS - 90% are electrical equipment
    # -----------------------------------------------------------------------
    "transformers_electrical": [
        r"(?:power|electrical|distribution|voltage|current)\s+transformer",
        r"transformer\s+(?:station|substation|capacity|rating|oil)",
        r"(?:step[\s-]?up|step[\s-]?down)\s+transformer",
        r"transformer\s+(?:kV|kVA|MVA|voltage)",
        r"(?:high|low|medium)\s+voltage\s+transformer",
        r"transformer\s+(?:maintenance|inspection|failure|replacement)",
    ],

    # -----------------------------------------------------------------------
    # MULTIMODAL - 70% are transport/logistics
    # -----------------------------------------------------------------------
    "multimodal_transport": [
        r"multimodal\s+(?:transport|transportation|logistics|freight|shipping)",
        r"multimodal\s+(?:terminal|hub|corridor|network|chain)",
        r"multimodal\s+(?:solution|service)(?!\s+(?:AI|model|learning))",
        r"(?:rail|road|sea|air)\s+multimodal",
        r"intermodal|multi[\s-]?modal\s+(?:cargo|container)",
    ],

    # -----------------------------------------------------------------------
    # INTELLIGENT - largest FP category (4,953 instances in corpus)
    #
    # v1.0.x had a catch-all that missed many APAC filing patterns.
    # v1.2.0: explicit patterns for top-frequency Chinese corporate contexts
    # (energy/nuclear/port/mining/city/manufacturing) that dominated FP list.
    # Catch-all is kept as final fallback but AI-context bypass still applies.
    # -----------------------------------------------------------------------
    "intelligent_non_ai": [
        # Transportation & infrastructure
        r"intelligent\s+transport(?:ation)?\s+system",
        r"intelligent\s+(?:traffic|vehicle|highway|road)\s+(?:system|management)",
        r"intelligent\s+building(?:\s+system)?",
        r"intelligent\s+(?:grid|power|energy)\s+(?:system|network|management)",
        r"intelligent\s+(?:city|cities)(?!\s+(?:AI|platform))",
        r"(?:business|market|competitive)\s+intelligence(?!\s+(?:AI|ML|platform))",
        r"intelligent\s+manufactur(?:ing)?(?!\s+(?:AI|ML|robot|automat))",
        # Chinese corporate buzzwords (v7.2.0 original)
        r"intelligent\s+(?:warehouse|warehousing|storage|logistics|sorting)",
        r"intelligent\s+(?:terminal|port|container|customs)",
        r"intelligent\s+(?:mine|mining|drilling|extraction)",
        r"intelligent\s+(?:retail|store|shopping|branch|outlet)",
        r"intelligent\s+(?:agriculture|farming|irrigation|breeding)",
        r"intelligent\s+(?:campus|park|community|property)",
        r"intelligent\s+(?:inspection|monitoring|surveillance|patrol)",
        r"intelligent\s+(?:payment|finance|banking|settlement)",
        r"intelligent\s+(?:healthcare|hospital|medical|diagnosis)",
        r"intelligent\s+(?:education|teaching|learning|classroom)",
        r"intelligent\s+(?:security|safety|protection|fire)",
        r"intelligent\s+(?:driving|parking|charging|navigation)",
        r"intelligent\s+(?:home|appliance|lighting|HVAC)",
        r"intelligent\s+(?:network|connectivity|communication|IoT)",
        r"intelligent\s+(?:operation|operations|management|control|service|services)",
        # v1.2.0 NEW: energy/nuclear/utility patterns from APAC filings
        r"intelligent\s+(?:energy\s+system|power\s+grid|substation|meter|metering)",
        r"intelligent\s+(?:nuclear|hydro(?:power)?|wind\s+power|solar\s+power)",
        r"intelligent\s+(?:transmission|distribution\s+network|dispatching)",
        r"intelligent\s+(?:pipeline|gas\s+network|oil\s+field|refinery)",
        r"intelligent\s+(?:construction|engineering|project\s+management)",
        r"intelligent\s+(?:supply\s+chain|procurement|inventory)",
        r"intelligent\s+(?:risk\s+control|credit\s+management|underwriting)(?!\s+AI)",
        r"intelligent\s+(?:customer\s+service|call\s+center|contact\s+center)(?!\s+(?:AI|powered))",
        # Catch-all: "intelligent" NOT followed by AI-related terms
        r"\bintelligent\b(?!\s+(?:AI|artificial|machine|deep|neural|algorithm|model|agent|automation|robot))",
    ],

    # -----------------------------------------------------------------------
    # AUTOMATION - 1,484 FPs in corpus
    #
    # v1.0.x catch-all was too broad and also caught legitimate RPA/IPA.
    # v1.2.0: Explicit patterns for the three dominant FP contexts:
    #   1. Proxy statement compensation tables (executives describing "automation" as
    #      a supply-chain/operational goal without AI)
    #   2. Financial/annual report boilerplate (cost savings, process improvement)
    #   3. Factory/plant automation (PLC/SCADA — classical, not AI)
    # -----------------------------------------------------------------------
    "automation_non_ai": [
        # Factory / industrial automation without AI context
        r"(?:factory|plant|industrial)\s+automation(?!\s+(?:AI|ML|intelligent))",
        r"automation\s+(?:PLC|SCADA|DCS|controller)",
        # Proxy statement supply-chain boilerplate
        r"(?:supply\s+chain|replenishment|merchandising)\s+automation\s+and\s+accuracy",
        r"(?:supply\s+chain|fulfillment|warehouse)\s+(?:design|productivity)\s+(?:&|and)\s+automation",
        r"automation\s+(?:and\s+accuracy|and\s+efficiency)(?!\s+(?:AI|ML|intelligent|cognitive))",
        # Executive compensation narrative — "automation" as a CTO objective
        r"(?:technology\s+stack|legacy\s+system).*automation(?!\s+(?:AI|ML|intelligent))",
        r"automation.*(?:forecasting|replenishment)(?!\s+(?:AI|ML|model))",
        # Generic process efficiency — no AI signal
        r"\bautomation\b(?!\s+(?:AI|intelligent|cognitive|robotic\s+process|hyper|ML|machine\s+learning))",
    ],

    # -----------------------------------------------------------------------
    # FINE-TUNE / FINE-TUNING - non-AI uses  (v1.0.1 improvement C preserved)
    # -----------------------------------------------------------------------
    "fine_tune_non_ai": [
        r"\bfine[\s-]?tun(?:e|ed|ing)\b(?!\s+(?:AI|model|LLM|GPT|language\s+model|neural|transformer|PEFT|LoRA|adapter))",
        r"\bfine[\s-]?tun(?:e|ed|ing)\b.*(?:capital\s+structure|liquidity|debt|bond|interest\s+rate|dividend|buyback|leverage|covenant)",
        r"(?:capital\s+structure|liquidity|debt|bond|interest\s+rate|dividend|buyback|leverage).*\bfine[\s-]?tun(?:e|ed|ing)\b",
        r"\bfine[\s-]?tun(?:e|ed|ing)\b.*(?:operational|supply\s+chain|cost\s+structure|margin|efficiency)(?!.*(?:AI|model|LLM))",
    ],

    # -----------------------------------------------------------------------
    # KNOWLEDGE BASE / KNOWLEDGE MANAGEMENT - often non-AI
    # -----------------------------------------------------------------------
    "knowledge_non_ai": [
        r"\bknowledge\s+base\b(?!\s+(?:AI|ML|LLM|RAG|vector|retrieval))",
        r"\bknowledge\s+management\b(?!\s+(?:AI|ML|platform|system.*AI))",
    ],

    # -----------------------------------------------------------------------
    # EDGE COMPUTING - 70 FPs from Chinese telecom filings
    #
    # v1.2.0 NEW: "edge computing" in China Mobile, China Telecom, China Unicom
    # reports means 5G MEC (Multi-access Edge Computing) / network infra,
    # not AI edge inference. Requires explicit AI/ML context to pass.
    # -----------------------------------------------------------------------
    "edge_computing_telecom": [
        # Chinese telecom MEC context
        r"edge\s+computing.*(?:5G|MEC|base\s+station|OLT|network\s+slice|bandwidth|latency)",
        r"(?:5G|MEC|base\s+station|network\s+slice).*edge\s+computing",
        r"edge\s+computing.*(?:IaaS|PaaS|cloud\s+service|DICT|RMB|revenue)",
        r"edge\s+computing.*(?:telecom|carrier|operator|mobile\s+network)",
        # Generic infrastructure edge without AI signal
        r"edge\s+computing(?!\s+(?:AI|ML|inference|model|intelligence|vision))",
    ],

    # -----------------------------------------------------------------------
    # CHATBOT / VIRTUAL ASSISTANT boilerplate guard
    #
    # v1.2.0 NEW: 101 FP chatbot / 25 FP virtual assistant instances.
    # Pattern: mentioned in multi-channel contact lists, HR enrollment tools,
    # timeline summaries, or regulatory compliance lists — without evidence
    # that the company built or deployed an AI chatbot itself.
    # -----------------------------------------------------------------------
    "chatbot_boilerplate": [
        # Multi-channel contact list (phone, email, webchat, chatbot)
        r"(?:telephone|phone|email|webchat|sms|whatsapp|social\s+media)[,\s]+(?:chatbot|virtual\s+assistant)",
        r"(?:chatbot|virtual\s+assistant)[,\s]+(?:telephone|phone|email|webchat|connected\s+services)",
        # HR / career enrollment chatbot
        r"(?:career\s+website|apply|candidates?|new\s+hires?).*(?:chatbot|virtual\s+assistant)",
        r"(?:chatbot|virtual\s+assistant).*(?:frequently\s+asked|FAQ|explore.*positions?|apply)",
        # Annual report timeline / milestone list (year + bullet)
        r"20\d{2}[^\n]{0,60}(?:chatbot|virtual\s+assistant)[^\n]{0,60}(?:launch|introduc|implement|deploy)",
        # Compliance / regulatory product list — brief mentions with no context
        r"(?:MIF2|DDA|PRIIPs|MIFID).*(?:chatbot|virtual\s+assistant)",
        # Named internal chatbots without AI-implementation language
        r'(?:chatbot|virtual\s+assistant)\s+["\u201c\u2018]?\w+["\u201d\u2019]?\s+(?:has\s+seen|was\s+released|was\s+launch|was\s+introduc)',
    ],

    # -----------------------------------------------------------------------
    # FRAUD DETECTION / PREDICTIVE MAINTENANCE / PREDICTIVE ANALYTICS
    # boilerplate guard
    #
    # v1.2.0 NEW: These were Tier-1 keywords but appear as FP in:
    #   - Director/executive biography sections (listing expertise areas)
    #   - Risk factor disclosures (mentioning fraud risk without AI solution)
    #   - Capability lists in consulting/tech vendor annual reports
    # Guard: if surrounded by biography / risk-factor boilerplate language,
    # treat as FP regardless of the term's usual high-confidence status.
    # -----------------------------------------------------------------------
    "high_signal_term_boilerplate": [
        # Director biography context — expertise enumeration
        r"(?:LLB|ICAI|CPA|ACCA|chartered\s+accountant|forensic\s+account).*(?:fraud\s+detection|anti[- ]fraud)",
        r"(?:fraud\s+detection|anti[- ]fraud).*(?:LLB|ICAI|CPA|chartered\s+accountant|forensic)",
        # Risk factor section — fraud risk disclosure without AI solution
        r"(?:mandatory\s+e[- ]?learning|online\s+training|ethics\s+module).*(?:fraud|anti[- ]corruption)",
        r"fraud\s+(?:risk|prevention\s+program|awareness)(?!\s+(?:AI|ML|model|detection\s+(?:AI|model|system)))",
        # Predictive maintenance in industrial sales pitch (not implementation)
        r"(?:opportunities?\s+(?:arise|from)|market\s+segment|application\s+experience).*predictive\s+maintenance",
        r"predictive\s+maintenance.*(?:opportunity|opportunities|market\s+potential|sales|revenue)(?!\s+(?:AI|ML|model|system\s+(?:using|powered)))",
        # Predictive analytics in ESG/climate context (weather, not AI)
        r"predictive\s+analytics.*(?:weather|hurricane|climate|storm|flood|natural\s+disaster)",
        r"(?:weather|hurricane|climate|storm|flood).*predictive\s+analytics",
        # Predictive analytics in sustainability/philanthropy context
        r"predictive\s+analytics.*(?:farmer|agriculture|smallholder|crop\s+yield|livelihood)",
    ],

    # -----------------------------------------------------------------------
    # KNOWLEDGE GRAPH / FOUNDATION MODELS - telecom/hardware guard
    #
    # v1.2.0 NEW: Chinese telco and hardware companies (Huawei, China Mobile)
    # use "knowledge graph" for network topology and "foundation models" in
    # general digital-transformation narratives without specific ML deployment.
    # -----------------------------------------------------------------------
    "knowledge_graph_telecom": [
        r"knowledge\s+graph.*(?:network\s+topology|service\s+catalog|ITSM|IT\s+management)",
        r"(?:network\s+topology|service\s+catalog|ITSM).*knowledge\s+graph",
        r"knowledge\s+graph.*(?:bank(?:ing)?|loan|credit|financial\s+product)(?!\s+(?:AI|ML|recommendation))",
    ],
    "foundation_models_generic": [
        # "Foundation" as in philanthropic foundation — not AI foundation model
        r"(?:restoring|supporting|enabling|fueling|enhancing)\s+.*foundation\s+model",
        r"foundation\s+model.*(?:philanthropy|charitable|grant|NGO|nonprofit)",
        # Huawei/telecom "foundation" in network architecture context
        r"foundation\s+(?:network|platform|layer)(?!\s+(?:AI|model|LLM))",
    ],

    # -----------------------------------------------------------------------
    # CLAUDE - person names  (v1.0.1 original preserved)
    # -----------------------------------------------------------------------
    "claude_person_names": [
        r"(?:Mr|Mrs|Ms|Dr|Prof|Jean|Jean-|Pierre-|Marie-)[\s\.]?Claude",
        r"Claude\s+[A-Z][a-z]+(?:son|berg|mann|ier|eau|ard|ert|ini)",
        r"[A-Z]\.\s*Claude",
        r"Claude\s+(?:et\s+al|and\s+colleagues)",
        r"(?:Chairman|CEO|Director|President|Member)\s+Claude",
        r"Claude[,\s]+(?:Chairman|CEO|Director|President|Member)",
        # v1.2.0 NEW: French banking / insurance reports use "Claude" as first name
        # in supervisory board member paragraphs
        r"Claude\s+[A-Z][a-z]+\s*,\s*(?:member|chairman|director|president|vice)",
        r"(?:member|chairman|director|president|vice)[,\s]+Claude\s+[A-Z][a-z]+",
    ],

    # -----------------------------------------------------------------------
    # ROBOTS / ROBOTIC - only explicit industrial patterns  (v1.0.1 preserved)
    # -----------------------------------------------------------------------
    "robots_non_ai": [
        r"(?:welding|painting|palletizing|packaging)\s+robot",
        r"robot(?:ic)?\s+(?:arm|gripper|manipulator|welder|cell)",
        r"robot\s+(?:maintenance|installation|programming)(?!\s+(?:AI|learning|autonomous))",
        r"(?:PLC|conveyor|actuator).*robot",
        r"(?:assembly|production)\s+robot(?!.*(?:AI|learning|autonomous|intelligent|vision))",
        r"(?:da\s+vinci|surgical)\s+robot(?!.*(?:AI|learning|autonomous))",
    ],

    # -----------------------------------------------------------------------
    # MISC FORMAT / ENCODING FALSE POSITIVES
    # -----------------------------------------------------------------------
    "format_errors": [
        r"\bAI[\s-]?\d+",                   # AI-123
        r"\d+[\s-]?AI\b",                   # 123-AI
        r"#AI\b", r"@AI\b",
        r"\bAI\.(com|org|net|io)\b",
        r"\.ai\b",
        r"www\..*\.ai",
    ],

    # -----------------------------------------------------------------------
    # LOCATION NAMES
    # -----------------------------------------------------------------------
    "locations": [
        r"\bDubai\b", r"\bBangkok\b", r"\bThai(?:land)?\b",
        r"\bHawaii\b", r"\bSamurai\b", r"\bBonsai\b",
        r"\bShangh?ai\b", r"\bMumbai\b", r"\bChennai\b",
    ],

    # -----------------------------------------------------------------------
    # MEASUREMENT UNITS (ML / DL as milliliter / deciliter)
    # -----------------------------------------------------------------------
    "measurement_units": [
        r"\bML\b(?=\s*(?:of\s+)?(?:water|metric|tons?|gallons?|liters?))",
        r"\bDL\b(?=\s*(?:of\s+)?(?:water|waste|emissions?))",
        r"\d+\.?\d*\s*(?:ML|DL)\b",
        r"\bML\b(?=\s*\d)",
    ],

    # -----------------------------------------------------------------------
    # BOARD / GOVERNANCE BIOGRAPHY FALSE POSITIVES
    # -----------------------------------------------------------------------
    "board_biography": [
        r"(?:BOD|board\s+of\s+directors?).*?(?:composed|experience|knowledge).*?AI",
        r"(?:communication|media|security),?\s*AI,?\s*(?:and\s+)?(?:cloud|digital)",
        r"(?:spouses?|lineal|descendants?|ascendants?).*?(?:AI|artificial)",
        r"(?:CEO|CFO|CTO|COO|founder|president)\s+(?:of|at)\s+\w+\s*AI\b",
        r"career\s+highlights.*?(?:AI|artificial)",
        r"joined\s+the\s+board.*?(?:AI|artificial)",
    ],

    # -----------------------------------------------------------------------
    # DOCUMENT HEADER REPETITIONS
    # -----------------------------------------------------------------------
    "document_headers": [
        r"AI\s+Company\s+Business\s+Overview",
        r"Special\s+Report.*?AI\s+Company",
    ],

    # -----------------------------------------------------------------------
    # CORPORATE GOVERNANCE BOILERPLATE (v1.1.0 preserved)
    # -----------------------------------------------------------------------
    "governance_boilerplate": [
        r"supervisory\s+board\s+members\s+as\s+a\s+whole\s+must\s+(?:be\s+)?familiar",
        r"members\s+as\s+a\s+whole.*?familiar\s+with\s+the\s+sector",
        r"able\s+to\s+assess\s+the\s+business\s+conducted\s+by\s+the\s+company",
        r"(?:fiscal|fy)\s*20\d{2}\s+(?:target\s+)?(?:TDC|total\s+direct\s+compensation)",
        r"(?:below|above)\s+the\s+median.*?(?:peer\s+group|compensation)",
        r"proposal\s+no\.?\s*\d+\s+election\s+of\s+directors",
        r"safe\s+harbor\s+for\s+forward[- ]looking\s+statements",
        r"nature\s+of\s+forward[- ]looking\s+statements",
        r"forward[- ]looking\s+statements.*?exchange\s+act",
    ],

    # -----------------------------------------------------------------------
    # GEMINI / COPILOT / TITAN - astronomy, aerospace, non-AI uses
    # -----------------------------------------------------------------------
    "gemini_non_ai": [
        r"(?:Project|NASA|Apollo|constellation|zodiac|horoscope|astrolog)\s+Gemini",
        r"Gemini\s+(?:project|mission|spacecraft|capsule|astronaut|program|constellation)",
        r"Gemini\s+(?:clinical\s+trial|trial|study|protocol|phase)",
        r"Gemini\s+(?:software|application|client)(?!\s+(?:AI|model))",
    ],
    "copilot_non_ai": [
        r"co[- ]?pilot\b(?!\s+(?:AI|Microsoft|GitHub|365|azure|assistant))",
    ],
    "titan_non_ai": [
        r"(?:Titan\s+(?:missile|rocket|submersible|moon|saturn|arum|crane|cement|insurance))",
        r"(?:Atlas|Titan)\s+(?:IV|V|rocket|launch\s+vehicle)",
    ],

    # -----------------------------------------------------------------------
    # LORA / SORA - person names and financial table substrings  (v1.2.0 NEW)
    # Vendor gate in AMBIGUOUS_PRODUCT_VENDORS is primary; these catch the
    # most common non-AI surface forms before vendor gate is even checked.
    # -----------------------------------------------------------------------
    "lora_person_name": [
        # TSMC and other Asian corporate filings list "Lora Ho" as CFO/SVP
        r"Lora\s+Ho\b",
        r"\bLora\b\s+[A-Z][a-z]+\s*,\s*(?:Senior\s+Vice\s+President|CFO|Chief\s+Financial)",
        # "LoRA" as substring in entity lists (Lightsource SPV, etc.)
        r"Lightsource\s+(?:SPV|BP)\s+\d+",
        r"(?:SPV|Special\s+Purpose\s+Vehicle)\s+\d+.*Lora",
    ],
    "sora_financial_table": [
        # "Sora" as substring in bond/debt table rows — Italian/Spanish words,
        # or partial OCR of "SORA" benchmark rate, or SPV suffix
        r"\bSORA\b(?!\s+(?:OpenAI|video|generation|model))",   # SORA rate (Singapore)
        r"(?:bond|note|issuance|coupon|maturity).*\bSora\b",
        r"\bSora\b.*(?:bond|note|coupon|maturity|RMB|EUR|USD\s+\d)",
        r"\bSora\b.*(?:GmbH|Ltd|S\.A\.|S\.p\.A\.|KG|BV)\b",   # company name suffix
    ],

    # -----------------------------------------------------------------------
    # LLMS - Loan Lifecycle Management System  (v1.2.1 NEW)
    #
    # 28 FP confirmed: Indian/Asian bank annual reports use "LLMS" as acronym
    # for "Loan Life Cycle Management System" or "Loan Lifecycle Management".
    # ChatGPT Deep Thinking mislabeled all 28 as "LLM / Large Language Models".
    # Pattern requires NO AI/LLM context in the surrounding window.
    # -----------------------------------------------------------------------
    "llms_banking_system": [
        r"\bLLMS\b(?!.*(?:large\s+language|language\s+model|GPT|token|prompt|anthropic|openai|meta\s+llama|mistral))",
        r"Loan\s+Life\s*[Cc]ycle\s+Management\s+System",
        r"Loan\s+Lifecycle\s+Management\s+System",
        r"\bLLMS\b.*(?:loan|credit|LOS|lifecycle|origination|sanction|disbursement)",
        r"(?:loan|credit|LOS|lifecycle|origination).*\bLLMS\b",
    ],

    # -----------------------------------------------------------------------
    # ML / DL / RPA / NLP as abbreviations in non-AI contexts  (v1.2.1 NEW)
    #
    # Short acronyms detected as AI terms but appear frequently in non-AI text:
    #   ML = milliliter, mileage, or standalone in luxury/auto reporting
    #   DL = deciliter, or "DL" in telecom/logistics tables
    #   RPA = in ESG audit context without AI/automation signal
    #   NLP = Natural Language Processing only when in non-tech context
    # NOTE: These are CATCH-ALL guards; skipped when _AI_CONTEXT_RE fires.
    # -----------------------------------------------------------------------
    "acronym_non_ai": [
        # ML in luxury vehicle / geography context (Lincoln China, Ford market)
        r"\bML\b.*(?:luxury|vehicle|market|China|brand|segment|consumer)(?!.*(?:model|learning|algorithm))",
        r"(?:luxury|vehicle|China|brand|automotive)\b.*\bML\b(?!.*(?:model|learning))",
        # RPA in ESG audit / seminar context without automation/AI signal
        r"\bRPA\b.*(?:audit|ESG|seminar|awareness|compliance|trend)(?!.*(?:automat|process|bot|AI))",
        r"(?:audit|ESG|seminar|awareness|internal\s+audit).*\bRPA\b(?!.*(?:automat|process))",
        # DL in telecom/logistics table context
        r"\bDL\b.*(?:downlink|bandwidth|throughput|Mbps|Gbps|latency|5G|4G)",
        r"(?:uplink|downlink|UL/DL|UL\s*/\s*DL)\b",
    ],

    # -----------------------------------------------------------------------
    # MODEL SERVING / MODEL DEPLOYMENT / MODEL INFERENCE in banking context
    # (v1.2.1 NEW)
    #
    # 7 FP: "model" in "credit model", "risk model", "IRB model", "scoring model"
    # combined with "serving/deployment/inference" in bank risk reports.
    # Pattern: these appear in credit risk / Basel IRB sections.
    # Only fires when NOT followed by AI/cloud/production context within ~60 chars.
    # -----------------------------------------------------------------------
    "model_banking_context": [
        r"\bmodel\s+(?:serving|deployment|inference)\b.*(?:credit|loan|risk|IRB|Basel|scoring|rating)",
        r"(?:credit|loan|risk|IRB|Basel|scoring|rating\s+model).*\bmodel\s+(?:serving|deployment|inference)\b",
        # Catch-all: model serving/deployment/inference NOT followed by AI/cloud tech within 80 chars
        r"\bmodel\s+serving\b(?!.{0,80}(?:AI|ML|inference|endpoint|cloud|GPU|latency|container|docker))",
        r"\bmodel\s+deployment\b(?!.{0,80}(?:AI|ML|production|cloud|API|endpoint|monitoring|MLOps|docker))",
        r"\bmodel\s+inference\b(?!.{0,80}(?:AI|ML|GPU|latency|batch|real.time|endpoint|accelerat))",
    ],

    # -----------------------------------------------------------------------
    # RECOMMENDATION SYSTEM / ENGINE in non-AI HR or wellness context
    # (v1.2.1 NEW)
    #
    # 4 FP: "recommendation system/engine" in HR benefits, wellness platforms,
    # or generic business recommendation contexts without ML/personalization.
    # -----------------------------------------------------------------------
    "recommendation_non_ai": [
        r"\brecommendation\s+(?:system|engine)\b.*(?:HR|benefit|wellness|employee|compliance|audit)",
        r"(?:HR|benefit|wellness|employee\s+survey|compliance).*\brecommendation\s+(?:system|engine)\b",
        # "recommendation system" in water/environment tools (Water Risk Monetizer)
        r"\brecommendation\s+(?:system|engine)\b.*(?:water|environmental|sustainability|ESG|risk\s+tool)",
    ],

    # -----------------------------------------------------------------------
    # FRONTIER MODEL in export control / trade context  (v1.2.1 NEW)
    #
    # 1 FP: "frontier model" in semiconductor export control sections
    # (e.g. "frontier model" as in frontier/advanced technology subject to
    # export restrictions — not OpenAI o1 / Anthropic Claude 3).
    # -----------------------------------------------------------------------
    "frontier_model_export": [
        r"\bfrontier\s+model\b.*(?:export\s+control|trade\s+restriction|sanction|supply\s+chain|distributor)",
        r"(?:export\s+control|trade\s+restriction|BIS|EAR|ITAR).*\bfrontier\s+model\b",
    ],

    # -----------------------------------------------------------------------
    # MISTRAL in non-AI contexts  (v1.2.1 NEW)
    #
    # 1 FP: "Mistral" as wind / region name in French construction/energy
    # reports (Bouygues group). Vendor gate handles Mistral AI, but the
    # explicit pattern provides a faster catch for geographic/weather context.
    # -----------------------------------------------------------------------
    "mistral_non_ai": [
        r"\bMistral\b.*(?:wind|région|Bouygues|construction|subsidiary|infrastructure|energy\s+system)",
        r"(?:wind|région|Bouygues|construction|offshore).*\bMistral\b(?!\s+(?:AI|model|LLM|7B|8x7B))",
    ],
}



# -----------------------------------------------------------------------
# AI context signal - used by check_false_positive() to skip catch-all
# filters when strong AI context is present nearby.
#
# v1.0.1 extensions: autonomous, self-driving, SLAM, sensor fusion, drone,
#     path planning, reinforcement learning, computer use, agentic
# v1.2.0 extensions: APAC vendor names (Huawei AI, Alibaba AI, Baidu,
#     Tencent AI, Samsung AI, Kakao, Naver HyperCLOVA) that appear in
#     APAC filings and increase precision for catch-all bypass.
#     Also added: agentic workflow, AI-native, AI copilot, multimodal AI,
#     embedding model, vector search — to better catch true positives.
# -----------------------------------------------------------------------
_AI_CONTEXT_RE = re.compile(
    r"\b(?:AI\b|artificial intelligence|machine learning|ML\b|deep learning|neural|"
    r"NLP|LLM|generative|GenAI|chatbot|RPA|robotic process|cognitive computing|"
    r"computer vision|natural language|"
    r"GPT|BERT|transformer\s+model|foundation\s+model|language\s+model|"
    r"pre[\s-]?train|training\s+data|"
    r"ChatGPT|Copilot|Gemini|Claude|Watson|SageMaker|TensorFlow|PyTorch|"
    # v1.0.1 additions for robotics/autonomy context
    r"autonomous|self[- ]driving|SLAM|sensor\s+fusion|"
    r"path\s+planning|reinforcement\s+learning|computer\s+use|agentic|"
    r"drone|UAV|AGV|autonomous\s+vehicle|autonomous\s+robot|robot\s+learning|"
    # v1.2.0 additions — APAC vendor AI brands
    r"Huawei\s+(?:AI|Cloud|Ascend)|Alibaba\s+(?:AI|Cloud|DAMO)|Baidu\s+(?:AI|ERNIE)|"
    r"Tencent\s+(?:AI|Hunyuan)|Samsung\s+(?:AI|Gauss)|Kakao|HyperCLOVA|"
    r"Pangu|Wenxin|ERNIE\s+Bot|Tongyi|Qwen|Kimi|Doubao|"
    # v1.2.0 additions — modern AI architecture / product terms
    r"AI[- ]native|AI\s+copilot|multimodal\s+(?:AI|model)|"
    r"embedding\s+model|vector\s+(?:search|store|database|index)|"
    r"agentic\s+workflow|agent\s+framework|tool\s+(?:calling|use)|"
    r"inference\s+(?:engine|endpoint|server)|model\s+(?:fine[- ]?tun|deploy|serving))",
    re.IGNORECASE,
)

# Patterns that are "catch-alls" - skipped when AI context is nearby
# NOTE: \brobot... catch-all was REMOVED in v1.0.1 (improvement B)
_CATCHALL_PREFIXES = (
    r"\bintelligent\b(?!",
    r"\bautomation\b(?!",
    r"\bembedding\b(?!",
    r"\bknowledge\s+base\b(?!",
    r"\bknowledge\s+management\b(?!",
    r"\bfine[\s-]?tun",
    r"edge\s+computing(?!",           # v1.2.0
    r"(?:chatbot|virtual\s+assistant)",  # v1.2.0 — boilerplate check skipped if AI context
)


# ============================================================================
# AI APPLICATIONS (Dimension A) - A1 through A8
# ============================================================================

_AI_APPLICATIONS_RAW: Dict[str, Dict] = {
    "A1_Product_Innovation": {
        "name": "Product & Service Innovation",
        "description": "AI-enhanced products and services; R&D acceleration; New product features",
        "keywords": [
            "AI-powered product", "AI-enabled", "AI-powered", "smart product",
            "AI feature", "AI-first", "next-generation product",
            "R&D", "prototype", "design", "patent", "breakthrough", "discovery",
            "innovation center", "frontier research",
            "academic collaboration", "university partnership", "research center",
            "AI research", "ML research",
        ],
        "patterns": [
            r"AI[\s-]?powered.*(?:product|feature|solution)",
            r"(?:AI|ML)[\s-]?enabled.*(?:product|feature|solution)",
            r"R&D.*(?:AI|artificial intelligence|machine learning)",
            r"(?:develop|build|create).*AI.*(?:product|solution)",
            r"(?:AI|ML)\s+(?:patent|research|lab|center)",
            r"(?:research|innovation|lab).*(?:AI|artificial intelligence|machine learning)",
            r"(?:AI|ML)\s+(?:breakthrough|innovation|discovery)",
            r"(?:academic|university).*(?:AI|ML)\s+(?:research|collaboration)",
        ],
        "keyword_tiers": {
            "AI-powered product": 1, "AI-enabled": 1, "AI-powered": 1,
            "smart product": 1, "AI feature": 1, "AI-first": 1,
            "AI research": 1, "ML research": 1,
            "prototype": 2, "design": 2, "patent": 2,
            "breakthrough": 2, "discovery": 2,
            "R&D": 3, "innovation center": 3, "frontier research": 3,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["人工智能产品", "AI产品", "智能产品", "人工智能研发", "AI创新"],
    },
    "A2_Operational_Excellence": {
        "name": "Operational Excellence & Automation",
        "description": "Process automation; Supply chain optimization; Operational efficiency",
        "keywords": [
            "intelligent automation", "hyperautomation", "IPA", "process automation",
            "RPA", "robotic process automation",
            "efficiency", "streamline", "optimize",
            "process improvement", "implementation", "rollout", "integration",
            "operational excellence", "productivity",
            "supply chain", "logistics optimization", "predictive maintenance",
            "quality control", "resource allocation",
        ],
        "patterns": [
            r"deploy(?:ed|ing|ment).*(?:AI|ML|machine learning)",
            r"(?:AI|ML).*automat(?:e|ed|ing|ion)",
            r"(?:AI|ML).*(?:efficiency|optimization|productivity)",
            r"(?:intelligent|hyper)[\s-]?automation",
            r"(?:supply chain|logistics).*(?:AI|ML|optimization)",
            r"(?:predictive|preventive)\s+maintenance",
            r"(?:process|operational).*(?:optimization|excellence).*(?:AI|ML)",
            r"(?:RPA|robotic\s+process\s+automation)",
        ],
        "keyword_tiers": {
            "intelligent automation": 1, "hyperautomation": 1, "RPA": 1,
            "robotic process automation": 1, "process automation": 1,
            "predictive maintenance": 1, "supply chain": 1, "logistics optimization": 1,
            "efficiency": 2, "optimize": 2, "streamline": 2,
            "productivity": 2, "operational excellence": 2,
            "implementation": 3, "rollout": 3, "integration": 3,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["智能化运营", "流程自动化", "机器人流程自动化", "智能自动化", "预测性维护"],
    },
    "A3_Customer_Experience": {
        "name": "Customer Experience & Engagement",
        "description": "Customer service automation; Personalization; Marketing optimization",
        "keywords": [
            "chatbot", "virtual assistant", "conversational AI", "customer service AI",
            "support automation", "customer experience", "CX", "customer journey",
            "personalization", "recommendation engine", "recommender system",
            "targeted", "customized experience",
            "marketing automation", "sales optimization", "lead scoring",
            "customer insights", "sentiment analysis", "voice of customer",
        ],
        "patterns": [
            r"(?:AI|ML).*(?:chatbot|virtual assistant|conversational)",
            r"personali[zs](?:e|ed|ation).*(?:customer|experience|AI|ML)",
            r"customer.*(?:experience|service).*(?:AI|ML)",
            r"(?:recommendation|recommender)\s+(?:system|engine)",
            r"(?:AI|ML).*(?:sentiment|customer\s+insights)",
        ],
        "keyword_tiers": {
            "chatbot": 1, "virtual assistant": 1, "conversational AI": 1,
            "recommendation engine": 1, "recommender system": 1,
            "personalization": 2, "customer experience": 2,
            "marketing automation": 2, "sentiment analysis": 2,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["智能客服", "聊天机器人", "虚拟助手", "个性化推荐", "智能推荐"],
    },
    "A4_Risk_Compliance": {
        "name": "Risk Management & Compliance",
        "description": "Fraud detection; Cybersecurity AI; Regulatory compliance; Risk assessment",
        "keywords": [
            "fraud detection", "fraud prevention", "threat detection",
            "anomaly detection", "intrusion detection",
            "AML", "KYC", "anti-money laundering", "regulatory compliance",
            "risk assessment", "credit risk", "model risk",
            "risk scoring", "AI underwriting",
        ],
        "patterns": [
            r"(?:fraud|anomaly)\s+detection.*(?:AI|ML|model)?",
            r"(?:cyber|threat)\s+(?:detection|intelligence).*(?:AI|ML)",
            r"(?:AML|KYC|anti[\s-]?money\s+laundering)",
            r"(?:compliance|regulatory).*(?:AI|ML|automation)",
            r"(?:risk|credit)\s+(?:assessment|scoring).*(?:AI|ML|model)",
        ],
        "keyword_tiers": {
            "fraud detection": 1, "fraud prevention": 1, "anomaly detection": 1,
            "threat detection": 1, "AML": 1, "KYC": 1,
            "risk assessment": 2, "credit risk": 2, "risk scoring": 2,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["风险管理AI", "智能风控", "反欺诈", "信用风险AI", "异常检测"],
    },
    "A5_Data_Analytics": {
        "name": "Data Analytics & Business Intelligence",
        "description": "Predictive analytics; Business intelligence; Data-driven decision making",
        "keywords": [
            "predictive analytics", "forecasting model", "prediction model",
            "advanced analytics", "augmented analytics",
            "data science", "ML modeling", "algorithm",
            "pattern recognition", "trend analysis",
            "decision support", "performance analytics",
        ],
        "patterns": [
            r"predictive\s+(?:analytics|model|insight)",
            r"(?:machine learning|ML).*(?:model|algorithm|prediction)",
            r"data[\s-]?driven.*(?:insight|decision|strategy).*(?:AI|ML)?",
            r"(?:advanced|augmented)\s+analytics",
            r"business\s+intelligence.*(?:AI|ML)",
        ],
        "keyword_tiers": {
            "predictive analytics": 1, "advanced analytics": 1, "augmented analytics": 1,
            "forecasting model": 1, "prediction model": 1,
            "data science": 2, "ML modeling": 2, "pattern recognition": 2,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["数据分析", "机器学习分析", "预测分析", "大数据分析", "商业智能AI"],
    },
    "A6_Strategy_Investment": {
        "name": "AI Strategy & Investment",
        "description": "Strategic initiatives; Budget allocation; Partnerships; ROI measurement",
        "keywords": [
            "AI investment", "AI spending", "AI budget",
            "AI acquisition", "AI funding",
            "AI strategy", "AI transformation", "AI roadmap",
            "strategic AI initiative", "AI-first strategy",
            "AI partnership", "AI venture", "AI ecosystem",
            "pilot project", "proof of concept", "POC",
            "AI center", "center of excellence", "AI program",
        ],
        "patterns": [
            r"invest(?:ed|ing|ment).*(?:AI|artificial intelligence)",
            r"(?:AI|artificial intelligence).*(?:budget|funding|capital)",
            r"acquir(?:ed|ing|ition).*(?:AI|machine learning)",
            r"\$[\d,.]+\s*(?:million|billion|M|B).*(?:AI|artificial intelligence)",
            r"(?:AI|ML)\s+(?:strategy|transformation|initiative)",
            r"(?:center\s+of\s+excellence|innovation\s+lab).*AI",
        ],
        "keyword_tiers": {
            "AI investment": 1, "AI strategy": 1, "AI transformation": 1,
            "AI center": 1, "center of excellence": 1,
            "AI partnership": 2, "pilot project": 2, "POC": 2,
            "AI roadmap": 2, "AI program": 2,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["人工智能战略", "AI投资", "人工智能布局", "数字化转型", "AI规划"],
    },
    "A7_Governance_Ethics": {
        "name": "AI Governance & Ethics",
        "description": "Responsible AI; Bias mitigation; Explainability; Ethical guidelines",
        "keywords": [
            "responsible AI", "ethical AI", "trustworthy AI", "AI ethics",
            "AI governance", "AI policy", "governance framework",
            "bias detection", "bias mitigation", "fairness in AI",
            "algorithmic fairness", "inclusive AI",
            "explainability", "XAI", "interpretability", "model interpretability",
            "AI transparency", "AI accountability",
            "AI safety", "guardrails", "hallucination", "alignment",
            "red teaming", "model evaluation",
        ],
        "patterns": [
            r"responsible\s+(?:AI|artificial intelligence)",
            r"(?:AI|ML).*(?:ethics|governance|compliance)",
            r"(?:bias|fairness).*(?:detection|mitigation|audit).*(?:AI|ML|model)",
            r"AI\s+(?:risk|policy|governance)\s+framework",
            r"(?:explainab|interpretab)(?:le|ility).*(?:AI|ML|model)",
            r"(?:trustworthy|transparent|accountable)\s+(?:AI|ML)",
        ],
        "keyword_tiers": {
            "responsible AI": 1, "ethical AI": 1, "AI ethics": 1,
            "AI governance": 1, "explainability": 1, "XAI": 1,
            "bias detection": 2, "bias mitigation": 2, "AI safety": 2,
            "guardrails": 2, "hallucination": 2, "red teaming": 2,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["AI治理", "算法监管", "负责任AI", "AI伦理", "可信AI"],
    },
    "A8_Talent_Workforce": {
        "name": "AI Talent & Workforce Development",
        "description": "Training; Upskilling; Talent acquisition; Change management",
        "keywords": [
            "AI upskilling", "AI reskilling", "AI training",
            "AI literacy", "capability building", "AI skill development",
            "AI talent", "data scientist", "ML engineer",
            "AI team", "AI expertise", "AI hiring",
            "AI adoption", "AI workforce", "AI culture",
        ],
        "patterns": [
            r"(?:AI|ML).*(?:talent|skill|training|team)",
            r"(?:upskill|reskill)(?:ing)?.*(?:AI|data|digital)",
            r"hir(?:e|ed|ing).*(?:AI|ML|data).*(?:engineer|scientist)",
            r"(?:workforce|employee).*(?:training|development).*AI",
            r"AI\s+(?:literacy|capability|adoption)\s+program",
        ],
        "keyword_tiers": {
            "AI talent": 1, "data scientist": 1, "ML engineer": 1,
            "AI team": 1, "AI literacy": 1,
            "AI upskilling": 2, "AI reskilling": 2, "AI training": 2,
            "AI adoption": 2, "capability building": 2,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["AI人才", "智能招聘", "数字员工", "AI技能培训", "人工智能人才"],
    },
}


# ============================================================================
# AI TECHNOLOGIES (Dimension B) - B1 through B8
#
# v1.0.1 (improvement D): B8_General_AI reduced by promoting specific patterns
# to the appropriate B1-B7 categories. B8 is now truly "generic/unspecified".
# ============================================================================

_AI_TECHNOLOGIES_RAW: Dict[str, Dict] = {
    "B1_Traditional_ML": {
        "name": "Traditional Machine Learning",
        "description": "Supervised/unsupervised learning; Ensemble methods; Statistical models; Predictive/forecasting models",
        "keywords": [
            "machine learning", "supervised learning", "unsupervised learning",
            "classification model", "regression model", "clustering algorithm",
            "random forest", "decision tree", "logistic regression",
            "support vector machine", "SVM", "k-means", "XGBoost",
            "gradient boosting", "ensemble method", "bagging", "boosting",
            "feature engineering", "model training", "cross-validation",
            "hyperparameter tuning",
            # v1.0.1: promoted from B8
            "predictive model", "forecasting model", "prediction model",
            "predictive analytics", "predictive algorithm",
            "recommendation algorithm", "recommendation model",
            "anomaly detection model",
        ],
        "patterns": [
            r"machine\s+learning(?!\s+(?:operation|platform|infrastructure))",
            r"\bML\b\s+(?:model|algorithm|technique|approach|system|pipeline)",
            r"(?:supervised|unsupervised)\s+learning",
            r"(?:random\s+forest|decision\s+tree|XGBoost)",
            r"(?:classification|regression|clustering)\s+(?:model|algorithm)",
            r"(?:train|build|develop).*ML.*(?:model|algorithm)",
            # v1.0.1: predictive/forecasting patterns promoted from B8
            r"predictive\s+(?:model|algorithm|system|analytics)",
            r"forecasting\s+(?:model|algorithm|system)",
            r"recommendation\s+(?:model|algorithm|engine|system)",
            r"anomaly\s+detection\s+(?:model|algorithm|system)",
        ],
        "keyword_tiers": {
            "machine learning": 1, "supervised learning": 1, "unsupervised learning": 1,
            "random forest": 1, "decision tree": 1, "XGBoost": 1,
            "gradient boosting": 1, "support vector machine": 1, "SVM": 1,
            "predictive model": 1, "forecasting model": 1, "predictive analytics": 1,
            "recommendation model": 1, "anomaly detection model": 1,
            "classification model": 2, "regression model": 2, "clustering algorithm": 2,
            "ensemble method": 2, "bagging": 2, "boosting": 2,
            "feature engineering": 3, "model training": 3, "cross-validation": 3,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["机器学习", "监督学习", "预测模型", "决策树", "随机森林"],
    },
    "B2_Deep_Learning": {
        "name": "Deep Learning & Neural Networks",
        "description": "DNN, CNN, RNN; Reinforcement learning; Transfer learning",
        "keywords": [
            "deep learning", "neural network", "artificial neural network",
            "deep neural network", "DNN",
            "convolutional neural network", "CNN",
            "recurrent neural network", "RNN", "LSTM", "GRU",
            "autoencoder", "GAN", "generative adversarial network",
            "backpropagation", "gradient descent", "activation function",
            "dropout", "batch normalization", "transfer learning",
            "reinforcement learning", "Q-learning", "policy gradient",
        ],
        "patterns": [
            r"deep\s+learning",
            r"(?:deep\s+)?neural\s+network",
            r"\b(?:CNN|RNN|LSTM|GRU|DNN)\b",
            r"(?:convolutional|recurrent)\s+neural",
            r"reinforcement\s+learning",
            r"(?:generative\s+adversarial|GAN)\s+(?:network|model)",
        ],
        "keyword_tiers": {
            "deep learning": 1, "neural network": 1, "DNN": 1,
            "CNN": 1, "RNN": 1, "LSTM": 1, "GRU": 1,
            "reinforcement learning": 1, "transfer learning": 1,
            "autoencoder": 2, "GAN": 2,
            "backpropagation": 3, "gradient descent": 3,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["深度学习", "神经网络", "深度神经网络", "卷积神经网络", "循环神经网络"],
    },
    "B3_NLP_NonLLM": {
        "name": "Natural Language Processing (Non-LLM)",
        "description": "Text classification; NER; Traditional NLP; Sentiment analysis",
        "keywords": [
            "natural language processing", "NLP", "text analysis",
            "text mining", "text classification", "information extraction",
            "named entity recognition", "NER", "part-of-speech", "POS tagging",
            "dependency parsing", "tokenization", "lemmatization",
            "sentiment analysis", "opinion mining",
            "Word2Vec", "GloVe", "fastText",
            "topic modeling", "TF-IDF",
        ],
        "patterns": [
            r"natural\s+language\s+processing(?!\s+(?:model|LLM))",
            r"\bNLP\b(?!\s+(?:model|LLM))",
            r"(?:text|sentiment)\s+(?:analysis|classification|mining)",
            r"named\s+entity\s+recognition",
            r"(?:Word2Vec|GloVe|fastText)",
            r"topic\s+model(?:ing)?",
            # v1.0.1: NER / sentiment promoted from B8
            r"(?:NER|named\s+entity)\s+(?:extraction|model|system)",
            r"sentiment\s+(?:model|scoring|classification)(?!\s+AI)",
        ],
        "keyword_tiers": {
            "natural language processing": 1, "NLP": 1,
            "named entity recognition": 1, "NER": 1,
            "sentiment analysis": 1, "text classification": 1,
            "Word2Vec": 2, "GloVe": 2, "topic modeling": 2,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["自然语言处理", "NLP技术", "语音识别", "文本分类", "情感分析"],
    },
    "B4_GenAI_LLMs": {
        "name": "Generative AI & Large Language Models",
        "description": "ChatGPT, Copilot, Gemini; Text/code generation; Foundation models",
        "keywords": [
            "generative AI", "GenAI", "generative model",
            "large language model", "LLM", "foundation model",
            "transformer model", "transformer architecture",
            "attention mechanism", "self-attention",
            "GPT", "ChatGPT", "GPT-3", "GPT-4", "GPT-4o", "OpenAI",
            "Gemini", "PaLM", "Google AI",
            "Llama", "LLaMA", "Mistral", "Cohere",
            "Copilot", "GitHub Copilot", "Microsoft 365 Copilot",
            "Amazon Q", "Amazon Bedrock", "CodeWhisperer",
            "prompt engineering", "few-shot", "zero-shot", "chain-of-thought",
            "fine-tuning", "RLHF", "instruction tuning",
            "RAG", "retrieval-augmented", "vector database",
            "text generation", "content generation", "code generation",
            "text-to-image", "image generation", "DALL-E", "Stable Diffusion", "Midjourney",
        ],
        "patterns": [
            r"generativ(?:e)?\s*(?:AI|artificial intelligence)",
            r"(?:large\s+)?language\s+model(?:s)?",
            r"\bLLM(?:s)?\b",
            r"\bGPT[\s-]?[3-5]?(?:o)?\b",
            r"\bChatGPT\b",
            r"(?:Anthropic|Anthropic's)\s+Claude",
            r"Claude\s+(?:AI|model|assistant|chatbot|2|3|Opus|Sonnet|Haiku)",
            r"\b(?:Gemini|PaLM)\b(?!\s+(?:project|program|constellation|mission|trial|study|clinical))",
            r"foundation\s+model(?:s)?",
            r"\bGenAI\b",
            r"(?:retrieval[\s-]?augmented|RAG)",
            r"vector\s+(?:database|store|search)",
            r"(?:prompt|instruction)\s+(?:engineering|tuning)",
            r"(?:text|image|code)\s+generation",
            r"transformer\s+(?:model|architecture|network|layer|based)",
            r"(?:multi[\s-]?modal|vision[\s-]?language)\s+(?:AI|model|LLM)",
            # v1.0.1: LLM/RAG/prompt patterns promoted from B8
            r"\b(?:few|zero)[\s-]shot\b",
            r"chain[\s-]of[\s-]thought",
            r"prompt\s+(?:injection|template|design)",
            r"context\s+window\b",
            r"token(?:ization|izer|s)\s+(?:model|LLM)",
        ],
        "keyword_tiers": {
            "generative AI": 1, "GenAI": 1, "LLM": 1, "large language model": 1,
            "ChatGPT": 1, "GPT-4": 1, "Copilot": 1, "GitHub Copilot": 1,
            "Gemini": 1, "foundation model": 1, "RAG": 1,
            "prompt engineering": 2, "fine-tuning": 2, "RLHF": 2,
            "text generation": 2, "code generation": 2,
            "vector database": 2, "retrieval-augmented": 2,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["大语言模型", "生成式AI", "大模型", "生成式人工智能", "基础模型"],
    },
    "B5_Computer_Vision": {
        "name": "Computer Vision",
        "description": "Image recognition; Object detection; Video analytics; Visual inspection",
        "keywords": [
            "computer vision", "image recognition", "image processing",
            "visual recognition", "image analysis",
            "object detection", "object recognition", "image classification",
            "semantic segmentation", "instance segmentation",
            "facial recognition", "face detection", "face recognition",
            "OCR", "optical character recognition",
            "video analytics", "video analysis", "visual inspection",
            "quality inspection", "defect detection", "visual quality control",
            "image segmentation", "3D vision", "depth estimation",
            "ResNet", "VGG", "YOLO", "R-CNN", "EfficientNet",
        ],
        "patterns": [
            r"computer\s+vision",
            r"(?:image|object|face|facial)\s+(?:recognition|detection|classification)",
            r"visual\s+(?:recognition|inspection|analysis)",
            r"video\s+analytics",
            r"(?:OCR|optical\s+character\s+recognition)",
            r"\b(?:YOLO|ResNet|VGG|R-CNN)\b",
        ],
        "keyword_tiers": {
            "computer vision": 1, "image recognition": 1, "object detection": 1,
            "facial recognition": 1, "YOLO": 1, "R-CNN": 1,
            "video analytics": 2, "visual inspection": 2, "OCR": 2,
            "semantic segmentation": 2, "defect detection": 2,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["计算机视觉", "图像识别", "人脸识别", "目标检测", "视觉检测"],
    },
    "B6_Robotics_Autonomous": {
        "name": "Robotics & Autonomous Systems",
        "description": "Autonomous vehicles; AI-powered robotics; Agentic AI; Multi-agent systems",
        "keywords": [
            "autonomous vehicle", "self-driving", "autonomous driving",
            "autonomous navigation", "path planning",
            "intelligent robot", "cognitive robot", "autonomous robot",
            "collaborative robot AI", "cobot AI", "robot learning",
            "AI-powered robot", "robot vision",
            "agentic AI", "AI agent", "autonomous agent",
            "multi-agent system", "agent orchestration", "agent framework",
            "tool use", "function calling", "computer use",
            "drone AI", "UAV AI", "autonomous drone",
            "AGV", "automated guided vehicle",
            # v1.0.1: added autonomy terms (previously missing from B6)
            "SLAM", "sensor fusion", "lidar AI", "autonomous system",
        ],
        "patterns": [
            r"(?:autonomous|self[\s-]?driving)\s+(?:vehicle|system|car|truck)",
            r"(?:agentic)\s+(?:AI|system|workflow)",
            r"(?:autonomous\s+)?AI\s+agent(?:s)?",
            r"multi[\s-]?agent\s+(?:system|orchestration)",
            r"agent\s+(?:orchestration|framework|coordination)",
            r"(?:tool|function)\s+(?:use|calling)",
            r"(?:drone|UAV|AGV).*(?:AI|autonomous|intelligent)",
            r"(?:cognitive|intelligent|autonomous)\s+robot",
            r"(?:collaborative\s+)?robot.*(?:AI|learning|intelligent|vision)",
            r"(?:AI|ML)[\s-]?powered\s+robot",
            # v1.0.1: autonomy/navigation patterns (improvement B)
            r"\bSLAM\b",
            r"sensor\s+fusion.*(?:AI|autonomous|robot)",
            r"(?:autonomous|self[\s-]?driving).*(?:navigation|planning)",
            r"(?:path|motion)\s+planning.*(?:AI|robot|autonomous)",
            r"lidar.*(?:AI|autonomous|perception)",
        ],
        "keyword_tiers": {
            "autonomous vehicle": 1, "self-driving": 1, "autonomous driving": 1,
            "intelligent robot": 1, "cognitive robot": 1, "autonomous robot": 1,
            "agentic AI": 1, "multi-agent system": 1,
            "SLAM": 1, "sensor fusion": 1, "autonomous system": 1,
            "AI agent": 2, "autonomous agent": 2, "cobot AI": 2,
            "agent orchestration": 2, "agent framework": 2,
            "tool use": 3, "function calling": 3, "robot learning": 3,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["机器人", "自动驾驶", "无人驾驶", "无人机", "智能机器人"],
    },
    "B7_Infrastructure_Platforms": {
        "name": "AI Infrastructure & Platforms",
        "description": "Cloud AI; MLOps; GPU infrastructure; Model deployment; Enterprise platforms",
        "keywords": [
            "MLOps", "ML operations", "model ops", "AI ops", "AIOps",
            "model deployment", "model serving", "model hosting", "model monitoring",
            "cloud AI", "AI platform", "AI infrastructure", "ML platform",
            "AI-as-a-service", "AIaaS", "ML-as-a-service",
            "SageMaker", "Amazon Bedrock",
            "Azure AI", "Azure ML", "Azure OpenAI",
            "Vertex AI", "Google AI Platform",
            "GPU cluster", "TPU", "NPU", "AI accelerator",
            "NVIDIA AI", "CUDA", "inference engine",
            "edge AI", "edge inference", "edge ML",
            "IBM Watson", "Watsonx",
            "MLflow", "Kubeflow",
            "model registry", "feature store",
        ],
        "patterns": [
            r"\bMLOps\b",
            r"(?:AI|ML)\s+(?:platform|infrastructure|stack)",
            r"edge\s+(?:AI|inference|ML)",
            r"(?:model|AI)\s+(?:deployment|serving|hosting|monitoring)",
            r"(?:GPU|TPU|NPU)\s+(?:compute|cluster|infrastructure)",
            r"AI[\s-]?as[\s-]?a[\s-]?service",
            r"(?:SageMaker|Vertex\s+AI|Watson|Bedrock)",
            r"(?:Azure|Google\s+Cloud)\s+(?:AI|ML|OpenAI)",
            # v1.0.1: cloud AI platform patterns promoted from B8
            r"(?:AWS|Azure|GCP|Google\s+Cloud).*(?:AI|ML)\s+(?:service|platform)",
            r"(?:Databricks|Snowflake|Palantir).*(?:AI|ML)",
        ],
        "keyword_tiers": {
            "MLOps": 1, "AI platform": 1, "ML platform": 1,
            "SageMaker": 1, "Vertex AI": 1, "Azure ML": 1, "Watsonx": 1,
            "model deployment": 2, "model serving": 2, "edge AI": 2,
            "GPU cluster": 2, "AI accelerator": 2,
            "MLflow": 3, "Kubeflow": 3, "model registry": 3,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["AI芯片", "云计算AI", "GPU集群", "AI基础设施", "模型部署"],
    },
    "B8_General_AI": {
        "name": "AI (General/Unspecified)",
        "description": (
            "Generic AI references without specific technical classification. "
            "v1.0.1: Predictive/forecasting → B1, Neural/deep → B2, "
            "NER/sentiment → B3, LLM/prompt/RAG → B4, SageMaker/Vertex → B7. "
            "B8 is now strictly for unclassifiable generic mentions."
        ),
        "keywords": [
            "artificial intelligence", "AI system", "AI solution",
            "AI technology", "AI capability", "AI-powered", "AI-driven",
            "AI-enabled",
        ],
        "patterns": [
            r"\bartificial\s+intelligence\b",
            r"AI[\s-](?:powered|driven|enabled|based)",
            r"AI\s+(?:system|solution|technology|capability|platform)",
        ],
        "keyword_tiers": {
            "artificial intelligence": 1, "AI-powered": 1, "AI-driven": 1,
            "AI system": 2, "AI solution": 2, "AI technology": 2,
        },
        # Chinese (zh) — AISA v1.1.0
        "keywords_zh": ["人工智能", "智能化", "AI技术", "智能系统", "人工智能技术"],
    },
}


# ============================================================================
# CATEGORY MAPPING (legacy v6.1 → v7.0 codes)
# ============================================================================

CATEGORY_MAPPING_V6_TO_V7: Dict[str, str] = {
    "Product Development":          "A1_Product_Innovation",
    "Research & Innovation":        "A1_Product_Innovation",
    "Operational Implementation":   "A2_Operational_Excellence",
    "Customer Experience":          "A3_Customer_Experience",
    "Risk & Compliance":            "A4_Risk_Compliance",
    "Data & Analytics":             "A5_Data_Analytics",
    "Strategic Investment":         "A6_Strategy_Investment",
    "Explainability & AI Safety":   "A7_Governance_Ethics",
    "Talent & Workforce":           "A8_Talent_Workforce",
    "Generative AI & LLMs":         "B4_GenAI_LLMs",
    "AI Infrastructure & MLOps":    "B7_Infrastructure_Platforms",
    "AI Coding & Development":      "B7_Infrastructure_Platforms",
    "Agentic AI Systems":           "B6_Robotics_Autonomous",
}


# ============================================================================
# BUILTIN TAXONOMY PROVIDER
# ============================================================================

def _build_category_info(raw: Dict[str, Dict], dimension: str) -> Dict[str, CategoryInfo]:
    """Convert raw dict definitions to CategoryInfo objects.

    Also merges any 'keywords_zh' list (Chinese keywords added in v1.1.0)
    into the main keywords list and keyword_tiers dict, so the detection
    pipeline picks them up without any changes to 05_detect.py.
    """
    result = {}
    for code, data in raw.items():
        kw      = list(data.get("keywords", []))
        tiers   = dict(data.get("keyword_tiers", {}))
        kw_zh   = data.get("keywords_zh", [])
        for w in kw_zh:
            if w not in kw:
                kw.append(w)
            if w not in tiers:
                tiers[w] = 1   # Chinese exact terms: high confidence

        result[code] = CategoryInfo(
            code=code,
            name=data["name"],
            description=data["description"],
            dimension=dimension,
            keywords=kw,
            patterns=data.get("patterns", []),
            keyword_tiers=tiers,
        )
    return result


class BuiltinTaxonomy(TaxonomyProvider):
    """
    Concrete TaxonomyProvider backed by hardcoded keyword/pattern data.

    v1.2.0 changes (based on manual review of 9,528 FPs, 51.1% FP rate):
      E) "intelligent" patterns expanded for APAC energy/nuclear/utility contexts
      F) "automation" catch-all replaced with explicit proxy-statement /
         supply-chain / financial-table patterns
      G) LoRA and Sora added to AMBIGUOUS_PRODUCT_VENDORS
      H) "edge computing" telecom-only patterns (Chinese telco 5G MEC)
      I) "chatbot"/"virtual assistant" boilerplate guard
      J) "fraud detection"/"predictive maintenance"/"predictive analytics"
         boilerplate guard for bio sections and risk disclosures
      K) "knowledge graph"/"foundation models" telecom/hardware guard
      L) LoRA person-name and Sora financial-table fast-path FP patterns
      M) _AI_CONTEXT_RE strengthened with APAC vendor AI brands

    v1.0.1 changes:
      - Vendor gate for ambiguous product names (Gemini, Claude, Copilot, etc.)
      - robot* catch-all FP removed; specific industrial patterns only
      - _AI_CONTEXT_RE extended with autonomy/navigation terms
      - fine-tune + finance FP patterns added
      - B8 reduced: predictive→B1, neural→B2, NER/sentiment→B3, LLM/RAG→B4,
        cloud platforms→B7

    Import the module-level singleton TAXONOMY instead of instantiating directly.
    """

    def __init__(self):
        self._applications  = _build_category_info(_AI_APPLICATIONS_RAW, "A")
        self._technologies  = _build_category_info(_AI_TECHNOLOGIES_RAW, "B")

        # Compile FP patterns once
        self._fp_compiled: Dict[str, List[re.Pattern]] = {}
        for cat, patterns in FALSE_POSITIVE_PATTERNS.items():
            self._fp_compiled[cat] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        # Compile detection patterns once
        self._det_apps:  List[Tuple[re.Pattern, str]] = []
        self._det_techs: List[Tuple[re.Pattern, str]] = []

        for code, cat in self._applications.items():
            for p in cat.patterns:
                try:
                    self._det_apps.append((re.compile(p, re.IGNORECASE), code))
                except re.error:
                    pass

        for code, cat in self._technologies.items():
            for p in cat.patterns:
                try:
                    self._det_techs.append((re.compile(p, re.IGNORECASE), code))
                except re.error:
                    pass

    # --- TaxonomyProvider interface ---

    def get_version(self) -> str:
        return TAXONOMY_VERSION

    def get_dimensions(self) -> Dict[str, Dict[str, CategoryInfo]]:
        return {
            "Application": self._applications,
            "Technology":  self._technologies,
        }

    # Kept for code that imports these directly (backward compat)
    def get_applications(self) -> Dict[str, CategoryInfo]:
        return self._applications

    def get_technologies(self) -> Dict[str, CategoryInfo]:
        return self._technologies

    def get_fp_patterns(self) -> Dict[str, List[str]]:
        return FALSE_POSITIVE_PATTERNS

    def check_false_positive(self, text: str, context: str = "") -> FPResult:
        """
        Check if text+context is a false positive.

        Catch-all patterns (intelligent, automation, fine-tune, etc.) are
        skipped when strong AI context is present nearby.

        v1.0.1: robot* catch-all removed — only explicit industrial patterns
        remain for robots_non_ai.
        """
        combined = f"{text} {context}".lower()
        has_ai = bool(_AI_CONTEXT_RE.search(combined))

        for fp_cat, patterns in self._fp_compiled.items():
            for pattern in patterns:
                if pattern.search(combined):
                    ps = pattern.pattern
                    is_catchall = any(ps.startswith(p) for p in _CATCHALL_PREFIXES)
                    if is_catchall and has_ai:
                        continue
                    return FPResult(is_fp=True, category=fp_cat, pattern=ps)

        return FPResult(is_fp=False)

    def classify(self, text: str, context: str = "") -> ClassificationResult:
        """
        Classify text into dual taxonomy (Application + Technology dimension).

        Strategy:
            1. Vendor gate check for ambiguous product names (NEW v1.0.1)
            2. Pattern matching (compiled regexes) - primary
            3. Keyword matching - secondary
            4. Best confidence wins per dimension

        Vendor gate (improvement A):
            If text/context mentions an ambiguous product name (Gemini, Claude,
            Copilot, Titan, Nova, Llama, Mistral, Falcon) but no vendor or
            strong AI context keyword is found in the window → confidence
            penalty applied, category forced to B8 as fallback.
        """
        combined = f"{text} {context}"
        combined_lower = combined.lower()

        # --- Vendor gate for ambiguous product names ---
        vendor_penalty = False
        for product, data in AMBIGUOUS_PRODUCT_VENDORS.items():
            product_re = re.compile(r"\b" + re.escape(product) + r"\b", re.IGNORECASE)
            if product_re.search(combined):
                # Product name found — check vendor/context
                if not _VENDOR_GATE_RE[product].search(combined):
                    vendor_penalty = True
                    break

        cat_a, conf_a = self._match_dimension(combined, self._det_apps, self._applications)
        cat_b, conf_b = self._match_dimension(combined, self._det_techs, self._technologies)

        # Apply vendor penalty: downgrade ambiguous detections with no vendor context
        if vendor_penalty:
            # If the only B classification is B4 (GenAI/LLMs) → penalize heavily
            if cat_b == "B4_GenAI_LLMs":
                conf_b = max(0.0, conf_b - 0.4)
                if conf_b < 0.3:
                    cat_b  = "B8_General_AI"
                    conf_b = 0.15

        dims: Dict[str, Tuple[str, float]] = {}
        for dim_name, (cat, conf) in zip(
            self.get_dimensions().keys(),
            [(cat_a, conf_a), (cat_b, conf_b)],
        ):
            dims[dim_name] = (cat, conf)
        return ClassificationResult(dimensions=dims)

    # --- Internal helpers ---

    def _match_dimension(
        self,
        text: str,
        compiled_patterns: List[Tuple[re.Pattern, str]],
        categories: Dict[str, CategoryInfo],
    ) -> Tuple[str, float]:
        """
        Find best-matching category for one dimension.

        Returns (category_code, confidence) or ("", 0.0) if no match.
        """
        scores: Dict[str, float] = {}

        # Pattern matches
        for pattern, code in compiled_patterns:
            if pattern.search(text):
                scores[code] = scores.get(code, 0.0) + 0.5

        # Keyword matches (weighted by tier)
        text_lower = text.lower()
        for code, cat in categories.items():
            for kw in cat.keywords:
                if kw.lower() in text_lower:
                    tier = cat.keyword_tiers.get(kw, 2)
                    weight = {1: 0.4, 2: 0.25, 3: 0.1}.get(tier, 0.25)
                    scores[code] = scores.get(code, 0.0) + weight

        if not scores:
            return ("", 0.0)

        best = max(scores, key=scores.__getitem__)
        conf = min(scores[best], 1.0)
        return (best, round(conf, 3))


# ============================================================================
# MODULE-LEVEL SINGLETONS
# Importers should use TAXONOMY and PATTERN_CACHE directly.
# ============================================================================

TAXONOMY = BuiltinTaxonomy()

PATTERN_CACHE = CompiledPatternCache()
PATTERN_CACHE.compile_from_provider(TAXONOMY)


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    from version import get_version_string
    print(get_version_string())
    print()

    t = TAXONOMY

    # Version
    assert t.get_version() == TAXONOMY_VERSION
    print(f"  version OK: {t.get_version()}")

    # Category counts
    apps  = t.get_applications()
    techs = t.get_technologies()
    assert len(apps)  == 8, f"Expected 8 applications, got {len(apps)}"
    assert len(techs) == 8, f"Expected 8 technologies, got {len(techs)}"
    print(f"  applications: {len(apps)}  technologies: {len(techs)}")

    # FP check - embedding ESG
    fp = t.check_false_positive("embedding", "embedding ESG values in our culture")
    assert fp.is_fp, "Should be FP: embedding ESG"
    print(f"  FP (embedding ESG): is_fp={fp.is_fp}, cat={fp.category}")

    # FP check - Claude person name
    fp2 = t.check_false_positive("Claude", "CEO Claude Monet joined the board")
    assert fp2.is_fp, "Should be FP: Claude person name"
    print(f"  FP (Claude person): is_fp={fp2.is_fp}")

    # FP check - Claude with Anthropic context should NOT be FP
    fp3 = t.check_false_positive("Claude", "We use Anthropic's Claude AI model")
    assert not fp3.is_fp, "Should NOT be FP: Anthropic's Claude"
    print(f"  FP (Anthropic Claude): is_fp={fp3.is_fp}")

    # Vendor gate - Gemini without vendor context → FP caught by gemini_non_ai pattern
    fp_gem_nv = t.check_false_positive("Gemini", "Gemini clinical trial phase II results 2020")
    assert fp_gem_nv.is_fp, "Gemini clinical trial should be FP"
    print(f"  FP (Gemini clinical trial): is_fp={fp_gem_nv.is_fp}, cat={fp_gem_nv.category}")

    # Vendor gate - Gemini with Google context → accepted
    clf_gem_v = t.classify("Gemini", "We deployed Google Gemini for our enterprise chatbot")
    print(f"  Vendor gate (Google Gemini): B={clf_gem_v.category_b}({clf_gem_v.confidence_b})")

    # Vendor gate - Copilot without Microsoft context → penalized
    clf_cop_nv = t.classify("co-pilot", "The co-pilot landed the aircraft safely")
    print(f"  Vendor gate (aviation co-pilot): B={clf_cop_nv.category_b}({clf_cop_nv.confidence_b})")

    # Vendor gate - Copilot with Microsoft context → accepted
    clf_cop_v = t.classify("Copilot", "Microsoft Copilot is integrated into our M365 workflow")
    print(f"  Vendor gate (Microsoft Copilot): B={clf_cop_v.category_b}({clf_cop_v.confidence_b})")

    # robot* - industrial robot should NOT classify as B6
    fp_rob = t.check_false_positive("robot", "welding robot arm in assembly line")
    assert fp_rob.is_fp, "Should be FP: welding robot arm"
    print(f"  FP (welding robot): is_fp={fp_rob.is_fp}, cat={fp_rob.category}")

    # robot* - autonomous robot WITH AI context should NOT be FP
    fp_rob2 = t.check_false_positive("robot", "autonomous robot using reinforcement learning")
    assert not fp_rob2.is_fp, "Should NOT be FP: autonomous robot with RL"
    print(f"  FP (autonomous robot RL): is_fp={fp_rob2.is_fp}")

    # fine-tune + finance → FP
    fp_ft = t.check_false_positive("fine-tune", "fine-tune capital structure and leverage ratios")
    assert fp_ft.is_fp, "Should be FP: fine-tune capital structure"
    print(f"  FP (fine-tune capital structure): is_fp={fp_ft.is_fp}, cat={fp_ft.category}")

    # fine-tune + AI model → NOT FP
    fp_ft2 = t.check_false_positive("fine-tune", "fine-tune the LLM model on our dataset")
    assert not fp_ft2.is_fp, "Should NOT be FP: fine-tune LLM"
    print(f"  FP (fine-tune LLM): is_fp={fp_ft2.is_fp}")

    # B1 - predictive model (was B8 in v1.0.0)
    clf_pred = t.classify("predictive model", "predictive model for demand forecasting")
    print(f"  classify (predictive model): B={clf_pred.category_b}({clf_pred.confidence_b})")
    assert clf_pred.category_b == "B1_Traditional_ML", f"Expected B1, got {clf_pred.category_b}"

    # B4 - standard classification
    clf = t.classify("ChatGPT", "We deployed ChatGPT for customer support automation")
    assert clf.category_b == "B4_GenAI_LLMs", f"Expected B4, got {clf.category_b}"
    print(f"  classify (ChatGPT): A={clf.category_a}({clf.confidence_a}) B={clf.category_b}({clf.confidence_b})")

    # A4 - fraud detection
    clf2 = t.classify("fraud detection", "AI-powered fraud detection model deployed")
    assert clf2.category_a == "A4_Risk_Compliance", f"Expected A4, got {clf2.category_a}"
    print(f"  classify (fraud detection): A={clf2.category_a}({clf2.confidence_a})")

    # Pattern cache
    assert PATTERN_CACHE.is_compiled()
    det = PATTERN_CACHE.get_detection_patterns()
    fp_p = PATTERN_CACHE.get_fp_patterns()
    print(f"  PATTERN_CACHE: {len(det)} detection patterns, {len(fp_p)} FP categories")

    # Keyword tier
    tier = t.get_keyword_tier("B4_GenAI_LLMs", "ChatGPT")
    assert tier == 1
    print(f"  keyword tier ChatGPT in B4: {tier}")

    # Legacy mapping
    assert "Generative AI & LLMs" in CATEGORY_MAPPING_V6_TO_V7
    print(f"  legacy mapping OK: {len(CATEGORY_MAPPING_V6_TO_V7)} entries")

    # Ambiguous products dict — now includes LoRA, Sora, Mistral
    assert "gemini" in AMBIGUOUS_PRODUCT_VENDORS
    assert "claude" in AMBIGUOUS_PRODUCT_VENDORS
    assert "copilot" in AMBIGUOUS_PRODUCT_VENDORS
    assert "lora" in AMBIGUOUS_PRODUCT_VENDORS, "LoRA missing from vendor gate"
    assert "sora" in AMBIGUOUS_PRODUCT_VENDORS, "Sora missing from vendor gate"
    assert "mistral" in AMBIGUOUS_PRODUCT_VENDORS, "Mistral missing from vendor gate"
    print(f"  AMBIGUOUS_PRODUCT_VENDORS: {len(AMBIGUOUS_PRODUCT_VENDORS)} entries")

    # v1.2.0 NEW TESTS --------------------------------------------------

    # E) intelligent - APAC energy context → FP
    fp_en = t.check_false_positive("intelligent", "intelligent energy system for nuclear power dispatch")
    assert fp_en.is_fp, "Should be FP: intelligent energy system"
    print(f"  FP (intelligent energy system): is_fp={fp_en.is_fp}")

    # F) automation - proxy statement supply chain → FP
    fp_auto = t.check_false_positive("automation", "improved forecasting, replenishment, and merchandising automation and accuracy")
    assert fp_auto.is_fp, "Should be FP: merchandising automation and accuracy"
    print(f"  FP (merchandising automation and accuracy): is_fp={fp_auto.is_fp}")

    # F) automation with RPA context → NOT FP
    fp_auto2 = t.check_false_positive("automation", "robotic process automation using AI")
    assert not fp_auto2.is_fp, "Should NOT be FP: RPA with AI"
    print(f"  FP (RPA with AI): is_fp={fp_auto2.is_fp}")

    # G) LoRA - person name → FP
    fp_lora = t.check_false_positive("Lora", "Lora Ho, Senior Vice President, CFO, TSMC")
    assert fp_lora.is_fp, "Should be FP: Lora Ho person name"
    print(f"  FP (Lora Ho person name): is_fp={fp_lora.is_fp}")

    # G) LoRA - fine-tuning context → NOT FP (vendor gate passes)
    fp_lora2 = t.check_false_positive("LoRA", "We fine-tuned the LLM using LoRA adapters on Hugging Face")
    assert not fp_lora2.is_fp, "Should NOT be FP: LoRA fine-tuning with HF context"
    print(f"  FP (LoRA fine-tuning HF): is_fp={fp_lora2.is_fp}")

    # H) edge computing - telecom context → FP
    fp_edge = t.check_false_positive("edge computing", "edge computing with 5G base stations MEC network slice bandwidth")
    assert fp_edge.is_fp, "Should be FP: edge computing 5G MEC"
    print(f"  FP (edge computing 5G): is_fp={fp_edge.is_fp}")

    # H) edge computing - AI inference context → NOT FP
    fp_edge2 = t.check_false_positive("edge computing", "edge computing AI inference on-device neural processing")
    assert not fp_edge2.is_fp, "Should NOT be FP: edge AI inference"
    print(f"  FP (edge AI inference): is_fp={fp_edge2.is_fp}")

    # I) chatbot - multi-channel list → FP
    fp_chat = t.check_false_positive("chatbot", "customers can contact us via telephone, email, webchat, chatbot or connected services")
    assert fp_chat.is_fp, "Should be FP: chatbot in channel list"
    print(f"  FP (chatbot channel list): is_fp={fp_chat.is_fp}")

    # J) fraud detection - director bio → FP
    fp_fraud = t.check_false_positive("fraud detection", "LLB (Gen) and holds ICAI certificate on Forensic Accounting fraud detection expert")
    assert fp_fraud.is_fp, "Should be FP: fraud detection in director bio"
    print(f"  FP (fraud detection director bio): is_fp={fp_fraud.is_fp}")

    # J) predictive analytics - weather/climate context → FP
    fp_pred = t.check_false_positive("predictive analytics", "Emergency Management uses predictive analytics to gauge the path and likely severity of hurricanes")
    assert fp_pred.is_fp, "Should be FP: predictive analytics weather"
    print(f"  FP (predictive analytics hurricane): is_fp={fp_pred.is_fp}")

    # J) predictive maintenance - market opportunity → FP
    fp_pm = t.check_false_positive("predictive maintenance", "opportunities arise from industrial connectivity predictive maintenance market potential")
    assert fp_pm.is_fp, "Should be FP: predictive maintenance market opportunity"
    print(f"  FP (predictive maintenance market): is_fp={fp_pm.is_fp}")

    # K) Sora - financial table → FP
    fp_sora = t.check_false_positive("Sora", "bond coupon maturity date Sora GmbH Ltd issuance")
    assert fp_sora.is_fp, "Should be FP: Sora in financial table"
    print(f"  FP (Sora financial table): is_fp={fp_sora.is_fp}")

    # v1.2.1 NEW TESTS --------------------------------------------------

    # N) LLMS = Loan Lifecycle Management System → FP
    fp_llms = t.check_false_positive("LLMS", "Loan Life Cycle Management System (LLMS) automates credit origination")
    assert fp_llms.is_fp, "Should be FP: LLMS banking system"
    print(f"  FP (LLMS banking): is_fp={fp_llms.is_fp}, cat={fp_llms.category}")

    # N) LLMS with LLM context → NOT FP (AI context bypass fires)
    fp_llms2 = t.check_false_positive("LLMS", "We fine-tuned LLMS using GPT-4 for customer support AI assistant")
    assert not fp_llms2.is_fp, "Should NOT be FP: LLMS with GPT AI context"
    print(f"  FP (LLMS GPT context): is_fp={fp_llms2.is_fp}")

    # O) ML in luxury auto context → FP
    fp_ml = t.check_false_positive("ML", "ML is embracing China's status as the world's largest luxury vehicle market")
    assert fp_ml.is_fp, "Should be FP: ML in luxury vehicle market context"
    print(f"  FP (ML luxury vehicle): is_fp={fp_ml.is_fp}")

    # O) RPA in ESG audit context → FP
    fp_rpa = t.check_false_positive("RPA", "Chief Audit Executive attended RPA ESG trends seminar on audit awareness")
    assert fp_rpa.is_fp, "Should be FP: RPA in ESG audit seminar"
    print(f"  FP (RPA ESG audit seminar): is_fp={fp_rpa.is_fp}")

    # O) RPA with automation/AI context → NOT FP
    fp_rpa2 = t.check_false_positive("RPA", "robotic process automation RPA bots using AI")
    assert not fp_rpa2.is_fp, "Should NOT be FP: RPA with automation AI"
    print(f"  FP (RPA automation AI): is_fp={fp_rpa2.is_fp}")

    # P) model deployment in credit risk context → FP
    fp_mdc = t.check_false_positive("model deployment", "credit risk IRB Basel model deployment for scoring")
    assert fp_mdc.is_fp, "Should be FP: model deployment in credit risk"
    print(f"  FP (model deployment credit risk): is_fp={fp_mdc.is_fp}")

    # P) model deployment with AI/cloud context → NOT FP
    fp_mdc2 = t.check_false_positive("model deployment", "model deployment to cloud API endpoint monitoring")
    assert not fp_mdc2.is_fp, "Should NOT be FP: model deployment cloud API"
    print(f"  FP (model deployment cloud): is_fp={fp_mdc2.is_fp}")

    # Q) recommendation system in HR/wellness → FP
    fp_rec = t.check_false_positive("recommendation system", "employee wellness HR benefit recommendation system compliance")
    assert fp_rec.is_fp, "Should be FP: recommendation system HR wellness"
    print(f"  FP (recommendation system HR): is_fp={fp_rec.is_fp}")

    # R) frontier model in export control → FP
    fp_fm = t.check_false_positive("frontier model", "export control trade restriction BIS frontier model semiconductor")
    assert fp_fm.is_fp, "Should be FP: frontier model export control"
    print(f"  FP (frontier model export): is_fp={fp_fm.is_fp}")

    # S) Mistral = wind/construction context → FP
    fp_mist = t.check_false_positive("Mistral", "Mistral wind offshore Bouygues construction subsidiary infrastructure")
    assert fp_mist.is_fp, "Should be FP: Mistral wind/construction"
    print(f"  FP (Mistral wind): is_fp={fp_mist.is_fp}")

    # S) Mistral with AI/model context → NOT FP (vendor gate)
    fp_mist2 = t.check_false_positive("Mistral", "We use Mistral 7B language model for document processing")
    assert not fp_mist2.is_fp, "Should NOT be FP: Mistral 7B LLM"
    print(f"  FP (Mistral 7B model): is_fp={fp_mist2.is_fp}")

    print()
    print("  04_taxonomy_builtin.py v1.2.1 — all checks passed.")

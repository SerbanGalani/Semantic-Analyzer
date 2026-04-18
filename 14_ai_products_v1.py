"""
===============================================================================
AISA - AI Semantic Analyzer
14_ai_products_v1.py - AI Product Dictionary & Extraction
===============================================================================

Dictionary of known AI products for granularity in TPDI and
Adoption Memory. Contains KNOWN_PRODUCTS (products grouped by B categories),
VENDOR_PATTERNS (vendor detection without specific product), INTERNAL_PATTERNS
(proprietary solutions) and associated extraction functions.

Folosit de:
    09_tpdi.py  — product-level diffusion tracking
    08_memory.py — Adoption Memory portfolio

KNOWN_PRODUCTS STRUCTURE:
    KNOWN_PRODUCTS[category_b] = {
        "product_name": {
            "vendor":             "Vendor Name",
            "release_year":       2023,
            "aliases":            ["alias1", "alias2"],
            "patterns":           [r"regex_pattern"],
            "category_a_typical": ["A2", "A3"],
            "require_ai_context": True,          # optional — ambiguous names
            "ai_context_patterns": [r"pattern"], # optional — required if above
        }
    }

USAGE:
    _m14 = importlib.import_module("14_ai_products_v1")
    product, vendor, granularity = _m14.extract_product_info(text, category_b)

MIGRATION NOTE:
    This module replaces the v6.3 ai_products_v1.py + shim combination.
    The original content is preserved verbatim; only the header and
    module number have changed.
    To update 09_tpdi.py: change "ai_products_v1" → "14_ai_products_v1"
    in the importlib.import_module() call (line ~69).

CHANGELOG:
    v1.2.0 (2026-03) - Migrated to AISA v1.1 module numbering (14_*)
                       Header updated; all content identical to v1.1.0.
                       Added: Mistral require_ai_context flag (aligned with
                       04_taxonomy_builtin.py v1.2.1 mistral_non_ai pattern).
    v1.1.0 (2026-02) - Temporal validation (report_year < release_year → skip)
                       AI context validation for Gemini, PaLM, Mistral.
    v1.0.0 (2026-02) - Initial release. Part of AI Semantic Analyzer v6.3.

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ============================================================================
# VERSION
# ============================================================================

PRODUCTS_VERSION = "1.6.0"
PRODUCTS_DATE    = "March 2026"


# ============================================================================
# GRANULARITY LEVELS
# ============================================================================

class GranularityLevel:
    SPECIFIC      = "SPECIFIC"       # Specific product mentioned (ChatGPT, Copilot)
    VENDOR_ONLY   = "VENDOR_ONLY"    # Vendor only mentioned (OpenAI, Anthropic)
    CATEGORY_ONLY = "CATEGORY_ONLY"  # Generic category only (generative AI)
    INTERNAL      = "INTERNAL"       # Internal / proprietary product


# ============================================================================
# KNOWN PRODUCTS
# ============================================================================

KNOWN_PRODUCTS: Dict[str, Dict] = {

    # -------------------------------------------------------------------------
    # B4_GenAI_LLMs — Generative AI & Large Language Models
    # -------------------------------------------------------------------------
    "B4_GenAI_LLMs": {

        # OpenAI
        "ChatGPT": {
            "vendor": "OpenAI",
            "release_year": 2022,
            "aliases": ["Chat GPT", "ChatGPT Plus", "ChatGPT Enterprise"],
            "patterns": [
                r"\bChatGPT\b",
                r"\bChat[\s-]?GPT\b",
            ],
            "category_a_typical": ["A2_Operational_Excellence", "A3_Customer_Experience"],
        },
        "GPT-4": {
            "vendor": "OpenAI",
            "release_year": 2023,
            "aliases": ["GPT4", "GPT-4o", "GPT-4 Turbo", "GPT-4V"],
            "patterns": [
                r"\bGPT[\s-]?4(?:o|[\s-]?Turbo|V)?\b",
            ],
            "category_a_typical": ["A1_Product_Innovation", "A2_Operational_Excellence"],
        },
        "GPT-3": {
            "vendor": "OpenAI",
            "release_year": 2020,
            "aliases": ["GPT3", "GPT-3.5", "GPT-3.5 Turbo"],
            "patterns": [
                r"\bGPT[\s-]?3(?:\.5)?(?:[\s-]?Turbo)?\b",
            ],
            "category_a_typical": ["A2_Operational_Excellence"],
        },
        "DALL-E": {
            "vendor": "OpenAI",
            "release_year": 2021,
            "aliases": ["DALLE", "DALL-E 2", "DALL-E 3"],
            "patterns": [
                r"\bDALL[\s-]?E(?:[\s-]?[23])?\b",
            ],
            "category_a_typical": ["A1_Product_Innovation", "A3_Customer_Experience"],
        },
        "Whisper": {
            "vendor": "OpenAI",
            "release_year": 2022,
            "aliases": [],
            "patterns": [
                r"\bOpenAI[\s\']?s?\s+Whisper\b",
                r"\bWhisper\s+(?:API|model)\b",
            ],
            "category_a_typical": ["A2_Operational_Excellence"],
        },
        "Codex": {
            "vendor": "OpenAI",
            "release_year": 2021,
            "aliases": ["OpenAI Codex"],
            "patterns": [r"\bCodex\b(?=.*(?:OpenAI|code|programming))"],
            "category_a_typical": ["A1_Product_Innovation", "A2_Operational_Excellence"],
        },
        "Sora": {
            "vendor": "OpenAI",
            "release_year": 2024,
            "aliases": [],
            "patterns": [r"\bSora\b(?=.*(?:OpenAI|video|generation))"],
            "category_a_typical": ["A1_Product_Innovation"],
        },

        # Anthropic
        "Claude": {
            "vendor": "Anthropic",
            "release_year": 2023,
            "aliases": [
                "Claude 2", "Claude 3", "Claude 3.5",
                "Claude Instant", "Claude Opus", "Claude Sonnet", "Claude Haiku",
            ],
            "patterns": [
                r"\bClaude(?:[\s-]?(?:2|3|3\.5|Instant|Opus|Sonnet|Haiku))?\b"
                r"(?![\s\.](?:[A-Z][a-z]+|[A-Z]\.))",
            ],
            "category_a_typical": ["A2_Operational_Excellence", "A3_Customer_Experience"],
        },

        # Google
        "Gemini": {
            "vendor": "Google",
            "release_year": 2023,
            "aliases": ["Gemini Pro", "Gemini Ultra", "Gemini Nano", "Gemini 1.5", "Gemini 2.0"],
            "patterns": [
                r"\bGemini(?:[\s-]?(?:Pro|Ultra|Nano|1\.5|2\.0))?\b",
            ],
            "require_ai_context": True,
            "ai_context_patterns": [
                r"(?:AI|artificial intelligence|language model|LLM|machine learning|Google|DeepMind)",
                r"(?:chatbot|assistant|generative|foundation model)",
            ],
            "category_a_typical": ["A1_Product_Innovation", "A2_Operational_Excellence"],
        },
        "Bard": {
            "vendor": "Google",
            "release_year": 2023,
            "aliases": [],
            "patterns": [
                r"\bGoogle[\s\']?s?\s+Bard\b",
                r"\bBard\s+(?:AI|chatbot)\b",
            ],
            "category_a_typical": ["A3_Customer_Experience"],
        },
        "PaLM": {
            "vendor": "Google",
            "release_year": 2022,
            "aliases": ["PaLM 2"],
            "patterns": [r"\bPaLM(?:[\s-]?2)?\b"],
            "require_ai_context": True,
            "ai_context_patterns": [
                r"(?:AI|artificial intelligence|language model|LLM|Google|DeepMind|generative)",
                r"(?:foundation model|large language|neural|transformer)",
            ],
            "category_a_typical": ["A1_Product_Innovation"],
        },

        # Microsoft
        "Copilot": {
            "vendor": "Microsoft",
            "release_year": 2023,
            "aliases": [
                "Microsoft Copilot", "Microsoft 365 Copilot",
                "GitHub Copilot", "Windows Copilot", "Copilot for Microsoft 365",
            ],
            "patterns": [
                r"\b(?:Microsoft\s+)?Copilot\b",
                r"\bGitHub\s+Copilot\b",
                r"\bCopilot\s+for\s+(?:Microsoft\s+)?365\b",
            ],
            "category_a_typical": ["A2_Operational_Excellence", "A8_Talent_Workforce"],
        },
        "Bing Chat": {
            "vendor": "Microsoft",
            "release_year": 2023,
            "aliases": ["Bing AI", "New Bing"],
            "patterns": [
                r"\bBing\s+(?:Chat|AI)\b",
                r"\bNew\s+Bing\b",
            ],
            "category_a_typical": ["A3_Customer_Experience"],
        },

        # Meta
        "Llama": {
            "vendor": "Meta",
            "release_year": 2023,
            "aliases": ["Llama 2", "Llama 3", "LLaMA", "Code Llama"],
            "patterns": [
                r"\bLlama(?:[\s-]?[23])?\b",
                r"\bLLaMA\b",
                r"\bCode\s+Llama\b",
            ],
            "category_a_typical": ["A1_Product_Innovation"],
        },

        # Other GenAI
        "Mistral": {
            "vendor": "Mistral AI",
            "release_year": 2023,
            "aliases": ["Mistral 7B", "Mixtral"],
            "patterns": [
                r"\bMistral(?:[\s-]?(?:7B|AI))?\b",
                r"\bMixtral\b",
            ],
            "require_ai_context": True,
            "ai_context_patterns": [
                r"(?:AI|artificial intelligence|language model|LLM|machine learning|generative)",
                r"(?:open[\s-]?source|foundation model|neural|chatbot|7B|8x7B|Mixtral)",
            ],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "Stable Diffusion": {
            "vendor": "Stability AI",
            "release_year": 2022,
            "aliases": ["SD", "SDXL", "Stable Diffusion XL"],
            "patterns": [
                r"\bStable\s+Diffusion(?:\s+XL)?\b",
                r"\bSDXL\b",
            ],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "Midjourney": {
            "vendor": "Midjourney",
            "release_year": 2022,
            "aliases": [],
            "patterns": [r"\bMidjourney\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "Cohere": {
            "vendor": "Cohere",
            "release_year": 2022,
            "aliases": ["Cohere Command", "Command R"],
            "patterns": [r"\bCohere\b", r"\bCommand\s+R\b"],
            "category_a_typical": ["A2_Operational_Excellence"],
        },
        "Perplexity": {
            "vendor": "Perplexity AI",
            "release_year": 2023,
            "aliases": [],
            "patterns": [r"\bPerplexity(?:\s+AI)?\b"],
            "category_a_typical": ["A5_Data_Analytics"],
        },
    },

    # -------------------------------------------------------------------------
    # B7_Infrastructure_Platforms — AI/ML Platforms & Cloud Services
    # -------------------------------------------------------------------------
    "B7_Infrastructure_Platforms": {

        # AWS
        "SageMaker": {
            "vendor": "AWS",
            "release_year": 2017,
            "aliases": ["Amazon SageMaker", "AWS SageMaker"],
            "patterns": [r"\b(?:Amazon\s+|AWS\s+)?SageMaker\b"],
            "category_a_typical": ["A1_Product_Innovation", "A5_Data_Analytics"],
        },
        "Bedrock": {
            "vendor": "AWS",
            "release_year": 2023,
            "aliases": ["Amazon Bedrock", "AWS Bedrock"],
            "patterns": [r"\b(?:Amazon\s+|AWS\s+)?Bedrock\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "Rekognition": {
            "vendor": "AWS",
            "release_year": 2016,
            "aliases": ["Amazon Rekognition"],
            "patterns": [r"\b(?:Amazon\s+)?Rekognition\b"],
            "category_a_typical": ["A4_Risk_Compliance"],
        },
        "Comprehend": {
            "vendor": "AWS",
            "release_year": 2017,
            "aliases": ["Amazon Comprehend"],
            "patterns": [r"\b(?:Amazon\s+)?Comprehend\b"],
            "category_a_typical": ["A5_Data_Analytics"],
        },
        "Lex": {
            "vendor": "AWS",
            "release_year": 2017,
            "aliases": ["Amazon Lex"],
            "patterns": [r"\bAmazon\s+Lex\b"],
            "category_a_typical": ["A3_Customer_Experience"],
        },

        # Microsoft Azure
        "Azure ML": {
            "vendor": "Microsoft",
            "release_year": 2015,
            "aliases": ["Azure Machine Learning", "Azure ML Studio"],
            "patterns": [r"\bAzure\s+(?:Machine\s+Learning|ML(?:\s+Studio)?)\b"],
            "category_a_typical": ["A1_Product_Innovation", "A5_Data_Analytics"],
        },
        "Azure OpenAI": {
            "vendor": "Microsoft",
            "release_year": 2023,
            "aliases": ["Azure OpenAI Service"],
            "patterns": [r"\bAzure\s+OpenAI(?:\s+Service)?\b"],
            "category_a_typical": ["A1_Product_Innovation", "A2_Operational_Excellence"],
        },
        "Azure Cognitive Services": {
            "vendor": "Microsoft",
            "release_year": 2016,
            "aliases": ["Cognitive Services", "Azure AI Services"],
            "patterns": [
                r"\b(?:Azure\s+)?Cognitive\s+Services\b",
                r"\bAzure\s+AI\s+Services\b",
            ],
            "category_a_typical": ["A3_Customer_Experience"],
        },

        # Google Cloud
        "Vertex AI": {
            "vendor": "Google",
            "release_year": 2021,
            "aliases": ["Google Vertex AI"],
            "patterns": [r"\b(?:Google\s+)?Vertex\s+AI\b"],
            "category_a_typical": ["A1_Product_Innovation", "A5_Data_Analytics"],
        },
        "Google Cloud AI": {
            "vendor": "Google",
            "release_year": 2017,
            "aliases": ["GCP AI", "Google AI Platform"],
            "patterns": [
                r"\bGoogle\s+Cloud\s+AI\b",
                r"\bGoogle\s+AI\s+Platform\b",
            ],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "AutoML": {
            "vendor": "Google",
            "release_year": 2018,
            "aliases": ["Google AutoML", "Cloud AutoML"],
            "patterns": [r"\b(?:Google\s+|Cloud\s+)?AutoML\b"],
            "category_a_typical": ["A5_Data_Analytics"],
        },
        "Document AI": {
            "vendor": "Google",
            "release_year": 2020,
            "aliases": ["Google Document AI"],
            "patterns": [r"\b(?:Google\s+)?Document\s+AI\b"],
            "category_a_typical": ["A2_Operational_Excellence"],
        },

        # IBM
        "Watson": {
            "vendor": "IBM",
            "release_year": 2011,
            "aliases": ["IBM Watson", "Watson AI", "Watson Assistant", "Watson Discovery"],
            "patterns": [
                r"\b(?:IBM\s+)?Watson(?:\s+(?:AI|Assistant|Discovery|Studio))?\b",
            ],
            "category_a_typical": ["A3_Customer_Experience", "A5_Data_Analytics"],
        },
        "Watsonx": {
            "vendor": "IBM",
            "release_year": 2023,
            "aliases": ["watsonx.ai", "watsonx.data", "watsonx.governance"],
            "patterns": [r"\b[Ww]atsonx(?:\.(?:ai|data|governance))?\b"],
            "category_a_typical": ["A1_Product_Innovation", "A7_AI_Governance"],
        },

        # Databricks
        "Databricks": {
            "vendor": "Databricks",
            "release_year": 2013,
            "aliases": ["Databricks Lakehouse", "Databricks ML", "MLflow"],
            "patterns": [r"\bDatabricks\b", r"\bMLflow\b"],
            "category_a_typical": ["A5_Data_Analytics", "A1_Product_Innovation"],
        },

        # Snowflake
        "Snowflake": {
            "vendor": "Snowflake",
            "release_year": 2014,
            "aliases": ["Snowflake AI", "Snowpark"],
            "patterns": [r"\bSnowflake(?:\s+(?:AI|ML))?\b", r"\bSnowpark\b"],
            "category_a_typical": ["A5_Data_Analytics"],
        },

        # Hugging Face
        "Hugging Face": {
            "vendor": "Hugging Face",
            "release_year": 2018,
            "aliases": ["HuggingFace", "Transformers library"],
            "patterns": [r"\bHugging\s*Face\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
    },

    # -------------------------------------------------------------------------
    # B1_Traditional_ML — Machine Learning Frameworks & Tools
    # -------------------------------------------------------------------------
    "B1_Traditional_ML": {
        "TensorFlow": {
            "vendor": "Google",
            "release_year": 2015,
            "aliases": ["TF", "TensorFlow 2"],
            "patterns": [r"\bTensorFlow(?:\s+2)?\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "PyTorch": {
            "vendor": "Meta",
            "release_year": 2016,
            "aliases": [],
            "patterns": [r"\bPyTorch\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "scikit-learn": {
            "vendor": "Open Source",
            "release_year": 2007,
            "aliases": ["sklearn", "Scikit Learn"],
            "patterns": [r"\b(?:scikit[\s-]?learn|sklearn)\b"],
            "category_a_typical": ["A5_Data_Analytics"],
        },
        "XGBoost": {
            "vendor": "Open Source",
            "release_year": 2014,
            "aliases": [],
            "patterns": [r"\bXGBoost\b"],
            "category_a_typical": ["A5_Data_Analytics", "A4_Risk_Compliance"],
        },
        "LightGBM": {
            "vendor": "Microsoft",
            "release_year": 2017,
            "aliases": [],
            "patterns": [r"\bLightGBM\b"],
            "category_a_typical": ["A5_Data_Analytics"],
        },
        "H2O.ai": {
            "vendor": "H2O.ai",
            "release_year": 2012,
            "aliases": ["H2O", "H2O AutoML"],
            "patterns": [r"\bH2O(?:\.ai)?\b"],
            "category_a_typical": ["A5_Data_Analytics"],
        },
        "DataRobot": {
            "vendor": "DataRobot",
            "release_year": 2012,
            "aliases": [],
            "patterns": [r"\bDataRobot\b"],
            "category_a_typical": ["A5_Data_Analytics"],
        },
    },

    # -------------------------------------------------------------------------
    # B6_Robotics_Autonomous — AI Agents & Autonomous Systems
    # -------------------------------------------------------------------------
    "B6_Robotics_Autonomous": {
        "AutoGPT": {
            "vendor": "Open Source",
            "release_year": 2023,
            "aliases": ["Auto-GPT"],
            "patterns": [r"\bAuto[\s-]?GPT\b"],
            "category_a_typical": ["A2_Operational_Excellence"],
        },
        "LangChain": {
            "vendor": "LangChain",
            "release_year": 2022,
            "aliases": ["LangGraph"],
            "patterns": [r"\bLang(?:Chain|Graph)\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "CrewAI": {
            "vendor": "CrewAI",
            "release_year": 2023,
            "aliases": [],
            "patterns": [r"\bCrewAI\b"],
            "category_a_typical": ["A2_Operational_Excellence"],
        },
        "AutoGen": {
            "vendor": "Microsoft",
            "release_year": 2023,
            "aliases": [],
            "patterns": [r"\bAutoGen\b"],
            "category_a_typical": ["A2_Operational_Excellence"],
        },
    },

    # -------------------------------------------------------------------------
    # B3_NLP — Natural Language Processing Tools
    # -------------------------------------------------------------------------
    "B3_NLP": {
        "spaCy": {
            "vendor": "Explosion",
            "release_year": 2015,
            "aliases": [],
            "patterns": [r"\bspaCy\b"],
            "category_a_typical": ["A5_Data_Analytics"],
        },
        "NLTK": {
            "vendor": "Open Source",
            "release_year": 2001,
            "aliases": [],
            "patterns": [r"\bNLTK\b"],
            "category_a_typical": ["A5_Data_Analytics"],
        },
        "BERT": {
            "vendor": "Google",
            "release_year": 2018,
            "aliases": ["FinBERT", "RoBERTa", "DistilBERT"],
            "patterns": [r"\b(?:Fin|Ro|Distil)?BERT[Aa]?\b"],
            "category_a_typical": ["A5_Data_Analytics"],
        },
    },

    # -------------------------------------------------------------------------
    # B5_Computer_Vision — Computer Vision Tools
    # -------------------------------------------------------------------------
    "B5_Computer_Vision": {
        "OpenCV": {
            "vendor": "Open Source",
            "release_year": 2000,
            "aliases": [],
            "patterns": [r"\bOpenCV\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "YOLO": {
            "vendor": "Ultralytics",
            "release_year": 2016,
            "aliases": ["YOLOv5", "YOLOv8"],
            "patterns": [r"\bYOLO(?:v[5-9])?\b"],
            "category_a_typical": ["A4_Risk_Compliance"],
        },
    },

    # -------------------------------------------------------------------------
    # Enterprise SaaS AI — Round 2 addition (2026-03)
    # Salesforce, SAP, ServiceNow, Oracle, Workday, Adobe
    # -------------------------------------------------------------------------
    "B7_Infrastructure_Platforms_Enterprise": {

        # Salesforce
        "Einstein": {
            "vendor": "Salesforce",
            "release_year": 2016,
            "aliases": ["Salesforce Einstein", "Einstein GPT", "Einstein AI", "Einstein Copilot"],
            "patterns": [
                r"\b(?:Salesforce\s+)?Einstein(?:\s+(?:GPT|AI|Copilot|Analytics))?\b",
            ],
            "require_ai_context": True,
            "ai_context_patterns": [
                r"(?:Salesforce|CRM|AI|machine learning|analytics|prediction|generative)",
            ],
            "category_a_typical": ["A3_Customer_Experience", "A5_Data_Analytics"],
        },

        # SAP
        "SAP AI Core": {
            "vendor": "SAP",
            "release_year": 2021,
            "aliases": ["SAP AI", "SAP Business AI", "SAP Joule"],
            "patterns": [
                r"\bSAP\s+AI\s+Core\b",
                r"\bSAP\s+Business\s+AI\b",
                r"\bSAP\s+Joule\b",
                r"\bSAP\s+(?:ML|machine\s+learning|AI\s+Foundation)\b",
            ],
            "category_a_typical": ["A2_Operational_Excellence", "A5_Data_Analytics"],
        },

        # ServiceNow
        "Now Intelligence": {
            "vendor": "ServiceNow",
            "release_year": 2019,
            "aliases": ["ServiceNow AI", "Now Assist", "ServiceNow Now Assist"],
            "patterns": [
                r"\bNow\s+(?:Intelligence|Assist)\b",
                r"\bServiceNow\s+(?:AI|ML|intelligence|Assist)\b",
            ],
            "category_a_typical": ["A2_Operational_Excellence"],
        },

        # Oracle
        "Oracle AI": {
            "vendor": "Oracle",
            "release_year": 2018,
            "aliases": ["Oracle Cloud AI", "OCI AI", "Oracle AI Services"],
            "patterns": [
                r"\bOracle\s+(?:AI|Cloud\s+AI|AI\s+Services|AI\s+Infrastructure)\b",
                r"\bOCI\s+(?:AI|ML|Generative\s+AI)\b",
            ],
            "category_a_typical": ["A5_Data_Analytics", "A2_Operational_Excellence"],
        },

        # Workday
        "Workday AI": {
            "vendor": "Workday",
            "release_year": 2020,
            "aliases": ["Workday Machine Learning", "Workday ML"],
            "patterns": [
                r"\bWorkday\s+(?:AI|ML|machine\s+learning|Illuminate)\b",
                r"\bWorkday\s+Illuminate\b",
            ],
            "category_a_typical": ["A8_Talent_Workforce", "A2_Operational_Excellence"],
        },

        # Adobe
        "Adobe Firefly": {
            "vendor": "Adobe",
            "release_year": 2023,
            "aliases": ["Firefly", "Adobe Sensei GenAI"],
            "patterns": [
                r"\bAdobe\s+Firefly\b",
                r"\bAdobe\s+Sensei(?:\s+GenAI)?\b",
                r"\bFirefly\b(?=.*(?:Adobe|AI|generative|image|creative))",
            ],
            "category_a_typical": ["A1_Product_Innovation"],
        },

        # Palantir
        "Palantir AIP": {
            "vendor": "Palantir",
            "release_year": 2023,
            "aliases": ["AIP", "Palantir AI Platform", "Palantir Foundry", "Palantir Gotham"],
            "patterns": [
                r"\bPalantir\s+(?:AIP|AI\s+Platform|Foundry|Gotham)\b",
                r"\bPalantir\b(?=.*(?:AI|ML|intelligence|defense|analytics))",
            ],
            "category_a_typical": ["A5_Data_Analytics", "A4_Risk_Compliance"],
        },

        # C3.ai
        "C3.ai": {
            "vendor": "C3.ai",
            "release_year": 2009,
            "aliases": ["C3 AI", "C3 AI Suite"],
            "patterns": [r"\bC3\.ai\b", r"\bC3\s+AI\b"],
            "category_a_typical": ["A5_Data_Analytics", "A2_Operational_Excellence"],
        },
    },
    # Huawei, Baidu, Alibaba, Tencent, Samsung — frequent in Fortune Global 500
    # -------------------------------------------------------------------------

    "B4_GenAI_LLMs_APAC": {

        # Baidu
        "ERNIE Bot": {
            "vendor": "Baidu",
            "release_year": 2023,
            "aliases": ["文心一言", "Wenxin Yiyan", "ERNIE 4.0"],
            "patterns": [
                r"\bERNIE\s*(?:Bot|4\.0|3\.5)?\b",
                r"\b文心一言\b",
                r"\bWenxin\s*(?:Yiyan)?\b",
            ],
            "require_ai_context": True,
            "ai_context_patterns": [
                r"(?:Baidu|AI|language model|LLM|generative|chatbot|中文)",
            ],
            "category_a_typical": ["A3_Customer_Experience", "A1_Product_Innovation"],
        },
        "Qianfan": {
            "vendor": "Baidu",
            "release_year": 2023,
            "aliases": ["千帆", "Baidu Qianfan"],
            "patterns": [r"\bQianfan\b", r"\b千帆\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },

        # Alibaba
        "Qwen": {
            "vendor": "Alibaba",
            "release_year": 2023,
            "aliases": ["通义千问", "Tongyi Qianwen", "Qwen2", "Qwen-VL"],
            "patterns": [
                r"\bQwen(?:\d+|-VL|-Audio)?\b",
                r"\b通义千问\b",
                r"\bTongyi\s+Qianwen\b",
            ],
            "category_a_typical": ["A1_Product_Innovation", "A2_Operational_Excellence"],
        },
        "Tongyi": {
            "vendor": "Alibaba",
            "release_year": 2023,
            "aliases": ["通义", "Alibaba Tongyi"],
            "patterns": [r"\bTongyi\b", r"\b通义\b"],
            "require_ai_context": True,
            "ai_context_patterns": [
                r"(?:Alibaba|AI|language model|LLM|generative|Qwen)",
            ],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "Alibaba Cloud AI": {
            "vendor": "Alibaba",
            "release_year": 2019,
            "aliases": ["Aliyun AI", "阿里云AI"],
            "patterns": [
                r"\bAlibaba\s+Cloud\s+(?:AI|ML|intelligence)\b",
                r"\bAliyun\s+(?:AI|ML)\b",
            ],
            "category_a_typical": ["A1_Product_Innovation", "A5_Data_Analytics"],
        },

        # Tencent
        "Hunyuan": {
            "vendor": "Tencent",
            "release_year": 2023,
            "aliases": ["混元", "Tencent Hunyuan"],
            "patterns": [
                r"\bHunyuan\b",
                r"\b混元\b",
            ],
            "require_ai_context": True,
            "ai_context_patterns": [
                r"(?:Tencent|AI|language model|LLM|generative)",
            ],
            "category_a_typical": ["A1_Product_Innovation", "A3_Customer_Experience"],
        },

        # Samsung
        "Samsung Gauss": {
            "vendor": "Samsung",
            "release_year": 2023,
            "aliases": ["Gauss"],
            "patterns": [
                r"\bSamsung\s+Gauss\b",
                r"\bGauss\b(?=.*(?:Samsung|AI|language model|LLM))",
            ],
            "category_a_typical": ["A1_Product_Innovation"],
        },

        # Kakao
        "KoGPT": {
            "vendor": "Kakao",
            "release_year": 2021,
            "aliases": ["HyperCLOVA"],
            "patterns": [
                r"\bKoGPT\b",
                r"\bHyperCLOVA(?:\s*X)?\b",
            ],
            "category_a_typical": ["A1_Product_Innovation"],
        },

        # Naver
        "HyperCLOVA X": {
            "vendor": "Naver",
            "release_year": 2023,
            "aliases": ["HyperCLOVA"],
            "patterns": [r"\bHyperCLOVA\s*X?\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },

        # Zhipu AI
        "GLM": {
            "vendor": "Zhipu AI",
            "release_year": 2022,
            "aliases": ["ChatGLM", "GLM-4"],
            "patterns": [r"\bChatGLM\b", r"\bGLM[- ]?4\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },

        # Moonshot
        "Kimi": {
            "vendor": "Moonshot AI",
            "release_year": 2023,
            "aliases": [],
            "patterns": [r"\bKimi\b(?=.*(?:AI|Moonshot|language model|LLM|chatbot))"],
            "category_a_typical": ["A1_Product_Innovation"],
        },

        # ByteDance
        "Doubao": {
            "vendor": "ByteDance",
            "release_year": 2023,
            "aliases": ["豆包"],
            "patterns": [r"\bDoubao\b", r"\b豆包\b"],
            "category_a_typical": ["A3_Customer_Experience", "A1_Product_Innovation"],
        },
    },

    # -------------------------------------------------------------------------
    # B7_Infrastructure_Platforms_APAC — Huawei AI Cloud
    # -------------------------------------------------------------------------
    "B7_Infrastructure_Platforms_APAC": {

        "Pangu": {
            "vendor": "Huawei",
            "release_year": 2023,
            "aliases": ["盘古", "Pangu Model", "Pangu-α"],
            "patterns": [
                r"\bPangu(?:[-\s]?(?:Model|Alpha|α|\d))?\b",
                r"\b盘古\b",
            ],
            "require_ai_context": True,
            "ai_context_patterns": [
                r"(?:Huawei|AI|language model|LLM|foundation model|generative)",
            ],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "Huawei Cloud AI": {
            "vendor": "Huawei",
            "release_year": 2019,
            "aliases": ["ModelArts", "Huawei AI Cloud"],
            "patterns": [
                r"\bHuawei\s+Cloud\s+(?:AI|ML|intelligence)\b",
                r"\bModelArts\b",
                r"\bAscend\s+(?:AI|computing|processor|chip)\b",
            ],
            "category_a_typical": ["A1_Product_Innovation", "A5_Data_Analytics"],
        },
        "Mindspore": {
            "vendor": "Huawei",
            "release_year": 2020,
            "aliases": ["MindSpore"],
            "patterns": [r"\bMind[Ss]pore\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
    },

    # -------------------------------------------------------------------------
    # B2_Intelligent_Automation — RPA & Process Automation  (Round 3, 2026-03)
    # -------------------------------------------------------------------------
    "B2_Intelligent_Automation": {
        "UiPath": {
            "vendor": "UiPath",
            "release_year": 2005,
            "aliases": ["UiPath RPA", "UiPath Autopilot"],
            "patterns": [r"\bUiPath\b"],
            "category_a_typical": ["A2_Operational_Excellence"],
        },
        "Automation Anywhere": {
            "vendor": "Automation Anywhere",
            "release_year": 2003,
            "aliases": ["AutomationAnywhere", "A360", "Automation 360"],
            "patterns": [
                r"\bAutomation\s+Anywhere\b",
                r"\bAutomation\s+360\b",
                r"\bA360\b(?=.*(?:RPA|automation|bot|AI))",
            ],
            "category_a_typical": ["A2_Operational_Excellence"],
        },
        "Blue Prism": {
            "vendor": "Blue Prism",
            "release_year": 2001,
            "aliases": ["BluePrism"],
            "patterns": [r"\bBlue\s*Prism\b"],
            "category_a_typical": ["A2_Operational_Excellence"],
        },
        "Power Automate": {
            "vendor": "Microsoft",
            "release_year": 2016,
            "aliases": ["Microsoft Power Automate", "Microsoft Flow"],
            "patterns": [r"\bPower\s+Automate\b", r"\bMicrosoft\s+Flow\b"],
            "category_a_typical": ["A2_Operational_Excellence"],
        },
    },

    # -------------------------------------------------------------------------
    # B7_Infrastructure_Platforms_GPU — Nvidia + AMD + Intel  (Round 3)
    # -------------------------------------------------------------------------
    "B7_Infrastructure_Platforms_GPU": {
        "TensorRT": {
            "vendor": "Nvidia",
            "release_year": 2016,
            "aliases": ["NVIDIA TensorRT"],
            "patterns": [r"\bTensorRT\b"],
            "category_a_typical": ["A1_Product_Innovation", "A2_Operational_Excellence"],
        },
        "Nvidia NIM": {
            "vendor": "Nvidia",
            "release_year": 2024,
            "aliases": ["NIM", "NVIDIA NIM"],
            "patterns": [
                r"\bNVIDIA\s+NIM\b",
                r"\bNIM\b(?=.*(?:Nvidia|NVIDIA|inference|microservice|AI))",
            ],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "DGX": {
            "vendor": "Nvidia",
            "release_year": 2016,
            "aliases": ["NVIDIA DGX", "DGX A100", "DGX H100"],
            "patterns": [r"\b(?:NVIDIA\s+)?DGX(?:\s+(?:A100|H100|H200))?\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "CUDA": {
            "vendor": "Nvidia",
            "release_year": 2007,
            "aliases": ["NVIDIA CUDA"],
            "patterns": [r"\bCUDA\b(?=.*(?:GPU|AI|ML|training|inference|Nvidia|NVIDIA))"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "ROCm": {
            "vendor": "AMD",
            "release_year": 2016,
            "aliases": ["AMD ROCm"],
            "patterns": [r"\bROCm\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "Intel Gaudi": {
            "vendor": "Intel",
            "release_year": 2019,
            "aliases": ["Gaudi2", "Gaudi3", "Habana Gaudi"],
            "patterns": [
                r"\bIntel\s+Gaudi(?:[23])?\b",
                r"\bHabana\s+Gaudi\b",
            ],
            "require_ai_context": True,
            "ai_context_patterns": [
                r"(?:AI|ML|training|inference|accelerator|Intel|Habana)",
            ],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "OpenVINO": {
            "vendor": "Intel",
            "release_year": 2018,
            "aliases": ["Intel OpenVINO"],
            "patterns": [r"\bOpenVINO\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
    },

    # -------------------------------------------------------------------------
    # B4_GenAI_LLMs_2025 — New GenAI entrants 2024-2025  (Round 3)
    # -------------------------------------------------------------------------
    "B4_GenAI_LLMs_2025": {
        "Grok": {
            "vendor": "xAI",
            "release_year": 2023,
            "aliases": ["Grok-1", "Grok-2"],
            "patterns": [r"\bGrok(?:[-\s]?[12])?\b(?=.*(?:xAI|Elon|AI|chatbot|LLM))"],
            "category_a_typical": ["A3_Customer_Experience"],
        },
        "DeepSeek": {
            "vendor": "DeepSeek",
            "release_year": 2023,
            "aliases": ["DeepSeek-R1", "DeepSeek-V2", "DeepSeek-V3", "DeepSeek Coder"],
            "patterns": [r"\bDeepSeek(?:[-\s]?(?:R1|V2|V3|Coder|Math))?\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "Gemma": {
            "vendor": "Google",
            "release_year": 2024,
            "aliases": ["Gemma 2"],
            "patterns": [r"\bGemma(?:[\s-]?2)?\b(?=.*(?:Google|AI|model|open.source))"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "Phi": {
            "vendor": "Microsoft",
            "release_year": 2023,
            "aliases": ["Phi-2", "Phi-3", "Phi-4"],
            "patterns": [r"\bPhi[-\s]?[234]\b(?=.*(?:Microsoft|AI|model|SLM))"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "Amazon Nova": {
            "vendor": "AWS",
            "release_year": 2024,
            "aliases": ["Nova Pro", "Nova Lite", "Nova Micro"],
            "patterns": [r"\bAmazon\s+Nova(?:\s+(?:Pro|Lite|Micro|Canvas|Reel))?\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "Amazon Titan": {
            "vendor": "AWS",
            "release_year": 2023,
            "aliases": ["Titan Text", "Titan Embeddings"],
            "patterns": [r"\bAmazon\s+Titan(?:\s+(?:Text|Embeddings|Image))?\b"],
            "category_a_typical": ["A1_Product_Innovation"],
        },
        "NotebookLM": {
            "vendor": "Google",
            "release_year": 2023,
            "aliases": [],
            "patterns": [r"\bNotebookLM\b"],
            "category_a_typical": ["A5_Data_Analytics", "A8_Talent_Workforce"],
        },
    },
}


# ============================================================================
# VENDOR PATTERNS
# For cases where only the vendor is mentioned (no specific product)
# ============================================================================

VENDOR_PATTERNS: Dict[str, Dict] = {
    "OpenAI": {
        "patterns": [
            r"\bOpenAI\b",
            r"\bOpenAI[\s\']?s?\s+(?:model|API|platform|technology)\b",
        ],
        "default_category_b": "B4_GenAI_LLMs",
        "default_products":   ["ChatGPT", "GPT-4"],
    },
    "Anthropic": {
        "patterns": [
            r"\bAnthropic\b",
            r"\bAnthropic[\s\']?s?\s+(?:model|API|Claude)\b",
        ],
        "default_category_b": "B4_GenAI_LLMs",
        "default_products":   ["Claude"],
    },
    "Google": {
        "patterns": [
            r"\bGoogle\s+AI\b",
            r"\bGoogle[\s\']?s?\s+(?:AI|ML|machine\s+learning)\b",
            r"\bDeepMind\b",
        ],
        "default_category_b": "B4_GenAI_LLMs",
        "default_products":   ["Gemini"],
    },
    "Microsoft": {
        "patterns": [
            r"\bMicrosoft\s+AI\b",
            r"\bMicrosoft[\s\']?s?\s+(?:AI|ML|artificial\s+intelligence)\b",
        ],
        "default_category_b": "B7_Infrastructure_Platforms",
        "default_products":   ["Azure ML", "Copilot"],
    },
    "AWS": {
        "patterns": [
            r"\bAWS\s+(?:AI|ML)\b",
            r"\bAmazon\s+(?:AI|ML|machine\s+learning)\b",
        ],
        "default_category_b": "B7_Infrastructure_Platforms",
        "default_products":   ["SageMaker"],
    },
    "IBM": {
        "patterns": [
            r"\bIBM\s+(?:AI|Watson)\b",
        ],
        "default_category_b": "B7_Infrastructure_Platforms",
        "default_products":   ["Watson"],
    },
    "Meta": {
        "patterns": [
            r"\bMeta\s+AI\b",
            r"\bMeta[\s\']?s?\s+(?:AI|Llama)\b",
        ],
        "default_category_b": "B4_GenAI_LLMs",
        "default_products":   ["Llama"],
    },
    "Nvidia": {
        "patterns": [
            r"\bNVIDIA\s+(?:AI|GPU|CUDA)\b",
            r"\bNVIDIA[\s\']?s?\s+AI\b",
        ],
        "default_category_b": "B7_Infrastructure_Platforms",
        "default_products":   [],
    },
    # Round 1: APAC vendors
    "Baidu": {
        "patterns": [
            r"\bBaidu\s+(?:AI|ML|intelligence)\b",
            r"\bBaidu[\s\']?s?\s+(?:AI|ERNIE|Wenxin)\b",
        ],
        "default_category_b": "B4_GenAI_LLMs_APAC",
        "default_products":   ["ERNIE Bot"],
    },
    "Alibaba": {
        "patterns": [
            r"\bAlibaba\s+(?:AI|ML|Cloud\s+AI)\b",
            r"\bAlibaba[\s\']?s?\s+(?:AI|Qwen|Tongyi)\b",
        ],
        "default_category_b": "B4_GenAI_LLMs_APAC",
        "default_products":   ["Qwen"],
    },
    "Tencent": {
        "patterns": [
            r"\bTencent\s+(?:AI|ML|Hunyuan)\b",
            r"\bTencent[\s\']?s?\s+AI\b",
        ],
        "default_category_b": "B4_GenAI_LLMs_APAC",
        "default_products":   ["Hunyuan"],
    },
    "Huawei": {
        "patterns": [
            r"\bHuawei\s+(?:AI|Cloud\s+AI|Ascend|Pangu)\b",
            r"\bHuawei[\s\']?s?\s+AI\b",
        ],
        "default_category_b": "B7_Infrastructure_Platforms_APAC",
        "default_products":   ["Pangu", "Huawei Cloud AI"],
    },
    "Samsung": {
        "patterns": [
            r"\bSamsung\s+(?:AI|Gauss|ML)\b",
            r"\bSamsung[\s\']?s?\s+AI\b",
        ],
        "default_category_b": "B4_GenAI_LLMs_APAC",
        "default_products":   ["Samsung Gauss"],
    },
    "ByteDance": {
        "patterns": [
            r"\bByteDance\s+(?:AI|Doubao)\b",
            r"\bTikTok\s+AI\b",
        ],
        "default_category_b": "B4_GenAI_LLMs_APAC",
        "default_products":   ["Doubao"],
    },
    # Round 2: Enterprise SaaS
    "Salesforce": {
        "patterns": [
            r"\bSalesforce\s+(?:AI|Einstein|ML)\b",
            r"\bSalesforce[\s\']?s?\s+AI\b",
        ],
        "default_category_b": "B7_Infrastructure_Platforms_Enterprise",
        "default_products":   ["Einstein"],
    },
    "SAP": {
        "patterns": [
            r"\bSAP\s+(?:AI|Joule|ML|Hana\s+AI)\b",
        ],
        "default_category_b": "B7_Infrastructure_Platforms_Enterprise",
        "default_products":   ["SAP AI Core"],
    },
    "ServiceNow": {
        "patterns": [
            r"\bServiceNow\s+(?:AI|ML|intelligence)\b",
        ],
        "default_category_b": "B7_Infrastructure_Platforms_Enterprise",
        "default_products":   ["Now Intelligence"],
    },
    "Oracle": {
        "patterns": [
            r"\bOracle\s+(?:AI|Cloud\s+AI|OCI\s+AI)\b",
        ],
        "default_category_b": "B7_Infrastructure_Platforms_Enterprise",
        "default_products":   ["Oracle AI"],
    },
    "Palantir": {
        "patterns": [r"\bPalantir\b"],
        "default_category_b": "B7_Infrastructure_Platforms_Enterprise",
        "default_products":   ["Palantir AIP"],
    },
    # Round 3: Infrastructure & Automation
    "UiPath": {
        "patterns": [r"\bUiPath\b"],
        "default_category_b": "B2_Intelligent_Automation",
        "default_products":   ["UiPath"],
    },
    "Nvidia": {
        "patterns": [
            r"\bNVIDIA\s+(?:AI|GPU|CUDA|DGX|TensorRT|NIM|H100|A100)\b",
            r"\bNVIDIA[\s\']?s?\s+AI\b",
        ],
        "default_category_b": "B7_Infrastructure_Platforms_GPU",
        "default_products":   ["DGX", "TensorRT"],
    },
    "xAI": {
        "patterns": [r"\bxAI\b(?=.*(?:Grok|Elon|AI|chatbot))", r"\bGrok\b"],
        "default_category_b": "B4_GenAI_LLMs_2025",
        "default_products":   ["Grok"],
    },
    "DeepSeek": {
        "patterns": [r"\bDeepSeek\b"],
        "default_category_b": "B4_GenAI_LLMs_2025",
        "default_products":   ["DeepSeek"],
    },
}


# ============================================================================
# INTERNAL PRODUCT PATTERNS
# For proprietary / in-house solutions
# ============================================================================

INTERNAL_PATTERNS = [
    r"\b(?:proprietary|in[\s-]?house|internal|custom[\s-]?built)\s+(?:AI|ML|model|platform|solution)\b",
    r"\b(?:our|their)\s+(?:own|proprietary)\s+(?:AI|ML|algorithm|model)\b",
    r"\bdeveloped\s+(?:internally|in[\s-]?house)\b.*(?:AI|ML|model)",
    r"\b(?:built|created|designed)\s+(?:our|an?\s+internal)\s+(?:AI|ML)\b",
]


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

# ============================================================================
# TEMPORAL EXCLUSION LOG
# Records every anachronistic detection (report_year < product release_year).
# Useful for thesis validation: should be 0% after correct temporal filtering.
# Reset between runs with reset_temporal_log().
# ============================================================================

import logging as _logging
_logger = _logging.getLogger("AISA")

TEMPORAL_EXCLUSION_LOG: List[Dict] = []
# Each entry: {product, vendor, release_year, report_year, text_snippet}


def reset_temporal_log() -> None:
    """Clear the temporal exclusion log. Call before each corpus run."""
    TEMPORAL_EXCLUSION_LOG.clear()


def get_temporal_exclusion_report() -> Dict:
    """
    Return a summary report of all temporal exclusions since last reset.

    Returns a dict with:
        total           — total exclusions
        by_product      — {product_name: count}
        by_year         — {report_year: count}
        by_vendor       — {vendor: count}
        details         — full log list (for export to Excel/JSON)

    Example use after a full corpus run:
        report = get_temporal_exclusion_report()
        print(f"Temporal exclusions: {report['total']}")
        # → Should be 0 if temporal validation is working correctly
    """
    if not TEMPORAL_EXCLUSION_LOG:
        return {
            "total":       0,
            "by_product":  {},
            "by_year":     {},
            "by_vendor":   {},
            "details":     [],
        }

    by_product: Dict[str, int] = {}
    by_year:    Dict[int, int] = {}
    by_vendor:  Dict[str, int] = {}

    for entry in TEMPORAL_EXCLUSION_LOG:
        p = entry["product"] or "Unknown"
        v = entry["vendor"]  or "Unknown"
        y = entry["report_year"]

        by_product[p] = by_product.get(p, 0) + 1
        by_vendor[v]  = by_vendor.get(v, 0) + 1
        by_year[y]    = by_year.get(y, 0) + 1

    return {
        "total":      len(TEMPORAL_EXCLUSION_LOG),
        "by_product": dict(sorted(by_product.items(), key=lambda x: -x[1])),
        "by_year":    dict(sorted(by_year.items())),
        "by_vendor":  dict(sorted(by_vendor.items(), key=lambda x: -x[1])),
        "details":    list(TEMPORAL_EXCLUSION_LOG),
    }


def print_temporal_exclusion_report() -> None:
    """Print a human-readable temporal exclusion report to stdout."""
    r = get_temporal_exclusion_report()
    print(f"\n{'='*60}")
    print(f"  TEMPORAL EXCLUSION REPORT")
    print(f"  Total anachronistic detections blocked: {r['total']}")
    print(f"{'='*60}")
    if r["total"] == 0:
        print("  ✓ No temporal exclusions — validation passed.")
        return
    print(f"\n  By product (top 10):")
    for product, count in list(r["by_product"].items())[:10]:
        print(f"    {product:<35} {count:>4}")
    print(f"\n  By report year:")
    for year, count in r["by_year"].items():
        print(f"    {year}    {count:>4} exclusions")
    print(f"\n  By vendor:")
    for vendor, count in r["by_vendor"].items():
        print(f"    {vendor:<25} {count:>4}")
    print()


def extract_product_info(
    text:        str,
    category_b:  str = None,
    context:     str = None,
    report_year: int = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Extract product and vendor information from text.

    Args:
        text:        The matched AI reference text.
        category_b:  Optional category B if already classified.
        context:     Extended context around the match.
        report_year: Year of the annual report (for temporal validation).

    Returns:
        Tuple of (product_name, vendor, granularity_level).

    Side effects:
        Anachronistic detections (report_year < release_year) are recorded
        in TEMPORAL_EXCLUSION_LOG for post-run reporting and thesis validation.
    """
    full_text     = f"{text} {context or ''}".lower()
    original_text = f"{text} {context or ''}"

    # Step 1: Internal / proprietary solutions
    for pattern in INTERNAL_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            name_match = re.search(
                r'(?:called|named|dubbed)\s+["\']?([A-Za-z][A-Za-z0-9\s]{2,20})["\']?',
                original_text, re.IGNORECASE,
            )
            product_name = name_match.group(1).strip() if name_match else None
            return (product_name, "Internal", GranularityLevel.INTERNAL)

    # Step 2: Match specific products
    categories_to_search = (
        [category_b] if category_b and category_b in KNOWN_PRODUCTS
        else KNOWN_PRODUCTS.keys()
    )

    for cat_b in categories_to_search:
        products = KNOWN_PRODUCTS.get(cat_b, {})
        for product_name, product_info in products.items():
            matched = False

            for pattern in product_info.get("patterns", []):
                if re.search(pattern, original_text, re.IGNORECASE):
                    matched = True
                    break

            if not matched:
                for alias in product_info.get("aliases", []):
                    if re.search(rf"\b{re.escape(alias)}\b", original_text, re.IGNORECASE):
                        matched = True
                        break

            if not matched:
                continue

            # Temporal validation: skip if report is older than product release
            release_year = product_info.get("release_year")
            if release_year and report_year and report_year < release_year:
                TEMPORAL_EXCLUSION_LOG.append({
                    "product":      product_name,
                    "vendor":       product_info.get("vendor"),
                    "release_year": release_year,
                    "report_year":  report_year,
                    "text_snippet": text[:120],
                })
                _logger.debug(
                    "Temporal exclusion: %s (released %d) in %d report — skipped",
                    product_name, release_year, report_year,
                )
                continue

            # AI context validation for ambiguous product names
            if product_info.get("require_ai_context", False):
                ai_context_found = any(
                    re.search(p, full_text, re.IGNORECASE)
                    for p in product_info.get("ai_context_patterns", [])
                )
                if not ai_context_found:
                    continue

            return (product_name, product_info["vendor"], GranularityLevel.SPECIFIC)

    # Step 3: Vendor only
    for vendor, vendor_info in VENDOR_PATTERNS.items():
        for pattern in vendor_info.get("patterns", []):
            if re.search(pattern, original_text, re.IGNORECASE):
                return (None, vendor, GranularityLevel.VENDOR_ONLY)

    # Step 4: Category only
    return (None, None, GranularityLevel.CATEGORY_ONLY)


def get_vendor_for_product(product_name: str) -> Optional[str]:
    """Return vendor name for a known product, or None."""
    for category_products in KNOWN_PRODUCTS.values():
        if product_name in category_products:
            return category_products[product_name].get("vendor")
    return None


def get_product_category(product_name: str) -> Optional[str]:
    """Return category B code for a known product, or None."""
    for category_b, products in KNOWN_PRODUCTS.items():
        if product_name in products:
            return category_b
    return None


def get_product_release_year(product_name: str) -> Optional[int]:
    """Return release year for a known product, or None."""
    for category_products in KNOWN_PRODUCTS.values():
        if product_name in category_products:
            return category_products[product_name].get("release_year")
    return None


def get_typical_applications(product_name: str) -> List[str]:
    """Return typical application categories (A-dimension) for a product."""
    for category_products in KNOWN_PRODUCTS.values():
        if product_name in category_products:
            return category_products[product_name].get("category_a_typical", [])
    return []


def search_products(query: str) -> List[Dict]:
    """
    Search for products matching a query string (name or alias).

    Returns a list of dicts with keys: product, vendor, category_b, match_type.
    """
    results    = []
    query_lower = query.lower()

    for category_b, products in KNOWN_PRODUCTS.items():
        for product_name, product_info in products.items():
            if query_lower in product_name.lower():
                results.append({
                    "product":    product_name,
                    "vendor":     product_info["vendor"],
                    "category_b": category_b,
                    "match_type": "name",
                })
                continue
            for alias in product_info.get("aliases", []):
                if query_lower in alias.lower():
                    results.append({
                        "product":       product_name,
                        "vendor":        product_info["vendor"],
                        "category_b":    category_b,
                        "match_type":    "alias",
                        "matched_alias": alias,
                    })
                    break
    return results


def get_product_statistics() -> Dict:
    """Return summary statistics about the product dictionary."""
    stats: Dict = {
        "total_products":       0,
        "total_vendors":        set(),
        "products_by_category": {},
        "vendors_by_category":  {},
    }
    for category_b, products in KNOWN_PRODUCTS.items():
        stats["products_by_category"][category_b] = len(products)
        stats["total_products"] += len(products)
        vendors = {p["vendor"] for p in products.values() if p.get("vendor")}
        stats["total_vendors"].update(vendors)
        stats["vendors_by_category"][category_b] = sorted(vendors)
    stats["total_vendors"] = len(stats["total_vendors"])
    return stats


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    print(f"AISA 14_ai_products_v1.py v{PRODUCTS_VERSION}")
    print("=" * 60)

    stats = get_product_statistics()
    print(f"\n  Total products : {stats['total_products']}")
    print(f"  Total vendors  : {stats['total_vendors']}")
    print("\n  Products by category:")
    for cat, count in stats["products_by_category"].items():
        print(f"    {cat:<35} {count} products")

    print("\n" + "=" * 60)
    print("  Extraction tests:")

    test_cases = [
        ("We deployed Microsoft Copilot across finance",       "Copilot",   "Microsoft"),
        ("Implemented ChatGPT for customer support",           "ChatGPT",   "OpenAI"),
        ("Partnered with Anthropic for AI solutions",          None,        "Anthropic"),
        ("Using generative AI to improve efficiency",          None,        None),
        ("Developed proprietary ML platform called DataBrain", "DataBrain", "Internal"),
        ("Leveraging OpenAI's API for automation",             None,        "OpenAI"),
        ("Running models on AWS SageMaker",                    "SageMaker", "AWS"),
        ("Gemini clinical trial Phase III results",            None,        None),   # FP
        ("Mistral wind turbine offshore installation",         None,        None),   # FP
        ("GPT-4 model deployed on Azure OpenAI Service",      "GPT-4",     "OpenAI"),
        # Round 1: APAC
        ("Baidu launched ERNIE Bot for enterprise AI",         "ERNIE Bot", "Baidu"),
        ("Alibaba released Qwen2 open-source model",           "Qwen",      "Alibaba"),
        ("Tencent Hunyuan AI platform for developers",         "Hunyuan",   "Tencent"),
        ("Huawei Pangu Model for industry applications",       "Pangu",     "Huawei"),
        ("Samsung Gauss generative AI model announced",        "Samsung Gauss", "Samsung"),
        ("ByteDance Doubao chatbot expansion",                 "Doubao",    "ByteDance"),
        ("Huawei ModelArts cloud AI platform",                 "Huawei Cloud AI", "Huawei"),
        ("通义千问 deployed in Alibaba operations",             "Qwen",      "Alibaba"),
        # Round 2: Enterprise SaaS
        ("Salesforce Einstein GPT for sales automation",       "Einstein",       "Salesforce"),
        ("SAP Joule AI assistant embedded in ERP",             "SAP AI Core",    "SAP"),
        ("ServiceNow Now Assist improves IT workflows",        "Now Intelligence","ServiceNow"),
        ("Oracle OCI AI Services for cloud workloads",         "Oracle AI",      "Oracle"),
        ("Workday Illuminate AI for HR analytics",             "Workday AI",     "Workday"),
        ("Adobe Firefly generative AI for creative teams",     "Adobe Firefly",  "Adobe"),
        ("Palantir AIP deployed for defense analytics",        "Palantir AIP",   "Palantir"),
        ("C3.ai enterprise AI suite for energy sector",        "C3.ai",          "C3.ai"),
        # Round 3: Infrastructure & Automation
        ("UiPath RPA bots automate finance workflows",          "UiPath",          "UiPath"),
        ("Automation Anywhere 360 deployed in back office",     "Automation Anywhere", "Automation Anywhere"),
        ("Blue Prism digital workers for claims processing",    "Blue Prism",      "Blue Prism"),
        ("Microsoft Power Automate for HR onboarding",         "Power Automate",  "Microsoft"),
        ("NVIDIA DGX H100 cluster for LLM training",           "DGX",             "Nvidia"),
        ("TensorRT optimizes inference latency on GPU",         "TensorRT",        "Nvidia"),
        ("DeepSeek-R1 open-source model released",             "DeepSeek",        "DeepSeek"),
        ("Grok-2 chatbot by xAI for enterprise",               "Grok",            "xAI"),
        ("Amazon Nova Pro multimodal model for enterprise AI",  "Amazon Nova",     "AWS"),
        ("Amazon Titan Embeddings for RAG applications",        "Amazon Titan",    "AWS"),
    ]

    all_ok = True
    for text, expected_product, expected_vendor in test_cases:
        product, vendor, granularity = extract_product_info(text)
        ok = (product == expected_product) and (vendor == expected_vendor)
        status = "  [OK]  " if ok else "  [FAIL]"
        print(f"{status} {text[:55]:<55}")
        if not ok:
            print(f"         Expected: product={expected_product!r}, vendor={expected_vendor!r}")
            print(f"         Got:      product={product!r}, vendor={vendor!r}")
            all_ok = False

    # Helper functions
    assert get_vendor_for_product("ChatGPT") == "OpenAI"
    assert get_product_category("SageMaker") == "B7_Infrastructure_Platforms"
    assert get_product_release_year("GPT-4") == 2023
    assert "A2_Operational_Excellence" in get_typical_applications("ChatGPT")
    assert len(search_products("GPT")) >= 2
    print("\n  Helper functions [OK]")

    # Temporal exclusion log test
    reset_temporal_log()
    # GPT-4 released 2023 — should be excluded from a 2021 report
    r1 = extract_product_info("We use GPT-4 for operations", report_year=2021)
    assert r1[0] is None, f"GPT-4 should be excluded from 2021 report, got {r1}"
    assert len(TEMPORAL_EXCLUSION_LOG) == 1
    assert TEMPORAL_EXCLUSION_LOG[0]["product"] == "GPT-4"
    assert TEMPORAL_EXCLUSION_LOG[0]["report_year"] == 2021
    # GPT-4 released 2023 — should be included in a 2024 report
    reset_temporal_log()
    r2 = extract_product_info("We use GPT-4 for operations", report_year=2024)
    assert r2[0] == "GPT-4", f"GPT-4 should be detected in 2024 report, got {r2}"
    assert len(TEMPORAL_EXCLUSION_LOG) == 0
    print("  Temporal exclusion log [OK]")
    print_temporal_exclusion_report()

    assert all_ok, "\nSome extraction tests failed — check output above."
    print()
    print("  14_ai_products_v1.py all checks passed.")

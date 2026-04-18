"""
===============================================================================
AISA - AI Semantic Analyzer
04_taxonomy_digitalization_relational_v2_2_0.py
Digitalizare Relationala & Ecosisteme de Business
===============================================================================

Taxonomie tri-axiala v2.2.0:
    Axa D — Application  (unde se digitalizeaza relatia): D1-D8
    Axa T — Technology   (cu ce tehnologie):              T1-T8
    Axa G — Governance   (cum este guvernata relatia):    G1-G6

GATE LOGIC (implementata in check_false_positive + classify):
    Co-occurrence Gate    (T2-T8): keyword T acceptat DOAR cu actor extern
                                   in fereastra +/-500 chars
    Co-occur + Verb Gate  (T1,T4): actor extern + verb relational obligatoriu
    Soft Gate             (G1-G6): marker digital in fereastra +/-400 chars
                                   SAU hit D/T anterior in document
    No Gate               (D1-D8): detectabile independent

CHANGELOG:
    v2.2.0 (2026-04) - +oportunism/power imbalance/relationship transparency
                       pe G1/G2/G5; +Catena-X/data spaces/sovereign exchange
                       pe D4/G3/G4; +legacy relationships/platform resistance
                       pe D7/G5; +DPP/battery passport/carbon footprint pe D8;
                       +4 FP patterns noi (opportunism_generic,
                       power_imbalance_generic, passport_travel, catena_x_generic);
                       +19 anchor phrases noi (★); +9 surse bibliografice R13-R21
    v2.1.0 (2026-04) - Soft Gate pentru G; Verb relational pentru T1/T4;
                       D6 ingustata la B2B/industrial; +7 FP patterns G;
                       T1/T4 keywords reformulate relational
    v2.0.0 (2026-04) - Model tri-axial D/T/G; arhitectura ChatGPT +
                       profunzime keywords/anchors AISA
    v1.1.0 (2026-04) - +D9 Governance, +D10 Sustainability
    v1.0.0 (2026-04) - Versiune initiala D1-D8 + T1-T8

Author: TeRa0
License: MIT
===============================================================================
"""

from __future__ import annotations

import importlib
import re
import sys
import os
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAXONOMY_NAME    = "Digitalization_Relational_v2"
TAXONOMY_VERSION = "2.2.0"

_m3 = importlib.import_module("03_taxonomy_base")
TaxonomyProvider     = _m3.TaxonomyProvider
CategoryInfo         = _m3.CategoryInfo
FPResult             = _m3.FPResult
ClassificationResult = _m3.ClassificationResult
CompiledPatternCache = _m3.CompiledPatternCache


# ============================================================================
# GATE SETS
# ============================================================================

# T2-T8: actor extern obligatoriu in fereastra ±500 chars
EXTERNAL_ACTORS: Set[str] = {
    "supplier", "vendor", "partner", "dealer", "distributor", "customer",
    "ecosystem", "third-party", "third party", "network", "complementor",
    "channel partner", "trading partner", "logistics partner", "service partner",
    "reseller", "developer", "marketplace participant",
}

# T1, T4: actor extern + verb relational obligatoriu
RELATIONAL_VERBS: Set[str] = {
    "integrate", "integrates", "integrated", "integrating",
    "connect", "connects", "connected", "connecting",
    "share", "shares", "shared", "sharing",
    "exchange", "exchanges", "exchanged", "exchanging",
    "onboard", "onboards", "onboarded", "onboarding",
    "collaborate", "collaborates", "collaborated", "collaborating",
    "synchronize", "synchronizes", "synchronized", "synchronizing",
    "transact", "transacts", "transacted",
    "extend to", "link to", "link suppliers", "link partners",
    "enable suppliers", "enable partners", "enable dealers",
    "connect suppliers", "connect partners", "connect dealers",
}

# G1-G6: marker digital obligatoriu in fereastra ±400 chars
DIGITAL_MARKERS: Set[str] = {
    "data", "platform", "api", "portal", "analytics", "system", "digital",
    "interoperab", "control tower", "shared", "exchange", "integration",
    "ecosystem", "partner", "supplier", "network", "cloud", "iot",
    "blockchain", "erp", "crm", "workflow", "dashboard", "dlt",
    "marketplace", "interface", "automation", "edi", "middleware",
}

# Categorii per tip de gate
COOCCUR_T_VERB: Set[str] = {"T1", "T4"}           # Co-occur + relational verb
COOCCUR_T_ONLY: Set[str] = {"T2","T3","T5","T6","T7","T8"}  # Co-occur only
SOFT_GATE_G:    Set[str] = {"G1","G2","G3","G4","G5","G6"}  # Soft gate


# ============================================================================
# FALSE POSITIVE PATTERNS
# ============================================================================

FALSE_POSITIVE_PATTERNS: Dict[str, List[str]] = {
    "erp_internal_rollout_only": [
        r"\bERP\b.*\b(rollout|upgrade|migration|implementation)\b(?!.*\b(supplier|vendor|dealer|partner|customer|distributor|network|ecosystem|API|EDI|integrate|connect|synchronize|exchange)\b)",
    ],
    "cloud_generic_hosting": [
        r"\bcloud\b.*\b(hosting|server|backup|storage|compute|data center)\b(?!.*\b(partner|supplier|customer|shared|ecosystem|platform|exchange|interoperab|connect)\b)",
    ],
    "mobile_app_consumer_only": [
        r"\b(?:mobile app|application|app)\b.*\b(loyalty|consumer|shopping|gaming|entertainment)\b(?!.*\b(partner|dealer|distributor|supplier)\b)",
    ],
    "ecommerce_consumer_only": [
        r"\b(e-commerce|online store|webshop|direct-to-consumer)\b(?!.*\b(B2B|dealer|distributor|partner portal|marketplace|channel)\b)",
    ],
    "blockchain_speculative": [
        r"\bblockchain\b.*\b(pilot|research|exploratory|whitepaper|concept)\b(?!.*\b(traceability|trade|supplier|provenance|smart contract)\b)",
    ],
    "iot_consumer": [
        r"\bconsumer\s+IoT\b|\bsmart\s+(?:home|watch|speaker|TV|appliance)\b(?!.*\b(industrial|partner|supplier|logistics)\b)",
    ],
    "cyber_incident_only": [
        r"\b(cybersecurity|security)\b.*\b(incident|breach|attack|patch|CVE|vulnerability)\b(?!.*\b(partner|ecosystem|federated|supplier|access|exchange)\b)",
    ],
    "hr_onboarding": [
        r"\bonboarding\b.*\b(employee|talent|HR|staff|workforce)\b(?!.*\b(partner|supplier|vendor|dealer|distributor|ecosystem)\b)",
    ],
    "investor_portal": [
        r"\bportal\b.*\b(investor|shareholder|IR|investor relations)\b(?!.*\b(supplier|partner|dealer|distributor|ecosystem)\b)",
    ],
    "compliance_internal_only": [
        r"\bcompliance\b.*\b(training|internal policy|code of conduct|employee)\b(?!.*\b(partner|supplier|chain|network|ecosystem)\b)",
    ],
    "automation_internal_rpa": [
        r"\b(RPA|robotic process automation|automation)\b.*\b(back office|HR|payroll|accounting|finance)\b(?!.*\b(partner|supplier|customer|ecosystem|inter-company|cross-company)\b)",
    ],
    "sustainability_reporting_only": [
        r"\b(ESG|sustainability|scope 3)\b.*\b(reporting|disclosure|rating|score)\b(?!.*\b(supplier|partner|chain|traceability|digital|platform|blockchain)\b)",
    ],
    "partnership_generic_nondigital": [
        r"\bpartner(ship)?\b(?!.*\b(platform|API|portal|data|integration|digital|ecosystem|onboarding|workflow|exchange|connect)\b)",
    ],
    "cloud_meteorological": [
        r"\bcloud\s+(?:cover|formation|seeding|layer|ceiling)\b",
    ],
    "ecosystem_ecological": [
        r"\becosystem\b.*\b(forest|marine|biodiversity|climate|nature|habitat|species)\b",
    ],
    "platform_political": [
        r"\bplatform\b.*\b(political|party|policy|electoral|campaign)\b",
    ],
    "governance_board_only": [
        r"\bgovernance\b.*\b(board|director|shareholder|proxy|AGM)\b(?!.*\b(platform|digital|ecosystem|data|API|partner)\b)",
    ],
    "sustainability_non_digital": [
        r"\bsustainab\w+\b.*\b(agriculture|farming|fishing|forestry|biodiversity)\b(?!.*\b(digital|platform|data|reporting|blockchain)\b)",
    ],
    # v2.1 — G-category specific FPs
    "data_governance_internal_only": [
        r"\bdata\s+governance\b(?!.*\b(partner|supplier|ecosystem|exchange|sharing|interoper|cross-company|external|platform|third-party)\b)",
    ],
    "learning_internal_LD_only": [
        r"\b(learning|knowledge sharing|upskilling|training)\b.*\b(employee|workforce|internal|staff|L&D|HR|academy)\b(?!.*\b(partner|supplier|ecosystem|cross-company|external|platform|joint|collaborative)\b)",
    ],
    "control_tower_internal_only": [
        r"\bcontrol\s+tower\b(?!.*\b(supplier|partner|network|ecosystem|multi-tier|visibility|coordination|external)\b)",
    ],
    "transparency_investor_relations": [
        r"\btransparency\b.*\b(investor|shareholder|market|financial reporting|ESG disclosure|governance report)\b(?!.*\b(partner|supplier|platform|digital|ecosystem|data|API)\b)",
    ],
    "standards_internal_IT_only": [
        r"\b(standards|standard)\b.*\b(IT architecture|internal|security framework|ISO|NIST|internal compliance)\b(?!.*\b(partner|ecosystem|interoperab|exchange|data model|shared|external|supplier)\b)",
    ],
    "customer_experience_consumer_only": [
        r"\b(customer experience|CX|customer journey|consumer experience)\b(?!.*\b(B2B|industrial|partner|dealer|distributor|service partner|co-innovation|servitization)\b)",
    ],
    "developer_platform_internal_only": [
        r"\b(developer platform|developer tools|developer portal)\b(?!.*\b(third-party|external|partner|ecosystem|complementor|API marketplace|open)\b)",
    ],
    # v2.2 — 4 FP patterns noi
    "opportunism_generic": [
        r"\bopportunism\b(?!.*\b(digital|platform|supplier|partner|ecosystem|data|governance|monitoring)\b)",
    ],
    "power_imbalance_generic": [
        r"\bpower\s+imbalance\b(?!.*\b(platform|digital|ecosystem|supplier|partner|data|governance)\b)",
    ],
    "passport_travel": [
        r"\bpassport\b(?!.*\b(product|digital|battery|carbon|data|supply chain|circular|traceability)\b)",
    ],
    "catena_x_generic": [
        r"\bcatena\b(?!.*\b(x|X|-X|data|automotive|supply chain|ecosystem|sovereign)\b)",
    ],
}


# ============================================================================
# ANCHOR PHRASES
# ============================================================================

ANCHOR_PHRASES: Dict[str, List[str]] = {
    "D1_Partner_Onboarding_Access": [
        "We launched a partner onboarding portal that digitizes supplier and distributor registration end to end.",
        "New dealers can self-register, upload compliance documents and receive system access through a digital workflow.",
        "Vendor onboarding is managed through a self-service portal integrated with approval and identity checks.",
        "Partner activation time decreased after digitizing onboarding and access provisioning.",
        "Our ecosystem onboarding platform allows third-party partners to integrate within days rather than weeks.",
        "Supplier self-registration workflows replaced manual onboarding and reduced activation lead times significantly.",
    ],
    "D2_Transactional_Integration": [
        "Orders and invoices now flow automatically between our systems and those of suppliers through EDI and API connections.",
        "We digitized procure-to-pay processes with trading partners to reduce manual transaction handling.",
        "Our order-to-cash integration connects customers and logistics partners on a shared digital workflow.",
        "Electronic invoicing and purchase order automation improved transaction accuracy across partner networks.",
        "B2B transaction integration eliminated paper-based processes in our supplier order management flows.",
        "The inter-company transaction platform processes over 2 million supplier invoices per year automatically.",
    ],
    "D3_Collaborative_Planning_Visibility": [
        "We introduced a shared visibility platform that gives suppliers and logistics partners real-time planning signals.",
        "Collaborative planning and forecasting are now performed through a digital control tower used across the network.",
        "Inventory and replenishment decisions are coordinated with partners through shared dashboards.",
        "Digital visibility tools improved synchronization across our extended supply network.",
        "Our partner control tower provides real-time demand signals to tier-1 and tier-2 suppliers simultaneously.",
        "Multi-tier supply chain visibility reduced stockouts by 30% through shared replenishment planning.",
    ],
    "D4_Shared_Data_Interoperability": [
        "We created a shared data platform that enables structured exchange of operational data with partners.",
        "Common data models and interoperable interfaces support cross-company collaboration in the ecosystem.",
        "Master data is synchronized across suppliers and distributors through a governed data exchange layer.",
        "Our interoperability program standardizes data flows across partners and platforms.",
        "The federated data space connects 1,200 ecosystem partners under common data standards and sovereignty rules.",
        "GAIA-X-compliant data exchange infrastructure governs how operational data is shared across our network.",
        # ★ v2.2
        "Catena-X membership enables sovereign data exchange with automotive suppliers under federated governance rules.",
        "Our industrial data ecosystem connects 400 tier-1 suppliers through standardized data space connectors.",
        "Sovereign data exchange agreements govern how operational data flows between firms without compromising proprietary information.",
    ],
    "D5_Channel_Partner_Interface": [
        "We digitized the dealer and distributor interface through a unified partner portal.",
        "Channel partners access pricing, orders and service cases through a digital management platform.",
        "Our reseller ecosystem now operates through self-service partner tools and integrated workflows.",
        "Digital channel interfaces improved visibility and responsiveness across the distribution network.",
        "The dealer portal handles 95% of order placement, returns and warranty claims without manual intervention.",
        "Distributor management digitalization reduced order error rates by 40% across our channel network.",
    ],
    "D6_CoInnovation_Digital_Servitization": [
        "We use connected assets and shared operational data to deliver digital services jointly with industrial customers.",
        "Predictive maintenance and remote monitoring created new B2B service relationships with equipment partners.",
        "Outcome-based contracts with industrial customers are supported by digital platforms connecting provider and customer data.",
        "Our digital servitization model depends on continuous collaboration and data exchange with installed-base partners.",
        "Digital servitization transformed B2B equipment sales into recurring outcome-based contracts with 200+ industrial customers.",
        "Co-innovation with industrial partners on digital service platforms generated $350 million in new B2B service revenue.",
    ],
    "D7_Ecosystem_Orchestration_Complementors": [
        "We expanded our digital ecosystem by onboarding complementors through a dedicated partner platform.",
        "Third-party developers build on our APIs under a structured ecosystem governance model.",
        "The company orchestrates a network of partners and complementors through shared digital infrastructure.",
        "Marketplace and API programs were used to scale partner participation in the ecosystem.",
        "Our multi-sided platform connects 4,500 complementors with enterprise customers across 40 markets.",
        "Ecosystem orchestration through shared APIs generated $2.3 billion in partner-driven revenue.",
        # ★ v2.2
        "We manage a multi-platform ecosystem where incumbent platform assets serve both product and service complementors.",
        "Legacy relationship digitalization enabled long-standing channel partners to participate in our new platform ecosystem.",
        "Relational challenges to B2B platform adoption were addressed through co-design of onboarding rules with key partners.",
    ],
    "D8_Traceability_Compliance_Network": [
        "We built end-to-end traceability with suppliers through shared digital records and compliance interfaces.",
        "The product passport program depends on digital data exchange across multiple value-chain partners.",
        "Chain-of-custody information is captured through a network platform linking suppliers and downstream partners.",
        "Digital traceability tools support regulatory and sustainability compliance across the ecosystem.",
        "Blockchain-based traceability covers 100% of our Scope 3 supply chain for ESG reporting purposes.",
        "The digital product passport enables full provenance tracking from raw material to end consumer.",
        # ★ v2.2
        "Battery passports enable end-to-end lifecycle traceability from raw material sourcing to recycling across our partner network.",
        "Product carbon footprint data is exchanged between suppliers and OEMs through our interoperable passport data carrier.",
        "Decentralized and tamper-proof digital product passports ensure transparent provenance across all value chain participants.",
        "Our DPP ecosystem connects manufacturers, logistics partners and recyclers under shared circular economy data governance.",
    ],
    "T1_ERP_SCM_Backbone": [
        "Our ERP and supply chain backbone connects suppliers and procurement partners on shared workflows.",
        "The digital core was extended beyond the enterprise boundary to support partner transactions and integrate supplier data.",
        "SCM and procurement backbone systems are integrated and synchronized with external supplier processes.",
        "ERP-to-supplier integration reduced order processing time from 5 days to 4 hours across our network.",
        "SAP Ariba connects our procurement backbone with 4.5 million suppliers through shared transaction flows.",
        "Backbone modernization enabled structured collaboration with partners, connecting our ERP to dealer and distributor networks.",
    ],
    "T2_API_EDI_iPaaS_Middleware": [
        "APIs, EDI and middleware are used to connect our systems with those of suppliers and partners.",
        "A B2B integration layer enables real-time data exchange across the ecosystem.",
        "The company relies on iPaaS and API gateways to integrate third-party partner workflows.",
        "Middleware and service layers are described as enablers of inter-company collaboration.",
        "MuleSoft enabled us to connect 340 legacy systems to partner APIs through a unified integration layer.",
        "Our B2B gateway processes 2 billion API calls per month across our ecosystem partner network.",
    ],
    "T3_B2B_Platforms_Marketplaces": [
        "We use a B2B platform to coordinate transactions and interactions across multiple business partners.",
        "The digital marketplace serves as the shared interface for suppliers, customers and complementors.",
        "Partner participation is mediated through a common platform rather than bilateral manual processes.",
        "Coupa deployment standardized procurement across 60 countries on a single supplier network platform.",
        "B2B marketplace volumes exceeded $12 billion in digitally mediated partner transactions in 2023.",
        "The platform is described as infrastructure for business ecosystem coordination across supply chain partners.",
    ],
    "T4_Cloud_Data_Platforms": [
        "A shared cloud platform enables collaboration and data exchange with ecosystem partners.",
        "Our data platform connects internal and partner datasets for joint operational use cases across the supply chain.",
        "Industry cloud services were adopted to standardize collaboration and data sharing across the partner network.",
        "Cloud technologies are framed as shared infrastructure for inter-organizational processes with suppliers and distributors.",
        "Our industry cloud connects 800 manufacturing partners on a shared data exchange environment.",
        "Cloud-based supply chain platform reduced integration lead times from months to days for new partner onboarding.",
    ],
    "T5_IoT_Digital_Twins_Connected_Assets": [
        "Connected assets and IoT platforms enable shared visibility and service coordination with partners.",
        "Digital twins are used to exchange operational information across company boundaries with service partners.",
        "Remote monitoring links providers, customers and service partners through connected equipment.",
        "IoT technologies are described as enablers of ecosystem-level coordination across partner firms.",
        "Our IIoT platform connects 2.3 million sensors and shares operational data with service partners in real time.",
        "Digital twins of customer assets enable predictive service delivery coordinated across supplier ecosystems.",
    ],
    "T6_Identity_Access_Security": [
        "External partner collaboration is enabled through federated identity and controlled access management.",
        "Supplier and partner users access shared platforms through secure identity federation.",
        "Zero-trust controls are used to protect data exchange across business boundaries with ecosystem partners.",
        "Federated identity management allows 85,000 partner users to securely access our digital platforms.",
        "Zero-trust architecture governs data exchange with 500+ ecosystem partners across 40 jurisdictions.",
        "Security capabilities are described as enablers of trusted ecosystem participation across our partner network.",
    ],
    "T7_Blockchain_DLT_Traceability": [
        "Blockchain is used to create shared records and traceability across multiple supply chain partners.",
        "Distributed ledger technology supports provenance and trade documentation between trading firms.",
        "Smart contracts are mentioned in the context of supplier payment triggers and logistics partner collaboration.",
        "Blockchain traceability platform covers 100% of our food supply chain provenance for partner ESG compliance.",
        "Smart contracts with suppliers automate payment triggers on confirmed delivery across our partner network.",
        "The DLT platform is framed as multi-party infrastructure connecting buyers, suppliers and logistics partners.",
    ],
    "T8_Workflow_Automation_LowCode": [
        "Low-code workflow tools automate approvals and cases involving external partners and suppliers.",
        "Digital forms and workflow orchestration streamline cross-company processes with dealers and distributors.",
        "Automation is applied to shared partner workflows rather than only internal back-office routines.",
        "Partner workflow automation reduced onboarding approval cycles from 14 days to 2 days.",
        "Low-code partner portal configurators allowed business teams to deploy new partner journeys in days.",
        "The company uses low-code tools to configure partner-facing approval processes quickly at scale.",
    ],
    "G1_Transparency_Visibility_Control": [
        "The partnership model relies on shared dashboards, transparency and real-time monitoring across the digital platform.",
        "Control towers and shared KPI visibility are used as governance mechanisms across the partner network.",
        "Digital transparency reduces coordination frictions with suppliers and logistics partners on shared systems.",
        "Shared metrics and exception alerts govern how ecosystem partners respond to disruptions on our platform.",
        "Our multi-tier visibility platform provides shared KPIs to all ecosystem participants in real time.",
        "Algorithmic transparency in our marketplace ensures fair and auditable pricing across all platform partners.",
        # ★ v2.2
        "Real-time supplier monitoring through shared dashboards reduces information asymmetry and opportunistic behavior.",
        "Behavior monitoring and verification mechanisms govern partner compliance on our digital platform.",
        "Relationship transparency tools allow us to track supplier commitments against contractual obligations in real time.",
    ],
    "G2_Trust_Security_Assurance": [
        "The digital relationship depends on trusted data sharing and assurance mechanisms between ecosystem partners.",
        "Auditability and secure collaboration are emphasized as prerequisites for participation in our digital ecosystem.",
        "Digital records and verifiable audit trails are used to increase confidence between platform counterparties.",
        "Our digital trust framework governs data exchange with 85 ecosystem partners across 40 jurisdictions.",
        "Verifiable audit trails and assurance mechanisms enabled onboarding of regulated financial and supply chain partners.",
        "Trusted data sharing protocols govern how partners access and use data on our shared digital infrastructure.",
        # ★ v2.2
        "We implemented anti-opportunism mechanisms through digital audit trails and verifiable partner records.",
        "Power imbalance mitigation strategies are embedded in our platform governance to protect smaller ecosystem partners.",
        "Mutual dependence governance frameworks ensure fair data exchange between ecosystem participants of different sizes.",
    ],
    "G3_Data_Sovereignty_Ownership_Rights": [
        "Partners share data under explicit ownership, access-right and usage-control rules on our platform.",
        "The company emphasizes data sovereignty when building shared digital interfaces with ecosystem partners.",
        "Usage policies and rights management govern how ecosystem partners access and reuse shared data.",
        "Our data sovereignty platform ensures all partner data remains within contractually agreed jurisdictions.",
        "Consent-based data sharing agreements govern how each ecosystem participant can access shared datasets.",
        "Cross-border data governance rules protect partner data across 40 jurisdictions in our digital ecosystem.",
    ],
    "G4_Standards_Interoperability_Rules": [
        "Digital collaboration is governed by common data standards and interoperability rules across ecosystem partners.",
        "The ecosystem uses shared reference architectures and standard interfaces to coordinate partners digitally.",
        "A common data taxonomy and model reduce integration friction across participating firms on our platform.",
        "GAIA-X-compliant standards govern data interoperability across our 1,200-member digital ecosystem.",
        "Common ontologies and data model harmonization reduced partner integration costs by 60% on our platform.",
        "Standards are presented as a digital governance mechanism enabling ecosystem scale across all participants.",
    ],
    "G5_Orchestration_Roles_Incentives": [
        "The platform owner defines partner roles, incentives and participation rules for the digital ecosystem.",
        "Ecosystem orchestration is described through governance logic governing platform roles and complementor access.",
        "Revenue-sharing, API access rules and complementor policies shape digital ecosystem participation.",
        "Our platform governance framework defines fair revenue-sharing rules for all complementors on the system.",
        "Ecosystem role-setting and incentive design enabled rapid scaling to 4,500 digital partner participants.",
        "The company explicitly manages platform roles and access rules to coordinate ecosystem growth and partner incentives.",
        # ★ v2.2
        "Legacy relationship management protocols were digitalized to integrate long-standing partners into our ecosystem platform.",
        "Platform adoption resistance was overcome through incentive redesign and role clarification for channel partners.",
        "Ecosystem boundary reconfiguration enabled new complementor roles and redefined value-sharing rules across the network.",
    ],
    "G6_Joint_Learning_Knowledge_Sharing": [
        "The company uses shared digital spaces to exchange knowledge and build partner capabilities across the ecosystem.",
        "Joint learning routines and collaborative analytics support deeper digital collaboration across partner firms.",
        "Partner enablement programs are linked to knowledge sharing on shared digital platforms.",
        "Our shared knowledge platform connects 800 partners for collaborative analytics and cross-company problem solving.",
        "Digital partner enablement programs reduced time-to-capability for new ecosystem participants by 50%.",
        "Cross-company learning routines on our digital platform institutionalize knowledge exchange between partner firms.",
    ],
}


# ============================================================================
# CATEGORIES — D (Application)
# ============================================================================

_D_APPLICATIONS: Dict[str, CategoryInfo] = {
    "D1_Partner_Onboarding_Access": CategoryInfo(
        code="D1_Partner_Onboarding_Access",
        name="Partner Onboarding & Access",
        description="Digital onboarding, access provisioning and self-service interfaces for suppliers, distributors, dealers and ecosystem partners",
        dimension="D",
        keywords=[
            "partner onboarding portal","supplier onboarding portal","vendor registration portal",
            "dealer onboarding portal","partner self-service portal","digital onboarding workflow",
            "ecosystem partner onboarding","third-party onboarding platform","partner activation portal",
            "supplier self-registration platform",
            "partner portal","supplier registration workflow","channel partner portal",
            "partner access provisioning","vendor access management","distributor onboarding",
            "partner identity verification","onboarding automation","digital partner registration",
            "portal access","partner activation","vendor onboarding","supplier access",
            "partner registration","onboarding platform",
        ],
        patterns=[
            r"\bpartner\s+onboarding\s+(?:portal|platform|workflow)\b",
            r"\bsupplier\s+(?:onboarding|self-registration|registration)\s+(?:portal|platform|workflow)\b",
            r"\bvendor\s+registration\s+portal\b",
            r"\bdealer\s+onboarding\s+(?:portal|platform)\b",
            r"\bpartner\s+self-service\s+portal\b",
            r"\becosystem\s+partner\s+onboarding\b",
            r"\bthird-party\s+onboarding\s+platform\b",
        ],
        keyword_tiers={
            "partner onboarding portal":1,"supplier onboarding portal":1,"vendor registration portal":1,
            "dealer onboarding portal":1,"partner self-service portal":1,"digital onboarding workflow":1,
            "ecosystem partner onboarding":1,"third-party onboarding platform":1,
            "partner activation portal":1,"supplier self-registration platform":1,
            "partner portal":2,"supplier registration workflow":2,"channel partner portal":2,
            "partner access provisioning":2,"vendor access management":2,"distributor onboarding":2,
            "partner identity verification":2,"onboarding automation":2,"digital partner registration":2,
            "portal access":3,"partner activation":3,"vendor onboarding":3,"supplier access":3,
            "partner registration":3,"onboarding platform":3,
        },
    ),
    "D2_Transactional_Integration": CategoryInfo(
        code="D2_Transactional_Integration",
        name="Transactional Integration",
        description="Digital integration of inter-company transactions: ordering, invoicing, procurement, payments and fulfillment",
        dimension="D",
        keywords=[
            "EDI integration","API-based order integration","e-invoicing with suppliers",
            "procure-to-pay integration","order-to-cash integration","electronic data interchange",
            "B2B transaction integration","supplier invoice automation","digital purchase order exchange",
            "inter-company transaction platform",
            "digital ordering","invoice automation","procurement integration",
            "purchase order automation","digital invoicing with partners","transaction digitalization",
            "e-invoicing platform","order management integration","supply chain transaction automation",
            "transaction automation","electronic orders","invoice platform",
            "digital transactions","order integration","payment automation",
        ],
        patterns=[
            r"\bEDI\s+integration\b|\belectronic\s+data\s+interchange\b",
            r"\bAPI-based\s+order\s+integration\b",
            r"\be-?invoicing\b.*\b(?:supplier|partner|trading|B2B)\b",
            r"\bprocure-to-pay\s+integration\b",
            r"\border-to-cash\s+integration\b",
            r"\bB2B\s+transaction\s+integration\b",
            r"\binter-company\s+transaction\s+platform\b",
            r"\bsupplier\s+invoice\s+automation\b",
        ],
        keyword_tiers={
            "EDI integration":1,"API-based order integration":1,"e-invoicing with suppliers":1,
            "procure-to-pay integration":1,"order-to-cash integration":1,
            "electronic data interchange":1,"B2B transaction integration":1,
            "supplier invoice automation":1,"digital purchase order exchange":1,
            "inter-company transaction platform":1,
            "digital ordering":2,"invoice automation":2,"procurement integration":2,
            "purchase order automation":2,"digital invoicing with partners":2,
            "transaction digitalization":2,"e-invoicing platform":2,
            "order management integration":2,"supply chain transaction automation":2,
            "transaction automation":3,"electronic orders":3,"invoice platform":3,
            "digital transactions":3,"order integration":3,"payment automation":3,
        },
    ),
    "D3_Collaborative_Planning_Visibility": CategoryInfo(
        code="D3_Collaborative_Planning_Visibility",
        name="Collaborative Planning & Visibility",
        description="Shared planning, forecasting, replenishment and real-time visibility across suppliers, logistics providers and network partners",
        dimension="D",
        keywords=[
            "supply chain visibility platform","collaborative planning","shared demand planning",
            "joint forecasting portal","partner control tower","collaborative replenishment",
            "multi-tier supply chain visibility","shared S&OP platform","demand signal sharing platform",
            "real-time partner inventory visibility",
            "inventory visibility","shared planning","replenishment collaboration","network visibility",
            "visibility dashboard","demand signal sharing","planning platform",
            "collaborative forecasting","supply chain transparency","shared operational data",
            "visibility","shared planning","demand planning","supply chain planning",
            "forecasting platform","planning collaboration",
        ],
        patterns=[
            r"\bsupply\s+chain\s+visibility\s+platform\b",
            r"\bcollaborative\s+planning\b(?=.*(?:supplier|partner|network))",
            r"\bshared\s+demand\s+planning\b",
            r"\bjoint\s+forecasting\s+(?:portal|platform)\b",
            r"\bpartner\s+control\s+tower\b",
            r"\bmulti-tier\s+supply\s+chain\s+visibility\b",
            r"\bdemand\s+signal\s+sharing\s+platform\b",
            r"\breal-time\s+partner\s+inventory\s+visibility\b",
        ],
        keyword_tiers={
            "supply chain visibility platform":1,"collaborative planning":1,"shared demand planning":1,
            "joint forecasting portal":1,"partner control tower":1,"collaborative replenishment":1,
            "multi-tier supply chain visibility":1,"shared S&OP platform":1,
            "demand signal sharing platform":1,"real-time partner inventory visibility":1,
            "inventory visibility":2,"shared planning":2,"replenishment collaboration":2,
            "network visibility":2,"visibility dashboard":2,"demand signal sharing":2,
            "planning platform":2,"collaborative forecasting":2,
            "supply chain transparency":2,"shared operational data":2,
            "visibility":3,"demand planning":3,"supply chain planning":3,
            "forecasting platform":3,"planning collaboration":3,
        },
    ),
    "D4_Shared_Data_Interoperability": CategoryInfo(
        code="D4_Shared_Data_Interoperability",
        name="Shared Data & Interoperability",
        description="Cross-company data sharing, interoperability, common data models and structured data exchange between ecosystem participants",
        dimension="D",
        keywords=[
            "data sharing platform","shared data space","common data model","master data synchronization",
            "partner data exchange","GAIA-X","federated data platform","data mesh",
            "cross-company data sharing","interoperability platform",
            # ★ v2.2
            "Catena-X","industrial data ecosystem","data space connector","sovereign data exchange",
            "interoperability framework","data exchange layer","federated data sharing",
            "open data interfaces","data sovereignty","data portability","partner data platform",
            "data interoperability","shared datasets","open data standards",
            "data access rights","usage policy enforcement","federated data sharing agreement",
            "cross-company data governance platform",
            "data integration","interoperable systems","data exchange","data sharing",
            "shared data","data space","data ecosystem",
            "data access policy","industrial data sharing","data space participation",
        ],
        patterns=[
            r"\bdata\s+sharing\s+platform\b",
            r"\bshared\s+data\s+space\b",
            r"\bcommon\s+data\s+model\b",
            r"\bmaster\s+data\s+synchronization\b",
            r"\bGAIA-X\b",
            r"\bfederated\s+data\s+platform\b|\bdata\s+mesh\b",
            r"\bcross-company\s+data\s+sharing\b",
            r"\binteroperability\s+platform\b",
            r"\bdata\s+exchange\s+layer\b",
            # ★ v2.2
            r"\bCatena-X\b",
            r"\bindustrial\s+data\s+ecosystem\b",
            r"\bdata\s+space\s+connector\b",
            r"\bsovereign\s+data\s+exchange\b",
        ],
        keyword_tiers={
            "data sharing platform":1,"shared data space":1,"common data model":1,
            "master data synchronization":1,"partner data exchange":1,"GAIA-X":1,
            "federated data platform":1,"data mesh":1,"cross-company data sharing":1,
            "interoperability platform":1,
            # ★ v2.2
            "Catena-X":1,"industrial data ecosystem":1,"data space connector":1,
            "sovereign data exchange":1,
            "interoperability framework":2,"data exchange layer":2,"federated data sharing":2,
            "open data interfaces":2,"data sovereignty":2,"data portability":2,
            "partner data platform":2,"data interoperability":2,"shared datasets":2,
            "open data standards":2,
            "data access rights":2,"usage policy enforcement":2,
            "federated data sharing agreement":2,"cross-company data governance platform":2,
            "data integration":3,"interoperable systems":3,"data exchange":3,
            "data sharing":3,"shared data":3,"data space":3,"data ecosystem":3,
            "data access policy":3,"industrial data sharing":3,"data space participation":3,
        },
    ),
    "D5_Channel_Partner_Interface": CategoryInfo(
        code="D5_Channel_Partner_Interface",
        name="Channel & Partner Interface",
        description="Digital interfaces for dealers, distributors, resellers and channel partners: portals, co-selling, partner management",
        dimension="D",
        keywords=[
            "dealer portal","distributor portal","reseller platform","partner relationship management",
            "channel management platform","PRM platform","dealer management system",
            "distributor management platform","channel partner portal","dealer network platform",
            "channel portal","partner management portal","co-selling portal",
            "distributor management system","channel digitization","partner network portal",
            "dealer network","reseller portal","channel integration platform",
            "channel platform","partner portal","dealer system",
            "distributor interface","channel management","reseller interface",
        ],
        patterns=[
            r"\bdealer\s+portal\b|\bdealer\s+management\s+system\b",
            r"\bdistributor\s+(?:portal|management\s+platform|management\s+system)\b",
            r"\breseller\s+platform\b",
            r"\bpartner\s+relationship\s+management\b|\bPRM\s+platform\b",
            r"\bchannel\s+management\s+platform\b",
            r"\bchannel\s+partner\s+portal\b",
            r"\bdealer\s+network\s+platform\b",
            r"\bco-selling\s+portal\b",
        ],
        keyword_tiers={
            "dealer portal":1,"distributor portal":1,"reseller platform":1,
            "partner relationship management":1,"channel management platform":1,
            "PRM platform":1,"dealer management system":1,
            "distributor management platform":1,"channel partner portal":1,
            "dealer network platform":1,
            "channel portal":2,"partner management portal":2,"co-selling portal":2,
            "distributor management system":2,"channel digitization":2,
            "partner network portal":2,"dealer network":2,
            "reseller portal":2,"channel integration platform":2,
            "channel platform":3,"partner portal":3,"dealer system":3,
            "distributor interface":3,"channel management":3,"reseller interface":3,
        },
    ),
    "D6_CoInnovation_Digital_Servitization": CategoryInfo(
        code="D6_CoInnovation_Digital_Servitization",
        name="Co-Innovation & Digital Servitization",
        description="Digital service models, connected offerings and joint value creation with B2B/industrial partners (not consumer digital services)",
        dimension="D",
        keywords=[
            "digital servitization","B2B remote monitoring service","connected industrial service",
            "predictive maintenance platform","digital service contract with customers",
            "outcome-based B2B service","digital twin for supply chain",
            "industrial asset-as-a-service","product-as-a-service B2B",
            "co-innovation with industrial partners",
            "asset performance platform","customer operations data sharing",
            "joint solution development","connected asset service",
            "industrial service digitalization","servitization platform",
            "outcome-based contract","B2B digital service model",
            "co-innovation","connected product","service platform",
            "digital service","remote monitoring","predictive service",
        ],
        patterns=[
            r"\bdigital\s+serviti[sz]ation\b",
            r"\bB2B\s+remote\s+monitoring\s+service\b",
            r"\bconnected\s+industrial\s+service\b",
            r"\bpredictive\s+maintenance\s+platform\b",
            r"\boutcome-based\s+(?:B2B\s+service|contract)\b",
            r"\bco-innovation\s+with\s+(?:industrial|B2B)\s+partners?\b",
            r"\bindustrial\s+asset-as-a-service\b",
            r"\bproduct-as-a-service\s+B2B\b",
        ],
        keyword_tiers={
            "digital servitization":1,"B2B remote monitoring service":1,
            "connected industrial service":1,"predictive maintenance platform":1,
            "digital service contract with customers":1,"outcome-based B2B service":1,
            "digital twin for supply chain":1,"industrial asset-as-a-service":1,
            "product-as-a-service B2B":1,"co-innovation with industrial partners":1,
            "asset performance platform":2,"customer operations data sharing":2,
            "joint solution development":2,"connected asset service":2,
            "industrial service digitalization":2,"servitization platform":2,
            "outcome-based contract":2,"B2B digital service model":2,
            "co-innovation":3,"connected product":3,"service platform":3,
            "digital service":3,"remote monitoring":3,"predictive service":3,
        },
    ),
    "D7_Ecosystem_Orchestration_Complementors": CategoryInfo(
        code="D7_Ecosystem_Orchestration_Complementors",
        name="Ecosystem Orchestration & Complementors",
        description="Management of complementors, third-party developers and ecosystem participants through platform APIs and governance",
        dimension="D",
        keywords=[
            "ecosystem orchestration","developer ecosystem","partner ecosystem platform",
            "complementor onboarding","third-party integration ecosystem","API marketplace",
            "platform ecosystem","B2B ecosystem","multi-sided platform","ecosystem play",
            # ★ v2.2
            "incumbent platform ecosystem","B2B platform orchestration","multi-platform ecosystem",
            "hybrid service platform","product ecosystem platform",
            "partner enablement","ecosystem APIs","marketplace ecosystem",
            "partner network platform","digital ecosystem","ecosystem participants",
            "third-party integrations","open innovation ecosystem","complementors",
            "ecosystem value creation",
            "legacy relationship digitalization","platform inertia management",
            "relational challenge platform","complementor role allocation",
            "platform boundary management",
            "platform","ecosystem","partner network","API economy",
            "open innovation","network effects",
            "platform inertia","legacy partner","platform boundary",
        ],
        patterns=[
            r"\becosystem\s+orchestration\b",
            r"\bdeveloper\s+ecosystem\b",
            r"\bpartner\s+ecosystem\s+platform\b",
            r"\bcomplementor\s+onboarding\b",
            r"\bAPI\s+marketplace\b",
            r"\bplatform\s+ecosystem\b",
            r"\bmulti-sided\s+platform\b",
            r"\bB2B\s+ecosystem\b",
            r"\bthird-party\s+integration\s+ecosystem\b",
            # ★ v2.2
            r"\bincumbent\s+platform\s+ecosystem\b",
            r"\bmulti-platform\s+ecosystem\b",
            r"\bhybrid\s+service\s+platform\b",
            r"\blegacy\s+relationship\s+digitali[sz]ation\b",
            r"\bplatform\s+inertia\s+management\b",
        ],
        keyword_tiers={
            "ecosystem orchestration":1,"developer ecosystem":1,"partner ecosystem platform":1,
            "complementor onboarding":1,"third-party integration ecosystem":1,
            "API marketplace":1,"platform ecosystem":1,"B2B ecosystem":1,
            "multi-sided platform":1,"ecosystem play":1,
            # ★ v2.2
            "incumbent platform ecosystem":1,"B2B platform orchestration":1,
            "multi-platform ecosystem":1,"hybrid service platform":1,
            "product ecosystem platform":1,
            "partner enablement":2,"ecosystem APIs":2,"marketplace ecosystem":2,
            "partner network platform":2,"digital ecosystem":2,"ecosystem participants":2,
            "third-party integrations":2,"open innovation ecosystem":2,
            "complementors":2,"ecosystem value creation":2,
            "legacy relationship digitalization":2,"platform inertia management":2,
            "relational challenge platform":2,"complementor role allocation":2,
            "platform boundary management":2,
            "platform":3,"ecosystem":3,"partner network":3,
            "API economy":3,"open innovation":3,"network effects":3,
            "platform inertia":3,"legacy partner":3,"platform boundary":3,
        },
    ),
    "D8_Traceability_Compliance_Network": CategoryInfo(
        code="D8_Traceability_Compliance_Network",
        name="Traceability & Compliance Network",
        description="End-to-end traceability, chain-of-custody, digital product passports and compliance data exchange across partners",
        dimension="D",
        keywords=[
            "end-to-end traceability","digital product passport","supplier compliance portal",
            "chain of custody platform","provenance tracking","blockchain traceability",
            "multi-tier traceability","ESG supply chain traceability","blockchain ESG traceability",
            "supply chain compliance platform",
            # ★ v2.2
            "battery passport","product carbon footprint exchange","passport data carrier",
            "decentralized passport","tamper-proof traceability","passport interoperability",
            "traceability platform","compliance data exchange","batch traceability",
            "audit trail across partners","digital traceability","provenance",
            "sustainability traceability","regulatory traceability","CSRD traceability",
            "product lifecycle data sharing","stakeholder data contribution",
            "circular economy traceability","DPP ecosystem",
            "carbon footprint sharing supply chain",
            "traceability data","compliance platform","provenance data",
            "audit trail","traceability","supply chain transparency",
            "product passport","lifecycle traceability","carbon passport","circular traceability",
        ],
        patterns=[
            r"\bend-to-end\s+traceability\b",
            r"\bdigital\s+product\s+passport\b",
            r"\bsupplier\s+compliance\s+portal\b",
            r"\bchain\s+of\s+custody\s+platform\b",
            r"\bprovenance\s+tracking\b",
            r"\bblockchain\s+(?:traceability|ESG\s+traceability)\b",
            r"\bmulti-tier\s+traceability\b",
            r"\bESG\s+supply\s+chain\s+traceability\b",
            r"\bsupply\s+chain\s+compliance\s+platform\b",
            # ★ v2.2
            r"\bbattery\s+passport\b",
            r"\bproduct\s+carbon\s+footprint\s+exchange\b",
            r"\bpassport\s+data\s+carrier\b",
            r"\bDPP\s+ecosystem\b",
            r"\bdecentralized\s+(?:passport|product\s+passport)\b",
            r"\btamper-proof\s+traceability\b",
        ],
        keyword_tiers={
            "end-to-end traceability":1,"digital product passport":1,
            "supplier compliance portal":1,"chain of custody platform":1,
            "provenance tracking":1,"blockchain traceability":1,
            "multi-tier traceability":1,"ESG supply chain traceability":1,
            "blockchain ESG traceability":1,"supply chain compliance platform":1,
            # ★ v2.2
            "battery passport":1,"product carbon footprint exchange":1,
            "passport data carrier":1,"decentralized passport":1,
            "tamper-proof traceability":1,"passport interoperability":1,
            "traceability platform":2,"compliance data exchange":2,"batch traceability":2,
            "audit trail across partners":2,"digital traceability":2,"provenance":2,
            "sustainability traceability":2,"regulatory traceability":2,"CSRD traceability":2,
            "product lifecycle data sharing":2,"stakeholder data contribution":2,
            "circular economy traceability":2,"DPP ecosystem":2,
            "carbon footprint sharing supply chain":2,
            "traceability data":3,"compliance platform":3,"provenance data":3,
            "audit trail":3,"traceability":3,"supply chain transparency":3,
            "product passport":3,"lifecycle traceability":3,
            "carbon passport":3,"circular traceability":3,
        },
    ),
}


# ============================================================================
# CATEGORIES — T (Technology)
# NOTE: All T categories require Co-occurrence Gate in detect.py
#       T1 and T4 additionally require Relational Verb Gate
# ============================================================================

_T_TECHNOLOGIES: Dict[str, CategoryInfo] = {
    "T1_ERP_SCM_Backbone": CategoryInfo(
        code="T1_ERP_SCM_Backbone",
        name="ERP / SCM Backbone",
        description="ERP/SCM backbone extended beyond enterprise to support partner transactions [Co-occur + relational verb required]",
        dimension="T",
        keywords=[
            "ERP-to-supplier integration","supplier relationship management","SRM platform",
            "procurement ERP integration","SAP Ariba","integrated ERP partner network",
            "SAP S/4HANA supplier integration","Oracle SCM partner connect",
            "ERP supply chain integration","SCM suite with supplier connectivity",
            "SAP S/4HANA","Oracle ERP","Dynamics 365","Workday","ERP system",
            "order management system","digital core","ERP backbone","procurement system",
            "ERP","SCM platform","core system","SAP","Oracle","enterprise system",
        ],
        patterns=[
            r"\bERP-to-supplier\s+integration\b",
            r"\bsupplier\s+relationship\s+management\b|\bSRM\s+platform\b",
            r"\bprocurement\s+ERP\s+integration\b",
            r"\bSAP\s+Ariba\b(?=.*(?:supplier|partner|procurement|network))",
            r"\bintegrated\s+ERP\s+partner\s+network\b",
            r"\bSAP\s+S/4HANA\b.*\b(?:supplier|partner|integration|connect)\b",
            r"\bSCM\s+suite\b.*\b(?:supplier|partner|connect)\b",
            r"\bERP\s+supply\s+chain\s+integration\b",
        ],
        keyword_tiers={
            "ERP-to-supplier integration":1,"supplier relationship management":1,
            "SRM platform":1,"procurement ERP integration":1,"SAP Ariba":1,
            "integrated ERP partner network":1,"SAP S/4HANA supplier integration":1,
            "Oracle SCM partner connect":1,"ERP supply chain integration":1,
            "SCM suite with supplier connectivity":1,
            "SAP S/4HANA":2,"Oracle ERP":2,"Dynamics 365":2,"Workday":2,
            "ERP system":2,"order management system":2,"digital core":2,
            "ERP backbone":2,"procurement system":2,
            "ERP":3,"SCM platform":3,"core system":3,"SAP":3,
            "Oracle":3,"enterprise system":3,
        },
    ),
    "T2_API_EDI_iPaaS_Middleware": CategoryInfo(
        code="T2_API_EDI_iPaaS_Middleware",
        name="API / EDI / iPaaS / Middleware",
        description="Integration technologies connecting firms in real time [Co-occurrence Gate]",
        dimension="T",
        keywords=[
            "API integration","electronic data interchange","integration middleware","iPaaS",
            "B2B gateway","MuleSoft","Boomi","Azure Integration Services","AWS EventBridge",
            "B2B integration platform",
            "API gateway","enterprise service bus","integration hub","message broker",
            "REST API","microservices integration","event-driven architecture",
            "API management platform","integration platform as a service",
            "APIs","middleware","connectors","ESB","web services","integration",
        ],
        patterns=[
            r"\bAPI\s+integration\b|\bAPI-first\s+(?:architecture|platform)\b",
            r"\belectronic\s+data\s+interchange\b",
            r"\biPaaS\b|\bintegration\s+platform\s+as\s+a\s+service\b",
            r"\bB2B\s+(?:gateway|integration\s+platform)\b",
            r"\bMuleSoft\b|\bBoomi\b|\bAWS\s+EventBridge\b",
            r"\bAzure\s+Integration\s+Services\b",
            r"\bB2B\s+integration\s+layer\b",
        ],
        keyword_tiers={
            "API integration":1,"electronic data interchange":1,"integration middleware":1,
            "iPaaS":1,"B2B gateway":1,"MuleSoft":1,"Boomi":1,
            "Azure Integration Services":1,"AWS EventBridge":1,"B2B integration platform":1,
            "API gateway":2,"enterprise service bus":2,"integration hub":2,
            "message broker":2,"REST API":2,"microservices integration":2,
            "event-driven architecture":2,"API management platform":2,
            "integration platform as a service":2,
            "APIs":3,"middleware":3,"connectors":3,"ESB":3,
            "web services":3,"integration":3,
        },
    ),
    "T3_B2B_Platforms_Marketplaces": CategoryInfo(
        code="T3_B2B_Platforms_Marketplaces",
        name="B2B Platforms & Marketplaces",
        description="Digital B2B platforms, trading hubs and procurement marketplaces [Co-occurrence Gate]",
        dimension="T",
        keywords=[
            "SAP Ariba","Coupa","Jaggaer","Tradeshift","Basware","Ivalua","Zycus",
            "B2B marketplace","procurement marketplace","supplier network platform",
            "partner platform","digital trade platform","marketplace platform",
            "ecosystem platform","partner marketplace","network platform",
            "procure-to-pay platform","P2P platform","e-procurement platform",
            "platform","marketplace","portal","B2B platform","trade platform",
        ],
        patterns=[
            r"\bSAP\s+Ariba\b|\bCoupa\b|\bJaggaer\b|\bTradeshift\b|\bBasware\b|\bIvalua\b|\bZycus\b",
            r"\bB2B\s+(?:marketplace|platform)\b",
            r"\bprocurement\s+marketplace\b",
            r"\bsupplier\s+network\s+platform\b",
            r"\bdigital\s+trade\s+platform\b",
            r"\bprocure-to-pay\s+platform\b|\bP2P\s+platform\b",
        ],
        keyword_tiers={
            "SAP Ariba":1,"Coupa":1,"Jaggaer":1,"Tradeshift":1,"Basware":1,
            "Ivalua":1,"Zycus":1,"B2B marketplace":1,"procurement marketplace":1,
            "supplier network platform":1,
            "partner platform":2,"digital trade platform":2,"marketplace platform":2,
            "ecosystem platform":2,"partner marketplace":2,"network platform":2,
            "procure-to-pay platform":2,"P2P platform":2,"e-procurement platform":2,
            "platform":3,"marketplace":3,"portal":3,"B2B platform":3,"trade platform":3,
        },
    ),
    "T4_Cloud_Data_Platforms": CategoryInfo(
        code="T4_Cloud_Data_Platforms",
        name="Cloud & Data Platforms",
        description="Shared cloud and data platforms enabling cross-company collaboration [Co-occur + relational verb required]",
        dimension="T",
        keywords=[
            "shared cloud data platform","industry cloud for partners","data exchange cloud with partners",
            "collaborative cloud workspace","cloud platform connecting suppliers",
            "cloud-based supply chain platform","partner data lake",
            "shared analytics platform with partners","multi-cloud partner integration",
            "cloud ecosystem platform",
            "cloud platform","data platform","shared analytics platform",
            "common cloud environment","hybrid cloud partner integration",
            "SaaS partner platform","cloud-native ecosystem","cloud data sharing",
            "cloud","data lake","cloud infrastructure","Azure","AWS","Google Cloud",
        ],
        patterns=[
            r"\bshared\s+cloud\s+(?:data\s+)?platform\b",
            r"\bindustry\s+cloud\b.*\b(?:partner|supplier|ecosystem)\b",
            r"\bdata\s+exchange\s+cloud\b.*\b(?:partner|supplier)\b",
            r"\bcollaborative\s+cloud\s+workspace\b",
            r"\bcloud\s+platform\b.*\b(?:connect|supplier|partner|ecosystem)\b",
            r"\bcloud-based\s+supply\s+chain\s+platform\b",
            r"\bpartner\s+data\s+lake\b",
            r"\bshared\s+analytics\s+platform\b.*\b(?:partner|supplier)\b",
        ],
        keyword_tiers={
            "shared cloud data platform":1,"industry cloud for partners":1,
            "data exchange cloud with partners":1,"collaborative cloud workspace":1,
            "cloud platform connecting suppliers":1,"cloud-based supply chain platform":1,
            "partner data lake":1,"shared analytics platform with partners":1,
            "multi-cloud partner integration":1,"cloud ecosystem platform":1,
            "cloud platform":2,"data platform":2,"shared analytics platform":2,
            "common cloud environment":2,"hybrid cloud partner integration":2,
            "SaaS partner platform":2,"cloud-native ecosystem":2,"cloud data sharing":2,
            "cloud":3,"data lake":3,"cloud infrastructure":3,
            "Azure":3,"AWS":3,"Google Cloud":3,
        },
    ),
    "T5_IoT_Digital_Twins_Connected_Assets": CategoryInfo(
        code="T5_IoT_Digital_Twins_Connected_Assets",
        name="IoT / Digital Twins / Connected Assets",
        description="Connected assets and IoT platforms enabling shared visibility across firm boundaries [Co-occurrence Gate]",
        dimension="T",
        keywords=[
            "industrial IoT platform","connected asset monitoring","digital twin for supply chain",
            "remote asset diagnostics","telematics with partners","IIoT platform",
            "OT-IT convergence","smart factory IoT","SCADA integration","connected factory",
            "IoT sensors","asset connectivity","condition monitoring","connected operations",
            "Internet of Things","digital twin","remote monitoring",
            "predictive maintenance IoT","industrial sensors",
            "IoT","connected devices","sensors","monitoring","smart devices",
        ],
        patterns=[
            r"\bindustrial\s+IoT\s+platform\b|\bIIoT\s+platform\b",
            r"\bconnected\s+asset\s+monitoring\b",
            r"\bdigital\s+twin\b.*\b(?:supply chain|partner|supplier|customer)\b",
            r"\bremote\s+asset\s+diagnostics\b",
            r"\btelematics\b.*\b(?:partner|supplier|customer)\b",
            r"\bOT-IT\s+convergence\b",
            r"\bSCADA\s+integration\b",
        ],
        keyword_tiers={
            "industrial IoT platform":1,"connected asset monitoring":1,
            "digital twin for supply chain":1,"remote asset diagnostics":1,
            "telematics with partners":1,"IIoT platform":1,"OT-IT convergence":1,
            "smart factory IoT":1,"SCADA integration":1,"connected factory":1,
            "IoT sensors":2,"asset connectivity":2,"condition monitoring":2,
            "connected operations":2,"Internet of Things":2,"digital twin":2,
            "remote monitoring":2,"predictive maintenance IoT":2,"industrial sensors":2,
            "IoT":3,"connected devices":3,"sensors":3,"monitoring":3,"smart devices":3,
        },
    ),
    "T6_Identity_Access_Security": CategoryInfo(
        code="T6_Identity_Access_Security",
        name="Identity / Access / Security",
        description="Federated identity, zero-trust and secure data exchange enabling trusted ecosystem participation [Co-occurrence Gate]",
        dimension="T",
        keywords=[
            "federated identity","partner identity management","zero trust partner access",
            "secure data exchange","PKI for suppliers","IAM platform","SSO platform",
            "zero trust architecture","identity and access management","digital identity platform",
            "single sign-on for partners","identity federation","access control",
            "secure onboarding","zero trust","multi-factor authentication",
            "cybersecurity platform","privileged access management","endpoint security platform",
            "IAM","secure access","cybersecurity platform","SSO","identity","authentication",
        ],
        patterns=[
            r"\bfederated\s+identity\b",
            r"\bpartner\s+identity\s+management\b",
            r"\bzero\s+trust\b.*\b(?:partner|supplier|ecosystem|access)\b",
            r"\bsecure\s+data\s+exchange\b",
            r"\bPKI\s+for\s+suppliers\b",
            r"\bIAM\s+platform\b|\bidentity\s+and\s+access\s+management\b",
            r"\bSSO\s+platform\b|\bsingle\s+sign-on\s+for\s+partners\b",
            r"\bidentity\s+federation\b",
        ],
        keyword_tiers={
            "federated identity":1,"partner identity management":1,
            "zero trust partner access":1,"secure data exchange":1,
            "PKI for suppliers":1,"IAM platform":1,"SSO platform":1,
            "zero trust architecture":1,"identity and access management":1,
            "digital identity platform":1,
            "single sign-on for partners":2,"identity federation":2,"access control":2,
            "secure onboarding":2,"zero trust":2,"multi-factor authentication":2,
            "cybersecurity platform":2,"privileged access management":2,
            "endpoint security platform":2,
            "IAM":3,"secure access":3,"SSO":3,"identity":3,"authentication":3,
        },
    ),
    "T7_Blockchain_DLT_Traceability": CategoryInfo(
        code="T7_Blockchain_DLT_Traceability",
        name="Blockchain / DLT / Traceability",
        description="Blockchain and DLT for multi-party provenance, trade documentation and smart contracts [Co-occurrence Gate]",
        dimension="T",
        keywords=[
            "blockchain traceability","distributed ledger trade platform",
            "smart contracts with suppliers","provenance blockchain",
            "tokenized trade documentation","blockchain ESG","supply chain blockchain",
            "DLT platform","permissioned blockchain","carbon credits blockchain",
            "DLT network","ledger-based audit trail","digital trade ledger",
            "blockchain consortium","smart contracts","distributed ledger",
            "blockchain integration","blockchain network",
            "blockchain","distributed ledger","smart contract","ledger","tokens",
        ],
        patterns=[
            r"\bblockchain\s+traceability\b",
            r"\bdistributed\s+ledger\s+trade\s+platform\b",
            r"\bsmart\s+contracts\b.*\b(?:supplier|partner|logistics)\b",
            r"\bprovenance\s+blockchain\b",
            r"\btokenized\s+trade\s+documentation\b",
            r"\bblockchain\s+ESG\b",
            r"\bsupply\s+chain\s+blockchain\b",
            r"\bDLT\s+platform\b|\bpermissioned\s+blockchain\b",
        ],
        keyword_tiers={
            "blockchain traceability":1,"distributed ledger trade platform":1,
            "smart contracts with suppliers":1,"provenance blockchain":1,
            "tokenized trade documentation":1,"blockchain ESG":1,
            "supply chain blockchain":1,"DLT platform":1,
            "permissioned blockchain":1,"carbon credits blockchain":1,
            "DLT network":2,"ledger-based audit trail":2,"digital trade ledger":2,
            "blockchain consortium":2,"smart contracts":2,"distributed ledger":2,
            "blockchain integration":2,"blockchain network":2,
            "blockchain":3,"smart contract":3,"ledger":3,"tokens":3,
        },
    ),
    "T8_Workflow_Automation_LowCode": CategoryInfo(
        code="T8_Workflow_Automation_LowCode",
        name="Workflow Automation / Low-Code",
        description="Workflow orchestration and low-code tools automating cross-company partner processes [Co-occurrence Gate]",
        dimension="T",
        keywords=[
            "low-code workflow orchestration","partner workflow automation",
            "supplier case management automation","no-code partner portal",
            "automated approval flows","RPA partner process","hyperautomation",
            "Power Platform","robotic process automation","intelligent automation partner",
            "workflow automation","business process automation","digital forms",
            "collaborative workflow","low-code","no-code","RPA","Power Automate",
            "citizen developer",
            "automation platform","workflow app","automation","bots",
        ],
        patterns=[
            r"\blow-code\s+workflow\s+orchestration\b",
            r"\bpartner\s+workflow\s+automation\b",
            r"\bsupplier\s+case\s+management\s+automation\b",
            r"\bno-code\s+partner\s+portal\b",
            r"\bautomated\s+approval\s+flows\b",
            r"\bRPA\s+partner\s+process\b",
            r"\bPower\s+Platform\b(?=.*(?:partner|supplier|ecosystem))",
        ],
        keyword_tiers={
            "low-code workflow orchestration":1,"partner workflow automation":1,
            "supplier case management automation":1,"no-code partner portal":1,
            "automated approval flows":1,"RPA partner process":1,"hyperautomation":1,
            "Power Platform":1,"robotic process automation":1,
            "intelligent automation partner":1,
            "workflow automation":2,"business process automation":2,"digital forms":2,
            "collaborative workflow":2,"low-code":2,"no-code":2,"RPA":2,
            "Power Automate":2,"citizen developer":2,
            "automation platform":3,"workflow app":3,"automation":3,"bots":3,
        },
    ),
}


# ============================================================================
# CATEGORIES — G (Governance)
# NOTE: All G categories require Soft Gate in detect.py
#       (digital marker in ±400 chars window OR D/T hit in document)
# ============================================================================

_G_GOVERNANCE: Dict[str, CategoryInfo] = {
    "G1_Transparency_Visibility_Control": CategoryInfo(
        code="G1_Transparency_Visibility_Control",
        name="Transparency, Visibility & Control",
        description="Shared metrics, dashboards and control towers as governance mechanisms across partner networks [Soft Gate: digital marker required]",
        dimension="G",
        keywords=[
            "real-time partner visibility","shared KPI dashboard","multi-tier visibility",
            "partner performance monitoring","control tower governance","shared performance metrics",
            "ecosystem transparency","algorithmic transparency","platform monitoring",
            "partner performance dashboard",
            # ★ v2.2
            "relationship transparency","supplier transparency","behavior monitoring",
            "verification mechanism","real-time supply chain monitoring",
            "shared metrics","exception management","monitoring","dashboard transparency",
            "partner scorecard","network visibility","supply chain transparency",
            "real-time monitoring","performance visibility",
            "opportunism monitoring","partner behavior tracking",
            "audit-based governance","information sharing transparency",
            "visibility","transparency","monitoring","dashboard","KPI sharing",
            "supplier monitoring","partner verification","behavior control",
        ],
        patterns=[
            r"\breal-time\s+partner\s+visibility\b",
            r"\bshared\s+KPI\s+dashboard\b",
            r"\bcontrol\s+tower\s+governance\b",
            r"\bpartner\s+performance\s+(?:monitoring|dashboard)\b",
            r"\becosystem\s+transparency\b",
            r"\balgorithmic\s+transparency\b",
            r"\bmulti-tier\s+visibility\b",
            # ★ v2.2
            r"\brelationship\s+transparency\b",
            r"\bbehavior\s+monitoring\b(?=.*(?:partner|supplier|digital|platform|ecosystem))",
            r"\bverification\s+mechanism\b(?=.*(?:partner|supplier|digital|platform))",
            r"\bopportunism\s+monitoring\b",
        ],
        keyword_tiers={
            "real-time partner visibility":1,"shared KPI dashboard":1,
            "multi-tier visibility":1,"partner performance monitoring":1,
            "control tower governance":1,"shared performance metrics":1,
            "ecosystem transparency":1,"algorithmic transparency":1,
            "platform monitoring":1,"partner performance dashboard":1,
            # ★ v2.2
            "relationship transparency":1,"supplier transparency":1,
            "behavior monitoring":1,"verification mechanism":1,
            "real-time supply chain monitoring":1,
            "shared metrics":2,"exception management":2,"monitoring":2,
            "dashboard transparency":2,"partner scorecard":2,"network visibility":2,
            "supply chain transparency":2,"real-time monitoring":2,"performance visibility":2,
            "opportunism monitoring":2,"partner behavior tracking":2,
            "audit-based governance":2,"information sharing transparency":2,
            "visibility":3,"transparency":3,"dashboard":3,"KPI sharing":3,
            "supplier monitoring":3,"partner verification":3,"behavior control":3,
        },
    ),
    "G2_Trust_Security_Assurance": CategoryInfo(
        code="G2_Trust_Security_Assurance",
        name="Trust, Security & Assurance",
        description="Auditability, verifiable records and assurance mechanisms enabling trusted ecosystem participation [Soft Gate: digital marker required]",
        dimension="G",
        keywords=[
            "trusted data sharing","partner assurance","secure collaboration","auditability",
            "verifiable records","digital trust framework","ecosystem trust mechanism",
            "trust-based data exchange","platform assurance","zero-trust ecosystem",
            # ★ v2.2
            "information leakage prevention","anti-opportunism mechanism",
            "mutual dependence governance","power imbalance mitigation",
            "trust framework","assurance mechanism","cybersecurity assurance",
            "secure collaboration model","partner trust","digital trust","assured data exchange",
            "capability asymmetry","partner opportunism","power asymmetry platform",
            "dependency governance",
            "trust","assurance","auditability","secure exchange","trusted platform",
            "opportunism","power imbalance","mutual dependence",
        ],
        patterns=[
            r"\btrusted\s+data\s+sharing\b",
            r"\bpartner\s+assurance\b",
            r"\bdigital\s+trust\s+framework\b",
            r"\becosystem\s+trust\s+mechanism\b",
            r"\btrust-based\s+data\s+exchange\b",
            r"\bplatform\s+assurance\b",
            r"\bzero-trust\s+ecosystem\b",
            r"\bverifiable\s+records\b(?=.*(?:partner|digital|platform|ecosystem))",
            # ★ v2.2
            r"\banti-opportunism\s+mechanism\b",
            r"\bmutual\s+dependence\s+governance\b",
            r"\bpower\s+imbalance\s+mitigation\b",
            r"\binformation\s+leakage\s+prevention\b",
        ],
        keyword_tiers={
            "trusted data sharing":1,"partner assurance":1,"secure collaboration":1,
            "auditability":1,"verifiable records":1,"digital trust framework":1,
            "ecosystem trust mechanism":1,"trust-based data exchange":1,
            "platform assurance":1,"zero-trust ecosystem":1,
            # ★ v2.2
            "information leakage prevention":1,"anti-opportunism mechanism":1,
            "mutual dependence governance":1,"power imbalance mitigation":1,
            "trust framework":2,"assurance mechanism":2,"cybersecurity assurance":2,
            "secure collaboration model":2,"partner trust":2,"digital trust":2,
            "assured data exchange":2,
            "capability asymmetry":2,"partner opportunism":2,
            "power asymmetry platform":2,"dependency governance":2,
            "trust":3,"assurance":3,"auditability":3,"secure exchange":3,
            "trusted platform":3,
            "opportunism":3,"power imbalance":3,"mutual dependence":3,
        },
    ),
    "G3_Data_Sovereignty_Ownership_Rights": CategoryInfo(
        code="G3_Data_Sovereignty_Ownership_Rights",
        name="Data Sovereignty, Ownership & Rights",
        description="Data ownership agreements, usage control and consent-based data sharing between firms [Soft Gate: digital marker required]",
        dimension="G",
        keywords=[
            "data sovereignty","usage control","access rights management",
            "data ownership agreement","consent-based data sharing","data sovereignty platform",
            "cross-border data governance","GDPR data portability","sovereign cloud",
            "privacy by design platform",
            # ★ v2.2
            "sovereign data exchange","data space sovereignty","industrial data sovereignty",
            "Catena-X data governance",
            "data rights","policy-based access","data governance","usage policies",
            "data ownership","access rights","data localization","data portability",
            "data residency","trusted data exchange",
            "data access rights enforcement","usage policy","data sharing agreement",
            "federated data rights",
            "data control","sovereignty",
            "data access control","sovereign exchange","usage rights",
        ],
        patterns=[
            r"\bdata\s+sovereignty\b",
            r"\busage\s+control\b(?=.*(?:data|partner|platform|ecosystem))",
            r"\baccess\s+rights\s+management\b",
            r"\bdata\s+ownership\s+agreement\b",
            r"\bconsent-based\s+data\s+sharing\b",
            r"\bdata\s+sovereignty\s+platform\b",
            r"\bcross-border\s+data\s+governance\b",
            r"\bsovereign\s+cloud\b",
            r"\bprivacy\s+by\s+design\s+platform\b",
            # ★ v2.2
            r"\bdata\s+space\s+sovereignty\b",
            r"\bindustrial\s+data\s+sovereignty\b",
            r"\bCatena-X\s+data\s+governance\b",
            r"\bsovereign\s+data\s+exchange\b(?=.*(?:governance|agreement|policy|rights))",
        ],
        keyword_tiers={
            "data sovereignty":1,"usage control":1,"access rights management":1,
            "data ownership agreement":1,"consent-based data sharing":1,
            "data sovereignty platform":1,"cross-border data governance":1,
            "GDPR data portability":1,"sovereign cloud":1,"privacy by design platform":1,
            # ★ v2.2
            "sovereign data exchange":1,"data space sovereignty":1,
            "industrial data sovereignty":1,"Catena-X data governance":1,
            "data rights":2,"policy-based access":2,"data governance":2,
            "usage policies":2,"data ownership":2,"access rights":2,
            "data localization":2,"data portability":2,"data residency":2,
            "trusted data exchange":2,
            "data access rights enforcement":2,"usage policy":2,
            "data sharing agreement":2,"federated data rights":2,
            "data control":3,"sovereignty":3,
            "data access control":3,"sovereign exchange":3,"usage rights":3,
        },
    ),
    "G4_Standards_Interoperability_Rules": CategoryInfo(
        code="G4_Standards_Interoperability_Rules",
        name="Standards, Interoperability & Rules",
        description="Common data standards, shared ontologies and interoperability rules governing ecosystem coordination [Soft Gate: digital marker required]",
        dimension="G",
        keywords=[
            "common data standards","interoperability framework","shared ontology",
            "industry standard interfaces","data model harmonization","GAIA-X standards",
            "open data standards","canonical data model","ecosystem standard",
            "interoperability protocol",
            # ★ v2.2
            "Catena-X standards","data space interoperability","industrial data standard",
            "sovereign interoperability framework",
            "standards compliance","common taxonomy","reference architecture",
            "interoperability rules","standard interfaces",
            "ecosystem protocol","data standard","open standard",
            "data ecosystem standard","federated interoperability",
            "usage policy standard","data space protocol",
            "standards","interoperability","common model","protocol",
            "industrial standard","data space","federated standard",
        ],
        patterns=[
            r"\bcommon\s+data\s+standards\b",
            r"\binteroperability\s+framework\b(?=.*(?:ecosystem|partner|digital|platform))",
            r"\bshared\s+ontology\b",
            r"\bdata\s+model\s+harmonization\b",
            r"\bGAIA-X\s+standards\b",
            r"\bcanonical\s+data\s+model\b",
            r"\becosystem\s+standard\b",
            r"\binteroperability\s+protocol\b",
            # ★ v2.2
            r"\bCatena-X\s+standards\b",
            r"\bdata\s+space\s+interoperability\b",
            r"\bindustrial\s+data\s+standard\b",
            r"\bsovereign\s+interoperability\s+framework\b",
        ],
        keyword_tiers={
            "common data standards":1,"interoperability framework":1,"shared ontology":1,
            "industry standard interfaces":1,"data model harmonization":1,
            "GAIA-X standards":1,"open data standards":1,"canonical data model":1,
            "ecosystem standard":1,"interoperability protocol":1,
            # ★ v2.2
            "Catena-X standards":1,"data space interoperability":1,
            "industrial data standard":1,"sovereign interoperability framework":1,
            "standards compliance":2,"common taxonomy":2,"reference architecture":2,
            "interoperability rules":2,"standard interfaces":2,
            "ecosystem protocol":2,"data standard":2,"open standard":2,
            "data ecosystem standard":2,"federated interoperability":2,
            "usage policy standard":2,"data space protocol":2,
            "standards":3,"interoperability":3,"common model":3,"protocol":3,
            "industrial standard":3,"data space":3,"federated standard":3,
        },
    ),
    "G5_Orchestration_Roles_Incentives": CategoryInfo(
        code="G5_Orchestration_Roles_Incentives",
        name="Orchestration, Roles & Incentives",
        description="Role-setting, participation rules, complementor incentives and orchestration logic for ecosystem governance [Soft Gate: digital marker required]",
        dimension="G",
        keywords=[
            "ecosystem governance model","partner roles and rules","complementor incentives",
            "revenue-sharing rules","orchestration logic","platform governance framework",
            "ecosystem governance","fair value distribution","platform rules",
            "complementor governance",
            # ★ v2.2
            "legacy relationship management","platform adoption incentives",
            "boundary reconfiguration","role allocation ecosystem",
            "ecosystem participation rules",
            "partner governance","participation model","ecosystem policy",
            "platform fairness","platform stewardship","ecosystem compliance",
            "platform accountability","partner participation rules",
            "partner resistance management","platform adoption resistance",
            "complementor role definition","ecosystem boundary dynamics",
            "legacy partner integration",
            "governance","roles","incentives","platform policy","ecosystem rules",
            "role allocation","platform resistance","boundary dynamics",
        ],
        patterns=[
            r"\becosystem\s+governance\s+(?:model|framework)\b",
            r"\bpartner\s+roles\s+and\s+rules\b",
            r"\bcomplementor\s+incentives\b",
            r"\brevenue-sharing\s+rules\b",
            r"\bplatform\s+governance\s+framework\b",
            r"\bfair\s+value\s+distribution\b",
            r"\bcomplementor\s+governance\b",
            r"\borchestration\s+logic\b",
            # ★ v2.2
            r"\blegacy\s+relationship\s+management\b(?=.*(?:digital|platform|ecosystem|partner))",
            r"\bplatform\s+adoption\s+(?:resistance|incentives)\b",
            r"\bboundary\s+reconfiguration\b",
            r"\brole\s+allocation\s+ecosystem\b",
        ],
        keyword_tiers={
            "ecosystem governance model":1,"partner roles and rules":1,
            "complementor incentives":1,"revenue-sharing rules":1,
            "orchestration logic":1,"platform governance framework":1,
            "ecosystem governance":1,"fair value distribution":1,
            "platform rules":1,"complementor governance":1,
            # ★ v2.2
            "legacy relationship management":1,"platform adoption incentives":1,
            "boundary reconfiguration":1,"role allocation ecosystem":1,
            "ecosystem participation rules":1,
            "partner governance":2,"participation model":2,"ecosystem policy":2,
            "platform fairness":2,"platform stewardship":2,"ecosystem compliance":2,
            "platform accountability":2,"partner participation rules":2,
            "partner resistance management":2,"platform adoption resistance":2,
            "complementor role definition":2,"ecosystem boundary dynamics":2,
            "legacy partner integration":2,
            "governance":3,"roles":3,"incentives":3,
            "platform policy":3,"ecosystem rules":3,
            "role allocation":3,"platform resistance":3,"boundary dynamics":3,
        },
    ),
    "G6_Joint_Learning_Knowledge_Sharing": CategoryInfo(
        code="G6_Joint_Learning_Knowledge_Sharing",
        name="Joint Learning & Knowledge Sharing",
        description="Shared knowledge platforms, joint learning and partner capability building across firm boundaries [Soft Gate: digital marker required]",
        dimension="G",
        keywords=[
            "shared knowledge platform","joint learning routines","partner capability building",
            "collaborative analytics learning","digital partner enablement",
            "joint innovation platform","partner co-development","ecosystem learning network",
            "cross-company knowledge sharing","collaborative capability program",
            "best practice sharing","learning network","knowledge repository",
            "collaborative problem solving","partner upskilling","joint capability development",
            "shared learning platform","ecosystem capability building",
            "knowledge sharing","joint learning","capability building",
            "partner training","learning platform",
        ],
        patterns=[
            r"\bshared\s+knowledge\s+platform\b",
            r"\bjoint\s+learning\s+routines\b",
            r"\bpartner\s+capability\s+building\b",
            r"\bcollaborative\s+analytics\s+learning\b",
            r"\bdigital\s+partner\s+enablement\b",
            r"\bjoint\s+innovation\s+platform\b",
            r"\bpartner\s+co-development\b",
            r"\becosystem\s+learning\s+network\b",
            r"\bcross-company\s+knowledge\s+sharing\b",
        ],
        keyword_tiers={
            "shared knowledge platform":1,"joint learning routines":1,
            "partner capability building":1,"collaborative analytics learning":1,
            "digital partner enablement":1,"joint innovation platform":1,
            "partner co-development":1,"ecosystem learning network":1,
            "cross-company knowledge sharing":1,"collaborative capability program":1,
            "best practice sharing":2,"learning network":2,"knowledge repository":2,
            "collaborative problem solving":2,"partner upskilling":2,
            "joint capability development":2,"shared learning platform":2,
            "ecosystem capability building":2,
            "knowledge sharing":3,"joint learning":3,"capability building":3,
            "partner training":3,"learning platform":3,
        },
    ),
}


# ============================================================================
# TAXONOMY PROVIDER
# ============================================================================

class DigitalizationRelationalTaxonomy(TaxonomyProvider):
    """
    TaxonomyProvider v2.2.0 — Digitalizare Relationala & Ecosisteme de Business.

    Trei dimensiuni: Application (D1-D8), Technology (T1-T8), Governance (G1-G6).

    Gate logic:
        D: fara gate — detectabile independent
        T2-T8: Co-occurrence Gate (actor extern in ±500 chars)
        T1,T4: Co-occurrence Gate + Relational Verb Gate
        G1-G6: Soft Gate (marker digital in ±400 chars sau hit D/T anterior)

    v2.2 delta: +Catena-X/DPP/battery passport/sovereign data exchange pe D4/D8/G3/G4;
                +oportunism/power imbalance/relationship transparency pe G1/G2/G5;
                +legacy relationships/platform resistance pe D7/G5;
                +4 FP patterns noi; +19 anchor phrases noi (★).

    Nota implementare 05_detect.py:
        - EXTERNAL_ACTORS, RELATIONAL_VERBS, DIGITAL_MARKERS sets din acest modul
        - Verifica gate-urile la Stage 1 (dupa keyword/pattern match, inainte de semantic)
    """

    def __init__(self):
        # Compile FP patterns
        self._fp_compiled: Dict[str, List[re.Pattern]] = {}
        for fp_cat, patterns in FALSE_POSITIVE_PATTERNS.items():
            self._fp_compiled[fp_cat] = []
            for ps in patterns:
                try:
                    self._fp_compiled[fp_cat].append(re.compile(ps, re.IGNORECASE))
                except re.error:
                    pass

        # Compile detection patterns per dimension
        self._det_d: List[Tuple[re.Pattern, str]] = []
        self._det_t: List[Tuple[re.Pattern, str]] = []
        self._det_g: List[Tuple[re.Pattern, str]] = []

        for code, cat in _D_APPLICATIONS.items():
            for ps in cat.patterns:
                try: self._det_d.append((re.compile(ps, re.IGNORECASE), code))
                except re.error: pass

        for code, cat in _T_TECHNOLOGIES.items():
            for ps in cat.patterns:
                try: self._det_t.append((re.compile(ps, re.IGNORECASE), code))
                except re.error: pass

        for code, cat in _G_GOVERNANCE.items():
            for ps in cat.patterns:
                try: self._det_g.append((re.compile(ps, re.IGNORECASE), code))
                except re.error: pass

    # ── TaxonomyProvider interface ────────────────────────────────────────

    def get_version(self) -> str:
        return TAXONOMY_VERSION

    def get_dimensions(self) -> Dict[str, Dict[str, CategoryInfo]]:
        return {
            "Application": _D_APPLICATIONS,
            "Technology":  _T_TECHNOLOGIES,
            "Governance":  _G_GOVERNANCE,
        }

    def get_fp_patterns(self) -> Dict[str, List[str]]:
        return FALSE_POSITIVE_PATTERNS

    def check_false_positive(self, text: str, context: str = "") -> FPResult:
        combined = f"{text} {context}".lower()
        for fp_cat, patterns in self._fp_compiled.items():
            for pattern in patterns:
                if pattern.search(combined):
                    return FPResult(is_fp=True, category=fp_cat, pattern=pattern.pattern)
        return FPResult(is_fp=False)

    def classify(self, text: str, context: str = "") -> ClassificationResult:
        """
        Clasificare in cele 3 dimensiuni D/T/G.
        Nota: gate-urile NU sunt aplicate in classify() — sunt aplicate
        in 05_detect.py la nivel de fereastra de context, nu de text fragment.
        classify() returneaza best match per dimensiune pe baza de score.
        """
        combined = f"{text} {context}"

        cat_d, conf_d = self._match_dimension(combined, self._det_d, _D_APPLICATIONS)
        cat_t, conf_t = self._match_dimension(combined, self._det_t, _T_TECHNOLOGIES)
        cat_g, conf_g = self._match_dimension(combined, self._det_g, _G_GOVERNANCE)

        return ClassificationResult(dimensions={
            "Application": (cat_d, conf_d),
            "Technology":  (cat_t, conf_t),
            "Governance":  (cat_g, conf_g),
        })

    # ── Gate helpers (called from 05_detect.py) ───────────────────────────

    def has_external_actor(self, window: str) -> bool:
        """True daca fereastra contine cel putin un actor extern. (Gate T)"""
        wl = window.lower()
        return any(actor in wl for actor in EXTERNAL_ACTORS)

    def has_relational_verb(self, window: str) -> bool:
        """True daca fereastra contine cel putin un verb relational. (Gate T1/T4)"""
        wl = window.lower()
        return any(verb in wl for verb in RELATIONAL_VERBS)

    def has_digital_marker(self, window: str) -> bool:
        """True daca fereastra contine cel putin un marker digital. (Soft Gate G)"""
        wl = window.lower()
        return any(marker in wl for marker in DIGITAL_MARKERS)

    def get_gate_type(self, category_code: str) -> str:
        """
        Returneaza tipul de gate pentru o categorie:
          'none'        — D categories, detectabile independent
          'cooccur'     — T2-T8, necesita actor extern
          'cooccur_verb'— T1, T4, necesita actor extern + verb relational
          'soft'        — G1-G6, necesita marker digital
        """
        prefix = category_code[:2].rstrip("_")
        cod = category_code.split("_")[0]
        if cod in COOCCUR_T_VERB:
            return "cooccur_verb"
        if cod in COOCCUR_T_ONLY:
            return "cooccur"
        if cod in SOFT_GATE_G:
            return "soft"
        return "none"

    # ── Internal helpers ──────────────────────────────────────────────────

    def _match_dimension(
        self,
        text: str,
        compiled_patterns: List[Tuple[re.Pattern, str]],
        categories: Dict[str, CategoryInfo],
    ) -> Tuple[str, float]:
        scores: Dict[str, float] = {}
        text_lower = text.lower()

        for compiled, code in compiled_patterns:
            if compiled.search(text):
                scores[code] = scores.get(code, 0.0) + 0.7

        for code, cat in categories.items():
            for kw in cat.keywords:
                if kw.lower() in text_lower:
                    tier  = cat.keyword_tiers.get(kw, 2)
                    bonus = {1: 0.8, 2: 0.6, 3: 0.4}.get(tier, 0.6)
                    scores[code] = scores.get(code, 0.0) + bonus

        if not scores:
            return ("", 0.0)
        best = max(scores, key=lambda c: scores[c])
        return (best, round(min(scores[best], 1.0), 3))


# ============================================================================
# MODULE-LEVEL SINGLETON
# ============================================================================

TAXONOMY      = DigitalizationRelationalTaxonomy()
PATTERN_CACHE = CompiledPatternCache()
PATTERN_CACHE.compile_from_provider(TAXONOMY)


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    print(f"AISA — {TAXONOMY_NAME} v{TAXONOMY_VERSION}")
    print()

    dims = TAXONOMY.get_dimensions()
    for dim_name, cats in dims.items():
        total_kw  = sum(len(c.keywords) for c in cats.values())
        total_pat = sum(len(c.patterns) for c in cats.values())
        print(f"  [{dim_name}] {len(cats)} categorii | {total_kw} kw | {total_pat} patterns")
        for code in sorted(cats.keys()):
            c  = cats[code]
            t1 = sum(1 for k in c.keywords if c.keyword_tiers.get(k) == 1)
            t2 = sum(1 for k in c.keywords if c.keyword_tiers.get(k) == 2)
            t3 = sum(1 for k in c.keywords if c.keyword_tiers.get(k) == 3)
            print(f"    {code}: T1={t1} T2={t2} T3={t3} | {len(c.patterns)} patterns | gate={TAXONOMY.get_gate_type(code)}")

    print()

    tests = [
        # D — no gate
        ("We launched a partner onboarding portal for supplier registration", "D1_Partner_Onboarding_Access"),
        ("EDI integration connects our systems with tier-1 supplier invoices", "D2_Transactional_Integration"),
        ("Our partner control tower provides real-time demand signals to suppliers", "D3_Collaborative_Planning_Visibility"),
        ("Federated data platform connects 1,200 partners under GAIA-X standards", "D4_Shared_Data_Interoperability"),
        ("Dealer portal handles 95% of order placement across our distributor network", "D5_Channel_Partner_Interface"),
        ("Digital servitization transformed B2B equipment sales with industrial customers", "D6_CoInnovation_Digital_Servitization"),
        ("We onboarded 4,500 complementors through our API marketplace ecosystem", "D7_Ecosystem_Orchestration_Complementors"),
        ("Blockchain traceability covers 100% of our ESG supply chain", "D8_Traceability_Compliance_Network"),
        # T — with external actors (gate satisfied)
        ("SAP Ariba connects our ERP to supplier procurement workflows", "T1_ERP_SCM_Backbone"),
        ("MuleSoft B2B gateway integrates partner APIs across our ecosystem", "T2_API_EDI_iPaaS_Middleware"),
        ("Coupa marketplace connects procurement with supplier network", "T3_B2B_Platforms_Marketplaces"),
        ("Shared cloud data platform connects 800 manufacturing partners", "T4_Cloud_Data_Platforms"),
        ("IIoT platform shares sensor data with service partners in real time", "T5_IoT_Digital_Twins_Connected_Assets"),
        ("Federated identity allows 85,000 partner users to access our platforms", "T6_Identity_Access_Security"),
        ("Blockchain traceability platform multi-party provenance with suppliers", "T7_Blockchain_DLT_Traceability"),
        ("Partner workflow automation reduced supplier onboarding cycles by 50%", "T8_Workflow_Automation_LowCode"),
        # G — with digital markers (soft gate satisfied)
        ("Our multi-tier visibility platform provides shared KPIs to ecosystem participants", "G1_Transparency_Visibility_Control"),
        ("Digital trust framework governs data exchange with 85 ecosystem partners", "G2_Trust_Security_Assurance"),
        ("Data sovereignty platform ensures partner data stays within EU jurisdictions", "G3_Data_Sovereignty_Ownership_Rights"),
        ("GAIA-X interoperability framework governs our ecosystem data standards", "G4_Standards_Interoperability_Rules"),
        ("Platform governance framework defines complementor incentives and revenue-sharing rules", "G5_Orchestration_Roles_Incentives"),
        ("Shared knowledge platform connects 800 partners for collaborative analytics", "G6_Joint_Learning_Knowledge_Sharing"),
    ]

    print("  Clasificare teste:")
    all_ok = True
    for text, expected in tests:
        clf  = TAXONOMY.classify(text)
        cats_found = [clf.category_a, clf.category_b]
        # v2.1 has 3rd dimension G
        if len(clf.dimensions) > 2:
            vals = list(clf.dimensions.values())
            cats_found = [v[0] for v in vals if v[0]]
        hit  = expected in cats_found
        icon = "✓" if hit else "✗"
        gate = TAXONOMY.get_gate_type(expected)
        print(f"  {icon} [{gate:<13}] {expected:<44} → {cats_found}")
        if not hit:
            all_ok = False

    print()

    # Gate helper tests
    print("  Gate helper teste:")
    assert TAXONOMY.has_external_actor("our supplier integration connects EDI") == True
    assert TAXONOMY.has_external_actor("our internal ERP migration completed") == False
    assert TAXONOMY.has_relational_verb("we integrated SAP with supplier network") == True
    assert TAXONOMY.has_relational_verb("SAP migration completed this quarter") == False
    assert TAXONOMY.has_digital_marker("our data platform governs exchange") == True
    assert TAXONOMY.has_digital_marker("board governance meeting this year") == False
    print("  ✓ Toate gate helper testele OK")

    print()
    print(f"  Clasificare: {'✓ OK' if all_ok else '⚠ unele erori'}")
    total_kw  = sum(len(c.keywords) for c in TAXONOMY.get_all_categories().values())
    total_pat = sum(len(c.patterns) for c in TAXONOMY.get_all_categories().values())
    print(f"\n  Sumar v{TAXONOMY_VERSION}:")
    print(f"    D categorii: {len(_D_APPLICATIONS)} (D1-D8, no gate)")
    print(f"    T categorii: {len(_T_TECHNOLOGIES)} (T2-T8 co-occur, T1/T4 co-occur+verb)")
    print(f"    G categorii: {len(_G_GOVERNANCE)} (G1-G6 soft gate)")
    print(f"    Keywords:    {total_kw}")
    print(f"    Patterns:    {total_pat}")
    print(f"    Anchors:     {sum(len(v) for v in ANCHOR_PHRASES.values())}")
    print(f"    FP groups:   {len(FALSE_POSITIVE_PATTERNS)}")
    print(f"    Det. patterns compiled: {len(PATTERN_CACHE.get_detection_patterns())}")

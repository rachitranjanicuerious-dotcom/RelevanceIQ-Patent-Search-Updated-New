import re
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from state import RelevanceState
from io import BytesIO
from collections import defaultdict
from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from langchain_core.output_parsers import PydanticOutputParser
from docx import Document
from docx.shared import (
    Inches,
    Pt,
    RGBColor
)

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import (
    WD_TABLE_ALIGNMENT,
    WD_CELL_VERTICAL_ALIGNMENT
)

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import textwrap
import difflib
from llm import llm

class FeatureRecommendation(BaseModel):

    product_feature: str
    priority: Literal[
        "Highest",
        "High",
        "Moderate",
        "Lower"
    ]

    analysis: str
    recommendation: str
    supporting_publication_numbers: list[str]

class AIRecommendationsOutput(BaseModel):

    executive_summary: str
    highest_priority_features: list[FeatureRecommendation]
    lower_priority_features: list[FeatureRecommendation]
    recommendations: list[str]
    potential_design_alternatives: list[str]
    additional_investigation_areas: list[str]
    conclusion: str

ai_recommendations_parser = PydanticOutputParser(
    pydantic_object=AIRecommendationsOutput
)

# ----------------- schemas for feature consolidation ------------------

class CanonicalFeatures(BaseModel):
    features: list[str]

class FeatureMapping(BaseModel):
    mapping: dict[str, str]

canonical_features_parser = PydanticOutputParser(
    pydantic_object=CanonicalFeatures
)

feature_mapping_parser = PydanticOutputParser(
    pydantic_object=FeatureMapping
)

OTHER_FEATURE_LABEL = "Other"

# ============================================================
# 6.2 HELPER FUNCTIONS
# ============================================================

def _get_value(record, key, default=None):

    """
    Safely retrieve values from dictionaries or Pydantic objects.
    """

    if isinstance(record, dict):

        return record.get(key, default)

    return getattr(record, key, default)


def _normalize_feature(feature):
    """
    Normalize product feature descriptions.
    """

    feature = str(feature or "").strip().lower()
    feature = re.sub(r"\s+", " ", feature)
    return feature

# ============================================================
# 6.1 PRODUCT FEATURE CANONICALIZATION
# ============================================================

def canonicalize_product_feature(feature):
    """
    Technology-independent string normalization.
    Used for exact/near-exact text matching, not semantic matching.
    """

    text = str(feature or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Keep compatibility if other parts of the application use this helper.
def _normalize_feature(feature):
    return canonicalize_product_feature(feature)


# ============================================================
# 6.2 EXTRACT ALL DISTINCT CANONICAL PRODUCT FEATURES
# ============================================================

def build_canonical_features(product_description):
    """
    Extract all distinct technical features from the product
    description.

    There is no maximum feature limit.

    Semantically equivalent descriptions should be consolidated
    by the LLM.
    """

    if not product_description or not product_description.strip():
        return []

    prompt = f"""
Break the product description below into ALL of its distinct
core technical features.

PRODUCT DESCRIPTION:
{product_description}

RULES:

Extract every meaningful technical feature explicitly supported by the product description. Each feature must be a concise label of no more than 15 words.

Each feature must identify a specific technical function, mechanism, component, architecture, or implementation that can support meaningful patent-to-product comparison.

Identify the underlying technical concept before creating a feature label. Different descriptions of the same concept must be represented by one canonical feature.

Do not create separate features for different levels of description of the same technology. In particular, do not separately extract:

- A technology and its general method of operation.
- A system and the general function it performs.
- A technical function and its intended purpose or benefit.
- A technology and its broad application or production context.
- A general process and alternative wording describing that same process.

Preserve technically distinct mechanisms, components, architectures, and implementations, even when they serve the same overall function. Do not merge distinct technical aspects merely because they belong to the same technology area.

A method, pathway, process, system, or implementation may be extracted separately only when the product description provides enough information to establish that it represents a technically distinct aspect.

Do not extract broad application areas, intended benefits, general objectives, or umbrella technology descriptions as independent features when their technical content is already represented by more specific features.

Use only information explicitly stated or directly supported by the product description. Do not invent features or infer unsupported technical details.

Before finalizing, compare every pair of extracted features. Ask whether they represent different technical capabilities or merely different descriptions, purposes, or levels of abstraction of the same capability. Merge the latter.

Return the complete list of distinct canonical technical features. Prioritize meaningful technical distinctions over the number of extracted features. Do not create additional features merely to make the list more comprehensive.

{canonical_features_parser.get_format_instructions()}
"""

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    parsed = canonical_features_parser.parse(
        response.content
    )

    features = []
    seen = set()

    for feature in parsed.features:

        cleaned = re.sub(
            r"\s+",
            " ",
            str(feature or "")
        ).strip()

        normalized = canonicalize_product_feature(cleaned)

        if cleaned and normalized and normalized not in seen:

            features.append(cleaned)
            seen.add(normalized)

    return features

# ============================================================
# 6.4 MAP PATENT FEATURES TO CANONICAL PRODUCT FEATURES
# ============================================================

def map_features_to_canonical(raw_features, canonical_features):
    """
    Map raw patent comparison feature descriptions to the
    canonical product features.

    A raw feature is mapped only when there is a genuine
    technical relationship.

    Unrelated or unmappable features receive None and are
    excluded from the feature concentration analysis.
    """

    raw_features = sorted(set(raw_features))

    if not raw_features or not canonical_features:
        return {}

    prompt = f"""
Map each RAW FEATURE to the most appropriate CANONICAL FEATURE.

CANONICAL FEATURES:
{json.dumps(canonical_features, indent=2)}

RAW FEATURES:
{json.dumps(raw_features, indent=2)}

RULES:

1. Map a raw feature only when it has a genuine technical
   relationship with a canonical feature.

2. Consider technical meaning, not just similarity in wording.

3. A raw feature may be expressed using synonyms, abbreviations,
   or different technical terminology.

4. Do not force unrelated features into a canonical category.

5. Do not map a feature merely because one canonical feature
   is the closest available option.

6. If no canonical feature is a reasonable technical match,
   return exactly "UNMAPPED".

7. Do not invent new canonical feature labels.

8. Every raw feature must appear as a key, exactly as written.

9. Every mapped value must be copied exactly from the
   canonical feature list, or be "UNMAPPED".

{feature_mapping_parser.get_format_instructions()}
"""

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    raw_mapping = feature_mapping_parser.parse(
        response.content
    ).mapping

    # Normalize mapping keys to tolerate case and whitespace.
    key_lookup = {
        canonicalize_product_feature(key): value
        for key, value in raw_mapping.items()
    }

    # Canonical label lookup.
    value_lookup = {
        canonicalize_product_feature(label): label
        for label in canonical_features
    }

    result = {}

    for raw in raw_features:

        candidate = raw_mapping.get(raw)

        if candidate is None:

            candidate = key_lookup.get(
                canonicalize_product_feature(raw)
            )

        label = None

        if candidate:

            candidate_text = str(candidate).strip()

            if candidate_text.upper() != "UNMAPPED":

                # First try exact normalized canonical label.
                label = value_lookup.get(
                    canonicalize_product_feature(candidate_text)
                )

        # Do not automatically map the original raw feature
        # to its closest canonical feature.
        result[raw] = label

    return result


# ============================================================
# 6.5 COLLECT RAW FEATURES FROM PATENT RESULTS
# ============================================================

def _collect_raw_features(patent_results):
    """
    Collect distinct patent comparison features that have
    an actual non-No-Match comparison classification.

    Blank match types are excluded because they do not provide
    an explicit feature-match assessment.
    """

    raw_features = set()

    for record in patent_results:

        comparisons = (
            _get_value(record, "comparison", []) or []
        )

        for comparison in comparisons:

            feature = str(
                _get_value(
                    comparison,
                    "product_feature",
                    ""
                ) or ""
            ).strip()

            match_type = str(
                _get_value(
                    comparison,
                    "match_type",
                    ""
                ) or ""
            ).strip().lower()

            if not feature:
                continue

            if not match_type:
                continue

            if match_type == "no match":
                continue

            raw_features.add(feature)

    return raw_features


# ============================================================
# 6.6 BUILD FEATURE MAP
# ============================================================

def build_feature_map( patent_results, product_description, canonical_features=None):
    """
    Returns:

        {
            raw patent feature wording: canonical product feature,
            ...
        }

    Features that cannot be mapped are assigned None.
    """

    raw_features = _collect_raw_features(patent_results)

    if not raw_features:
        return {}

    if not product_description:
        return {}

    try:

        # Reuse an already extracted canonical feature list.
        if canonical_features is None:

            canonical_features = build_canonical_features(
                product_description
            )

        if not canonical_features:
            return {}

        return map_features_to_canonical(
            raw_features,
            canonical_features
        )

    except Exception as e:

        print(
            "Feature consolidation failed. "
            "Unmapped features will be excluded from "
            "feature concentration analysis:",
            e
        )
        return {}


# ============================================================
# 6.7 CALCULATE PATENT ANALYTICS
# ============================================================

def calculate_patent_analytics(
    patent_results,
    product_description=None,
    feature_map=None,
    canonical_features=None
):
    """
    Calculate patent-level and consolidated product-feature
    analytics.

    Main requirements:

    1. Extract all distinct canonical product features.
    2. Include a product feature in the final feature analysis
       only if at least one patent has a qualifying comparison
       mapped to that feature.
    3. Exclude No Match comparisons.
    4. Exclude unmapped patent features.
    5. Count each publication number once per feature and rating.
    6. Retain original patent feature wording and patent details.

    Counts are based on unique publication numbers
    """

    ratings = ["H", "M+", "L", "NR"]

    rating_counts = {
        rating: 0
        for rating in ratings
    }

    unique_patents = {}

    feature_data = defaultdict(
        lambda: {
            "H": set(),
            "M+": set(),
            "L": set(),
            "NR": set(),
            "patent_details": {},
            "original_features": set()
        }
    )

    # --------------------------------------------------------
    # STEP 0: EXTRACT CANONICAL FEATURES
    # --------------------------------------------------------

    if canonical_features is None:

        canonical_features = (
            build_canonical_features(product_description)
            if product_description
            else []
        )

    # --------------------------------------------------------
    # STEP 1: BUILD FEATURE MAP
    # --------------------------------------------------------

    if feature_map is None:

        feature_map = build_feature_map(
            patent_results,
            product_description,
            canonical_features=canonical_features
        )

    # --------------------------------------------------------
    # STEP 2: COLLECT UNIQUE PUBLICATIONS
    # --------------------------------------------------------

    for record in patent_results:

        publication_number = str(
            _get_value(
                record,
                "publication_number",
                ""
            ) or ""
        ).strip()

        if not publication_number:
            continue

        relevance = str(
            _get_value(
                record,
                "relevance",
                "NR"
            ) or "NR"
        ).strip().upper()

        if relevance not in ratings:
            continue

        patent_data = {
            "publication_number": publication_number,
            "title": _get_value(record, "title", ""),
            "relevance": relevance,
            "confidence": _get_value(record, "confidence", 0),
            "rationale": _get_value(record, "rationale", ""),
            "reasoning": _get_value(record, "reasoning", "")
        }

        if publication_number not in unique_patents:

            unique_patents[publication_number] = patent_data

        else:

            existing = unique_patents[publication_number]

            # Replace only with a more complete record having
            # the same rating.
            if (
                existing["relevance"] == relevance
                and not existing.get("rationale")
                and patent_data.get("rationale")
            ):

                unique_patents[publication_number] = patent_data

            elif existing["relevance"] != relevance:

                print(
                    "WARNING: Conflicting relevance ratings "
                    f"for publication {publication_number}. "
                    f"Keeping first rating: {existing['relevance']}; "
                    f"later rating: {relevance}."
                )

    # --------------------------------------------------------
    # STEP 3: COUNT UNIQUE PUBLICATIONS BY RATING
    # --------------------------------------------------------

    for patent in unique_patents.values():

        rating_counts[patent["relevance"]] += 1

    total_patents = len(unique_patents)

    # --------------------------------------------------------
    # STEP 4: CONSOLIDATE PRODUCT FEATURES
    # --------------------------------------------------------

    canonical_lookup = {
        canonicalize_product_feature(feature): feature
        for feature in canonical_features
    }

    for record in patent_results:

        publication_number = str(
            _get_value(
                record,
                "publication_number",
                ""
            ) or ""
        ).strip()

        if publication_number not in unique_patents:
            continue

        # Use the final selected rating for this publication.
        rating = unique_patents[publication_number]["relevance"]

        comparisons = (
            _get_value(record, "comparison", []) or []
        )

        for comparison in comparisons:

            raw_feature = str(
                _get_value(
                    comparison,
                    "product_feature",
                    ""
                ) or ""
            ).strip()

            match_type = str(
                _get_value(
                    comparison,
                    "match_type",
                    ""
                ) or ""
            ).strip()

            if not raw_feature:
                continue

            # A blank match classification is not treated as
            # confirmed feature correspondence.
            if not match_type:
                continue

            # No Match is not evidence of feature correspondence.
            if match_type.lower() == "no match":
                continue

            # ------------------------------------------------
            # MAP TO CANONICAL PRODUCT FEATURE
            # ------------------------------------------------

            mapped_feature = feature_map.get(raw_feature)

            if not mapped_feature:
                continue

            # Ensure the mapped label belongs to the extracted
            # canonical product feature list.
            canonical_feature = canonical_lookup.get(
                canonicalize_product_feature(mapped_feature)
            )

            if not canonical_feature:
                continue

            # ------------------------------------------------
            # COUNT UNIQUE PUBLICATIONS
            # ------------------------------------------------

            feature_data[canonical_feature][rating].add(
                publication_number
            )
            
            # ------------------------------------------------
            # RETAIN ORIGINAL FEATURE DESCRIPTIONS
            # ------------------------------------------------

            feature_data[canonical_feature][
                "original_features"
            ].add(raw_feature)

            # ------------------------------------------------
            # RETAIN PATENT-LEVEL DETAILS
            # ------------------------------------------------

            patent_details = feature_data[
                canonical_feature
            ]["patent_details"]

            if publication_number not in patent_details:

                patent_details[publication_number] = {
                    "product_features": []
                }

            detail = {
                "product_feature": raw_feature,
                "match_type": match_type,
                "patent_feature": _get_value(
                    comparison,
                    "patent_feature",
                    ""
                ),
                "reasoning": _get_value(
                    comparison,
                    "reasoning",
                    ""
                )
            }

            if detail not in patent_details[
                publication_number
            ]["product_features"]:

                patent_details[publication_number][
                    "product_features"
                ].append(detail)

    # --------------------------------------------------------
    # STEP 5: CALCULATE FEATURE ANALYTICS
    # --------------------------------------------------------

    feature_analytics = []

    for canonical_feature, data in feature_data.items():

        h_count = len(data["H"])
        mplus_count = len(data["M+"])
        l_count = len(data["L"])
        nr_count = len(data["NR"])

        high_relevance_count = h_count + mplus_count

        all_feature_patents = (
            data["H"]
            | data["M+"]
            | data["L"]
            | data["NR"]
        )

        total_feature_patents = len(all_feature_patents)

        # ----------------------------------------------------
        # IMPORTANT FILTER
        # ----------------------------------------------------

        # Include only features associated with at least
        # one unique patent.
        if total_feature_patents == 0:
            continue

        h_share = (
            h_count / total_feature_patents * 100
            if total_feature_patents
            else 0
        )

        if h_count:
            highest_rating = "H"

        elif mplus_count:
            highest_rating = "M+"

        elif l_count:
            highest_rating = "L"

        else:
            highest_rating = "NR"

        # Short graph label.
        display_feature = canonical_feature

        if len(display_feature) > 60:

            display_feature = (
                display_feature[:57].rsplit(" ", 1)[0] + "..."
            )

        feature_analytics.append({

            "product_feature": canonical_feature,

            "display_feature": display_feature,

            "H": h_count,

            "M+": mplus_count,

            "L": l_count,

            "NR": nr_count,

            "total_patents": total_feature_patents,

            "high_relevance_count": high_relevance_count,

            "highest_rating": highest_rating, 

            "h_share": round(
                h_share,
                2
            ),

            "supporting_publication_numbers": sorted(
                data["H"] | data["M+"]
            ),

            "all_publication_numbers": sorted(
                all_feature_patents
            ),

            "original_feature_variants": sorted(
                data["original_features"]
            ),

            "patent_details": data["patent_details"]

        })

    # --------------------------------------------------------
    # STEP 6: SORT FEATURES
    # --------------------------------------------------------

    feature_analytics.sort(
        key=lambda x: (
            x["high_relevance_count"],
            x["H"],
            x["M+"],
            x["total_patents"]
        ),
        reverse=True
    )

    # --------------------------------------------------------
    # STEP 7: OVERALL RATING PERCENTAGES
    # --------------------------------------------------------

    rating_percentages = {

        rating: round(
            rating_counts[rating] / total_patents * 100,
            2
        ) if total_patents else 0

        for rating in ratings

    }

    # --------------------------------------------------------
    # STEP 8: HIGH-PRIORITY PATENT COUNT
    # --------------------------------------------------------

    high_priority_patents = (
        rating_counts["H"] + rating_counts["M+"]
    )

    high_priority_percentage = (
        high_priority_patents / total_patents * 100
        if total_patents
        else 0
    )

    # --------------------------------------------------------
    # STEP 9: RETURN ANALYTICS
    # --------------------------------------------------------

    return {

        "total_patents": total_patents,
        "rating_counts": rating_counts,
        "rating_percentages": rating_percentages,
        "high_priority_patents": high_priority_patents,
        "high_priority_percentage": round(
            high_priority_percentage,
            2
        ),

        # All extracted features, including those with no patents. Useful for auditing and internal checks.
        "canonical_features": canonical_features,
        "feature_analytics": feature_analytics,
        "patent_records": list(unique_patents.values())

    }


# ============================================================
# 6.10 GENERATE AI GENERATED RECOMMENDATIONS USING LLM
# ============================================================

def generate_ai_recommendations(
    patent_results,
    product_description
):
    """
    Generate consolidated portfolio-level AI recommendations.
    """

    analytics = calculate_patent_analytics(
        patent_results,
        product_description=product_description
    )

    if analytics["total_patents"] == 0:

        raise ValueError(
            "No valid completed patent analysis results were provided."
        )

    # --------------------------------------------------------
    # PREPARE FEATURE SUMMARY
    # --------------------------------------------------------

    feature_summary = []

    for feature in analytics["feature_analytics"]:

        feature_summary.append({

            "product_feature": feature["product_feature"],

            "H_patents": feature["H"],

            "Mplus_patents": feature["M+"],

            "L_patents": feature["L"],

            "NR_patents": feature["NR"],

            "total_associated_patents": feature["total_patents"],

            "supporting_publication_numbers":
                feature["supporting_publication_numbers"]

        })

    # --------------------------------------------------------
    # PREPARE PATENT SUMMARY
    # --------------------------------------------------------

    patent_summary = []

    for patent in analytics["patent_records"]:

        publication_number = patent["publication_number"]

        comparison_evidence = {}

        for feature in analytics["feature_analytics"]:

            patent_details = feature["patent_details"].get(
                publication_number
            )

            if patent_details:

                comparison_evidence[feature["product_feature"]] = (
                    patent_details["product_features"]
                )

        patent_summary.append({

            "publication_number": patent["publication_number"],

            "title": patent["title"],

            "rating": patent["relevance"],

            "confidence": patent["confidence"],

            "rationale": patent["rationale"],

            "comparison_evidence": comparison_evidence

        })

    # --------------------------------------------------------
    # AI PROMPT
    # --------------------------------------------------------

    recommendation_prompt = f"""
You are a senior patent portfolio analyst conducting
a technical FTO screening assessment.

Analyse the complete patent dataset against the provided
product description.

============================================================
PRODUCT DESCRIPTION
============================================================
{product_description}
============================================================
OVERALL PATENT STATISTICS
============================================================

Total unique publications:
{analytics["total_patents"]}

Rating distribution:
{json.dumps(analytics["rating_counts"], indent=2)}

Rating percentages:
{json.dumps(analytics["rating_percentages"], indent=2)}

Total H + M+ publications:
{analytics["high_priority_patents"]}

H + M+ portfolio percentage:
{analytics["high_priority_percentage"]}%

============================================================
PRODUCT FEATURE-WISE PATENT CONCENTRATION
============================================================

{json.dumps(feature_summary, indent=2)}

============================================================
INDIVIDUAL PATENT RESULTS
============================================================

{json.dumps(patent_summary, indent=2)}

============================================================
ANALYSIS INSTRUCTIONS
============================================================

1. Analyse the complete patent portfolio against the
   supplied product description.

2. Identify meaningful patterns among H-rated and M+-rated patents.

3. Examine L and NR findings without automatically
   treating them as technically irrelevant.

4. Examine recurring technical combinations, functional
   relationships, structural arrangements, processes,
   and claim limitations.

5. Consider actual claim correspondence, confidence, and
   rationale.

6. Identify areas requiring detailed patent claim review.

7. Distinguish patent concentration from infringement risk.

8. Identify technical investigation directions only
   where supported by the actual patent findings.

9. Use actual publication numbers when identifying
    supporting patents.

10. Never count duplicate publication numbers twice.

11. Feature-level patent counts may overlap because one
    patent may correspond to multiple features.

============================================================
PORTFOLIO-LEVEL CONSOLIDATION
============================================================

Recommendations must be based on DISTINCT technical or
claim-related findings, not on the total number of patents.

1. Identify distinct technical and claim-related findings
   across the complete dataset.

2. Group semantically similar findings into broader
   portfolio-level review areas. Treat different descriptions of the same underlying
   technical functionality as one finding where justified.

3. Multiple patents discussing the same technical area
   should generally strengthen the evidence for that area,
   rather than automatically generating additional
   recommendations.

4. Do not generate one recommendation per patent.

5. Do not generate one recommendation per product feature.

6. Do not split one underlying technical concern into
   multiple recommendations merely because it appears
   in different patents.

7. Generate a separate recommendation only when there is
   a genuinely distinct technical issue, claim limitation,
   functional relationship, implementation concern, or
   investigation requirement.

# ============================================================
# AVOID REPETITIVE PRODUCT FEATURES
# ============================================================

# The report already contains a separate section titled
# "Product Feature-Wise Patent Concentration".

# Therefore:

# 1. Do not reproduce individual product feature names as
#    recommendation headings.

# 2. Do not repeat the feature concentration table.

# 3. Do not create another list of product features with
#    slightly modified wording.

# 4. Focus on broader portfolio-level review areas, recurring
#    claim limitations, technical combinations, overlapping
#    patent concerns, and strategic findings.

# 5. Each recommendation must add analytical value beyond
#    the feature concentration section.

# 6. Do not manufacture relationships between unrelated
#    technologies or patent findings.

# The existing schema field "product_feature" must be used
# for a concise portfolio review area or strategic concern.
# It must NOT simply repeat an individual product feature.

============================================================
RECOMMENDATION PRIORITY
============================================================

Highest:
Strong H-rated evidence with meaningful claim correspondence
and substantial patent assessment concerns.

High:
Meaningful H and M+ evidence, or substantial claim-related
concerns requiring further examination.

Moderate:
Limited H/M+ evidence, or material technical and
claim-related uncertainty requiring further examination.

Lower:
Predominantly L or NR findings with limited evidence
of meaningful claim correspondence.

Priority must not be determined using patent counts alone.

Consider:

- Strength of claim correspondence.
- Nature of the claimed technical limitations.
- Confidence of the underlying patent assessment.
- Recurrence of a genuinely similar technical concern.
- Breadth and significance of the identified issue.

============================================================
OUTPUT REQUIREMENTS
============================================================

1. EXECUTIVE SUMMARY

Provide approximately 3-5 concise sentences summarizing
the overall portfolio landscape, important findings,
and investigation priorities.

2. HIGHEST-PRIORITY AREAS

Identify broader portfolio review areas requiring
significant attention.

Each entry must contain:

- product_feature
- priority
- analysis
- recommendation
- supporting_publication_numbers

Do not repeat individual product feature names.

3. LOWER-PRIORITY AREAS

Identify comparatively lower-priority portfolio review
areas and explain the evidence supporting the priority.

Do not assume L or NR means no technical relationship.

4. ACTIONABLE RECOMMENDATIONS

Provide practical recommendations for:

- Detailed claim review.
- Technical verification.
- Patent-family investigation.
- Implementation analysis.
- Portfolio monitoring.

Only include recommendations relevant to the supplied data.
Consolidate actions addressing the same underlying concern.

5. POTENTIAL DESIGN ALTERNATIVES

Identify technical directions worth investigating where
actual patent claim limitations provide a basis.

Do not invent unsupported alternatives.
Do not describe any alternative as a confirmed workaround.

6. ADDITIONAL AREAS TO BE VERIFIED MANUALLY

Identify missing technical information, implementation
details, claim limitations, patent status, or other relevant
matters requiring further examination.

7. CONCLUSION

Provide a concise overall portfolio assessment and
recommended next steps.

============================================================
IMPORTANT LIMITATIONS
============================================================

This is a technical portfolio screening assessment.

Do not state that:

- The product infringes.
- The product is free from infringement.
- A design alternative guarantees avoidance.
- The supplied portfolio represents all existing patents.

A complete FTO assessment requires additional claim
interpretation, jurisdiction-specific analysis, patent
status verification, and legal review.

============================================================
FINAL CONSISTENCY CHECK
============================================================

Before returning the output, verify:

1. Recommendations represent distinct portfolio concerns.
2. Similar findings across multiple patents are consolidated.
3. No recommendation exists solely because another patent
   was added to the dataset.
4. Supporting publication numbers actually support the
   corresponding recommendation.
5. Recommendation priority is evidence-based.
6. Individual product features have not been repeated as
   recommendation headings.
7. No unsupported technical conclusions are introduced.
8. The output is consistent with the supplied Pydantic schema.

============================================================
OUTPUT FORMAT
============================================================

Return valid JSON only.

Keep the analysis concise, specific, and evidence-based.

Follow the Pydantic schema exactly.

If insufficient evidence exists for an output category,
return an empty list rather than inventing content.

{ai_recommendations_parser.get_format_instructions()}
"""

    # --------------------------------------------------------
    # CALL LLM
    # --------------------------------------------------------

    start = time.time()

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an experienced patent portfolio "
                    "analyst. Provide concise, evidence-based "
                    "technical recommendations. Consolidate "
                    "semantically similar findings across patents. "
                    "Do not generate recommendations based merely "
                    "on the number of patents or repeated product "
                    "features. Identify only genuinely distinct "
                    "portfolio-level concerns."
                )
            ),
            HumanMessage(content=recommendation_prompt)
        ]
    )

    print(
        "AI recommendation duration:",
        time.time() - start
    )

    # --------------------------------------------------------
    # PARSE RESPONSE
    # --------------------------------------------------------

    try:

        recommendations = ai_recommendations_parser.parse(
            response.content
        )

    except Exception as e:

        print("=" * 80)
        print("AI RECOMMENDATION PARSING ERROR")
        print(e)
        print(response.content)

        raise

    return analytics, recommendations


# ============================================================
# 6.11 WORD DOCUMENT FORMATTING HELPERS
# ============================================================

def _set_cell_background(cell, color):

    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), color)

    cell._tc.get_or_add_tcPr().append(shading)


def _format_table_header(table):

    for cell in table.rows[0].cells:

        _set_cell_background(cell, "071426")

        for paragraph in cell.paragraphs:

            for run in paragraph.runs:

                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)


def _add_dataframe_style_table(
    document,
    headers,
    rows
):

    table = document.add_table(
        rows=1,
        cols=len(headers)
    )

    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for index, header in enumerate(headers):

        table.rows[0].cells[index].text = str(header)

    _format_table_header(table)

    for row in rows:

        cells = table.add_row().cells

        for index, value in enumerate(row):

            cells[index].text = str(value)

            cells[index].vertical_alignment = (
                WD_CELL_VERTICAL_ALIGNMENT.CENTER
            )

    return table


# ============================================================
# 6.12 GENERATE PATENT ANALYTICS GRAPHS
# ============================================================

def create_feature_concentration_figure(analytics):
    """
    Generate the feature-wise patent concentration chart.

    Only patent-supported canonical features are included.
    """

    features = analytics["feature_analytics"]

    if not features:
        return None

    names = [
        textwrap.fill(
            item["display_feature"],
            width=38
        )
        for item in features
    ]

    series = [
        ("H", "#1f77b4"),
        ("M+", "#ff7f0e"),
        ("L", "#2ca02c"),
        ("NR", "#d62728")
    ]

    fig, ax = plt.subplots(
        figsize=(
            9,
            max(3.5, 0.75 * len(names) + 1.5)
        )
    )

    y = np.arange(len(names))
    left = np.zeros(len(names), dtype=int)

    for key, color in series:

        values = np.array(
            [
                item[key]
                for item in features
            ],
            dtype=int
        )

        ax.barh(
            y,
            values,
            left=left,
            label=key,
            color=color
        )

        left += values

    max_total = int(left.max()) if len(left) else 0
    max_total = max(max_total, 1)

    # Whole-number X-axis intervals.
    ax.xaxis.set_major_locator(
        MultipleLocator(1)
    )

    ax.set_xlim(
        0,
        max_total + 1
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    ax.set_axisbelow(True)

    ax.set_xlabel("Number of Unique Publications")
    ax.set_ylabel("Product Features")

    ax.set_title(
        "Patent Concentration Across Product Features",
        fontweight="bold"
    )

    ax.legend(
        title="Relevance Rating",
        loc="lower right"
    )

    fig.tight_layout()

    return fig


def _create_recommendation_charts(analytics):
    """
    Generate:

    1. Patent relevance distribution.
    2. Product feature-wise patent concentration.

    Returns a dictionary containing BytesIO chart images.
    """

    chart_images = {}

    # --------------------------------------------------------
    # CHART 1: PATENT RELEVANCE DISTRIBUTION
    # --------------------------------------------------------

    counts = analytics["rating_counts"]

    fig, ax = plt.subplots(
        figsize=(7, 3.5)
    )

    ratings = list(counts.keys())
    values = list(counts.values())

    bars = ax.bar(
        ratings,
        values
    )

    ax.set_title(
        "Patent Relevance Distribution"
    )

    ax.set_xlabel(
        "Relevance Rating"
    )

    ax.set_ylabel(
        "Number of Publications"
    )

    max_value = max(values, default=0)

    ax.set_ylim(
        0,
        max_value * 1.15 + 1
    )

    for bar, value in zip(bars, values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    image_buffer = BytesIO()

    fig.savefig(
        image_buffer,
        format="png",
        dpi=180
    )

    image_buffer.seek(0)

    chart_images["rating"] = image_buffer

    plt.close(fig)

    # --------------------------------------------------------
    # CHART 2: FEATURE-WISE PATENT CONCENTRATION
    # --------------------------------------------------------

    fig = create_feature_concentration_figure(
        analytics
    )

    if fig is not None:

        image_buffer = BytesIO()

        fig.savefig(
            image_buffer,
            format="png",
            dpi=180
        )

        image_buffer.seek(0)

        chart_images["features"] = image_buffer

        plt.close(fig)

    return chart_images


# ============================================================
# 6.13 GENERATE PATENT ANALYTICS WORD REPORT
# ============================================================

def generate_ai_recommendations_report(
    patent_results,
    product_description,
    analytics=None,
    recommendations=None
):
    """
    Generate a downloadable Word document containing:

    1. Product overview.
    2. Executive summary.
    3. Patent relevance distribution.
    4. Patent analytics graphs.
    5. Product feature-wise patent concentration.
    6. Highest-priority portfolio review areas.
    7. Lower-priority portfolio review areas.
    8. AI-generated recommendations.
    9. Potential design alternatives.
    10. Additional investigation areas.
    11. Conclusion.
    12. Methodology and limitations.
    """

    # --------------------------------------------------------
    # GENERATE RECOMMENDATIONS IF NOT PROVIDED
    # --------------------------------------------------------

    if analytics is None or recommendations is None:

        analytics, recommendations = generate_ai_recommendations(
            patent_results,
            product_description
        )

    document = Document()

    # --------------------------------------------------------
    # DOCUMENT STYLING
    # --------------------------------------------------------

    section = document.sections[0]

    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)

    styles = document.styles

    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(9)

    styles["Title"].font.name = "Arial"
    styles["Title"].font.size = Pt(24)
    styles["Title"].font.bold = True
    styles["Title"].font.color.rgb = RGBColor(7, 20, 38)

    styles["Heading 1"].font.color.rgb = RGBColor(7, 20, 38)

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    title = document.add_heading(
        "RelevanceIQ",
        level=0
    )

    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = document.add_paragraph(
        "AI Generated Recommendations"
    )

    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].bold = True

    # --------------------------------------------------------
    # PRODUCT OVERVIEW
    # --------------------------------------------------------

    document.add_heading(
        "Product Overview",
        level=1
    )

    document.add_paragraph(
        product_description
    )

    document.add_paragraph(
        "Technical screening report based on the supplied "
        "product description and patent analysis results."
    )

    # --------------------------------------------------------
    # 1. EXECUTIVE SUMMARY
    # --------------------------------------------------------

    document.add_heading(
        "1. Executive Summary",
        level=1
    )

    document.add_paragraph(
        recommendations.executive_summary
    )

    # --------------------------------------------------------
    # 2. PATENT RELEVANCE DISTRIBUTION
    # --------------------------------------------------------

    document.add_heading(
        "2. Patent Relevance Distribution",
        level=1
    )

    _add_dataframe_style_table(
        document,
        [
            "Rating",
            "Patent Count",
            "Portfolio Percentage"
        ],
        [
            [
                rating,
                analytics["rating_counts"][rating],
                f'{analytics["rating_percentages"][rating]}%'
            ]
            for rating in ["H", "M+", "L", "NR"]
        ]
    )

    document.add_paragraph(
        f'Total unique publications analysed: '
        f'{analytics["total_patents"]}'
    )

    document.add_paragraph(
        f'H + M+ publications: '
        f'{analytics["high_priority_patents"]} '
        f'({analytics["high_priority_percentage"]}%)'
    )

    # --------------------------------------------------------
    # 3. PATENT ANALYTICS GRAPHS
    # --------------------------------------------------------

    document.add_heading(
        "3. Patent Analytics",
        level=1
    )

    charts = _create_recommendation_charts(
        analytics
    )

    if "rating" in charts:

        document.add_picture(
            charts["rating"],
            width=Inches(5.8)
        )

    if "features" in charts:

        document.add_picture(
            charts["features"],
            width=Inches(6.5)
        )

    # --------------------------------------------------------
    # 4. PRODUCT FEATURE-WISE PATENT CONCENTRATION
    # --------------------------------------------------------

    document.add_heading(
        "4. Product Feature-Wise Patent Concentration",
        level=1
    )

    document.add_paragraph(
        "The product description is decomposed into distinct "
        "canonical technical features. Only features associated "
        "with at least one qualifying patent comparison are "
        "included in this analysis. Differently worded patent "
        "feature descriptions that refer to the same canonical "
        "product feature are consolidated. Counts represent "
        "unique publication numbers within each feature. A "
        "publication may be associated with multiple features; "
        "therefore, feature-level counts may overlap."
    )

    feature_rows = []

    for feature in analytics["feature_analytics"]:

        feature_rows.append([
            feature["product_feature"],
            feature["H"],
            feature["M+"],
            feature["L"],
            feature["NR"],
            feature["total_patents"]
        ])

    if feature_rows:

        _add_dataframe_style_table(
            document,
            [
                "Product Feature",
                "H",
                "M+",
                "L",
                "NR",
                "Unique Publications"
            ],
            feature_rows
        )

    else:

        document.add_paragraph(
            "No product features were identified in at least "
            "one qualifying patent comparison."
        )

    # --------------------------------------------------------
    # 5. HIGHEST-PRIORITY PORTFOLIO REVIEW AREAS
    # --------------------------------------------------------

    document.add_heading(
        "5. Highest-Priority Portfolio Review Areas",
        level=1
    )

    if recommendations.highest_priority_features:

        for item in recommendations.highest_priority_features:

            document.add_heading(
                item.product_feature,
                level=2
            )

            document.add_paragraph(
                f"Priority: {item.priority}"
            )

            document.add_paragraph(
                f"Analysis: {item.analysis}"
            )

            document.add_paragraph(
                f"Recommendation: {item.recommendation}"
            )

            document.add_paragraph(
                "Supporting publications: "
                + ", ".join(
                    item.supporting_publication_numbers
                )
            )

    else:

        document.add_paragraph(
            "No significant highest-priority portfolio review "
            "areas were identified."
        )

    # --------------------------------------------------------
    # 6. LOWER-PRIORITY PORTFOLIO REVIEW AREAS
    # --------------------------------------------------------

    document.add_heading(
        "6. Lower-Priority Portfolio Review Areas",
        level=1
    )

    if recommendations.lower_priority_features:

        for item in recommendations.lower_priority_features:

            document.add_heading(
                item.product_feature,
                level=2
            )

            document.add_paragraph(
                f"Priority: {item.priority}"
            )

            document.add_paragraph(
                f"Analysis: {item.analysis}"
            )

            document.add_paragraph(
                f"Recommendation: {item.recommendation}"
            )

            document.add_paragraph(
                "Supporting publications: "
                + ", ".join(
                    item.supporting_publication_numbers
                )
            )

    else:

        document.add_paragraph(
            "No specific lower-priority portfolio review "
            "areas were identified."
        )

    # --------------------------------------------------------
    # 7. AI GENERATED RECOMMENDATIONS
    # --------------------------------------------------------

    document.add_heading(
        "7. AI Generated Recommendations",
        level=1
    )

    if recommendations.recommendations:

        for recommendation in recommendations.recommendations:

            document.add_paragraph(
                recommendation,
                style="List Bullet"
            )

    else:

        document.add_paragraph(
            "No additional recommendations were generated."
        )

    # --------------------------------------------------------
    # 8. POTENTIAL DESIGN ALTERNATIVES
    # --------------------------------------------------------

    document.add_heading(
        "8. Potential Design Alternatives",
        level=1
    )

    document.add_paragraph(
        "The following are areas for technical investigation. "
        "They are not confirmed design-arounds or guarantees "
        "of avoiding patent claims."
    )

    if recommendations.potential_design_alternatives:

        for alternative in recommendations.potential_design_alternatives:

            document.add_paragraph(
                alternative,
                style="List Bullet"
            )

    else:

        document.add_paragraph(
            "No potential design alternatives were identified."
        )

    # --------------------------------------------------------
    # 9. ADDITIONAL AREAS TO BE VERIFIED MANUALLY
    # --------------------------------------------------------

    document.add_heading(
        "9. Additional Areas to be Verified Manually",
        level=1
    )

    if recommendations.additional_investigation_areas:

        for investigation in recommendations.additional_investigation_areas:

            document.add_paragraph(
                investigation,
                style="List Bullet"
            )

    else:

        document.add_paragraph(
            "No additional investigation areas to be verified "
            "manually were identified."
        )

    # --------------------------------------------------------
    # 10. CONCLUSION
    # --------------------------------------------------------

    document.add_heading(
        "10. Conclusion",
        level=1
    )

    document.add_paragraph(
        recommendations.conclusion
    )

    # --------------------------------------------------------
    # 11. METHODOLOGY AND LIMITATIONS
    # --------------------------------------------------------

    document.add_heading(
        "11. Methodology and Limitations",
        level=1
    )

    document.add_paragraph(
        "This report consolidates the individual patent "
        "relevance assessments produced by RelevanceIQ. "
        "Portfolio-level patent counts are based on unique "
        "publication numbers. Product feature-wise counts "
        "include only canonical product features associated "
        "with at least one qualifying patent comparison. "
        "Feature-level counts may overlap because one patent "
        "publication can correspond to multiple product "
        "features. H and M+ classifications indicate screening "
        "priority and technical correspondence, not confirmed "
        "infringement. The analysis is limited to the supplied "
        "product description, patent records, claims, and "
        "portfolio coverage. A complete FTO assessment requires "
        "additional claim interpretation, jurisdiction-specific "
        "analysis, patent status verification, and legal review."
    )

    # --------------------------------------------------------
    # SAVE DOCUMENT TO MEMORY BUFFER
    # --------------------------------------------------------

    output = BytesIO()

    document.save(output)

    output.seek(0)

    print(
        "AI Generated Recommendations report "
        "generated successfully."
    )

    return output
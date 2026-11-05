from collections import defaultdict
from io import BytesIO
import json
import re

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import time
from typing import Literal
from state import RelevanceState
import pandas as pd
from llm import llm
from pathlib import Path

# Few-shot examples file
FEW_SHOT_FILE = Path(__file__).parent / "Few_shot_examples.xlsx"
def load_few_shot_examples():
    df = pd.read_excel(FEW_SHOT_FILE)

    examples = []

    for _, row in df.iterrows():
        example = f"""
============================================================
FEW-SHOT CALIBRATION EXAMPLE
============================================================

PRODUCT DESCRIPTION: 
{row["Product Description"]}

PATENT:
Publication Number: {row["Publication Number"]}
Title: {row["title"]}

ABSTRACT:
{row["Abstract"]}

INDEPENDENT CLAIMS:
{row["Independent Claims"]}

ALL CLAIMS:
{row["All Claims"]}

HUMAN-ANNOTATED RELEVANCE RATING:
{row["Relevance Rating"]}

============================================================
"""
        examples.append(example)
    return "\n".join(examples)

def build_common_fto_system_prompt(few_shot_examples):
    """
    Build the stable prompt prefix used by both FTO analyses.
    Keeping this prefix identical maximizes OpenAI prompt-cache hits.
    """
    return COMMON_FTO_INSTRUCTIONS.replace(
        "{few_shot_examples}",
        few_shot_examples
    )

# 1. PRODUCT FEATURE EXTRACTION
# ============================================================
class FeatureExtraction(BaseModel):
    primary_features: list[str]
    secondary_features: list[str]

feature_parser = PydanticOutputParser(
    pydantic_object=FeatureExtraction
)

# 2. PATENT FEATURE EXTRACTION
# ============================================================
def extract_patent_features(state: RelevanceState):
    
    prompt = f"""
You are an experienced patent analyst performing an FTO
(Freedom-to-Operate) technical analysis.

Extract technical features from the patent claims.

SOURCE PRIORITY:
1. Independent Claims — identify core mandatory limitations.
2. All Claims — identify additional dependent-claim limitations.

Classify every extracted technical feature into PRIMARY or SECONDARY.

PRIMARY FEATURES:
- Technical limitations appearing in independent claims.
- Features defining the core claimed combination.

SECONDARY FEATURES:
- Additional technical limitations.
- Limitations introduced by dependent claims.
- Supporting, optional, control, or auxiliary features.

IMPORTANT:
A dependent-claim limitation must NOT be treated as a mandatory
limitation of the independent claim.

Example:
Claim 1 = A + B + C
Claim 2 = Claim 1 + D

D is required for Claim 2, but not for Claim 1.

Rules:
- Do not infer claim limitations.
- Do not convert abstract/specification disclosure into claim
  limitations.
- Keep features concise.
- One feature per list item.

{feature_parser.get_format_instructions()}

PUBLICATION NUMBER:
{state["publication_number"]}

TITLE:
{state["title"]}

ABSTRACT:
{state["abstract"]}

INDEPENDENT CLAIMS:
{state["independent_claims"]}
      
ALL CLAIMS:
{state["all_claims"]}
"""

    start = time.time()
    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )
    print(
        "Patent feature extraction duration:",
         time.time() - start
    )
    try:
        result = feature_parser.parse(response.content)
    except Exception as e:
        print("=" * 80)
        print("PATENT FEATURE PARSING ERROR")
        print(e)
        print(response.content)

        state["primary_patent_features"] = []
        state["secondary_patent_features"] = []

        return state

    state["primary_patent_features"] = result.primary_features
    state["secondary_patent_features"] = result.secondary_features

    return state

# 3. OUTPUT SCHEMAS
# ===========================================================

class FeatureMatch(BaseModel):
    product_feature: str
    patent_feature: str
    match_type: Literal[
        "Exact Match",
        "Equivalent Match",
        "Partial Match",
        "No Match"
    ]

    reasoning: str

class DetailedAnalysisOutput(BaseModel):
    comparisons: list[FeatureMatch]
    relevance: Literal["H", "M+", "L", "NR"]
    rationale: str
    reasoning : str
    confidence: int = Field(ge=0, le=100)

class FrameworkOnlyOutput(BaseModel):
    relevance_framework_only: Literal["H", "M+", "L", "NR"]
    rationale_framework_only: str
    reasoning_framework_only : str
    confidence_framework_only: int = Field(ge=0, le=100)

detailed_parser = PydanticOutputParser(
    pydantic_object=DetailedAnalysisOutput
)

framework_parser = PydanticOutputParser(
    pydantic_object=FrameworkOnlyOutput
)

# ============================================================
# 4. FTO INSTRUCTIONS
# ============================================================

COMMON_FTO_INSTRUCTIONS = """
You are an experienced patent analyst performing a technical
Freedom-to-Operate (FTO) relevance screening.

The purpose of this analysis is to determine the technical relevance
and potential claim correspondence between the product and the patent claims.

This is a technical screening assessment only. It is NOT a legal
conclusion regarding infringement, validity, enforceability, or finals
Freedom to Operate.

============================================================
FEW-SHOT CALIBRATION EXAMPLES
============================================================

The following are human-annotated examples.

Use these examples to calibrate the H / M+ / L decision boundary.
Do not copy their conclusions blindly. Compare the technical
relationship in each example with the current product and patent.

{few_shot_examples}

============================================================
CLAIM-CENTRIC FTO ANALYSIS
============================================================

FTO analysis must be based primarily on the PATENT CLAIMS.

Analyze the strongest relevant patent claims individually.

Do not determine relevance from:
- the patent title, the abstract, the general technical field, specification-only disclosure;
- a simple count of matching product features.


============================================================
CLAIM AS A COMPLETE COMBINATION
============================================================

Each claim must be evaluated as the complete combination of its
required limitations.

Do not select a few matching limitations from a claim while ignoring
other material limitations.

At the same time, do not automatically treat every difference in
wording, terminology, or implementation as a material missing
limitation.

Determine which limitations materially define the claimed technical
invention.

============================================================
CLAIM DEPENDENCY
============================================================

Preserve the actual claim hierarchy.
An independent claim contains its own mandatory limitations.

A dependent claim includes:
- all limitations of its parent claim; and
- the additional limitations introduced by the dependent claim.

Example:
Claim 1 = A + B + C
Claim 2 = Claim 1 + D

D is mandatory for Claim 2.
D is NOT a mandatory limitation of Claim 1.

Therefore:
- do not transfer dependent-claim limitations to the parent claim;
- do not use a dependent-claim limitation to downgrade an otherwise relevant independent claim;
- evaluate dependent claims separately.

============================================================
NEVER CREATE A HYPOTHETICAL CLAIM
============================================================

Never combine limitations from separate claims to create a claim that
does not actually exist.

For example, do not take:
- A + B from Claim 1;
- C from Claim 3; and
- D from Claim 5

and treat A+B+C+D as one claimed combination.

Every relevance conclusion must be traceable to an actual claim.
============================================================
PRODUCT EVIDENCE CLASSIFICATION
============================================================

For each important claim limitation, classify the product evidence as:

A. CONFIRMED PRESENT
The product description explicitly supports the limitation or clearly
describes a corresponding technical mechanism.

B. NOT DISCLOSED / UNCERTAIN
The product description does not provide enough information to
determine whether the limitation is present.

This includes situations where:
- the product description is silent;
- the implementation is not described;
- terminology differs and correspondence cannot be conclusively
  established; or
- the available product description is incomplete.

C. CONFIRMED ABSENT / INCOMPATIBLE

The product description affirmatively indicates that:
- the product does not have the limitation;
- the product cannot satisfy the limitation; or
- the product uses a materially different technical mechanism that is
  incompatible with the claimed limitation.

CRITICAL RULE:
NOT DISCLOSED / UNCERTAIN is NOT the same as CONFIRMED ABSENT /
INCOMPATIBLE.

Do not infer absence solely because a limitation is not mentioned or is routine.

============================================================
DO NOT INVENT PRODUCT FEATURES
============================================================
Do not infer or invent product features that are not supported by the
product description.

Therefore:
No evidence of a feature
→ NOT DISCLOSED / UNCERTAIN.

Affirmative evidence contradicting the feature
→ potentially CONFIRMED ABSENT / INCOMPATIBLE.

============================================================
MATERIAL CLAIM LIMITATIONS
============================================================

A claim limitation is material when it materially defines the claimed
technical invention or the required technical combination.

Potentially material limitations include:
- a required technical component;
- a required technical mechanism;
- a required processing step;
- a required control relationship;
- a required sensor or input;
- a required physical arrangement;
- a required data relationship;
- a required technical constraint

Do not automatically treat the following as material differences:
- terminology;
- naming;
- ordinary contextual language;
- field labels;
- routine implementation details;
- unspecified implementation details; or
- additional product features that are not required by the claim.

Materiality must be determined from the actual claim language and the
product's described technical operation.

============================================================
TECHNICAL EQUIVALENCE
============================================================
Different terminology does not automatically mean different technology.

Where appropriate, treat a product feature as technically equivalent
when it performs substantially the same technical function, through
substantially the same technical mechanism, to achieve substantially
the same technical result.

Use technical equivalence carefully.

Do NOT infer equivalence merely because:
- the purpose is similar;
- the products operate in the same field;
- the result is broadly similar

Technical equivalence is a technical analytical concept only.
Do not make or imply a legal Doctrine of Equivalents or infringement
conclusion.

============================================================
IMPORTANT : EXTRA PRODUCT FEATURES
============================================================

Additional product features do not reduce relevance merely because
they are not present in the patent claim.

For example:
Claim = A + B + C
Product = A + B + C + D + E

D and E do not by themselves make the claim less relevant.

The analysis should focus on whether the product corresponds to
the claimed technical combination. The product can have additional features, the anlysis has to be on whether the claimed patent features are in the product.

============================================================
CLAIM-BY-CLAIM ANALYSIS PROCESS
============================================================

For each candidate claim:

Step 1:
Read the complete actual claim.

Step 2:
Identify its mandatory limitations.

Step 3:
Preserve its independent/dependent claim relationship.

Step 4:
Identify its core technical concept.

Step 5:
Map the product's technical functionality against the important
claim limitations.

Step 6:
Classify important product evidence as:

- CONFIRMED PRESENT;
- NOT DISCLOSED / UNCERTAIN; or
- CONFIRMED ABSENT / INCOMPATIBLE.

Step 7:
Determine whether any confirmed absent limitation is material to the
claimed technical invention.

Step 8:
Consider technical equivalence where justified.

Step 9:
Determine the technical relevance of the actual claim.

Step 10:
Compare the strongest relevant patent claims when necessary.

============================================================
DEPENDENT CLAIM CALIBRATION
============================================================

- If an independent claim substantially corresponds to the product,
evaluate that independent claim based on its own limitations.
- Do not downgrade the independent claim merely because a dependent claim
introduces an additional limitation that is absent or uncertain.
- For a dependent claim, however, all parent limitations plus the
dependent limitation must be considered.

============================================================
STRONGEST CLAIM PRINCIPLE
============================================================

Identify the strongest 1–3 patent claims relevant to the product.

The final rating should be based primarily on the strongest meaningful
claim-to-product correspondence rather than an average across
unrelated claims.

A single highly relevant  claim can be sufficient to make the
patent highly relevant when the applicable rating framework supports
that conclusion.

============================================================
NO FEATURE COUNTING
============================================================

Do not use a numerical feature-match score as the rating mechanism.

For example:
"8 out of 10 features match, therefore H"
is not valid FTO reasoning.

Instead evaluate:
- the patent claims;
- the complete claimed combination;
- the materiality of limitations;
- the core technical concept; and
- the strength of the product-to-claim correspondence.

============================================================
FTO VS LEGAL CONCLUSION
============================================================

This analysis identifies technical relevance and potential claim
correspondence.

Do not conclude that:
- the product infringes;
- the product does not infringe;
- the patent is valid;
- the patent is invalid;

Use technical screening language such as:
- technically relevant;
- potentially corresponding;
- substantial technical overlap;
- material distinction;

============================================================
CONFIDENCE
============================================================

Confidence represents certainty in the assigned rating, NOT the
strength of relevance.

Consider:

1. Is there sufficient evidence supporting the rating?
2. Is a neighboring rating reasonably possible?
3. Are important claim limitations ambiguous?
4. Is the product description incomplete?
5. Could another reasonable patent analyst disagree?

90–100:
Strong evidence and little reasonable disagreement.
75–89:
Well supported with some uncertainty.
50–74:
Plausible, but a neighboring rating is reasonably possible.
25–49:
Material ambiguity or incomplete evidence.
0–24:
Insufficient evidence to confidently distinguish the rating.

Confidence may be high or low for any rating.

============================================================
RATIONALE
============================================================

The rationale must be exactly 2–3 sentences.
It must be based on specific facts from the product and patent.

The rationale must:
- identify the relevant actual claim or claimed subject matter;
- identify the relevant product functionality;
- explain the technical correspondence;
- identify important material distinctions where supported;
- distinguish uncertainty from confirmed absence;

Avoid generic statements such as:

"The patent is relevant because it relates to similar technology."

Do not invent facts.

============================================================
IMPORTANT : CHARACTER AND ENCODING REQUIREMENTS
============================================================

The rationale MUST contain ASCII characters only.

Use only:
A-Z, a-z, 0-9, spaces, and:
. , : ; ' " - / ( ) [ ] + = % &

Allowed characters are ONLY:
A-Z, a-z, 0-9, space, and:
. , : ; ' " - / ( ) [ ] + = % &

Do NOT output:
- Greek or other Unicode characters
- Smart quotes or curly apostrophes
- En dash or em dash
- Superscripts or subscripts
- Mathematical symbols
- Unicode arrows
- HTML or Markdown
- Bullet characters
- Non-ASCII punctuation
- Any corrupted encoding such as:
  Î²
  â€™
  â€“
  Ã©
  Â°

IMPORTANT:
This requirement applies to EVERY generated text field, including:
- rationale
- reasoning
- rationale_framework_only
- reasoning_framework_only

If any character is outside the allowed ASCII set, replace it with the closest ASCII equivalent.

Before returning the rationale, verify that it contains ONLY ASCII
characters and exactly 2–3 sentences.
============================================================
REASONING INSTRUCTIONS:
============================================================

The reasoning field explains why the final relevance rating is M+ or L under the applicable rating framework.

If the final relevance rating is H:
- Return an empty string "" for the reasoning field.

If the final relevance rating is NR 
- Return an empty string "" for the reasoning field.

If the final relevance rating is M+ or L:
- Identify the specific actual claim limitation(s) or claim-related distinction that prevents the reference from receiving an H rating under the applicable rating framework.
- Explain specifically what is missing, not disclosed, uncertain, or materially different in the product compared with the relevant claim.
- Do not combine limitations from separate claims.
- Do not treat a limitation from a dependent claim as a limitation of the independent claim.
- Focus on mandatory claim limitations rather than merely optional or disclosed features.
- The reasoning must be specific to the actual patent claims and the product evidence.

For M+ or L, the reasoning must clearly answer:
"What specific claim limitation or claim-related distinction is responsible for this rating?"

"""
# ============================================================
# 5. FINAL ANALYSIS
# ============================================================

def final_analysis(state: RelevanceState):

    few_shot_examples = load_few_shot_examples()
    # The common FTO instructions + few-shot examples -- System Message

    common_fto_system_prompt = build_common_fto_system_prompt(
        few_shot_examples
    )
    # ========================================================
    # FRAMEWORK-ONLY FTO ANALYSIS
    # =======================================================
    
    framework_prompt = f"""
You are an experienced Patent Analyst.    
You are performing the FRAMEWORK-ONLY FTO RELEVANCE ANALYSIS.

The common FTO methodology, claim rules, confidence rules, and
few-shot calibration examples are already provided in the system
message. Apply them exactly.

For determining the rating, use the definitions of relevance ratings only provided by the user not the one specified in detailed_prompt section.
============================================================
USER-PROVIDED RELEVANCE FRAMEWORK
============================================================

The user has supplied the following relevance framework:

{state["relevance_framework"]}

After completing the FTO analysis, use this user-provided
framework to calibrate the final H / M+ / L / NR rating.

The user-provided framework determines the meaning and threshold of
H, M+, L, NR for this analysis.

Do not invent or use a different rating framework.

The relevance rating must be exactly one of:

H
M+
L
NR

============================================================
ANALYSIS INDEPENDENCE
============================================================

Perform this analysis independently as per the relevance framework provided by the user.

Perform the complete claim-centric FTO analysis required by the common
FTO methodology before assigning the Framework-Only rating.

# Do not downgrade H to M+ or L solely because a product limitation is
# not explicitly stated. Do not treat silence as affirmative absence.

Do not downgrade H merely because a minor, routine, contextual, or
implementation-specific product limitation is not explicitly stated.

============================================================
INPUTS
============================================================

PRODUCT DESCRIPTION:
{state["product_description"]}

PRIMARY PRODUCT FEATURES:
{state["primary_product_features"]}

SECONDARY PRODUCT FEATURES:
{state["secondary_product_features"]}

============================================================
PATENT
============================================================

Publication Number:
{state["publication_number"]}

Title:
{state["title"]}

Abstract:
{state["abstract"]}

INDEPENDENT CLAIMS:
{state["independent_claims"]}

ALL CLAIMS:
{state["all_claims"]}

PRIMARY PATENT FEATURES:
{state["primary_patent_features"]}

SECONDARY PATENT FEATURES:
{state["secondary_patent_features"]}

============================================================
ANALYSIS TASK
============================================================

Identify the strongest actual claim or claims relevant to the product.

Perform the complete claim analysis according to the FTO
methodology.

Then apply the USER-PROVIDED RELEVANCE FRAMEWORK to determine the
final rating.

The final rating must be:

H
M+
L
NR

============================================================
RATIONALE
============================================================

Write exactly 2–3 sentences.

The rationale must:
- identify the relevant actual claimed subject matter;
- identify the corresponding product functionality;
- explain the technical relationship;
- apply the user's relevance framework;
- identify the most important material distinction where supported;

The rationale must be specific to the actual product and patent.

Do not make a legal infringement, validity, enforceability, or final
FTO conclusion.

============================================================
OUTPUT
============================================================

Return:

1. relevance_framework_only
2. rationale_framework_only
3. confidence_framework_only
4. reasoning_framework_only

Return ONLY valid JSON.

{framework_parser.get_format_instructions()}
"""

    try:

        start = time.time()

        framework_response = llm.invoke(
            [
                SystemMessage(content=common_fto_system_prompt),
                HumanMessage(content=framework_prompt),
            ],
            prompt_cache_key="relevanceiq-fto-common-v1",
            prompt_cache_retention="24h",
        )

        print(
            "Framework only duration:",
            time.time() - start
        )

        try:
            cache_read = (
                framework_response.usage_metadata
                .get("input_token_details", {})
                .get("cache_read", 0)
            )
            print(
                "Framework analysis cached input tokens:",
                cache_read
            )
        except Exception:
            pass

        framework_result = framework_parser.parse(
            framework_response.content
        )

        # Keep the original Framework-Only result unchanged.
        state["relevance_framework_only"] = (
            framework_result.relevance_framework_only
        )

        state["rationale_framework_only"] = (
            framework_result.rationale_framework_only
        )

        state["confidence_framework_only"] = (
            framework_result.confidence_framework_only
        )

        state["reasoning_framework_only"] = (
            framework_result.reasoning_framework_only
        )

    except Exception as e:

        print("=" * 80)
        print("FRAMEWORK-ONLY PARSING ERROR")
        print(e)

        try:
            print(framework_response.content)
        except Exception:
            pass

        state["relevance_framework_only"] = "error"
        state["rationale_framework_only"] = ""
        state["confidence_framework_only"] = 0
        state["reasoning_framework_only"] = ""

    # ========================================================
    # DETAILED FTO ANALYSIS
    # ========================================================

    framework_only_rating = state.get(
        "relevance_framework_only",
        "error"
    )

    framework_only_rationale = state.get(
        "rationale_framework_only",
        ""
    )

    detailed_prompt = f"""

You are an experienced Patent Analyst.

You are performing the DETAILED FTO RELEVANCE ANALYSIS.

The common FTO methodology, claim rules, confidence rules, and
few-shot calibration examples are already provided in the system
message. Apply them exactly.

============================================================
TWO-STAGE RATING PROCESS
============================================================

This analysis is the SECOND STAGE of a two-stage process.

STAGE 1 - FRAMEWORK-ONLY ANALYSIS

A separate analysis first evaluated the patent using the
USER-PROVIDED RELEVANCE FRAMEWORK.

Its provisional result was:

Framework-Only Rating:
{framework_only_rating}

Framework-Only Rationale:
{framework_only_rationale}

For the detailed analyis assume the Framework-Only result as provisional.

STAGE 2 - DETAILED FTO VALIDATION

Your task is now to perform the detailed FTO analysis and
validate the Framework-Only rating against the actual patent claims
and product evidence.

Do NOT automatically agree with the Framework-Only rating.

Do NOT automatically downgrade it either.

Instead:

1. Perform the detailed FTO claim analysis.
2. Identify the strongest actual relevant claims.
3. Identify their mandatory technical limitations.
4. Compare those limitations against the product evidence.
5. Determine whether the Framework-Only rating is supported by the
   actual claim evidence.
6. If the Framework-Only rating is supported, KEEP THE SAME RATING.
7. If the Framework-Only rating is not supported because the detailed
   analysis establishes a material technical distinction or another
   claim-centric reason that changes the applicable rating, CORRECT
   THE FINAL DETAILED RATING.
8. The final detailed rating must be stored in the existing
   `relevance` field.
9. The original Framework-Only rating must remain unchanged in the
   existing `relevance_framework_only` field.

IMPORTANT:

The Framework-Only analysis is the starting rating.
The detailed FTO analysis is the validation/correction stage.

If the Framework-Only rating is technically supported after the
detailed analysis, the final `relevance` MUST equal the
Framework-Only rating.

Do NOT change the rating merely because:
- a claim contains additional details;
- a product detail is not mentioned;
- the product description is incomplete;
- terminology differs;
- an implementation detail is different;
- a routine/conventional feature differs;
- the Framework-Only rationale is less detailed than your analysis.

Silence is NOT affirmative absence if it is very obvious or routine.
============================================================
DETAILED RATING CALIBRATION
============================================================

For the final detailed rating, assign exactly one:

H
M+
L
NR

------------------------------------------------------------
H — HIGHLY RELEVANT
------------------------------------------------------------

Assign H when at least one actual patent claim is directed to the same
or substantially aligned core technical functionality as the product,
and there is no clearly established material limitation of at least one patent claim that the
product lacks, cannot satisfy, or is technically incompatible with.

H can still be appropriate when some claim details are:
- not explicitly disclosed;
- uncertain;
- described using different terminology;
- implementation-specific;
- contextual;
- routine technical details;
- not disclosed but are obvious in that context

Do not require the product description to reproduce every minor/ routine claim detail
before assigning H.

However, do not assign H when the product evidence clearly establishes
that a material limitation required by the relevant claim is absent or
technically incompatible.

# Added

An important limitation that is NOT DISCLOSED / UNCERTAIN may prevent H
when that limitation materially narrows the claimed technical solution.

------------------------------------------------------------
M+ — RELEVANT
------------------------------------------------------------

Assign M+ when there is substantial technical overlap between the
product and an actual patent claim, but the correspondence is not
strong enough for H under the applicable relevance standard.

This may occur when:
- a meaningful material claim limitation is affirmatively shown to
  be absent or technically different; or
- the core functionality substantially overlaps but an important
  claim limitation remains unresolved and prevents a high-confidence H
  classification under the applicable relevance standard.

Do NOT assign M+ merely because a limitation is not mentioned.

Do NOT treat uncertainty as confirmed absence.

------------------------------------------------------------
L — LESS RELEVANT
------------------------------------------------------------

Assign L when the patent's patent claims are not meaningfully directed
to the product's core technical functionality.

L is appropriate when:
- the claimed technical concept is fundamentally different;
- important material claim limitations clearly distinguish the claimed
  invention from the product;
- the strongest correspondence is only generic or peripheral; 
- the overlap is primarily at the level of field or purpose;
- the product does not technically correspond to the actual claimed
  combination of even at least one claim.
  
A limitation should support an L rating only when it is technically
meaningful and materially distinguishes the claimed solution.

Do not assign L solely because one or more product details are
undisclosed or uncertain or routine.

-------------------------------------------------------------
NR - Non Relevant
-------------------------------------------------------------

Assign NR only when the patent claims do not meaningfully correspond to any
primary technical feature or core technical functionality of the product.

If an actual claim shares a meaningful primary product functionality or
technical architecture but also contains material limitations that the
product does not establish, the rating should generally be M+ or L rather
than NR, depending on the degree of technical overlap.

NR should be reserved for cases where the strongest actual claims are
substantially unrelated to the product's primary technical functionality.
-------------------------------------------------------------

============================================================
INPUTS
============================================================

PRODUCT DESCRIPTION:
{state["product_description"]}

PRIMARY PRODUCT FEATURES:
{state["primary_product_features"]}

SECONDARY PRODUCT FEATURES:
{state["secondary_product_features"]}

============================================================
PATENT
============================================================

Publication Number:
{state["publication_number"]}

Title:
{state["title"]}

Abstract:
{state["abstract"]}

INDEPENDENT CLAIMS:
{state["independent_claims"]}

ALL CLAIMS:
{state["all_claims"]}

PRIMARY PATENT FEATURES:
{state["primary_patent_features"]}

SECONDARY PATENT FEATURES:
{state["secondary_patent_features"]}

============================================================
ANALYSIS TASK
============================================================

Identify the strongest 1–3 actual patent claims relevant to the product.

Analyze those claims according to the shared FTO methodology.

For the strongest relevant claims, identify the important technical
limitations and compare them against the product.

Return comparisons only for important limitations of the strongest
relevant claims.

Each comparison must contain:

- product_feature
- patent_feature
- match_type
- reasoning

MATCH TYPES:

Exact Match:
The product explicitly supports the patent limitation.

Equivalent Match:
The product uses a technically equivalent mechanism, function, and
result despite different terminology or implementation.

Partial Match:
Meaningful technical correspondence exists, but the particular
limitation is not fully established.

No Match:
The available product evidence affirmatively indicates that the
limitation is absent, incompatible, or materially different.

Do not use No Match merely because the product description is silent.

============================================================
FINAL RATING DECISION
============================================================

After completing the detailed analysis:

- If the Framework-Only rating is supported, set `relevance` to
  exactly the same rating.
- If the Framework-Only rating is not supported because of a material
  claim-supported distinction identified by the detailed analysis,
  correct `relevance`. The corrected relevance can be any one of H, M+, L, NR whatever the detailed analysis finds accurate (not just one rating below or above). 
- The final `relevance` must be exactly one of:

H
M+
L
NR

The `relevance_framework_only` field is NOT part of this output and
must NOT be overwritten by this analysis.

The rationale must explain the FINAL detailed rating, not merely repeat
the Framework-Only rationale.

============================================================
OUTPUT
============================================================

Return:

1. comparisons
2. relevance
3. rationale
4. confidence
5. reasoning

The relevance must be exactly one of:

H
M+
L
NR

The rationale must contain exactly 2–3 sentences.

Return ONLY valid JSON.

{detailed_parser.get_format_instructions()}
"""
    # ========================================================
    # CALL 2 — DETAILED ANALYSIS / VALIDATION
    # ========================================================

    try:

        start = time.time()

        detailed_response = llm.invoke(
            [
                SystemMessage(content=common_fto_system_prompt),
                HumanMessage(content=detailed_prompt),
            ],
            prompt_cache_key="relevanceiq-fto-common-v1",
            prompt_cache_retention="24h",
        )

        print(
            "Detailed analysis duration:",
            time.time() - start
        )

        try:
            cache_read = (
                detailed_response.usage_metadata
                .get("input_token_details", {})
                .get("cache_read", 0)
            )
            print(
                "Detailed analysis cached input tokens:",
                cache_read
            )
        except Exception:
            pass

        detailed_result = detailed_parser.parse(
            detailed_response.content
        )

        state["comparison"] = detailed_result.comparisons
        state["relevance"] = detailed_result.relevance
        state["rationale"] = detailed_result.rationale
        state["confidence"] = detailed_result.confidence
        state["reasoning"] = detailed_result.reasoning

    except Exception as e:

        print("=" * 80)
        print("DETAILED ANALYSIS PARSING ERROR")
        print(e)

        try:
            print(detailed_response.content)
        except Exception:
            pass

        state["comparison"] = []
        state["relevance"] = "error"
        state["rationale"] = "error"
        state["confidence"] = 0
        state["reasoning"] = ""

    return state



# ============================================================
# 6. AI GENERATED RECOMMENDATIONS & ANALYTICS
# ============================================================

# ============================================================
# 6.0 REQUIRED IMPORTS
# ============================================================

import re
import json
import time

import numpy as np
import matplotlib.pyplot as plt

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


# ============================================================
# 6.1 AI RECOMMENDATION OUTPUT SCHEMAS
# ============================================================


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
    Normalize product feature names for aggregation.
    """

    feature = str(feature).strip().lower()

    feature = re.sub(r"\s+", " ", feature)

    return feature


# ============================================================
# 6.3 PATENT PORTFOLIO ANALYTICS
# ============================================================


def calculate_patent_analytics(patent_results):

    """
    Calculate patent-level and product-feature-level analytics.

    Supported relevance ratings:

        H
        M+
        L
        NR

    Duplicate publication numbers are counted only once.

    Feature-level counts may overlap because one patent
    can correspond to multiple product features.
    """

    ratings = ["H", "M+", "L", "NR"]

    rating_counts = {
        rating: 0 for rating in ratings
    }

    unique_patents = {}

    feature_data = defaultdict(
        lambda: {
            "H": set(),
            "M+": set(),
            "L": set(),
            "NR": set(),
            "patent_details": {}
        }
    )

    # --------------------------------------------------------
    # STEP 1: COLLECT UNIQUE PATENTS
    # --------------------------------------------------------

    for record in patent_results:

        publication_number = str(
            _get_value(
                record,
                "publication_number",
                ""
            )
        ).strip()

        if not publication_number:

            continue

        relevance = str(
            _get_value(
                record,
                "relevance",
                "NR"
            )
        ).strip().upper()

        if relevance not in ratings:

            continue

        # Avoid duplicate publication counting.
        unique_patents[publication_number] = {

            "publication_number": publication_number,

            "title": _get_value(
                record,
                "title",
                ""
            ),

            "relevance": relevance,

            "confidence": _get_value(
                record,
                "confidence",
                0
            ),

            "rationale": _get_value(
                record,
                "rationale",
                ""
            ),

            "reasoning": _get_value(
                record,
                "reasoning",
                ""
            )
        }

    # --------------------------------------------------------
    # STEP 2: COUNT UNIQUE PATENTS BY RATING
    # --------------------------------------------------------

    for patent in unique_patents.values():

        rating_counts[
            patent["relevance"]
        ] += 1

    total_patents = len(unique_patents)

    # --------------------------------------------------------
    # STEP 3: PRODUCT FEATURE-WISE PATENT CONCENTRATION
    # --------------------------------------------------------

    for record in patent_results:

        publication_number = str(
            _get_value(
                record,
                "publication_number",
                ""
            )
        ).strip()

        if publication_number not in unique_patents:

            continue

        rating = unique_patents[
            publication_number
        ]["relevance"]

        comparisons = _get_value(
            record,
            "comparison",
            []
        ) or []

        for comparison in comparisons:

            product_feature = _get_value(
                comparison,
                "product_feature",
                ""
            )

            match_type = _get_value(
                comparison,
                "match_type",
                ""
            )

            if not product_feature:

                continue

            # No Match is not evidence of correspondence.
            if match_type == "No Match":

                continue

            normalized = _normalize_feature(
                product_feature
            )

            if not normalized:

                continue

            feature_data[normalized][rating].add(
                publication_number
            )

            feature_data[normalized]["patent_details"][
                publication_number
            ] = {

                "product_feature": str(
                    product_feature
                ),

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

    # --------------------------------------------------------
    # STEP 4: CALCULATE FEATURE ANALYTICS
    # --------------------------------------------------------

    feature_analytics = []

    for normalized, data in feature_data.items():

        h_count = len(data["H"])

        mplus_count = len(data["M+"])

        l_count = len(data["L"])

        nr_count = len(data["NR"])

        high_relevance_count = (
            h_count + mplus_count
        )

        total_feature_patents = len(
            set().union(
                data["H"],
                data["M+"],
                data["L"],
                data["NR"]
            )
        )

        portfolio_concentration = (

            high_relevance_count / total_patents * 100

            if total_patents else 0

        )

        h_share = (

            h_count / total_feature_patents * 100

            if total_feature_patents else 0

        )

        feature_analytics.append({

            "product_feature": normalized,

            "H": h_count,

            "M+": mplus_count,

            "L": l_count,

            "NR": nr_count,

            "total_patents": total_feature_patents,

            "high_relevance_count": high_relevance_count,

            "portfolio_concentration": round(
                portfolio_concentration,
                2
            ),

            "h_share": round(
                h_share,
                2
            ),

            "supporting_publication_numbers": sorted(
                data["H"] | data["M+"]
            ),

            "patent_details": data["patent_details"]

        })

    # --------------------------------------------------------
    # STEP 5: SORT FEATURES
    # --------------------------------------------------------

    feature_analytics.sort(

        key=lambda x: (

            x["high_relevance_count"],

            x["H"],

            x["M+"]

        ),

        reverse=True

    )

    # --------------------------------------------------------
    # STEP 6: OVERALL RATING PERCENTAGES
    # --------------------------------------------------------

    rating_percentages = {

        rating: round(

            rating_counts[rating] / total_patents * 100,

            2

        ) if total_patents else 0

        for rating in ratings

    }

    # --------------------------------------------------------
    # STEP 7: HIGH-PRIORITY PATENT COUNT
    # --------------------------------------------------------

    high_priority_patents = (

        rating_counts["H"]

        + rating_counts["M+"]

    )

    high_priority_percentage = (

        high_priority_patents / total_patents * 100

        if total_patents else 0

    )

    # --------------------------------------------------------
    # RETURN ANALYTICS
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

        "feature_analytics": feature_analytics,

        "patent_records": list(
            unique_patents.values()
        )

    }


# ============================================================
# 6.4 GENERATE AI GENERATED RECOMMENDATIONS USING LLM
# ============================================================


def generate_ai_recommendations(
    patent_results,
    product_description
):

    """
    Generate consolidated AI recommendations.

    The LLM receives:

        1. Product description
        2. Overall patent distribution
        3. Feature-wise patent concentration
        4. Individual patent analysis results

    Returns:

        analytics
        recommendations
    """

    analytics = calculate_patent_analytics(
        patent_results
    )

    if analytics["total_patents"] == 0:

        raise ValueError(
            "No valid completed patent analysis results "
            "were provided."
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

    patent_summary = [

        {

            "publication_number": patent["publication_number"],

            "title": patent["title"],

            "rating": patent["relevance"],

            "confidence": patent["confidence"],

            "rationale": patent["rationale"],

            "reasoning": patent["reasoning"]

        }

        for patent in analytics["patent_records"]

    ]

    # --------------------------------------------------------
    # AI PROMPT
    # --------------------------------------------------------

    recommendation_prompt = f"""

You are a senior patent portfolio analyst conducting
a technical FTO screening assessment.

Your task is to analyse the COMPLETE patent dataset
against the supplied product description.

This is a portfolio-level assessment and not an
individual patent recommendation exercise.

============================================================
PRODUCT DESCRIPTION
============================================================

{product_description}


============================================================
OVERALL PATENT STATISTICS
============================================================

Total unique patents:

{analytics["total_patents"]}


Rating distribution:

{json.dumps(analytics["rating_counts"], indent=2)}


Rating percentages:

{json.dumps(analytics["rating_percentages"], indent=2)}


Total H + M+ patents:

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

1. Identify product features associated with the
   largest concentration of H patents.

2. Identify features with significant M+ concentration.

3. Identify features attracting both H and M+ patents.

4. Identify features predominantly associated with
   L and NR patents.

5. Identify recurring technical combinations appearing
   across multiple H or M+ patents.

6. Explain which product features deserve the highest
   priority for further patent review.

7. Recommend areas where alternative technical
   implementations could be investigated.

8. Identify product features requiring additional
   technical documentation to resolve uncertainty.

9. Distinguish patent concentration from infringement risk.
   A higher patent count does not establish infringement.

10. Do not invent technical alternatives or claim limitations.
    Suggested design alternatives must be framed as areas
    for investigation, not confirmed workarounds.

11. Use actual publication numbers when identifying
    supporting patents.

12. Never count duplicate publication numbers twice.

13. Feature-level patent counts may overlap because one
    patent may correspond to multiple product features.

14. Do not infer that L or NR patents have no technical
    relationship whatsoever.

15. Consider actual claim correspondence and confidence,
    not patent counts alone.

============================================================
RECOMMENDATION PRIORITY
============================================================

Highest:

Features with the strongest H patent concentration
and meaningful claim correspondence.

High:

Features with substantial H + M+ concentration.

Moderate:

Features with limited H/M+ concentration or material
claim-related uncertainty.

Lower:

Features predominantly associated with L or NR patents.

============================================================
IMPORTANT LIMITATIONS
============================================================

This report is a technical portfolio screening assessment.

Do not state that:

- The product infringes.
- The product is free from infringement.
- A design alternative guarantees avoidance.
- The supplied portfolio represents all existing patents.

============================================================
OUTPUT REQUIREMENTS
============================================================

Generate:

1. Executive portfolio summary.

   Keep it crisp and concise, approximately 3-5 sentences.

2. Highest-priority product features.

3. Lower-priority product features.

4. Actionable portfolio recommendations.

5. Potential design alternatives for investigation.

6. Additional investigation areas.

7. Overall portfolio conclusion.

Keep feature-level analysis concise and evidence-based.

Return valid JSON only.

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
                    "technical recommendations."

                )

            ),

            HumanMessage(
                content=recommendation_prompt
            )

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
# 6.5 WORD DOCUMENT FORMATTING HELPERS
# ============================================================


def _set_cell_background(cell, color):

    shading = OxmlElement("w:shd")

    shading.set(
        qn("w:fill"),
        color
    )

    cell._tc.get_or_add_tcPr().append(
        shading
    )


def _format_table_header(table):

    for cell in table.rows[0].cells:

        _set_cell_background(
            cell,
            "071426"
        )

        for paragraph in cell.paragraphs:

            for run in paragraph.runs:

                run.font.bold = True

                run.font.color.rgb = RGBColor(
                    255,
                    255,
                    255
                )


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

    table.alignment = (
        WD_TABLE_ALIGNMENT.CENTER
    )

    for index, header in enumerate(headers):

        table.rows[0].cells[index].text = str(
            header
        )

    _format_table_header(table)

    for row in rows:

        cells = table.add_row().cells

        for index, value in enumerate(row):

            cells[index].text = str(
                value
            )

            cells[index].vertical_alignment = (

                WD_CELL_VERTICAL_ALIGNMENT.CENTER

            )

    return table


# ============================================================
# 6.6 GENERATE PATENT ANALYTICS GRAPHS
# ============================================================


def _create_recommendation_charts(analytics):

    """
    Generate graphs for:

        1. H / M+ / L / NR patent distribution.
        2. Feature-wise patent concentration.

    Returns:
        Dictionary containing BytesIO chart images.
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
        "Number of Patents"
    )

    ax.set_ylim(
        0,
        max(values, default=0) * 1.15 + 1
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

    features = analytics["feature_analytics"][:10]

    if features:

        names = [

            item["product_feature"]

            for item in features

        ]

        h_values = [

            item["H"]

            for item in features

        ]

        mplus_values = [

            item["M+"]

            for item in features

        ]

        l_values = [

            item["L"]

            for item in features

        ]

        nr_values = [

            item["NR"]

            for item in features

        ]

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        y = np.arange(
            len(names)
        )

        ax.barh(
            y,
            h_values,
            label="H"
        )

        ax.barh(

            y,

            mplus_values,

            left=h_values,

            label="M+"

        )

        left_l = (

            np.array(h_values)

            + np.array(mplus_values)

        )

        ax.barh(

            y,

            l_values,

            left=left_l,

            label="L"

        )

        left_nr = (

            left_l

            + np.array(l_values)

        )

        ax.barh(

            y,

            nr_values,

            left=left_nr,

            label="NR"

        )

        ax.set_yticks(y)

        ax.set_yticklabels(names)

        ax.invert_yaxis()

        ax.set_xlabel(
            "Number of Associated Patents"
        )

        ax.set_title(
            "Top 10 Product Features - Patent Concentration"
        )

        ax.legend()

        fig.tight_layout()

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
# 6.7 AI GENERATED RECOMMENDATIONS WORD REPORT
# ============================================================


def generate_ai_recommendations_report(

    patent_results,

    product_description,

    analytics=None,

    recommendations=None

):

    """
    Generate a downloadable Word document containing:

        1. Executive Summary
        2. Patent Distribution
        3. Patent Analytics Graphs
        4. Feature-wise Patent Concentration
        5. Highest-Priority Features
        6. Lower-Priority Features
        7. AI Recommendations
        8. Potential Design Alternatives
        9. Additional Investigation Areas
        10. Conclusion
        11. Assessment Limitations

    If analytics and recommendations are already supplied,
    no additional LLM call is made.
    """

    # --------------------------------------------------------
    # GENERATE RECOMMENDATIONS IF NOT PROVIDED
    # --------------------------------------------------------

    if analytics is None or recommendations is None:

        analytics, recommendations = (
            generate_ai_recommendations(

                patent_results,

                product_description

            )
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

    styles["Title"].font.color.rgb = RGBColor(
        7,
        20,
        38
    )

    styles["Heading 1"].font.color.rgb = RGBColor(
        7,
        20,
        38
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    title = document.add_heading(

        "RelevanceIQ",

        level=0

    )

    title.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    subtitle = document.add_paragraph(

        "AI Generated Recommendations"

    )

    subtitle.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

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

        f'Total unique patents analysed: '

        f'{analytics["total_patents"]}'

    )

    document.add_paragraph(

        f'H + M+ patents: '

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
    # 4. FEATURE-WISE PATENT CONCENTRATION
    # --------------------------------------------------------

    document.add_heading(

        "4. Product Feature-Wise Patent Concentration",

        level=1

    )

    document.add_paragraph(

        "Feature-level counts represent distinct publication "
        "numbers associated with each product feature. "
        "A patent may appear under multiple product features. "
        "Therefore, feature-level counts may overlap."

    )

    feature_rows = []

    for feature in analytics["feature_analytics"]:

        feature_rows.append([

            feature["product_feature"],

            feature["H"],

            feature["M+"],

            feature["L"],

            feature["NR"],

            feature["high_relevance_count"],

            f'{feature["portfolio_concentration"]}%'

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

                "H + M+",

                "Portfolio %"

            ],

            feature_rows

        )

    else:

        document.add_paragraph(

            "No usable feature-level comparison data was available."

        )

    # --------------------------------------------------------
    # 5. HIGHEST-PRIORITY FEATURES
    # --------------------------------------------------------

    document.add_heading(

        "5. Highest-Priority Product Features",

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

            "No highest-priority features were identified."

        )

    # --------------------------------------------------------
    # 6. LOWER-PRIORITY FEATURES
    # --------------------------------------------------------

    document.add_heading(

        "6. Lower-Priority Product Features",

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

            "No lower-priority features were identified."

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

        for alternative in (
            recommendations.potential_design_alternatives
        ):

            document.add_paragraph(

                alternative,

                style="List Bullet"

            )

    else:

        document.add_paragraph(

            "No potential design alternatives were identified."

        )

    # --------------------------------------------------------
    # 9. ADDITIONAL INVESTIGATION AREAS
    # --------------------------------------------------------

    document.add_heading(

        "9. Additional Investigation Areas",

        level=1

    )

    if recommendations.additional_investigation_areas:

        for investigation in (
            recommendations.additional_investigation_areas
        ):

            document.add_paragraph(

                investigation,

                style="List Bullet"

            )

    else:

        document.add_paragraph(

            "No additional investigation areas were identified."

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
        "publication numbers. Feature-level counts may overlap "
        "because one patent can correspond to multiple product "
        "features. H and M+ classifications indicate screening "
        "priority and technical correspondence, not confirmed "
        "infringement. The analysis is limited to the supplied "
        "product description, patent records, claims and "
        "portfolio coverage. This report does not establish "
        "freedom to operate or constitute a legal opinion."

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




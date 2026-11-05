from typing import TypedDict

class RelevanceState(TypedDict) :
    # product description
    product_description : str

    # input from thr excel sheet rows
    publication_number  : str
    title   : str
    abstract  : str
    independent_claim : str
    all_claims    : str

    relevance_human : str

    # Extracted primary and secondary features
    primary_product_features : list[str]
    secondary_product_features : list[str]

    
    primary_patent_features : list[str]
    secondary_patent_features : list[str]

    # Extracted features
    # product_features : list[str]
    # patent_features : list[str]

    # User-provided framework
    relevance_framework : str

    #Comparison
    comparison : list

    #Final outputs (feature comparison method)
    relevance : str
    confidence : int
    rationale  : str
    reasoning   : str

    #Final outputs (framework-only method)
    relevance_framework_only: str
    confidence_framework_only: int
    reasoning_framework_only : str
    rationale_framework_only: str

    current_patent_index: int
    review_type: str



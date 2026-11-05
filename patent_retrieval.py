from __future__ import annotations
import textwrap
import os
import re
from typing import Any, Callable, Optional
import numpy as np
import templates as ui
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the project .env file so OPENAI_API_KEY is available to this app.
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# st.set_page_config(
#     page_title="RelevanceIQ | Patent Search",
#     page_icon="🔎",
#     layout="wide",
# )

PUBLICATION_COL = "Publication Number"
TITLE_COL = "Title"
ABSTRACT_COL = "Abstract"
DESCRIPTION_COL = "Description"
CLAIMS_COL = "All Claims"
EMBEDDING_MODEL = "text-embedding-3-small"
# Added for sending result to relevance iq
INDEPENDENT_CLAIMS_COL = "Independent Claims"

EMBEDDING_CHUNK_CHARS = 3_000
QUERY_MAX_CHARS = 6_000
EMBEDDING_BATCH_SIZE = 64

# A progress callback receives (fraction between 0 and 1, status message).
ProgressCallback = Callable[[float, str], None]

COLUMN_ALIASES = {
    PUBLICATION_COL: [
        "publication number", "publication_number",   "Publication_Number" ,
         "Publication Number"
    ],

    TITLE_COL: ["title", "patent title", "Title" , "invention title"],
    ABSTRACT_COL: ["abstract", "patent abstract" , "Abstract"],
    DESCRIPTION_COL: [
         "description", "Description" , "patent description", "detailed description",
        "full description", "Full Description"
    ],

    INDEPENDENT_CLAIMS_COL : [
         "Independent_Claims" , "Independent_Claim" , "Independent_claim" , "Independent_claims"
    ]
       ,
    CLAIMS_COL: [
           
            "Claims" , "All Claims" , "claims" , "All claims"
        ],
}

# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def render_html(content):
    """
    Safely render custom HTML.

    textwrap.dedent() prevents Streamlit Markdown
    from interpreting indented HTML as a code block.
    """

    content = textwrap.dedent(content)

    if hasattr(st, "html"):
        st.html(content)
    else:
        st.markdown(
            content,
            unsafe_allow_html=True
        )

def _report(callback: Optional[ProgressCallback], fraction: float, message: str) -> None:
    """Send a progress update if a callback was provided."""
    if callback is not None:
        callback(min(max(float(fraction), 0.0), 1.0), message)

def _scaled(
    callback: Optional[ProgressCallback], start: float, end: float
) -> Optional[ProgressCallback]:
    """Map a sub-task's 0..1 progress onto the [start, end] slice of the full run."""
    if callback is None:
        return None

    def inner(fraction: float, message: str) -> None:
        callback(start + (end - start) * min(max(float(fraction), 0.0), 1.0), message)

    return inner

def make_dataframe_serializable(df: pd.DataFrame) -> pd.DataFrame:
    """Convert NumPy/Pandas values into MessagePack-friendly Python values."""

    df = df.copy()

    # Convert NumPy integer and float columns to native Python-compatible values
    for column in df.columns:
        if pd.api.types.is_integer_dtype(df[column]):
            df[column] = df[column].astype(object)

        elif pd.api.types.is_float_dtype(df[column]):
            df[column] = df[column].astype(object)

    # Replace NaN and NaT with None
    df = df.astype(object).where(pd.notna(df), None)

    return df

def normalize_header(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()

def prepare_patents_dataframe(source_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the uploaded patent dataset and create searchable text."""
    if source_df is None or source_df.empty:
        raise ValueError("The uploaded patent dataset is empty.")

    df = source_df.copy()

    lookup = {normalize_header(column): column for column in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        source_column = next(
            (column for column in df.columns
             if normalize_header(column) == normalize_header(canonical)),
            None,
        )
        if source_column is None:
            for alias in aliases:
                matched = lookup.get(normalize_header(alias))
                if matched is not None:
                    source_column = matched
                    break

        if source_column is None:
            # Abstract and Description are display/context fields only.
            
            if canonical in {ABSTRACT_COL, DESCRIPTION_COL}:
                df[canonical] = ""
            else:
                raise ValueError(
                    f"Required column '{canonical}' was not found. "
                    f"Accepted alternatives: {', '.join(aliases)}"
                )
        elif source_column != canonical:
            df[canonical] = df[source_column]

    for column in [
        PUBLICATION_COL, TITLE_COL, ABSTRACT_COL, DESCRIPTION_COL, CLAIMS_COL, INDEPENDENT_CLAIMS_COL
    ]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["search_text"] = (
         
            df[ABSTRACT_COL]
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                    +
                    " "
                    +
             df[CLAIMS_COL]
                            .str.replace(r"\s+", " ", regex=True)
                            .str.strip() 
    )

    df = df[df["search_text"].str.len() > 0].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset contains no searchable patent records.")

    missing_ids = df[PUBLICATION_COL].eq("")

    if missing_ids.any():
        missing_count = int(missing_ids.sum())
        
        df.loc[missing_ids, PUBLICATION_COL] = [
            f"Unknown publication #{index + 1}" for index in range(missing_count)
        ]
    return df

def get_ranks(scores: list[float] | np.ndarray) -> list[int]:
    """Convert scores to descending, one-based ranks."""
    values = np.asarray(scores, dtype=float)
    ordering = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[ordering] = np.arange(1, len(values) + 1)
    return ranks.tolist()

def tfidf_search(query: str, documents: list[str]) -> list[float]:
    corpus = [query] + documents
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features= 300_000 , 
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    return cosine_similarity(matrix[0], matrix[1:]).ravel().tolist()

def get_openai_api_key() -> str | None:
    """Read the key from Streamlit secrets first, then environment variables."""
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("OPENAI_API_KEY")

def split_for_embeddings(text: str, max_chars: int = EMBEDDING_CHUNK_CHARS) -> list[str]:
    """Split long patent text into overlapping, API-safe embedding chunks."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    # Small overlap helps avoid splitting a technical phrase at a boundary.
    overlap = 200
    step = max_chars - overlap
    chunks = [text[start:start + max_chars] for start in range(0, len(text), step)]
    return [chunk for chunk in chunks if chunk.strip()]




def get_document_embeddings(
    documents: tuple[str, ...] | list[str],
    model_name: str,
    api_key: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed patent chunks once; return chunk vectors and their patent indexes."""
    if OpenAI is None:
        raise RuntimeError("OpenAI package is missing")


    chunks: list[str] = []
    patent_indexes: list[int] = []
    for patent_index, document in enumerate(documents):
        for chunk in split_for_embeddings(document):
            chunks.append(chunk)
            patent_indexes.append(patent_index)

    total_chunks = len(chunks)
    _report(
        progress_callback, 0.0,
        f"Embedding patent text: 0 of {total_chunks:,} sections"
    )

    client = OpenAI(api_key=api_key)
    vectors: list[list[float] | None] = [None] * total_chunks

    for start in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
        batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=model_name, input=batch)
        for item in response.data:
            vectors[start + item.index] = item.embedding

        done = min(start + len(batch), total_chunks)
        _report(
            progress_callback,
            done / total_chunks,
            f"Embedding patent text: {done:,} of {total_chunks:,} sections",
        )

    if any(vector is None for vector in vectors):
        raise RuntimeError("The embedding API did not return an embedding for every patent chunk.")

    result = (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(patent_indexes, dtype=np.int32),
    )


    return result

def semantic_search(
    query: str,
    documents: list[str],
    api_key: str,
    model_name: str = EMBEDDING_MODEL,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[float]:
    """Score each patent by its most semantically similar text chunk."""
    if OpenAI is None:
        raise RuntimeError("OpenAI package is missing")

    _report(progress_callback, 0.0, "Embedding the product description...")
    client = OpenAI(api_key=api_key)
    safe_query = str(query).strip()[:QUERY_MAX_CHARS]
    query_response = client.embeddings.create(model=model_name, input=safe_query)
    query_vector = np.asarray(query_response.data[0].embedding, dtype=np.float32)

    chunk_vectors, patent_indexes = get_document_embeddings(
        tuple(documents), model_name, api_key,
        progress_callback=_scaled(progress_callback, 0.05, 0.95),
    )

    _report(progress_callback, 0.95, "Scoring patents by semantic similarity...")

    # Cosine similarity between the query and every chunk.
    query_norm = np.linalg.norm(query_vector)
    chunk_norms = np.linalg.norm(chunk_vectors, axis=1)
    denominator = chunk_norms * query_norm
    denominator[denominator == 0] = 1.0
    chunk_scores = (chunk_vectors @ query_vector) / denominator

    # material found deep in a long description while returning one score per patent.
    patent_scores = np.full(len(documents), -1.0, dtype=np.float32)
    for chunk_index, patent_index in enumerate(patent_indexes):
        patent_scores[patent_index] = max(
            patent_scores[patent_index], float(chunk_scores[chunk_index])
        )

    _report(progress_callback, 1.0, "Semantic scoring complete")
    return patent_scores.astype(float).tolist()

def retrieve_patents(
    source_df: pd.DataFrame,
    product_description: str,
    similarity_threshold: float = 0.02,
    semantic_threshold: float = 0.30,
    use_openai_embeddings: bool = True,
    rrf_k: int = 60,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Retrieve candidates if they qualify via TF-IDF or semantic score."""
    query = str(product_description or "").strip()
    if not query:
        raise ValueError("Please enter a product description.")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("TF-IDF threshold must be between 0 and 1.")
    if not 0.0 <= semantic_threshold <= 1.0:
        raise ValueError("Semantic threshold must be between 0 and 1.")

    _report(progress_callback, 0.0, "Preparing the patent dataset...")
    df = prepare_patents_dataframe(source_df)
    documents = df["search_text"].tolist()
    _report(
        progress_callback, 0.03,
        f"Dataset ready: {len(df):,} searchable patents. Running TF-IDF search..."
    )

    tfidf_scores = tfidf_search(query, documents)
    df["TF-IDF Similarity"] = tfidf_scores
    rankings: dict[str, list[float]] = {"TF-IDF": tfidf_scores}
    _report(progress_callback, 0.20, "TF-IDF search complete")

    semantic_scores: list[float] | None = None
    if use_openai_embeddings:
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError(
                "OpenAI API key not found"
                
            )
        
        semantic_scores = semantic_search(
            query, documents, api_key,
            progress_callback=_scaled(progress_callback, 0.20, 0.90),
        )
        df["Semantic Similarity"] = semantic_scores
        rankings["Semantic"] = semantic_scores

    _report(progress_callback, 0.90, "Applying thresholds and ranking results...")

    # Combined retrieval rule: a patent qualifies through ANY enabled method.
    qualifying_mask = df["TF-IDF Similarity"] >= similarity_threshold
    if semantic_scores is not None:
        qualifying_mask |= df["Semantic Similarity"] >= semantic_threshold

    df["Search Score"] = 0.0
    for method_name, scores in rankings.items():
        ranks = get_ranks(scores)
        df[f"{method_name} Rank"] = ranks
        df["Search Score"] += [1.0 / (rrf_k + rank) for rank in ranks]

    candidates = df.loc[qualifying_mask].copy()
    candidates = candidates.sort_values(
        ["Search Score", "TF-IDF Similarity"], ascending=False
    ).reset_index(drop=True)
    candidates["Search Rank"] = range(1, len(candidates) + 1)

    metadata = {
        "methods_used": list(rankings.keys()),
        "threshold_method": "TF-IDF OR semantic similarity",
        "similarity_threshold": similarity_threshold,
        "semantic_threshold": semantic_threshold if semantic_scores is not None else None,
        "semantic_search_used": semantic_scores is not None,
        "embedding_model": EMBEDDING_MODEL if semantic_scores is not None else None,
        "cross_encoder_used": False,
        "corpus_size": len(df),
        "candidates_returned": len(candidates),
    }
    _report(progress_callback, 1.0, "Finalizing results...")
    return candidates, metadata

# Rename main to render_patent_retrieval
def render_patent_retrieval() -> None:
    st.title("Patent Search")
    st.caption(
        "Search the product description. This tool retrieves the relevant patent candidates corresponding to a product description"
    )
    st.divider()

    
    # ============================================================
    # INFORMATION AND TEMPLATE DOWNLOAD
    # ============================================================

    col1, col2 = st.columns([3, 1])

    with col1:
        render_html(ui.input_info())

    with col2:

        template_path = "Patent_Retrieval_Input_Data_Schema.xlsx"

        if os.path.exists(template_path):

            with open(template_path, "rb") as f:

                st.download_button(
                    "Download Template",
                    f.read(),
                    "Patent_Retrieval_Input_Data_Schema.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="download_patent_search_template"
                )

        else:
            st.warning("Patent Search template file not found.")

    # ============================================================
    # PATENT SEARCH FORM
    # ============================================================

    with st.form("patent_search_form"):

        product_description = st.text_area(
            "Product Description",
            height=220,
            placeholder=(
                "Describe the product, its technical operation, components, "
                "and intended functionality..."
            ),
        )

        uploaded_file = st.file_uploader(
            "Upload Patent Corpus (.xlsx)",
            type=["xlsx"],
            help=(
                "Required columns: Publication Number, Title, "
                "Abstract, Description and Claims."
            ),
        )

        similarity_threshold = 0.02
        semantic_threshold = 0.33

        submitted = st.form_submit_button(
            "Search Patents",
            type="primary",
            use_container_width=True
        )

    # ============================================================
    # PATENT SEARCH EXECUTION
    # ============================================================

    if submitted:

        if not product_description.strip():

            st.error("Please enter a product description.")

        elif uploaded_file is None:

            st.error("Please upload the patent corpus Excel file.")

        else:

            progress_slot = st.empty()

            progress_bar = progress_slot.progress(
                0.0,
                text="Starting patent search..."
            )

            def update_progress(fraction: float, message: str) -> None:

                progress_bar.progress(
                    min(max(float(fraction), 0.0), 1.0),
                    text=(
                        f"{int(min(max(fraction, 0.0), 1.0) * 100)}% | "
                        f"{message}"
                    ),
                )

            try:

                update_progress(
                    0.0,
                    "Reading the uploaded patent file (large files can take a while)..."
                )

                source_df = pd.read_excel(
                    uploaded_file,
                    engine="openpyxl"
                )

                update_progress(
                    0.05,
                    f"File loaded: {len(source_df):,} rows"
                )

                candidates, metadata = retrieve_patents(
                    source_df=source_df,
                    product_description=product_description,
                    similarity_threshold=similarity_threshold,
                    semantic_threshold=semantic_threshold,
                    progress_callback=_scaled(update_progress, 0.05, 1.0),
                )

                # Convert retrieved output into serialization-friendly values
                candidates = make_dataframe_serializable(candidates)

                st.session_state["patent_search_candidates"] = candidates
                st.session_state["patent_search_metadata"] = metadata

                st.session_state[
                    "patent_search_product_description"
                ] = product_description

                st.session_state[
                    "patent_search_corpus_name"
                ] = uploaded_file.name

                update_progress(1.0, "Search complete")

                st.success(
                    f"Retrieved {len(candidates)} qualifying patents from "
                    f"{metadata['corpus_size']} searchable records."
                )

            except Exception as exc:

                progress_slot.empty()

                st.error(f"Patent search failed: {exc}")

    candidates = st.session_state.get("patent_search_candidates")
    metadata = st.session_state.get("patent_search_metadata")
    if candidates is not None and metadata is not None:
        st.divider()
        st.subheader("Retrieved Patent Candidates")
        semantic_status = (
            f"OpenAI embeddings ({metadata['embedding_model']})"
            if metadata["semantic_search_used"] else "OpenAI embeddings disabled"
        )
        st.caption(
            f"Corpus: {st.session_state.get('patent_search_corpus_name', 'Uploaded file')} | "
            f"Records searched: {metadata['corpus_size']} | "
            f"Inclusion rule: {metadata['threshold_method']} | "
            f"TF-IDF threshold: {metadata['similarity_threshold']:.2f} | "
            f"Semantic: {semantic_status} | "
            f"Ranking methods: {', '.join(metadata['methods_used'])}"
        )
        st.metric("Qualifying patents", len(candidates))

        if candidates.empty:
            st.warning(
                "No patents met the selected retrieval rule"
            )
        else:
            preferred_columns = [
                PUBLICATION_COL, TITLE_COL, ABSTRACT_COL, CLAIMS_COL,
    INDEPENDENT_CLAIMS_COL, 
                "TF-IDF Similarity", "TF-IDF Rank",
                "Semantic Similarity", "Semantic Rank", "Search Rank", "Search Score",
            ]
            display_columns = [
                column for column in preferred_columns if column in candidates.columns
            ]
            st.dataframe(
                candidates[display_columns],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Search Rank": st.column_config.NumberColumn("Search Rank", format="%d"),
                    "Search Score": st.column_config.NumberColumn("RRF Search Score", format="%.6f"),
                    "TF-IDF Similarity": st.column_config.NumberColumn("TF-IDF Similarity", format="%.4f"),
                    "Semantic Similarity": st.column_config.NumberColumn("Semantic Similarity", format="%.4f"),
                },
            )
            csv_data = candidates[display_columns].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download All Retrieved Patents (CSV)",
                data=csv_data,
                file_name="RelevanceIQ_Patent_Search_Results.csv",
                mime="text/csv",
            )

            with st.expander("Inspect Patent Details"):
                for _, row in candidates.iterrows():
                    publication_number = row.get(PUBLICATION_COL, "Unknown publication")
                    title = row.get(TITLE_COL, "Untitled patent")
                    with st.expander(f"{publication_number} — {title}"):
                        st.markdown("**Abstract**")
                        st.write(row.get(ABSTRACT_COL, "") or "No abstract provided.")
                        st.markdown("**Independent Claims Col**")
                        st.write(row.get(INDEPENDENT_CLAIMS_COL, "") or "No Independent Claims provided.")
                        # st.markdown("**Description**")
                        # st.write(row.get(DESCRIPTION_COL, "") or "No description provided.")
                        st.markdown("**Claims**")
                        st.write(row.get(CLAIMS_COL, "") or "No Claims provided.")

        st.divider()
        st.subheader("Next Step")

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "Download Retrieved Patents",
                data=candidates.to_csv(index=False).encode("utf-8-sig"),
                file_name="Retrieved_Patents.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            
            if st.button(
                    "Continue to RelevanceIQ →",
                    type="primary",
                    use_container_width=True
                ):
                    st.session_state["analysis_source"] = "retrieval"
                    st.session_state["navigate_to_relevanceiq"] = True
                    st.session_state["page"] = "Analysis"
                    st.rerun()

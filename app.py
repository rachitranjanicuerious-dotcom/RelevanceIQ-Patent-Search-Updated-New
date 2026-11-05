# import streamlit as st
# import sqlite3
# import uuid 
# import os
# import textwrap
# import pandas as pd
# from langgraph.types import Command
# from langchain_core.messages import HumanMessage
# from node import llm, feature_parser
# from graph import workflow
# from database import create_table, save_result
# import templates as ui
# import numpy as np
# import pandas as pd
# from patent_retrieval import render_patent_retrieval

# # ============================================================
# # PAGE CONFIG
# # ============================================================

# st.set_page_config(
#     page_title="RelevanceIQ | iCuerious",
#     page_icon="Icon.png",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )


# def make_serializable(value):
#     """Recursively convert NumPy/Pandas values to msgpack-safe Python values."""

#     if isinstance(value, dict):
#         return {
#             make_serializable(key): make_serializable(val)
#             for key, val in value.items()
#         }

#     if isinstance(value, (list, tuple, set)):
#         return [make_serializable(item) for item in value]

#     if isinstance(value, np.ndarray):
#         return make_serializable(value.tolist())

#     if isinstance(value, np.generic):
#         return make_serializable(value.item())

#     if value is pd.NA or value is pd.NaT:
#         return None

#     if isinstance(value, (pd.Timestamp,)):
#         return value.isoformat()

#     # Convert pandas/numpy missing scalar values to None.
#     try:
#         if pd.isna(value):
#             return None
#     except (TypeError, ValueError):
#         pass

#     # Leave only ordinary msgpack-friendly scalar types.
#     if isinstance(value, (str, int, float, bool)) or value is None:
#         return value

#     return str(value)


# def normalize_dataframe_for_state(dataframe):
#     """Make every DataFrame cell safe before storing/passing it to LangGraph."""
#     dataframe = dataframe.copy().astype(object)
#     dataframe = dataframe.where(pd.notna(dataframe), None)

#     for column in dataframe.columns:
#         dataframe[column] = dataframe[column].map(make_serializable)

#     return dataframe

# def prepare_retrieval_for_relevanceiq(retrieved_df):

#     df = retrieved_df.copy()

#     # Map All Claims
#     if "All Claims" not in df.columns:
#         if "Claims" in df.columns:
#             df["All Claims"] = df["Claims"]
#         else:
#             df["All Claims"] = ""

#     # Map Independent Claims
#     if "Independent Claim" not in df.columns:
#         if "Independent Claims" in df.columns:
#             df["Independent Claim"] = df["Independent Claims"]
#         else:
#             df["Independent Claim"] = ""

#     # Human relevance is unavailable for newly retrieved patents
#     if "Relevance Human" not in df.columns:
#         df["Relevance Human"] = ""

#     return df

# # ============================================================
# # HTML HELPER
# # ============================================================

# def render_html(content):
#     """
#     Safely render custom HTML.

#     textwrap.dedent() prevents Streamlit Markdown
#     from interpreting indented HTML as a code block.
#     """

#     content = textwrap.dedent(content)

#     if hasattr(st, "html"):
#         st.html(content)
#     else:
#         st.markdown(
#             content,
#             unsafe_allow_html=True
#         )


# def count_rating(df, column, value):
#     return len(df[df[column] == value])


# def render_cards(items):
#     """Render a row of info cards from (title, value) pairs."""

#     columns = st.columns(len(items))

#     for column, (title, value) in zip(columns, items):
#         with column:
#             render_html(ui.info_card(title, value))


# # ============================================================
# # CUSTOM CSS  (styles.css)
# # ============================================================

# render_html(ui.load_css())

# # ============================================================
# # LOGIN
# # ============================================================

# def login():

#     render_html(ui.login_card())

#     username = st.text_input(
#         "Username",
#         placeholder="Enter username",
#         key="login_username"
#     )
#     password = st.text_input(
#         "Password",
#         type="password",
#         placeholder="Enter password",
#         key="login_password"
#     )
#     if st.button(
#         "Login",
#         use_container_width=True,
#         key="login_button"
#     ):

#         users = st.secrets["users"]

#         if username in users and password == users[username]:

#             st.session_state.logged_in = True

#             st.session_state.username = username

#             st.rerun()

#         else:

#             st.error(
#                 "Invalid username or password"
#             )


# # ============================================================
# # SESSION INITIALIZATION
# # ============================================================

# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False


# if not st.session_state.logged_in:

#     login()

#     st.stop()


# if "analysis_started" not in st.session_state:
#     st.session_state.analysis_started = False


# if "df" not in st.session_state:
#     st.session_state.df = None


# if "current_index" not in st.session_state:
#     st.session_state.current_index = 0


# if "results" not in st.session_state:
#     st.session_state.results = []

# if "product_description" not in st.session_state:
#     st.session_state.product_description = ""

# if "relevance_framework" not in st.session_state:
#     st.session_state.relevance_framework = ""

# if "primary_product_features" not in st.session_state:
#     st.session_state.primary_product_features = []

# if "secondary_product_features" not in st.session_state:
#     st.session_state.secondary_product_features = []

# if "product_features_approved" not in st.session_state:
#     st.session_state.product_features_approved = False

# if "analysis_error" not in st.session_state:
#     st.session_state.analysis_error = False

# if "active_tool" not in st.session_state:
#     st.session_state.active_tool = "Patent Retrieval"    

# # ============================================================
# # DATABASE
# # ============================================================

# create_table()

# # ============================================================
# # SIDEBAR
# # ============================================================

# with st.sidebar:

#     render_html(ui.sidebar_brand())

#     render_html(ui.sidebar_section("Tools"))

#     # Handle navigation from Patent Retrieval to RelevanceIQ
#     if st.session_state.pop("navigate_to_relevanceiq", False):
#         st.session_state["active_tool"] = "RelevanceIQ"

#     active_tool = st.radio(
#         "Select Tool",
#         [
#             "Patent Retrieval",
#             "RelevanceIQ"
#         ],
#         key = "active_tool" ,
#         label_visibility = "collapsed"
#     )

#     if active_tool == "RelevanceIQ":
#         render_html(ui.sidebar_section("Application"))

#         page = st.radio(
#             "Navigation",
#             [
#                 "Analysis",
#                 "Results",
#                 "Database"
#             ],
#             key = "page" , 
#             label_visibility="collapsed"
#         )

#     else:
#       page = None

#     # render_html(ui.sidebar_tool())

#     render_html(ui.sidebar_user(st.session_state.username))

#     # st.markdown(
#     #     "<br>",
#     #     unsafe_allow_html=True
#     # )

#     if st.button(
#         "Logout",
#         use_container_width=True,
#         key="logout_button"
#     ):

#         st.session_state.logged_in = False

#         st.session_state.pop(
#             "username",
#             None
#         )
#         st.rerun()

# # ============================================================
# # TOOL ROUTING
# # ============================================================

# if active_tool == "Patent Retrieval": 
#     render_patent_retrieval()

#     st.stop()        
    
# # ============================================================
# # TOP HEADER
# # ============================================================

# render_html(ui.top_header())

# # ============================================================
# # DATABASE PAGE
# # ============================================================

# if page == "Database":

#     render_html(
#         ui.hero(
#             "RelevanceIQ",
#             "Analysis Database",
#             "Review previously generated patent relevance "
#             "analyses stored by the application."
#         )
#     )

#     conn = sqlite3.connect(
#         "RelevanceIQ.db"
#     )

#     try:

#         database_df = pd.read_sql_query(
#             """
#             SELECT *
#             FROM FTO_results
#             ORDER BY id DESC
#             """,
#             conn
#         )

#     except Exception as e:

#         database_df = pd.DataFrame()

#         st.error(
#             f"Unable to load database records: {e}"
#         )

#     finally:

#         conn.close()

#     if not database_df.empty:

#         # Number of records
#         st.write(
#             f"**{len(database_df)} records found**"
#         )

#         st.dataframe(
#             database_df,
#             use_container_width=True,
#             hide_index=True
#         )

#         st.download_button(
#             "⬇ Download Database",
#             database_df.to_csv(index=False),
#             "RelevanceIQ_Database.csv",
#             "text/csv",
#             key="download_database"
#         )

#     else:

#         st.info(
#             "No database records are available."
#         )

#     st.stop()

# # ============================================================
# # RESULTS PAGE
# # ============================================================

# if page == "Results":

#     render_html(
#         ui.hero(
#             "RelevanceIQ",
#             "Analysis Results",
#             "Patent relevance results generated from the "
#             "current analysis session."
#         )
#     )

#     if st.session_state.results:

#         result_df = pd.DataFrame(
#             st.session_state.results
#         )

#         # ----------------------------------------------------
#         # MAIN METRICS
#         # ----------------------------------------------------

#         render_cards([
#             ("Patents Analysed", len(result_df)),
#             ("Highly Relevant", count_rating(result_df, "Relevance", "H")),
#             ("M+", count_rating(result_df, "Relevance", "M+")),
#             ("L", count_rating(result_df, "Relevance", "L")),
#             ("NR", count_rating(result_df, "Relevance", "NR")),
#         ])

#         # ----------------------------------------------------
#         # FRAMEWORK-ONLY SUMMARY
#         # ----------------------------------------------------

#         render_html(
#             ui.section(
#                 "F",
#                 "Framework-Only Classification",
#                 "Classification based specifically on the "
#                 "user-provided relevance framework."
#             )
#         )

#         render_cards([
#             ("H", count_rating(result_df, "Relevance_Framework_Only", "H")),
#             ("M+", count_rating(result_df, "Relevance_Framework_Only", "M+")),
#             ("L", count_rating(result_df, "Relevance_Framework_Only", "L")),
#             ("NR", count_rating(result_df, "Relevance_Framework_Only", "NR")),
#         ])

#         # ====================================================
#         # RESULTS TABLE
#         # ====================================================

#         render_html(
#             ui.section(
#                 "R",
#                 "Common Relevance Framework Results"
#             )
#         )

#         st.dataframe(
#             result_df,
#             use_container_width=True,
#             hide_index=True
#         )

#         # ====================================================
#         # DOWNLOAD
#         # ====================================================

#         st.download_button(
#             "Download Results",
#             result_df.to_csv(index=False),
#             "RelevanceIQ_Results.csv",
#             "text/csv",
#             key="download_results"
#         )

#     else:

#         st.info(
#             "No analysis results are available yet. "
#             "Run an analysis first."
#         )

#     st.stop()

# # ============================================================
# # ANALYSIS PAGE
# # ============================================================

# render_html(
#     ui.hero(
#         "Patent Intelligence • FTO",
#         "Patent Relevance Analysis",
#         "Evaluate patents against a target product using "
#         "a defined relevance framework and human-reviewed "
#         "technical feature extraction."
#     )
# )

# # ============================================================
# # INPUT SECTION
# # ============================================================

# render_html(
#     ui.section(
#         1,
#         "Target Product",
#         "Describe the technical product that you want to analyze."
#     )
# )


# product_description = st.text_area(
#     "Product Description",

#     value=st.session_state.product_description,

#     height=180,

#     placeholder=(
#         "Enter the technical description of the target product..."
#     ),

#     label_visibility="collapsed",

#     key="product_description_input"
# )

# # ============================================================
# # PATENT FILE
# # ============================================================

# render_html(
#     ui.section(
#         2,
#         "Patent Dataset",
#         "Upload the Excel file containing the patents to be evaluated."
#     )
# )

# analysis_source = st.session_state.get(
#     "analysis_source",
#     "independent"
# )

# if analysis_source == 'retrieval':

#     uploaded_df = prepare_retrieval_for_relevanceiq(
#         st.session_state["patent_search_candidates"]
#     )

#     product_description = st.session_state[
#         "patent_search_product_description"
#     ]

#     st.success(
#         f"{len(uploaded_df)} retrieved patents loaded for RelevanceIQ."
#     )

# else:    
#     # Keep your existing independent Excel upload

#     uploaded_file = st.file_uploader(
#         "Upload Patent Excel File",
#         type=["xlsx"],
#         label_visibility="collapsed",
#         key="patent_file"
#     )


# col1, col2 = st.columns([3, 1])

# with col1:

#     render_html(ui.input_info())

# with col2:

#     template_path = (
#         "RelevanceIQ_Input_Data_Schema.xlsx"
#     )

#     if os.path.exists(template_path):

#         with open(
#             template_path,
#             "rb"
#         ) as f:

#             st.download_button(
#                 "Download Template",

#                 f.read(),

#                 "RelevanceIQ_Input_Data_Schema.xlsx",
#                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

#                 use_container_width=True,
#                 key="download_template"
#             )

# # ============================================================
# # FRAMEWORK
# # ============================================================

# render_html(
#     ui.section(
#         3,
#         "Relevance Framework",
#         "Define the criteria that will be used to determine "
#         "patent relevance."
#     )
# )


# relevance_framework = st.text_area(
#     "Relevance Framework",

#     value=st.session_state.relevance_framework,

#     height=220,

#     placeholder=(
#         "Enter the relevance framework used to classify "
#         "patents as H, M+, L or NR..."
#     ),

#     label_visibility="collapsed",

#     key="relevance_framework_input"
# )


# # ============================================================
# # RUN ANALYSIS
# # ============================================================

# st.markdown(
#     "<br>",
#     unsafe_allow_html=True
# )


# run_col1, run_col2, run_col3 = st.columns(
#     [1, 2, 1]
# )


# with run_col2:

#     run = st.button(
#         "Run Relevance Analysis",

#         use_container_width=True,

#         key="run_analysis_button"
#     )

# if run:
#     # --------------------------------------------------------
#     # VALIDATION
#     # --------------------------------------------------------

#     if analysis_source != "retrieval" and uploaded_file is None:
#         st.warning(
#             "Please upload an Excel file containing the patents."
#         )
#         st.stop()


#     if not product_description.strip():
#         st.warning(
#             "Please enter the product description."
#         )
#         st.stop()

#     if not relevance_framework.strip():
#         st.warning(
#             "Please provide the relevance framework."
#         )
#         st.stop()


#     # --------------------------------------------------------
#     # READ EXCEL
#     # --------------------------------------------------------

#     if analysis_source == "retrieval":

#         uploaded_df =  prepare_retrieval_for_relevanceiq(
#             st.session_state["patent_search_candidates"]
#         )
        
#     else:
        
#         try:
#             # uploaded_df = pd.read_excel(
#             #     uploaded_file

#             uploaded_df = pd.read_excel(uploaded_file)

#             # Normalize all values before processing patents
#             uploaded_df = uploaded_df.astype(object)
#             uploaded_df = uploaded_df.where(pd.notna(uploaded_df), None
#             )

#         except Exception as e:

#             st.error(
#                 f"Unable to read the Excel file: {e}"
#             )
#             st.stop()

#     # Convert numpy/pandas scalar values (for example numpy.int64) before
#     # saving the dataset in Streamlit session state or sending rows to LangGraph.
#     uploaded_df = normalize_dataframe_for_state(uploaded_df)

#     if uploaded_df.empty:

#         st.warning(
#             "The uploaded Excel file does not contain "
#             "any patent rows."
#         )
#         st.stop()

#     # --------------------------------------------------------
#     # VALIDATE REQUIRED COLUMNS
#     # --------------------------------------------------------

#     required_columns = [
#         "Publication Number",
#         "Title",  
#         "Abstract",
#         "Independent Claim",
#         "All Claims",
#         "Relevance Human"
#     ]


#     missing_columns = [
#         column

#         for column in required_columns

#         if column not in uploaded_df.columns
#     ]


#     if missing_columns:

#         st.error(
#             "The uploaded Excel file is missing the following "
#             f"required columns: {', '.join(missing_columns)}"
#         )

#         st.stop()

#     # --------------------------------------------------------
#     # SAVE SESSION DATA
#     # --------------------------------------------------------

#     st.session_state.df = uploaded_df


#     st.session_state.product_description = (
#         product_description
#     )


#     st.session_state.relevance_framework = (
#         relevance_framework
#     )

#     st.session_state.results = []

#     st.session_state.current_index = 0

#     st.session_state.primary_product_features = []

#     st.session_state.secondary_product_features = []

#     st.session_state.product_features_approved = False

#     st.session_state.analysis_started = True

#     st.session_state.session_id = str(
#         uuid.uuid4()
#     )

#     st.rerun()

# # ============================================================
# # PRODUCT FEATURE EXTRACTION
# # ============================================================

# if (
#     st.session_state.analysis_started
#     and not st.session_state.product_features_approved
# ):

#     # --------------------------------------------------------
#     # EXTRACT FEATURES
#     # --------------------------------------------------------

#     if (
#         not st.session_state.primary_product_features
#         and not st.session_state.secondary_product_features
#     ):

#         render_html(
#             ui.status_card(
#                 "Extracting Product Features",
#                 "RelevanceIQ is identifying the primary "
#                 "and secondary technical features of the "
#                 "target product for human review."
#             )
#         )


#         product_prompt = f"""
# You are an experienced patent analyst performing an FTO
# (Freedom-to-Operate) technical analysis.

# Extract the technical features explicitly disclosed in the
# product description.

# PRIMARY FEATURES:
# - Core technical features and mechanisms.
# - Features important for satisfying potential patent claim
#   limitations.

# SECONDARY FEATURES:
# - Supporting, auxiliary, optional, control, communication,
#   interface, or peripheral features.

# Rules:
# - Extract only explicitly disclosed features.
# - Do not infer or invent features.
# - Keep features concise and technically specific.

# {feature_parser.get_format_instructions()}

# Product Description:
# {st.session_state.product_description}
# """


#         response = llm.invoke(
#             [
#                 HumanMessage(
#                     content=product_prompt
#                 )
#             ]
#         )


#         result = feature_parser.parse(
#             response.content
#         )


#         st.session_state.primary_product_features = (
#             result.primary_features
#         )


#         st.session_state.secondary_product_features = (
#             result.secondary_features
#         )

#         st.rerun()

#     # --------------------------------------------------------
#     # HUMAN REVIEW
#     # --------------------------------------------------------

#     render_html(
#         ui.section(
#             "✓",
#             "Review Product Features",
#             "Review and edit the extracted features before "
#             "continuing with patent analysis."
#         )
#     )


#     col1, col2 = st.columns(2)


#     with col1:

#         edited_primary = st.text_area(
#             "Primary Product Features",

#             value="\n".join(
#                 st.session_state.primary_product_features
#             ),

#             height=250,

#             key="product_primary_review"
#         )


#     with col2:

#         edited_secondary = st.text_area(
#             "Secondary Product Features",

#             value="\n".join(
#                 st.session_state.secondary_product_features
#             ),

#             height=250,

#             key="product_secondary_review"
#         )


#     approve_col1, approve_col2, approve_col3 = st.columns(
#         [1, 2, 1]
#     )


#     with approve_col2:

#         if st.button(
#             "Approve Product Features",

#             use_container_width=True,

#             key="approve_product_features"
#         ):

#             st.session_state.primary_product_features = [

#                 x.strip()

#                 for x in edited_primary.split("\n")

#                 if x.strip()
#             ]


#             st.session_state.secondary_product_features = [

#                 x.strip()

#                 for x in edited_secondary.split("\n")

#                 if x.strip()
#             ]


#             st.session_state.product_features_approved = True


#             st.rerun()


#     st.stop()


# # ============================================================
# # PATENT PROCESSING
# # ============================================================

# if st.session_state.analysis_started:

#     df = st.session_state.df

#     index = st.session_state.current_index


#     # ========================================================
#     # COMPLETED
#     # ========================================================

#     if index >= len(df):

#         render_html(
#             ui.hero(
#                 "Analysis Complete",
#                 "Patent Analysis Completed",
#                 "The selected patent dataset has been "
#                 "processed successfully."
#             )
#         )


#         result_df = pd.DataFrame(
#             st.session_state.results
#         )


#         # ====================================================
#         # SUMMARY METRICS
#         # ====================================================

#         render_cards([
#             ("Common Relevance Framework Results", len(result_df)),
#             ("H", count_rating(result_df, "Relevance", "H")),
#             ("M+", count_rating(result_df, "Relevance", "M+")),
#             ("L", count_rating(result_df, "Relevance", "L")),
#             ("NR", count_rating(result_df, "Relevance", "NR")),
#         ])


#         # ====================================================
#         # FRAMEWORK ONLY SUMMARY
#         # ====================================================

#         render_html(
#             ui.section(
#                 "F",
#                 "Framework-Only Results",
#                 "Classification based only on the "
#                 "user-provided relevance framework."
#             )
#         )

#         render_cards([
#             ("H", count_rating(result_df, "Relevance_Framework_Only", "H")),
#             ("M+", count_rating(result_df, "Relevance_Framework_Only", "M+")),
#             ("L", count_rating(result_df, "Relevance_Framework_Only", "L")),
#             ("NR", count_rating(result_df, "Relevance_Framework_Only", "NR")),
#         ])


#         # ====================================================
#         # RESULTS
#         # ====================================================

#         render_html(
#             ui.section(
#                 "R",
#                 "Results"
#             )
#         )


#         st.dataframe(
#             result_df,

#             use_container_width=True,
#             hide_index=True
#         )

#         # ====================================================
#         # DOWNLOAD
#         # ====================================================

#         st.download_button(
#             "Download Results",

#             result_df.to_csv(index=False),

#             "RelevanceIQ_Results.csv",

#             "text/csv",

#             key="download_final_results"
#         )


#     # ========================================================
#     # PROCESS CURRENT PATENT
#     # ========================================================

#     else:

#         try:
#             progress_value = (
#                 index / len(df)
#             )


#             st.progress(
#                 progress_value
#             )


#             render_html(
#                 ui.status_card(
#                     f"Patent {index + 1} of {len(df)}",
#                     "RelevanceIQ is processing the current patent."
#                 )
#             )


#             row = df.iloc[index]
#             row_data = make_serializable(row.to_dict())


#             config = {
#                 "configurable": {
#                     "thread_id":
#                     f"{st.session_state.session_id}_patent_{index}"
#                 }
#             }


#             snapshot = workflow.get_state(
#                 config
#             )


#             # ====================================================
#             # FIRST EXECUTION
#             # ====================================================

#             if not snapshot or not snapshot.values:

#                 state = {

#                     "product_description":
#                         st.session_state.product_description,

#                     "publication_number":
#                         str(row_data.get("Publication Number", "")),

#                     "title":
#                         row_data.get("Title", ""),

#                     "abstract":
#                         row_data.get("Abstract", ""),

#                     "independent_claim":
#                         row_data.get("Independent Claim", ""),

#                     "all_claims":
#                         row_data.get("All Claims", ""),

#                     "relevance_human":
#                         row_data.get("Relevance Human", ""),

#                     "relevance_framework":
#                         st.session_state.relevance_framework,

#                     "primary_product_features":
#                         st.session_state.primary_product_features,

#                     "secondary_product_features":
#                         st.session_state.secondary_product_features,

#                     "primary_patent_features":
#                         [],

#                     "secondary_patent_features":
#                         [],

#                     "comparison":
#                         [],

#                     "relevance":
#                         "",

#                     "confidence":
#                         0,

#                     "rationale":
#                         "",

#                     "reasoning":
#                         "",

#                     "relevance_framework_only":
#                         "",

#                     "confidence_framework_only":
#                         0,

#                     "rationale_framework_only":
#                         "",

#                     "reasoning_framework_only":
#                         "",

#                     "review_type":
#                         ""
#                 }


#                 # Final safety pass: LangGraph's checkpointer must receive
#                 # only native Python values, never numpy.int64/float32 scalars.
#                 state = make_serializable(state)

#                 workflow.invoke(
#                     state,
#                     config=config
#                 )


#                 snapshot = workflow.get_state(
#                     config
#                 )


#             # ====================================================
#             # HUMAN INTERRUPT
#             # ====================================================

#             if snapshot.interrupts:

#                 review = snapshot.interrupts[0].value


#                 if review["review_type"] == "product":

#                     render_html(
#                         ui.section(
#                             "!",
#                             "Human Review Required",
#                             "Review the extracted product features "
#                             "before patent relevance analysis continues."
#                         )
#                     )


#                     col1, col2 = st.columns(2)


#                     with col1:

#                         edited_primary = st.text_area(
#                             "Primary Product Features",

#                             value="\n".join(
#                                 review["primary_features"]
#                             ),

#                             height=230,

#                             key=f"primary_{index}_product"
#                         )


#                     with col2:

#                         edited_secondary = st.text_area(
#                             "Secondary Product Features",

#                             value="\n".join(
#                                 review["secondary_features"]
#                             ),

#                             height=230,

#                             key=f"secondary_{index}_product"
#                         )


#                     if st.button(
#                         "Approve & Continue",

#                         key=f"approve_{index}_product",

#                         use_container_width=True
#                     ):

#                         approved_features = {

#                             "primary_features": [

#                                 x.strip()

#                                 for x in edited_primary.split("\n")

#                                 if x.strip()
#                             ],

#                             "secondary_features": [

#                                 x.strip()

#                                 for x in edited_secondary.split("\n")

#                                 if x.strip()
#                             ]
#                         }


#                         approved_features = make_serializable(approved_features)

#                         workflow.invoke(
#                             Command(
#                                 resume=approved_features
#                             ),

#                             config=config
#                         )


#                         st.rerun()


#             # ====================================================
#             # CONTINUE WORKFLOW
#             # ====================================================

#             elif snapshot.next:

#                 st.info(
#                     "Continuing patent analysis..."
#                 )

#                 st.rerun()


#             # ====================================================
#             # WORKFLOW FINISHED
#             # ====================================================

#             else:

#                 result = make_serializable(snapshot.values)


#                 save_result(
#                     result
#                 )

#                 patent_result = {

#                     "Publication Number":
#                         result["publication_number"],

#                     "Title":
#                         result["title"],

#                     "Relevance Human":
#                         result["relevance_human"],   

#                     "Patent Primary Features":
#                         result["primary_patent_features"],

#                     "Patent Secondary Features":
#                         result["secondary_patent_features"],

#                     "Relevance":
#                         result["relevance"],

#                     "Confidence":
#                         result["confidence"],

#                     "Rationale":
#                         result["rationale"],

#                     "Reasoning":
#                         result["reasoning"],

#                     "Relevance_Framework_Only":
#                         result["relevance_framework_only"],

#                     "Confidence_Framework_Only":
#                         result["confidence_framework_only"],

#                     "Rationale_Framework_Only":
#                         result["rationale_framework_only"],

#                     "Reasoning_Framework_Only":
#                         result["reasoning_framework_only"]

#                 }

#                 st.session_state.results.append(make_serializable(patent_result))


#                 st.session_state.current_index += 1


#                 st.rerun()


#         except Exception as e:

#             st.session_state.analysis_started = False
#             st.session_state.analysis_error = True

#             st.error(
#                 f"Analysis stopped at Patent {index + 1}: "
#                 f"{row['Publication Number']}"
#             )

#             st.warning(
#                 f"Error: {str(e)}"
#             )

#             st.info(
#                 "Correct the issue and click Resume Analysis "
#                 "to continue from this patent."
#             )

# if st.session_state.get("analysis_error", False):

#     if st.button(
#         "▶ Resume Analysis",
#         use_container_width=True,
#         key="resume_analysis"
#     ):

#         st.session_state.analysis_error = False
#         st.session_state.analysis_started = True

#         st.rerun()


# # ============================================================
# # FOOTER
# # ============================================================

# render_html(ui.footer())


import streamlit as st
import sqlite3
import uuid
import os
import textwrap
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import HumanMessage
from node import (
    llm,
    feature_parser,
    generate_recommendations_report
)
from graph import workflow
from database import create_table, save_result
import templates as ui
import numpy as np
from patent_retrieval import render_patent_retrieval


# ============================================================
# PARALLEL PROCESSING CONFIGURATION
# ============================================================

MAX_WORKERS = 5


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RelevanceIQ | iCuerious",
    page_icon="Icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


def make_serializable(value):
    """Recursively convert NumPy/Pandas values to msgpack-safe Python values."""

    if isinstance(value, dict):
        return {
            make_serializable(key): make_serializable(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_serializable(item) for item in value]

    if isinstance(value, np.ndarray):
        return make_serializable(value.tolist())

    if isinstance(value, np.generic):
        return make_serializable(value.item())

    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def normalize_dataframe_for_state(dataframe):
    """Make every DataFrame cell safe before storing/passing it to LangGraph."""

    dataframe = dataframe.copy().astype(object)
    dataframe = dataframe.where(pd.notna(dataframe), None)

    for column in dataframe.columns:
        dataframe[column] = dataframe[column].map(make_serializable)

    return dataframe


def prepare_retrieval_for_relevanceiq(retrieved_df):

    df = retrieved_df.copy()

    # Map All Claims
    if "All Claims" not in df.columns:
        if "Claims" in df.columns:
            df["All Claims"] = df["Claims"]
        else:
            df["All Claims"] = ""

    # Map Independent Claims
    if "Independent Claim" not in df.columns:
        if "Independent Claims" in df.columns:
            df["Independent Claim"] = df["Independent Claims"]
        else:
            df["Independent Claim"] = ""

    # Human relevance is unavailable for newly retrieved patents
    if "Relevance Human" not in df.columns:
        df["Relevance Human"] = ""

    return df


# ============================================================
# PARALLEL PATENT PROCESSING FUNCTION
# ============================================================

def process_single_patent(
    index,
    row_data,
    product_description,
    relevance_framework,
    primary_product_features,
    secondary_product_features
):

    state = {

        "product_description":
            product_description,

        "publication_number":
            str(row_data.get("Publication Number", "")),

        "title":
            row_data.get("Title", ""),

        "abstract":
            row_data.get("Abstract", ""),

        "independent_claim":
            row_data.get("Independent Claim", ""),

        "all_claims":
            row_data.get("All Claims", ""),

        "relevance_human":
            row_data.get("Relevance Human", ""),

        "relevance_framework":
            relevance_framework,

        "primary_product_features":
            primary_product_features,

        "secondary_product_features":
            secondary_product_features,

        "primary_patent_features":
            [],

        "secondary_patent_features":
            [],

        "comparison":
            [],

        "relevance":
            "",

        "confidence":
            0,

        "rationale":
            "",

        "reasoning":
            "",

        "relevance_framework_only":
            "",

        "confidence_framework_only":
            0,

        "rationale_framework_only":
            "",

        "reasoning_framework_only":
            "",

        "review_type":
            ""
    }

    state = make_serializable(state)

    # Process one patent independently.
    result = workflow.invoke(state)

    return make_serializable(result)


# ============================================================
# HTML HELPER
# ============================================================

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


def count_rating(df, column, value):
    return len(df[df[column] == value])


def render_cards(items):
    """Render a row of info cards from (title, value) pairs."""

    columns = st.columns(len(items))

    for column, (title, value) in zip(columns, items):
        with column:
            render_html(ui.info_card(title, value))


# ============================================================
# CUSTOM CSS  (styles.css)
# ============================================================

render_html(ui.load_css())


# ============================================================
# LOGIN
# ============================================================

def login():

    render_html(ui.login_card())

    username = st.text_input(
        "Username",
        placeholder="Enter username",
        key="login_username"
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password",
        key="login_password"
    )

    if st.button(
        "Login",
        use_container_width=True,
        key="login_button"
    ):

        users = st.secrets["users"]

        if username in users and password == users[username]:

            st.session_state.logged_in = True

            st.session_state.username = username

            st.rerun()

        else:

            st.error(
                "Invalid username or password"
            )


# ============================================================
# SESSION INITIALIZATION
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:

    login()

    st.stop()


if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False


if "df" not in st.session_state:
    st.session_state.df = None


if "current_index" not in st.session_state:
    st.session_state.current_index = 0


if "results" not in st.session_state:
    st.session_state.results = []


# Results stored using original patent indices.
if "results_by_index" not in st.session_state:
    st.session_state.results_by_index = {}


# Failed patents stored using original patent indices.
if "failed_patents" not in st.session_state:
    st.session_state.failed_patents = {}


# COMPLETE PATENT RESULTS FOR PORTFOLIO ANALYSIS.
if "portfolio_results_by_index" not in st.session_state:
    st.session_state.portfolio_results_by_index = {}


# GENERATED WORD REPORT.
if "portfolio_recommendations_report" not in st.session_state:
    st.session_state.portfolio_recommendations_report = None


if "product_description" not in st.session_state:
    st.session_state.product_description = ""


if "relevance_framework" not in st.session_state:
    st.session_state.relevance_framework = ""


if "primary_product_features" not in st.session_state:
    st.session_state.primary_product_features = []


if "secondary_product_features" not in st.session_state:
    st.session_state.secondary_product_features = []


if "product_features_approved" not in st.session_state:
    st.session_state.product_features_approved = False


if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = False


if "active_tool" not in st.session_state:
    st.session_state.active_tool = "Patent Retrieval"


# ============================================================
# DATABASE
# ============================================================

create_table()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(ui.sidebar_brand())

    render_html(ui.sidebar_section("Tools"))

    # Handle navigation from Patent Retrieval to RelevanceIQ
    if st.session_state.pop("navigate_to_relevanceiq", False):
        st.session_state["active_tool"] = "RelevanceIQ"

    active_tool = st.radio(
        "Select Tool",
        [
            "Patent Retrieval",
            "RelevanceIQ"
        ],
        key="active_tool",
        label_visibility="collapsed"
    )

    if active_tool == "RelevanceIQ":

        render_html(ui.sidebar_section("Application"))

        page = st.radio(
            "Navigation",
            [
                "Analysis",
                "Results",
                "Database"
            ],
            key="page",
            label_visibility="collapsed"
        )

    else:

        page = None

    render_html(ui.sidebar_user(st.session_state.username))

    if st.button(
        "Logout",
        use_container_width=True,
        key="logout_button"
    ):

        st.session_state.logged_in = False

        st.session_state.pop(
            "username",
            None
        )

        st.rerun()


# ============================================================
# TOOL ROUTING
# ============================================================

if active_tool == "Patent Retrieval":

    render_patent_retrieval()

    st.stop()


# ============================================================
# TOP HEADER
# ============================================================

render_html(ui.top_header())


# ============================================================
# DATABASE PAGE
# ============================================================

if page == "Database":

    render_html(
        ui.hero(
            "RelevanceIQ",
            "Analysis Database",
            "Review previously generated patent relevance "
            "analyses stored by the application."
        )
    )

    conn = sqlite3.connect(
        "RelevanceIQ.db"
    )

    try:

        database_df = pd.read_sql_query(
            """
            SELECT *
            FROM FTO_results
            ORDER BY id DESC
            """,
            conn
        )

    except Exception as e:

        database_df = pd.DataFrame()

        st.error(
            f"Unable to load database records: {e}"
        )

    finally:

        conn.close()

    if not database_df.empty:

        st.write(
            f"**{len(database_df)} records found**"
        )

        st.dataframe(
            database_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download Database",
            database_df.to_csv(index=False),
            "RelevanceIQ_Database.csv",
            "text/csv",
            key="download_database"
        )

    else:

        st.info(
            "No database records are available."
        )

    st.stop()


# ============================================================
# RESULTS PAGE
# ============================================================

if page == "Results":

    render_html(
        ui.hero(
            "RelevanceIQ",
            "Analysis Results",
            "Patent relevance results generated from the "
            "current analysis session."
        )
    )

    if st.session_state.results:

        result_df = pd.DataFrame(
            st.session_state.results
        )

        # ----------------------------------------------------
        # MAIN METRICS
        # ----------------------------------------------------

        render_cards([
            ("Patents Analysed", len(result_df)),
            ("Highly Relevant", count_rating(result_df, "Relevance", "H")),
            ("M+", count_rating(result_df, "Relevance", "M+")),
            ("L", count_rating(result_df, "Relevance", "L")),
            ("NR", count_rating(result_df, "Relevance", "NR")),
        ])

        # ----------------------------------------------------
        # FRAMEWORK-ONLY SUMMARY
        # ----------------------------------------------------

        render_html(
            ui.section(
                "F",
                "Framework-Only Classification",
                "Classification based specifically on the "
                "user-provided relevance framework."
            )
        )

        render_cards([
            ("H", count_rating(result_df, "Relevance_Framework_Only", "H")),
            ("M+", count_rating(result_df, "Relevance_Framework_Only", "M+")),
            ("L", count_rating(result_df, "Relevance_Framework_Only", "L")),
            ("NR", count_rating(result_df, "Relevance_Framework_Only", "NR")),
        ])

        # ====================================================
        # RESULTS TABLE
        # ====================================================

        render_html(
            ui.section(
                "R",
                "Common Relevance Framework Results"
            )
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.download_button(
            "Download Results",
            result_df.to_csv(index=False),
            "RelevanceIQ_Results.csv",
            "text/csv",
            key="download_results"
        )

    else:

        st.info(
            "No analysis results are available yet. "
            "Run an analysis first."
        )

    st.stop()


# ============================================================
# ANALYSIS PAGE
# ============================================================

render_html(
    ui.hero(
        "Patent Intelligence • FTO",
        "Patent Relevance Analysis",
        "Evaluate patents against a target product using "
        "a defined relevance framework and human-reviewed "
        "technical feature extraction."
    )
)


# ============================================================
# INPUT SECTION
# ============================================================

render_html(
    ui.section(
        1,
        "Target Product",
        "Describe the technical product that you want to analyze."
    )
)


product_description = st.text_area(
    "Product Description",

    value=st.session_state.product_description,

    height=180,

    placeholder=(
        "Enter the technical description of the target product..."
    ),

    label_visibility="collapsed",

    key="product_description_input"
)


# ============================================================
# PATENT FILE
# ============================================================

render_html(
    ui.section(
        2,
        "Patent Dataset",
        "Upload the Excel file containing the patents to be evaluated."
    )
)

analysis_source = st.session_state.get(
    "analysis_source",
    "independent"
)

if analysis_source == 'retrieval':

    uploaded_df = prepare_retrieval_for_relevanceiq(
        st.session_state["patent_search_candidates"]
    )

    product_description = st.session_state[
        "patent_search_product_description"
    ]

    st.success(
        f"{len(uploaded_df)} retrieved patents loaded for RelevanceIQ."
    )

else:

    uploaded_file = st.file_uploader(
        "Upload Patent Excel File",
        type=["xlsx"],
        label_visibility="collapsed",
        key="patent_file"
    )


col1, col2 = st.columns([3, 1])

with col1:

    render_html(ui.input_info())

with col2:

    template_path = (
        "RelevanceIQ_Input_Data_Schema.xlsx"
    )

    if os.path.exists(template_path):

        with open(
            template_path,
            "rb"
        ) as f:

            st.download_button(
                "Download Template",

                f.read(),

                "RelevanceIQ_Input_Data_Schema.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                use_container_width=True,
                key="download_template"
            )


# ============================================================
# FRAMEWORK
# ============================================================

render_html(
    ui.section(
        3,
        "Relevance Framework",
        "Define the criteria that will be used to determine "
        "patent relevance."
    )
)


relevance_framework = st.text_area(
    "Relevance Framework",

    value=st.session_state.relevance_framework,

    height=220,

    placeholder=(
        "Enter the relevance framework used to classify "
        "patents as H, M+, L or NR..."
    ),

    label_visibility="collapsed",

    key="relevance_framework_input"
)


# ============================================================
# RUN ANALYSIS
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)


run_col1, run_col2, run_col3 = st.columns(
    [1, 2, 1]
)


with run_col2:

    run = st.button(
        "Run Relevance Analysis",

        use_container_width=True,

        key="run_analysis_button"
    )


if run:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if analysis_source != "retrieval" and uploaded_file is None:

        st.warning(
            "Please upload an Excel file containing the patents."
        )

        st.stop()


    if not product_description.strip():

        st.warning(
            "Please enter the product description."
        )

        st.stop()


    if not relevance_framework.strip():

        st.warning(
            "Please provide the relevance framework."
        )

        st.stop()


    # --------------------------------------------------------
    # READ EXCEL
    # --------------------------------------------------------

    if analysis_source == "retrieval":

        uploaded_df = prepare_retrieval_for_relevanceiq(
            st.session_state["patent_search_candidates"]
        )

    else:

        try:

            uploaded_df = pd.read_excel(uploaded_file)

            uploaded_df = uploaded_df.astype(object)

            uploaded_df = uploaded_df.where(
                pd.notna(uploaded_df),
                None
            )

        except Exception as e:

            st.error(
                f"Unable to read the Excel file: {e}"
            )

            st.stop()


    uploaded_df = normalize_dataframe_for_state(uploaded_df)


    if uploaded_df.empty:

        st.warning(
            "The uploaded Excel file does not contain "
            "any patent rows."
        )

        st.stop()


    # --------------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "Publication Number",
        "Title",
        "Abstract",
        "Independent Claim",
        "All Claims",
        "Relevance Human"
    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in uploaded_df.columns

    ]


    if missing_columns:

        st.error(
            "The uploaded Excel file is missing the following "
            f"required columns: {', '.join(missing_columns)}"
        )

        st.stop()


    # --------------------------------------------------------
    # SAVE SESSION DATA
    # --------------------------------------------------------

    st.session_state.df = uploaded_df

    st.session_state.product_description = (
        product_description
    )

    st.session_state.relevance_framework = (
        relevance_framework
    )

    st.session_state.results = []

    st.session_state.results_by_index = {}

    st.session_state.failed_patents = {}

    # Reset complete portfolio results for the new analysis.
    st.session_state.portfolio_results_by_index = {}

    # Reset any previously generated portfolio report.
    st.session_state.portfolio_recommendations_report = None

    st.session_state.current_index = 0

    st.session_state.primary_product_features = []

    st.session_state.secondary_product_features = []

    st.session_state.product_features_approved = False

    st.session_state.analysis_started = True

    st.session_state.analysis_error = False

    st.session_state.session_id = str(
        uuid.uuid4()
    )

    st.rerun()


# ============================================================
# PRODUCT FEATURE EXTRACTION
# ============================================================

if (
    st.session_state.analysis_started
    and not st.session_state.product_features_approved
):

    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    if (
        not st.session_state.primary_product_features
        and not st.session_state.secondary_product_features
    ):

        render_html(
            ui.status_card(
                "Extracting Product Features",
                "RelevanceIQ is identifying the primary "
                "and secondary technical features of the "
                "target product for human review."
            )
        )


        product_prompt = f"""
You are an experienced patent analyst performing an FTO
(Freedom-to-Operate) technical analysis.

Extract the technical features explicitly disclosed in the
product description.

PRIMARY FEATURES:
- Core technical features and mechanisms.
- Features important for satisfying potential patent claim
  limitations.

SECONDARY FEATURES:
- Supporting, auxiliary, optional, control, communication,
  interface, or peripheral features.

Rules:
- Extract only explicitly disclosed features.
- Do not infer or invent features.
- Keep features concise and technically specific.

{feature_parser.get_format_instructions()}

Product Description:
{st.session_state.product_description}
"""


        response = llm.invoke(
            [
                HumanMessage(
                    content=product_prompt
                )
            ]
        )


        result = feature_parser.parse(
            response.content
        )


        st.session_state.primary_product_features = (
            result.primary_features
        )


        st.session_state.secondary_product_features = (
            result.secondary_features
        )

        st.rerun()


    # --------------------------------------------------------
    # HUMAN REVIEW
    # --------------------------------------------------------

    render_html(
        ui.section(
            "✓",
            "Review Product Features",
            "Review and edit the extracted features before "
            "continuing with patent analysis."
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        edited_primary = st.text_area(
            "Primary Product Features",

            value="\n".join(
                st.session_state.primary_product_features
            ),

            height=250,

            key="product_primary_review"
        )


    with col2:

        edited_secondary = st.text_area(
            "Secondary Product Features",

            value="\n".join(
                st.session_state.secondary_product_features
            ),

            height=250,

            key="product_secondary_review"
        )


    approve_col1, approve_col2, approve_col3 = st.columns(
        [1, 2, 1]
    )


    with approve_col2:

        if st.button(
            "Approve Product Features",

            use_container_width=True,

            key="approve_product_features"
        ):

            st.session_state.primary_product_features = [

                x.strip()

                for x in edited_primary.split("\n")

                if x.strip()

            ]


            st.session_state.secondary_product_features = [

                x.strip()

                for x in edited_secondary.split("\n")

                if x.strip()

            ]


            st.session_state.product_features_approved = True

            st.rerun()


    st.stop()


# ============================================================
# PARALLEL PATENT PROCESSING
# ============================================================

if st.session_state.analysis_started:

    df = st.session_state.df

    total_patents = len(df)

    results_by_index = st.session_state.results_by_index

    failed_patents = st.session_state.failed_patents


    # Only patents that have not successfully completed
    # will be submitted to the worker pool.
    pending_indices = [

        i

        for i in range(total_patents)

        if i not in results_by_index

    ]


    # ========================================================
    # PARALLEL EXECUTION
    # ========================================================

    if pending_indices:

        render_html(
            ui.section(
                "P",
                "Parallel Patent Processing",
                f"Processing patents using {MAX_WORKERS} parallel workers."
            )
        )


        st.info(
            f"{len(pending_indices)} patents pending processing."
        )


        progress_bar = st.progress(
            len(results_by_index) / total_patents
        )


        status_text = st.empty()


        # Capture session values on the main Streamlit thread.
        product_description = (
            st.session_state.product_description
        )

        relevance_framework = (
            st.session_state.relevance_framework
        )

        primary_product_features = list(
            st.session_state.primary_product_features
        )

        secondary_product_features = list(
            st.session_state.secondary_product_features
        )


        # ----------------------------------------------------
        # THREAD POOL
        # ----------------------------------------------------

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            future_to_index = {}


            # Submit patents to the worker pool.
            for index in pending_indices:

                row_data = make_serializable(
                    df.iloc[index].to_dict()
                )


                future = executor.submit(
                    process_single_patent,

                    index,

                    row_data,

                    product_description,

                    relevance_framework,

                    primary_product_features,

                    secondary_product_features
                )


                future_to_index[future] = index


            # ------------------------------------------------
            # COLLECT COMPLETED RESULTS
            # ------------------------------------------------

            for future in as_completed(future_to_index):

                index = future_to_index[future]

                publication_number = str(
                    df.iloc[index]["Publication Number"]
                )


                try:

                    result = future.result()


                    # ----------------------------------------
                    # SAVE TO DATABASE
                    # ----------------------------------------

                    try:

                        save_result(result)

                    except Exception as db_error:

                        st.warning(
                            f"Database save failed for "
                            f"{publication_number}: {db_error}"
                        )


                    # ----------------------------------------
                    # PREPARE PATENT RESULT
                    # ----------------------------------------

                    patent_result = {

                        "Publication Number":
                            result["publication_number"],

                        "Title":
                            result["title"],

                        "Relevance Human":
                            result["relevance_human"],

                        "Patent Primary Features":
                            result["primary_patent_features"],

                        "Patent Secondary Features":
                            result["secondary_patent_features"],

                        "Relevance":
                            result["relevance"],

                        "Confidence":
                            result["confidence"],

                        "Rationale":
                            result["rationale"],

                        "Reasoning":
                            result["reasoning"],

                        "Relevance_Framework_Only":
                            result["relevance_framework_only"],

                        "Confidence_Framework_Only":
                            result["confidence_framework_only"],

                        "Rationale_Framework_Only":
                            result["rationale_framework_only"],

                        "Reasoning_Framework_Only":
                            result["reasoning_framework_only"]

                    }


                    # Store display results using original patent index.
                    results_by_index[index] = make_serializable(
                        patent_result
                    )


                    # ------------------------------------------------
                    # STORE COMPLETE PATENT RESULT FOR PORTFOLIO
                    # ------------------------------------------------

                    st.session_state.portfolio_results_by_index[index] = (
                        make_serializable(result)
                    )


                    # Remove patent from failed list if retried successfully.
                    failed_patents.pop(index, None)


                except Exception as e:

                    failed_patents[index] = str(e)

                    print(
                        f"Patent processing failed: "
                        f"{publication_number}"
                    )

                    print(e)


                # --------------------------------------------
                # UPDATE PROGRESS
                # --------------------------------------------

                total_completed = len(results_by_index)

                total_failed = len(failed_patents)

                total_processed = (
                    total_completed + total_failed
                )


                progress_bar.progress(
                    min(total_processed / total_patents, 1.0)
                )


                status_text.write(
                    f"Completed: {total_completed}/{total_patents} | "
                    f"Failed: {total_failed} | "
                    f"Remaining: {total_patents - total_processed}"
                )


        # ----------------------------------------------------
        # UPDATE SESSION RESULTS
        # ----------------------------------------------------

        st.session_state.results = [

            results_by_index[i]

            for i in sorted(results_by_index)

        ]


        st.session_state.current_index = len(
            results_by_index
        )


    # ========================================================
    # PROCESSING FINISHED
    # ========================================================

    st.session_state.analysis_started = False


    result_df = pd.DataFrame(
        st.session_state.results
    )


    # --------------------------------------------------------
    # FAILED PATENTS
    # --------------------------------------------------------

    if failed_patents:

        st.warning(
            f"{len(failed_patents)} patent(s) failed "
            "and can be retried."
        )


        failed_df = pd.DataFrame([

            {
                "Patent Index": index + 1,

                "Publication Number":
                    df.iloc[index]["Publication Number"],

                "Error": error

            }

            for index, error in sorted(failed_patents.items())

        ])


        st.dataframe(
            failed_df,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.success(
            "All patents have been processed successfully."
        )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    if not result_df.empty:

        render_html(
            ui.hero(
                "Analysis Complete",
                "Patent Analysis Results",
                "The completed patent analyses are available below."
            )
        )


        render_cards([
            ("Patents Analysed", len(result_df)),
            ("H", count_rating(result_df, "Relevance", "H")),
            ("M+", count_rating(result_df, "Relevance", "M+")),
            ("L", count_rating(result_df, "Relevance", "L")),
            ("NR", count_rating(result_df, "Relevance", "NR")),
        ])


        render_html(
            ui.section(
                "R",
                "Results"
            )
        )


        st.dataframe(
            result_df,

            use_container_width=True,

            hide_index=True
        )


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        st.download_button(
            "Download Results",

            result_df.to_csv(index=False),

            "RelevanceIQ_Results.csv",

            "text/csv",

            key="download_final_results"
        )


# ============================================================
# PORTFOLIO-LEVEL RECOMMENDATIONS
# ============================================================

if (
    not st.session_state.analysis_started
    and st.session_state.df is not None
    and len(st.session_state.portfolio_results_by_index)
        == len(st.session_state.df)
    and len(st.session_state.df) > 0
    and not st.session_state.failed_patents
):

    st.divider()

    render_html(
        ui.section(
            "P",
            "Portfolio-Level Recommendations",
            "Generate consolidated FTO insights across all analysed patents."
        )
    )


    if st.button(
        "Generate Portfolio Recommendations",

        use_container_width=True,

        key="generate_portfolio_recommendations"
    ):

        try:

            portfolio_results = [

                st.session_state.portfolio_results_by_index[i]

                for i in sorted(
                    st.session_state.portfolio_results_by_index
                )

            ]


            with st.spinner(
                "Analysing the complete patent portfolio and generating recommendations..."
            ):

                report = generate_recommendations_report(
                    portfolio_results=portfolio_results,

                    product_description=(
                        st.session_state.product_description
                    )
                )


                # Save report in session state so it remains
                # available after Streamlit reruns.
                st.session_state.portfolio_recommendations_report = report


            st.success(
                "Portfolio recommendations generated successfully."
            )


        except Exception as e:

            st.error(
                f"Unable to generate portfolio recommendations: {e}"
            )


    # --------------------------------------------------------
    # DOWNLOAD GENERATED WORD REPORT
    # --------------------------------------------------------

    if st.session_state.portfolio_recommendations_report is not None:

        st.download_button(
            label="Download Portfolio Recommendations",

            data=st.session_state.portfolio_recommendations_report,

            file_name="RelevanceIQ_Portfolio_Recommendations.docx",

            mime=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),

            use_container_width=True,

            key="download_portfolio_recommendations"
        )


# ============================================================
# RETRY FAILED PATENTS
# ============================================================

if (
    not st.session_state.analysis_started
    and st.session_state.failed_patents
):

    if st.button(
        "Retry Failed Patents",
        use_container_width=True,
        key="retry_failed_patents"
    ):

        st.session_state.analysis_started = True

        st.rerun()


# ============================================================
# FOOTER
# ============================================================c

render_html(ui.footer())
# import sqlite3
# import json

# DATABASE_NAME = "RelevanceIQ.db"

# def create_table():

#     conn = sqlite3.connect(DATABASE_NAME)
#     cursor = conn.cursor()

#     cursor.execute("""
#     CREATE TABLE IF NOT EXISTS FTO_result (

#         id INTEGER PRIMARY KEY AUTOINCREMENT,

#         publication_number TEXT,
#         title TEXT,
#         abstract TEXT,
#         independent_claims TEXT,
#         all_claims TEXT, 
#         product_description TEXT,
#         relevance_framework TEXT, 
#         relevance TEXT,
#         confidence INTEGER,
#         rationale TEXT,
#         reasoning TEXT,
#         relevance_framework_only TEXT,
#         confidence_framework_only INTEGER,
#         rationale_framework_only TEXT,
#         reasoning_framework_only TEXT,
#         primary_patent_features TEXT,
#         secondary_patent_features TEXT

#     )
#     """)

#     conn.commit()
#     conn.close()

# def save_result(result):

#     conn = sqlite3.connect(DATABASE_NAME)
#     cursor = conn.cursor()

#     cursor.execute(
#         """
#         INSERT INTO FTO_result(

#             publication_number,
#             title,
#             abstract,
#             independent_claims,
#             all_claims,
#             product_description,
#             relevance_framework,
#             relevance,
#             confidence,
#             rationale,
#             reasoning,
#             relevance_framework_only,
#             confidence_framework_only,
#             rationale_framework_only,
#             reasoning_framework_only,
#             primary_patent_features,
#             secondary_patent_features

#         )

#         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?, ?, ?, ?)
#         """,
#         (

#             result["publication_number"],
#             result["title"],
#             result["abstract"],
#             result["independent_claims"],
#             result["all_claims"],
    
#             result["product_description"],
#             result["relevance_framework"],
#             result["relevance"],
#             result["confidence"],
#             result["rationale"],
#             result["reasoning"],
#             result["relevance_framework_only"],
#             result["confidence_framework_only"],
#             result["rationale_framework_only"],
#             result["reasoning_framework_only"],
#             json.dumps(result["primary_patent_features"]),
#             json.dumps(result["secondary_patent_features"])


#         )
#     )

#     conn.commit()
#     conn.close()


# def fetch_all_results():

#     conn = sqlite3.connect(DATABASE_NAME)
#     cursor = conn.cursor()

#     cursor.execute(
#         """
#         SELECT *
#         FROM FTO_result
#         ORDER BY id DESC
#         """
#     )

#     rows = cursor.fetchall()

#     conn.close()

#     return rows

# import pandas as pd

# create_table()

# conn = sqlite3.connect(DATABASE_NAME)
# df = pd.read_sql_query(
#     "SELECT * FROM FTO_result",
#     conn
# )

# print(df)
# conn.close()




import sqlite3
import json
from datetime import datetime

# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DATABASE_NAME = "RelevanceIQ.db"


def get_connection():

    conn = sqlite3.connect(
        DATABASE_NAME,
        timeout=30
    )

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    return conn


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

def create_table():

    conn = get_connection()

    try:

        cursor = conn.cursor()

        # ----------------------------------------------------
        # PATENT ANALYSIS RESULTS
        # ----------------------------------------------------

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS FTO_result (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            publication_number TEXT,
            title TEXT,
            abstract TEXT,
            independent_claims TEXT,
            all_claims TEXT,

            product_description TEXT,
            relevance_framework TEXT,

            relevance TEXT,
            confidence INTEGER,
            rationale TEXT,
            reasoning TEXT,

            relevance_framework_only TEXT,
            confidence_framework_only INTEGER,
            rationale_framework_only TEXT,
            reasoning_framework_only TEXT,

            primary_patent_features TEXT,
            secondary_patent_features TEXT

        )
        """)

        # ----------------------------------------------------
        # ANALYSIS CHECKPOINTS
        # ----------------------------------------------------

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_checkpoints (

            session_id TEXT PRIMARY KEY,

            updated_at TEXT,

            status TEXT,

            checkpoint_data TEXT

        )
        """)

        conn.commit()

    finally:

        conn.close()


# ============================================================
# SAVE INDIVIDUAL PATENT RESULT
# ============================================================

def save_result(result):

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO FTO_result(

                publication_number,
                title,
                abstract,
                independent_claims,
                all_claims,

                product_description,
                relevance_framework,

                relevance,
                confidence,
                rationale,
                reasoning,

                relevance_framework_only,
                confidence_framework_only,
                rationale_framework_only,
                reasoning_framework_only,

                primary_patent_features,
                secondary_patent_features

            )

            VALUES (
                ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?
            )
            """,

            (

                result.get("publication_number"),
                result.get("title"),
                result.get("abstract"),
                result.get("independent_claims"),
                result.get("all_claims"),

                result.get("product_description"),
                result.get("relevance_framework"),

                result.get("relevance"),
                result.get("confidence"),
                result.get("rationale"),
                result.get("reasoning"),

                result.get("relevance_framework_only"),
                result.get("confidence_framework_only"),
                result.get("rationale_framework_only"),
                result.get("reasoning_framework_only"),

                json.dumps(
                    result.get("primary_patent_features", []),
                    ensure_ascii=False
                ),

                json.dumps(
                    result.get("secondary_patent_features", []),
                    ensure_ascii=False
                )

            )
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# FETCH ALL PATENT RESULTS
# ============================================================

def fetch_all_results():

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM FTO_result
            ORDER BY id DESC
        """)

        return cursor.fetchall()

    finally:

        conn.close()


# ============================================================
# SAVE ANALYSIS CHECKPOINT
# ============================================================

def save_analysis_checkpoint(
    session_id,
    checkpoint_data,
    status="in_progress"
):

    conn = get_connection()

    try:

        cursor = conn.cursor()

        serialized_data = json.dumps(
            checkpoint_data,
            ensure_ascii=False
        )

        cursor.execute(
            """
            INSERT INTO analysis_checkpoints (

                session_id,
                updated_at,
                status,
                checkpoint_data

            )

            VALUES (?, ?, ?, ?)

            ON CONFLICT(session_id)

            DO UPDATE SET

                updated_at = excluded.updated_at,

                status = excluded.status,

                checkpoint_data = excluded.checkpoint_data

            """,

            (

                session_id,

                datetime.now().isoformat(),

                status,

                serialized_data

            )
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# LOAD LATEST INCOMPLETE ANALYSIS CHECKPOINT
# ============================================================

def load_latest_checkpoint():

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                session_id,
                checkpoint_data,
                status

            FROM analysis_checkpoints

            WHERE status != 'completed'

            ORDER BY updated_at DESC

            LIMIT 1
        """)

        row = cursor.fetchone()

        if row:

            return {

                "session_id": row[0],

                "data": json.loads(row[1]),

                "status": row[2]

            }

        return None

    finally:

        conn.close()


# ============================================================
# LOAD CHECKPOINT USING SESSION ID
# ============================================================

def load_checkpoint(session_id):

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                session_id,
                checkpoint_data,
                status

            FROM analysis_checkpoints

            WHERE session_id = ?

            """,

            (session_id,)
        )

        row = cursor.fetchone()

        if row:

            return {

                "session_id": row[0],

                "data": json.loads(row[1]),

                "status": row[2]

            }

        return None

    finally:

        conn.close()


# ============================================================
# FETCH ALL CHECKPOINTS
# ============================================================

def fetch_all_checkpoints():

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                session_id,
                updated_at,
                status

            FROM analysis_checkpoints

            ORDER BY updated_at DESC
        """)

        return cursor.fetchall()

    finally:

        conn.close()


# ============================================================
# DELETE COMPLETED OR UNWANTED CHECKPOINT
# ============================================================

def delete_checkpoint(session_id):

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM analysis_checkpoints
            WHERE session_id = ?
            """,

            (session_id,)
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

if __name__ == "__main__":

    create_table()

    print("RelevanceIQ database initialized successfully.")

    conn = get_connection()

    try:

        import pandas as pd

        df = pd.read_sql_query(
            "SELECT * FROM FTO_result",
            conn
        )

        print("\nPatent Analysis Results:")
        print(df)

        checkpoints_df = pd.read_sql_query(
            """
            SELECT
                session_id,
                updated_at,
                status
            FROM analysis_checkpoints
            ORDER BY updated_at DESC
            """,
            conn
        )

        print("\nAnalysis Checkpoints:")
        print(checkpoints_df)

    finally:

        conn.close()

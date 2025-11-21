import os
import sys
import hashlib
import mysql.connector
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONFIG ----------
SQL_FILE_PATH = 'Vector_Based_Search_Engine\Search_Engine_db.sql'
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'search_engine',
    'autocommit': False
}
DEFAULT_TOP_K = 5
SNIPPET_LEN = 240
# ----------------------------

# Load model once
print("Loading embedding model...")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# ---------------- DB helpers ----------------
def connect_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        print("Database connection failed:", e)
        sys.exit(1)
        
def file_hash(path):
    try:
        import hashlib
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None

def parse_vector_text(vector_text):
    if not vector_text:
        return None
    return np.fromiter((float(x) for x in vector_text.split(',')), dtype=float)

# ---------------- Utility functions ----------------
def highlight_snippet(content, query_terms):
    if not content:
        return ""
    text = content.replace('\n', ' ')
    pattern = re.compile(r'(' + '|'.join(re.escape(t) for t in query_terms) + r')', re.IGNORECASE)
    m = pattern.search(text)
    if m:
        start = max(0, m.start() - SNIPPET_LEN // 3)
        end = min(len(text), start + SNIPPET_LEN)
        snippet = text[start:end]
    else:
        snippet = text[:SNIPPET_LEN]
    def bold(m): return '\033[1m' + m.group(0) + '\033[0m'
    snippet = pattern.sub(bold, snippet)
    return snippet + ('...' if len(snippet) < len(text) else '')

# ---------------- DB-side usage (silent) ----------------
def create_cli_stats_table_if_missing(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cli_stats (
            stat_id INT AUTO_INCREMENT PRIMARY KEY,
            stat_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            kind VARCHAR(50),
            details TEXT
        )
    """)
    conn.commit()
    cur.close()

def write_cli_stat(conn, kind, details):
    cur = conn.cursor()
    cur.execute("INSERT INTO cli_stats (kind, details) VALUES (%s, %s)", (kind, details))
    conn.commit()
    cur.close()

def try_call_proc(conn, proc_name, args=()):
    cur = conn.cursor()
    try:
        cur.callproc(proc_name, args)
        results = []
        for res in cur.stored_results():
            results.append(res.fetchall())
        cur.close()
        return results
    except mysql.connector.Error:
        cur.close()
        return None

def try_execute_func(conn, func_name, args=()):
    cur = conn.cursor()
    try:
        placeholder = ','.join(['%s'] * len(args))
        sql = f"SELECT {func_name}({placeholder})"
        cur.execute(sql, args)
        val = cur.fetchone()[0]
        cur.close()
        return val
    except mysql.connector.Error:
        cur.close()
        return None

# ---------------- Indexing / recompute ----------------
def fetch_documents(conn):
    cur = conn.cursor()
    cur.execute("SELECT document_id, content, title FROM document ORDER BY document_id")
    docs = cur.fetchall()
    cur.close()
    return docs

def fetch_all_document_embeddings(conn):
    cur = conn.cursor()
    cur.execute("SELECT document_id, title, content FROM document ORDER BY document_id")
    docs = cur.fetchall()
    if not docs:
        cur.close()
        return [], [], [], np.array([])
    doc_ids, titles, contents = [], [], []
    for d in docs:
        doc_ids.append(int(d[0])); titles.append(d[1]); contents.append(d[2])
    cur.execute("SELECT document_id, vector FROM embedding ORDER BY document_id")
    emb_rows = cur.fetchall()
    emb_map = {int(r[0]): r[1] for r in emb_rows}
    vectors = []
    for did in doc_ids:
        vtxt = emb_map.get(did)
        if vtxt is None:
            cur.close()
            return doc_ids, titles, contents, np.array([]) 
        vectors.append(parse_vector_text(vtxt))
    cur.close()
    return doc_ids, titles, contents, np.vstack(vectors)

def recompute_all(conn):
    cur = conn.cursor()
    docs = fetch_documents(conn)
    if not docs:
        cur.close(); return

    # 1) Clear via stored proc if present (preferred) — silent
    proc_result = try_call_proc(conn, 'sp_clear_embeddings', ())
    if proc_result is None:
        cur.execute("DELETE FROM embedding")
        conn.commit()

    # 2) Clear term and search_result
    cur.execute("DELETE FROM term")
    cur.execute("DELETE FROM search_result")
    conn.commit()

    # 3) Term frequencies (CountVectorizer)
    doc_ids = [int(r[0]) for r in docs]
    contents = [r[1] for r in docs]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(contents)
    terms = vectorizer.get_feature_names_out()
    freq_matrix = X.toarray()
    tid = 1
    insert_term_sql = "INSERT INTO term (term_id, document_id, frequency, term_text) VALUES (%s,%s,%s,%s)"
    for i, doc_id in enumerate(doc_ids):
        for j, term_text in enumerate(terms):
            f = int(freq_matrix[i][j])
            if f > 0:
                cur.execute(insert_term_sql, (tid, doc_id, f, term_text))
                tid += 1
    conn.commit()

    # 4) Document embeddings (compute and insert)
    doc_embs = MODEL.encode(contents, show_progress_bar=False)
    insert_emb_sql = "INSERT INTO embedding (embedding_id, document_id, vector, dimension, e_index) VALUES (%s,%s,%s,%s,%s)"
    for i, doc_id in enumerate(doc_ids):
        vec = doc_embs[i]
        vtxt = ','.join(format(float(x), '.6f') for x in vec)
        dim = int(len(vec))
        emb_id = i + 1
        cur.execute(insert_emb_sql, (emb_id, doc_id, vtxt, dim, emb_id))
    conn.commit()

    # 5) Build search_result for existing user_query rows
    cur.execute("SELECT query_id, query_text FROM user_query ORDER BY query_id")
    queries = cur.fetchall()
    if queries:
        qids = [int(q[0]) for q in queries]
        qtexts = [q[1] for q in queries]
        q_embs = MODEL.encode(qtexts, show_progress_bar=False)
        doc_emb_matrix = np.array(doc_embs)
        insert_sr_sql = "INSERT INTO search_result (query_id, sr_rank, document_id, score) VALUES (%s,%s,%s,%s)"
        for qi, qid in enumerate(qids):
            qvec = q_embs[qi].reshape(1, -1)
            sims = cosine_similarity(qvec, doc_emb_matrix)[0]
            ranked_idx = np.argsort(-sims)
            rank = 1
            for idx in ranked_idx:
                cur.execute(insert_sr_sql, (qid, rank, int(doc_ids[int(idx)]), float(sims[int(idx)])))
                rank += 1
    conn.commit()
    cur.close()

    # 6) CLI stat row summarizing the recompute event
    write_cli_stat(conn, 'recompute', f'docs_indexed={len(doc_ids)}, terms_inserted={tid-1}')
    # Done (silent to user)

# ---------------- Search / result filtering ----------------
def record_user_query(conn, qtext):
    cur = conn.cursor()
    cur.execute("INSERT INTO user_query (user_id, query_vector, query_text) VALUES (%s,%s,%s)", (1, None, qtext))
    conn.commit()
    qid = cur.lastrowid
    cur.close()
    write_cli_stat(conn, 'user_query', f'query_id={qid}')
    return qid

def store_search_results(conn, qid, doc_ids, scores):
    cur = conn.cursor()
    insert_sql = "INSERT INTO search_result (query_id, sr_rank, document_id, score) VALUES (%s,%s,%s,%s)"
    for rank, (did, score) in enumerate(zip(doc_ids, scores), start=1):
        cur.execute(insert_sql, (qid, rank, int(did), float(score)))
    conn.commit()
    cur.close()

def semantic_search(conn, query_text, top_k=DEFAULT_TOP_K):
    # 1) Ensure embeddings exist
    doc_ids, titles, contents, emb_matrix = fetch_all_document_embeddings(conn)
    if emb_matrix.size == 0:
        recompute_all(conn)
        doc_ids, titles, contents, emb_matrix = fetch_all_document_embeddings(conn)
        if emb_matrix.size == 0:
            return []

    # 2) Compute similarity
    qvec = MODEL.encode([query_text], show_progress_bar=False)[0].reshape(1, -1)
    sims = cosine_similarity(qvec, emb_matrix)[0]  # shape (n_docs,)
    ranked_idx_full = np.argsort(-sims)
    ranked_doc_ids = [doc_ids[i] for i in ranked_idx_full]
    ranked_scores = [float(sims[i]) for i in ranked_idx_full]

    # 3) Record query & insert results to search_result
    qid = record_user_query(conn, query_text)
    # insert top 2*top_k results into DB as preliminary
    pre_k = min(len(ranked_doc_ids), max(top_k * 2, top_k + 2))
    store_search_results(conn, qid, ranked_doc_ids[:pre_k], ranked_scores[:pre_k])

    # 4) Use DB function fn_max_score(qid) to get maximum score
    max_score = try_execute_func(conn, 'fn_max_score', (qid,))
    if max_score is None:
        # fallback to python calculation
        max_score = max(ranked_scores[:pre_k]) if ranked_scores else 0.0

    # 5) Compute adaptive threshold
    threshold = max_score * 0.6
    chosen_threshold = None
    for factor in (0.6, 0.5, 0.4, 0.3, 0.2):
        thr = max_score * factor
        cnt = try_execute_func(conn, 'fn_count_docs_above', (qid, float(thr)))
        if cnt is None:
            cnt = sum(1 for s in ranked_scores[:pre_k] if s > thr)
        if cnt >= top_k:
            chosen_threshold = thr
            break
    if chosen_threshold is None:
        chosen_threshold = max_score * 0.2

    # 6) Filter final results based on chosen_threshold, but ensure we return at least top_k
    final_pairs = [(did, score) for did, score in zip(ranked_doc_ids, ranked_scores) if score >= chosen_threshold]
    if len(final_pairs) < top_k:
        final_pairs = list(zip(ranked_doc_ids, ranked_scores))[:top_k]

    # 7) Build return results with snippets
    qterms = re.findall(r'\w+', query_text.lower())
    results = []
    for did, score in final_pairs[:top_k]:
        idx = doc_ids.index(did)
        title = titles[idx]
        snippet = highlight_snippet(contents[idx], qterms)
        results.append((did, title, score, snippet))

    # 8) Persist final count to CLI stats (silent DB usage)
    write_cli_stat(conn, 'search_executed', f'query_id={qid}, returned={len(results)}, threshold={chosen_threshold:.6f}')
    return results

# ---------------- CRUD ----------------
def create_document(conn):
    try:
        doc_id = int(input("Enter document_id (int): ").strip())
    except ValueError:
        print("Invalid id"); return
    author = input("Author: ").strip()
    title = input("Title: ").strip()
    content = input("Content: ").strip()
    upload_date = input("Upload date (YYYY-MM-DD): ").strip()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO document (document_id, author, content, title, upload_date) VALUES (%s,%s,%s,%s,%s)",
                    (doc_id, author, content, title, upload_date))
        conn.commit()
        print("[CRUD] Document inserted.")
        recompute_all(conn)
    except mysql.connector.Error as e:
        print("DB error:", e)
        conn.rollback()
    finally:
        cur.close()

def list_documents(conn):
    cur = conn.cursor()
    cur.execute("SELECT document_id, title, author, upload_date FROM document ORDER BY document_id")
    rows = cur.fetchall()
    if not rows:
        print("No documents.")
    else:
        for r in rows:
            print(f"ID: {r[0]} | Title: {r[1]} | Author: {r[2]} | Date: {r[3]}")
    cur.close()

def view_document(conn):
    try:
        doc_id = int(input("Enter document_id: ").strip())
    except ValueError:
        print("Bad id"); return
    cur = conn.cursor()
    cur.execute("SELECT document_id, author, title, upload_date, content FROM document WHERE document_id = %s", (doc_id,))
    r = cur.fetchone()
    if not r:
        print("Not found.")
    else:
        print(f"\n--- Document {r[0]} ---\nTitle: {r[2]}\nAuthor: {r[1]}\nDate: {r[3]}\n\n{r[4]}\n--- END ---\n")
    cur.close()

def update_document(conn):
    try:
        doc_id = int(input("Enter document_id to update: ").strip())
    except ValueError:
        print("Bad id"); return
    cur = conn.cursor()
    cur.execute("SELECT document_id FROM document WHERE document_id = %s", (doc_id,))
    if not cur.fetchone():
        print("Not found."); cur.close(); return
    author = input("New author (blank keep): ").strip()
    title = input("New title (blank keep): ").strip()
    content = input("New content (blank keep): ").strip()
    upload_date = input("New upload date (YYYY-MM-DD) (blank keep): ").strip()
    updates, params = [], []
    if author: updates.append("author=%s"); params.append(author)
    if title: updates.append("title=%s"); params.append(title)
    if content: updates.append("content=%s"); params.append(content)
    if upload_date: updates.append("upload_date=%s"); params.append(upload_date)
    if not updates:
        print("No updates."); cur.close(); return
    params.append(doc_id)
    sql = "UPDATE document SET " + ", ".join(updates) + " WHERE document_id = %s"
    try:
        cur.execute(sql, tuple(params)); conn.commit(); print("[CRUD] Updated.")
        # triggers will clear dependent rows; we re-index silently
        recompute_all(conn)
    except mysql.connector.Error as e:
        print("DB error:", e); conn.rollback()
    finally:
        cur.close()

def delete_document(conn):
    try:
        doc_id = int(input("Enter document_id to delete: ").strip())
    except ValueError:
        print("Bad id"); return
    cur = conn.cursor()
    cur.execute("SELECT document_id FROM document WHERE document_id = %s", (doc_id,))
    if not cur.fetchone():
        print("Not found."); cur.close(); return
    try:
        cur.execute("DELETE FROM document WHERE document_id = %s", (doc_id,))
        conn.commit(); print("[CRUD] Deleted.")
        # triggers will cascade-deletes; re-index silently
        recompute_all(conn)
    except mysql.connector.Error as e:
        print("DB error:", e); conn.rollback()
    finally:
        cur.close()

# ---------------- Menu (user DOES NOT see DB functions/proc internals) ----------------
def main():
    conn = connect_db()
    create_cli_stats_table_if_missing(conn)
    prev_hash = file_hash(SQL_FILE_PATH)
    if prev_hash:
        print("[init] SQL file hash detected.")
    else:
        print("[init] SQL file not found; SQL-change detection disabled.")

    menu = """
--- SEARCH ENGINE ---
1) Search
2) List documents
3) View document
4) Create document
5) Update document
6) Delete document
7) Manual re-index
8) Exit
Choose (1-8): """
    try:
        while True:
            # detect SQL file content change silently and re-index
            current_hash = file_hash(SQL_FILE_PATH)
            if current_hash and prev_hash and current_hash != prev_hash:
                recompute_all(conn)
                prev_hash = current_hash
            elif current_hash and prev_hash is None:
                recompute_all(conn)
                prev_hash = current_hash

            choice = input(menu).strip()
            if choice == '1':
                q = input("Enter search query: ").strip()
                if not q:
                    print("Empty query."); continue
                try:
                    k = int(input(f"Top-K (default {DEFAULT_TOP_K}): ").strip() or DEFAULT_TOP_K)
                except ValueError:
                    k = DEFAULT_TOP_K
                results = semantic_search(conn, q, top_k=k)
                if not results:
                    print("No results.")
                    continue
                print("\nTop results:")
                for i, (did, title, score, snippet) in enumerate(results, start=1):
                    print(f"{i}. [ID {did}] {title} (score: {score:.4f})")
                    print(f"   {snippet}\n")
                # simple open-by-number
                act = input("Enter result number to open, or press Enter to continue: ").strip()
                if act:
                    try:
                        idx = int(act) - 1
                        if 0 <= idx < len(results):
                            did = results[idx][0]
                            cur = conn.cursor()
                            cur.execute("SELECT document_id, author, title, upload_date, content FROM document WHERE document_id = %s", (did,))
                            r = cur.fetchone(); cur.close()
                            if r:
                                print(f"\n--- Document {r[0]} ---\nTitle: {r[2]}\nAuthor: {r[1]}\nDate: {r[3]}\n\n{r[4]}\n--- END ---\n")
                            else:
                                print("Document not found.")
                        else:
                            print("Invalid number.")
                    except ValueError:
                        print("Invalid input.")
            elif choice == '2':
                list_documents(conn)
            elif choice == '3':
                view_document(conn)
            elif choice == '4':
                create_document(conn)
            elif choice == '5':
                update_document(conn)
            elif choice == '6':
                delete_document(conn)
            elif choice == '7':
                ok = input("Re-indexing will clear derived data. Continue? (y/n): ").strip().lower()
                if ok == 'y':
                    recompute_all(conn)
                    print("Re-index complete.")
                else:
                    print("Cancelled.")
            elif choice == '8':
                print("Exiting.")
                break
            else:
                print("Invalid choice.")
    finally:
        conn.close()

if __name__ == '__main__':
    main()

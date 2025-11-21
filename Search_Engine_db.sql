CREATE DATABASE IF NOT EXISTS search_engine;
USE search_engine;

DROP TABLE IF EXISTS search_result;
DROP TABLE IF EXISTS embedding;
DROP TABLE IF EXISTS term;
DROP TABLE IF EXISTS user_query;
DROP TABLE IF EXISTS engine_user;
DROP TABLE IF EXISTS document;

CREATE TABLE engine_user(
    user_id INT PRIMARY KEY,
    username VARCHAR(30) NOT NULL
);

CREATE TABLE document(
    document_id INT PRIMARY KEY,
    author VARCHAR(60) NOT NULL,
    content TEXT NOT NULL,
    title VARCHAR(100) NOT NULL,
    upload_date DATE NOT NULL
);

CREATE TABLE term(
    term_id INT,
    document_id INT NOT NULL,
    frequency INT NOT NULL,
    term_text VARCHAR(100) NOT NULL,
    PRIMARY KEY(term_id, document_id),
    CONSTRAINT fk_term_document FOREIGN KEY(document_id) REFERENCES document(document_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE embedding(
    embedding_id INT PRIMARY KEY,
    document_id INT NOT NULL,
    vector TEXT NOT NULL,
    dimension INT NOT NULL,
    e_index INT UNIQUE NOT NULL,
    CONSTRAINT fk_embedding_document FOREIGN KEY(document_id) REFERENCES document(document_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE user_query(
    query_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    q_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    query_vector INT NULL,
    query_text VARCHAR(255) NOT NULL,
    CONSTRAINT fk_query_user FOREIGN KEY(user_id) REFERENCES engine_user(user_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE search_result(
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    query_id INT NOT NULL,
    sr_rank INT NOT NULL,
    document_id INT NOT NULL,
    score FLOAT NOT NULL DEFAULT 0,
    CONSTRAINT fk_search_result_query FOREIGN KEY(query_id) REFERENCES user_query(query_id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_search_result_doc FOREIGN KEY(document_id) REFERENCES document(document_id) ON DELETE CASCADE ON UPDATE CASCADE
);

INSERT IGNORE INTO engine_user(user_id, username) VALUES
(1, 'Alice'),
(2, 'Bob'),
(3, 'Charlie'),
(4, 'David'),
(5, 'Eve');

INSERT IGNORE INTO document(document_id, author, content, title, upload_date) VALUES
(1, 'John Doe', 'This paper discusses relational database concepts, indexing techniques, and query optimization.', 'DBMS Fundamentals', '2023-01-10'),
(2, 'Mary Johnson', 'A comprehensive survey on machine learning algorithms, including supervised and unsupervised methods.', 'ML Survey', '2023-02-15'),
(3, 'Robert Brown', 'Deep learning models for computer vision tasks such as image classification and object detection.', 'Deep Learning CV', '2023-03-20'),
(4, 'Patricia Miller', 'Database optimization strategies and advanced SQL tuning methods for large datasets.', 'DB Optimization', '2023-04-05'),
(5, 'Michael Wilson', 'Reinforcement learning techniques for robotics and autonomous systems.', 'Reinforcement Learning', '2023-05-12'),
(6, 'Linda Davis', 'Big data processing using Hadoop and Spark for scalable data analytics.', 'Big Data Analytics', '2023-06-18'),
(7, 'William Garcia', 'Natural language processing techniques including sentiment analysis and text summarization.', 'NLP Overview', '2023-07-22'),
(8, 'Elizabeth Martinez', 'Exploring generative adversarial networks and their applications in image generation.', 'GAN Applications', '2023-08-15'),
(9, 'James Rodriguez', 'Time series forecasting models using ARIMA, SARIMA, and Prophet.', 'Time Series Forecasting', '2023-09-10'),
(10, 'Barbara Hernandez', 'Cloud database systems and distributed database architectures for high availability.', 'Cloud DB Systems', '2023-10-05'),
(11, 'Thomas Anderson', 'Blockchain-based database systems and decentralized data storage.', 'Blockchain Databases', '2023-11-01'),
(12, 'Jennifer Taylor', 'Edge computing applications for low-latency data processing.', 'Edge Computing', '2023-11-15'),
(13, 'Christopher Moore', 'Cybersecurity in database systems and data encryption mechanisms.', 'Database Security', '2023-11-20'),
(14, 'Sarah Thompson', 'Optimization of SQL queries using cost-based optimization techniques.', 'SQL Optimization', '2023-11-25'),
(15, 'Andrew White', 'AI-assisted query generation using natural language interfaces.', 'AI Query Systems', '2023-11-30');
 
-- TRIGGER TO DELETE RELATED EMBEDDINGS, TERMS, AND SEARCH RESULTS WHEN A DOCUMENT IS DELETED
DELIMITER $$
CREATE TRIGGER trg_delete_doc_cascade
AFTER DELETE ON document
FOR EACH ROW
BEGIN
    DELETE FROM embedding WHERE document_id = OLD.document_id;
    DELETE FROM term WHERE document_id = OLD.document_id;
    DELETE FROM search_result WHERE document_id = OLD.document_id;
END$$
DELIMITER ;

-- TRIGGER TO DELETE RELATED EMBEDDINGS, TERMS, AND SEARCH RESULTS WHEN A DOCUMENT IS UPDATED
DELIMITER $$
CREATE TRIGGER trg_update_doc_remove_embedding
AFTER UPDATE ON document
FOR EACH ROW
BEGIN
    DELETE FROM embedding WHERE document_id = NEW.document_id;
    DELETE FROM term WHERE document_id = NEW.document_id;
    DELETE FROM search_result WHERE document_id = NEW.document_id;
END$$
DELIMITER ;

INSERT IGNORE INTO user_query(query_id, user_id, q_timestamp, query_vector, query_text) VALUES
(1, 1, NOW(), NULL, 'database optimization techniques'),
(2, 2, NOW(), NULL, 'machine learning algorithms'),
(3, 3, NOW(), NULL, 'deep learning computer vision');

INSERT IGNORE INTO search_result(query_id, sr_rank, document_id, score) VALUES
(1, 1, 1, 0.9),
(1, 2, 2, 0.7),
(2, 1, 3, 0.8);

-- FUNCTION TO GET MAX SCORE FOR A QUERY
DELIMITER $$
CREATE FUNCTION fn_max_score(q_id INT) RETURNS FLOAT
DETERMINISTIC
BEGIN
    DECLARE max_score FLOAT;
    SELECT MAX(score) INTO max_score FROM search_result WHERE query_id = q_id;
    RETURN IFNULL(max_score, 0);
END$$
DELIMITER ;

-- FUNCTION TO COUNT DOCUMENTS ABOVE A THRESHOLD SCORE
DELIMITER $$
CREATE FUNCTION fn_count_docs_above(q_id INT, thresh FLOAT) RETURNS INT
DETERMINISTIC
BEGIN
    DECLARE cnt INT;
    SELECT COUNT(*) INTO cnt FROM search_result WHERE query_id = q_id AND score > thresh;
    RETURN IFNULL(cnt,0);
END$$
DELIMITER ;

-- PROCEUDRE TO CLEAR EMBEDDINGS
DROP PROCEDURE IF EXISTS sp_clear_embeddings;
DELIMITER $$
CREATE PROCEDURE sp_clear_embeddings()
BEGIN
    DELETE FROM embedding;
    ALTER TABLE embedding AUTO_INCREMENT = 1;
END$$
DELIMITER ;

-- NESTED QUERY
SELECT * 
FROM document 
WHERE document_id IN (
    SELECT document_id FROM term WHERE term_text = 'database'
);

-- JOIN 
SELECT d.document_id, d.title, e.embedding_id, e.dimension
FROM document d 
JOIN embedding e ON d.document_id = e.document_id 
WHERE e.dimension > 0;

-- AGGREGATE QUERY
SELECT author, COUNT(*) AS doc_count 
FROM document 
GROUP BY author 
HAVING doc_count >= 1;
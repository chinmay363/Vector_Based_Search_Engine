# Vector_Based_Search_Engine

This project implements a CLI-based semantic search engine using:

1. Python

2. MySQL

2. Sentence Transformers (MiniLM embeddings for cosine similarity)

3. Database triggers, stored procedures, and functions

It supports semantic search, CRUD operations, auto-indexing, and SQL file change detection.

### To run, first clone the repo:

```
git clone https://github.com/chinmay363/Vector_Based_Search_Engine.git
```

### Install the required dependencies:

```python
pip install -r requirements.txt 
```

### Start a local instance of a MySQL server on your system and run this in the terminal (enter your MySQL password if prompted):

```python
mysql -u root -p < Search_Engine_db.sql
```

### Verify your database credentials in the Search_Engine.py file and then run:

```python
python search_engine.py
```

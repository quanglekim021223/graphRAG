import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import json

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="env2.env")


class Config:
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")


driver = GraphDatabase.driver(Config.neo4j_uri, auth=(
    Config.neo4j_username, Config.neo4j_password))

# Kiểm tra kết nối Neo4j
with driver.session() as session:
    result = session.run("RETURN 1")
    if result.single()[0] == 1:
        logger.info("Successfully connected to Neo4j")
    else:
        logger.error("Failed to connect to Neo4j")
        raise Exception("Failed to connect to Neo4j")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_nodes_without_embedding(tx, start_id=None, limit=1000):
    query = """
    MATCH (n)
    WHERE n.embedding IS NULL AND n.text IS NOT NULL 
    AND (n:Patient OR n:Disease OR n:Doctor OR n:Hospital OR n:InsuranceProvider OR n:Room OR n:Medication OR n:TestResults OR n:Billing)
    """
    if start_id:
        query += " AND elementId(n) > $start_id"
    query += " RETURN elementId(n) AS node_id, n.text AS text LIMIT $limit"
    result = tx.run(query, start_id=start_id, limit=limit)
    return [(record["node_id"], record["text"]) for record in result]


def set_embedding(tx, node_id, embedding):
    logger.info(
        f"Embedding for node {node_id}: {embedding[:10]}... (length: {len(embedding)})")
    query = """
    MATCH (n)
    WHERE elementId(n) = $node_id
    SET n.embedding = $embedding
    """
    tx.run(query, node_id=node_id, embedding=embedding)


def save_checkpoint(batch_num):
    with open("checkpoint_2.json", "w") as f:
        json.dump({"last_batch": batch_num}, f)


try:
    checkpoint = {}
    try:
        with open("checkpoint_2.json", "r") as f:
            checkpoint = json.load(f)
        start_batch = checkpoint.get("last_batch", 0) + 1
    except FileNotFoundError:
        start_batch = 0

    with driver.session() as session:
        nodes_without_embedding = session.execute_read(
            get_nodes_without_embedding, start_id=None, limit=1000)
        logger.info(
            f"Found {len(nodes_without_embedding)} nodes without embeddings.")

        batch_size = 180
        processed_nodes = 0
        for i in range(start_batch * batch_size, len(nodes_without_embedding), batch_size):
            batch = nodes_without_embedding[i:i + batch_size]
            if not batch:
                break
            node_ids, texts = zip(*batch)
            try:
                batch_embeddings = embeddings.embed_documents(list(texts))
                logger.info(
                    f"Batch embeddings: {batch_embeddings[0][:10]}... (length: {len(batch_embeddings[0])})")
                for node_id, embedding in zip(node_ids, batch_embeddings):
                    session.execute_write(set_embedding, node_id, embedding)
                processed_nodes += len(batch)
                logger.info(
                    f"Processed batch {i // batch_size + 1} with {len(batch)} nodes.")
                save_checkpoint(i // batch_size)
            except Exception as e:
                logger.error(f"Error in batch {i // batch_size + 1}: {str(e)}")
                raise
        logger.info(
            f"Embeddings created successfully for {processed_nodes} nodes.")
except Exception as e:
    logger.error(f"Failed to create embeddings: {str(e)}")
    raise
finally:
    driver.close()

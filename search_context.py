import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="env1.env")


class Config:
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")


driver = GraphDatabase.driver(Config.neo4j_uri, auth=(
    Config.neo4j_username, Config.neo4j_password))
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


def search_context(query_text, top_k=5):
    query_embedding = embeddings.embed_documents([query_text])[0]
    indexes = [
        "healthcare_vector_index_patient",
        "healthcare_vector_index_disease",
        "healthcare_vector_index_doctor",
        "healthcare_vector_index_hospital",
        "healthcare_vector_index_insuranceprovider",
        "healthcare_vector_index_room",
        "healthcare_vector_index_medication",
        "healthcare_vector_index_testresults",
        "healthcare_vector_index_billing"
    ]

    all_results = []
    with driver.session() as session:
        for index_name in indexes:
            query = """
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
            YIELD node, score
            RETURN node.text AS context, score
            ORDER BY score DESC
            LIMIT $top_k
            """
            result = session.run(query, index_name=index_name,
                                 top_k=top_k, query_embedding=query_embedding)
            all_results.extend([(record["context"], record["score"])
                               for record in result])

    # Sắp xếp và lấy top_k kết quả
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[:top_k]


try:
    query_text = "Which patients are has disease Hypertension?"
    contexts = search_context(query_text)
    logger.info("Retrieved contexts:")
    for context, score in contexts:
        logger.info(f"Context: {context} (Score: {score})")
except Exception as e:
    logger.error(f"Error during context search: {str(e)}")
finally:
    driver.close()

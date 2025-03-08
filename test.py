import os
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from neo4j.time import Date
from openai import OpenAI, OpenAIError
from langchain_neo4j import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from langsmith import Client  # Thêm LangSmith client

# Configure logging globally
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Cấu hình LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT", "HealthcareGraphRAG")


@dataclass
class Config:
    """Configuration class for Healthcare GraphRAG system."""
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    endpoint: str = "https://models.inference.ai.azure.com"
    model_name: str = "gpt-4o-mini"
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "12345678")

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.github_token:
            raise ValueError(
                "GITHUB_TOKEN must be provided in environment variables.")
        if not os.getenv("LANGCHAIN_API_KEY"):
            raise ValueError(
                "LANGCHAIN_API_KEY must be provided in environment variables.")


class HealthcareGraphRAG:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.config.validate()

        try:
            self.llm = OpenAI(base_url=config.endpoint,
                              api_key=config.github_token)
            self.graph = Neo4jGraph(
                url=config.neo4j_uri,
                username=config.neo4j_username,
                password=config.neo4j_password
            )
            self.schema = self.graph.get_structured_schema
            logger.info("Neo4j schema loaded successfully.")

            # Sử dụng HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Embeddings initialized successfully.")

            # Khởi tạo LangSmith client
            self.langsmith_client = Client()
            logger.info("LangSmith client initialized successfully.")
        except OpenAIError as e:
            logger.error(f"OpenAI initialization failed: {str(e)}")
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            raise ValueError(f"Failed to initialize GraphRAG system: {str(e)}")

    def search_context(self, query_text: str, top_k: int = 10) -> List[Any]:
        query_embedding = self.embeddings.embed_documents([query_text])[0]
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
        with self.graph._driver.session() as session:
            for index_name in indexes:
                query = """
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
                YIELD node, score
                RETURN node.text AS context, score
                ORDER BY score DESC
                LIMIT $top_k
                """
                result = session.run(
                    query, index_name=index_name, top_k=top_k, query_embedding=query_embedding)
                all_results.extend(
                    [(record["context"], record["score"]) for record in result])
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def _generate_cypher_query(self, question: str, schema: Dict[str, Any]) -> str:
        prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
            Based on the Neo4j schema:
            {schema}

            Generate an accurate Cypher query to answer: "{question}".
            - Use labels: Patient(name, age, gender, blood_type, admission_type, date_of_admission, discharge_date),
            Disease(name), Doctor(name), Hospital(name), InsuranceProvider(name), Room(room_number),
            Medication(name), TestResults(test_outcome), Billing(amount).
            - Relationships: HAS_DISEASE, TREATED_BY, ADMITTED_TO, COVERED_BY, STAY_IN, TAKE_MEDICATION,
            UNDERGOES, HAS_BILLING, WORKS_AT, PRESCRIBES, RELATED_TO_TEST, PARTNERS_WITH.
            - For name attributes, use case-insensitive matching by applying toLower() on both the node's property and the input value, e.g., WHERE toLower(n.name) = toLower('value').
            - Return only the Cypher query, no markdown or extra text.
            - Ensure valid syntax with MATCH, RETURN, LIMIT 5, matching the schema.
            """
        )
        try:
            response = self.llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt.format(
                        schema=schema, question=question)},
                ],
                temperature=0.3,
                max_tokens=1000,
                model=self.config.model_name
            )
            query = response.choices[0].message.content.strip()
            query = re.sub(r"```cypher|```", "", query).strip()
            logger.info(f"Generated Cypher query: {query}")
            return query
        except OpenAIError as e:
            logger.error(f"Failed to generate Cypher query: {str(e)}")
            raise ValueError(f"Cypher query generation failed: {str(e)}")

    def _validate_cypher_query(self, query: str, schema: Dict[str, Any]) -> str:
        prompt = PromptTemplate(
            input_variables=["schema", "query"],
            template="""
            Based on the Neo4j schema:
            {schema}

            Validate the following Cypher query:
            {query}

            Return a single line:
            - 'VALID' if the query is syntactically and semantically correct.
            - 'INVALID: <brief reason>' if invalid (e.g., 'INVALID: Missing MATCH').
            No additional explanation.
            """
        )
        try:
            response = self.llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt.format(
                        schema=schema, query=query)},
                ],
                temperature=0.3,
                max_tokens=100,
                model=self.config.model_name
            )
            result = response.choices[0].message.content.strip()
            if not result.startswith("VALID"):
                logger.warning(f"Invalid Cypher query detected: {result}")
                raise ValueError(f"Invalid Cypher query: {result}")
            return query
        except OpenAIError as e:
            logger.error(f"Failed to validate Cypher query: {str(e)}")
            raise ValueError(f"Cypher query validation failed: {str(e)}")

    def _execute_query(self, query: str) -> List[Dict[str, Any]]:
        try:
            result = self.graph.query(query)
            return [
                {k: v.iso_format() if isinstance(v, Date)
                 else v for k, v in record.items()}
                for record in result
            ]
        except Exception as e:
            logger.error(f"Failed to execute Cypher query: {str(e)}")
            raise ValueError(f"Query execution failed: {str(e)}")

    def _generate_response(self, question: str, query_result: List[Dict[str, Any]], context: List[Any]) -> Dict[str, Any]:
        prompt = PromptTemplate(
            input_variables=["question", "result", "context"],
            template="""
            Based on the question: "{question}"
            Neo4j results: {result}
            Vector search context: {context}
            Generate a concise, accurate response in English.
            If no results, return: "No information found."
            """
        )
        try:
            context_text = [doc[0] if isinstance(doc, tuple) and len(
                doc) > 0 else str(doc) for doc in context]
            response = self.llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt.format(
                        question=question, result=query_result, context=context_text)},
                ],
                temperature=0.3,
                max_tokens=1000,
                model=self.config.model_name
            )
            return {"query": query_result, "response": response.choices[0].message.content.strip()}
        except OpenAIError as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise ValueError(f"Response generation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in response generation: {str(e)}")
            raise ValueError(f"Unexpected error: {str(e)}")

    def get_pipeline(self) -> RunnableSequence:
        vector_retriever = RunnableLambda(self.search_context)
        pipeline = (
            {"question": RunnablePassthrough(), "schema": RunnableLambda(
                lambda _: self.schema)}
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "schema": x["schema"],
                "context": vector_retriever.invoke(x["question"])
            }).with_config(run_name="RetrieveContext")  # Đặt tên cho bước
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "schema": x["schema"],
                "context": (lambda ctx: (
                    print("Context details:"),
                    print(f"Raw context: {ctx}"),
                    print(f"Number of documents: {len(ctx)}"),
                    [print(f"Doc {i}: content={doc[0] if isinstance(doc, tuple) else doc}, metadata={None}")
                     for i, doc in enumerate(ctx)] if ctx else print("No documents found in context"),
                    ctx
                )[-1])(x["context"])
            }).with_config(run_name="DebugContext")  # Đặt tên cho bước
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "schema": x["schema"],
                "context": x["context"],
                "query": self._generate_cypher_query(x["question"], x["schema"])
            }).with_config(run_name="GenerateCypherQuery")  # Đặt tên cho bước
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "schema": x["schema"],
                "context": x["context"],
                "query": self._validate_cypher_query(x["query"], x["schema"])
            }).with_config(run_name="ValidateCypherQuery")  # Đặt tên cho bước
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "schema": x["schema"],
                "context": x["context"],
                "query": x["query"],
                "result": self._execute_query(x["query"])
            }).with_config(run_name="ExecuteCypherQuery")  # Đặt tên cho bước
            | RunnableLambda(lambda x: self._generate_response(x["question"], x["result"], x["context"])
                             ).with_config(run_name="GenerateResponse")  # Đặt tên cho bước
        )
        return pipeline

    def run(self, question: str) -> Dict[str, Any]:
        try:
            pipeline = self.get_pipeline()
            result = pipeline.invoke(question)
            logger.info(f"Successfully processed query: '{question}'")
            return result
        except ValueError as e:
            logger.error(f"Pipeline failed for '{question}': {str(e)}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in pipeline for '{question}': {str(e)}", exc_info=True)
            return {"query": None, "response": f"Error: {str(e)}"}


def main() -> None:
    config = Config()
    try:
        config.validate()
        chatbot = HealthcareGraphRAG(config)
        question = "Which patients are has disease Hypertension?"
        result = chatbot.run(question)
        print(f"🔍 Question: {question}")
        print(f"📝 Response: {result['response']}")
        print(f"🔗 Neo4j Results: {result['query']}")
    except ValueError as e:
        logger.error(f"Startup failed: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

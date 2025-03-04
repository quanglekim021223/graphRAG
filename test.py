import os
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from neo4j.time import Date
from openai import OpenAI, OpenAIError
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence

# Configure logging globally (can be overridden in production)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration class for Healthcare GraphRAG system.

    Attributes:
        github_token: GitHub token for OpenAI API authentication.
        endpoint: OpenAI API endpoint URL.
        model_name: Model name for text generation.
        embedding_model: Model name for embedding generation.
        neo4j_uri: Neo4j database URI.
        neo4j_username: Neo4j database username.
        neo4j_password: Neo4j database password.
    """
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    endpoint: str = "https://models.inference.ai.azure.com"
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "12345678")

    def validate(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        if not self.github_token:
            raise ValueError(
                "GITHUB_TOKEN must be provided in environment variables.")


class HealthcareGraphRAG:
    """A GraphRAG system for healthcare queries using Neo4j and OpenAI.

    Attributes:
        config: Configuration object for the system.
        llm: OpenAI client for text generation.
        graph: Neo4j database connection.
        schema: Neo4j schema dictionary.
        embeddings: Embedding generator for vector retrieval.
        vector_store: Vector store for Neo4j nodes.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the HealthcareGraphRAG system.

        Args:
            config: Configuration object with system settings.

        Raises:
            ValueError: If initialization fails due to connectivity or configuration issues.
        """
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

            self.embeddings = OpenAIEmbeddings(
                openai_api_base=config.endpoint,
                openai_api_key=config.github_token,
                model=config.embedding_model
            )

            self.vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=config.neo4j_uri,
                username=config.neo4j_username,
                password=config.neo4j_password,
                index_name="healthcare_vector_index",
                node_label=["Patient", "Disease", "Doctor", "Hospital",
                            "InsuranceProvider", "Room", "Medication", "TestResults", "Billing"],
                text_node_properties=["name", "age", "gender", "blood_type", "admission_type",
                                      "date_of_admission", "discharge_date", "test_outcome", "amount"],
                embedding_node_property="embedding"
            )
            logger.info("Vector store initialized successfully.")
        except OpenAIError as e:
            logger.error(f"OpenAI initialization failed: {str(e)}")
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            raise ValueError(f"Failed to initialize GraphRAG system: {str(e)}")

    def _generate_cypher_query(self, question: str, schema: Dict[str, Any]) -> str:
        """Generate a Cypher query from a user question.

        Args:
            question: The user's natural language question.
            schema: Neo4j schema dictionary.

        Returns:
            str: A valid Cypher query string.

        Raises:
            ValueError: If query generation fails.
        """
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
            - Use toLower() for name attributes to ensure case-insensitive matching.
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
        """Validate a Cypher query against the schema.

        Args:
            query: Cypher query string to validate.
            schema: Neo4j schema dictionary.

        Returns:
            str: Validated Cypher query string.

        Raises:
            ValueError: If the query is invalid.
        """
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
        """Execute a Cypher query on Neo4j.

        Args:
            query: Cypher query string.

        Returns:
            List[Dict[str, Any]]: Query results as a list of dictionaries.

        Raises:
            ValueError: If query execution fails.
        """
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
        """Generate a natural language response from query results and context.

        Args:
            question: User's natural language question.
            query_result: Results from Neo4j query.
            context: Context from vector retrieval.

        Returns:
            Dict[str, Any]: Dictionary with query results and response text.

        Raises:
            ValueError: If response generation fails.
        """
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
            context_text = [doc.page_content if hasattr(
                doc, 'page_content') else str(doc) for doc in context]
            response = self.llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt.format(
                        question=question,
                        result=query_result,
                        context=context_text
                    )},
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
        """Construct the GraphRAG pipeline.

        Returns:
            RunnableSequence: The pipeline for processing queries.
        """
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5})

        return (
            {"question": RunnablePassthrough(), "schema": RunnableLambda(
                lambda _: self.schema)}
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "schema": x["schema"],
                "context": vector_retriever.invoke(x["question"])
            })
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "schema": x["schema"],
                "context": x["context"],
                "query": self._generate_cypher_query(x["question"], x["schema"])
            })
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "schema": x["schema"],
                "context": x["context"],
                "query": self._validate_cypher_query(x["query"], x["schema"])
            })
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "schema": x["schema"],
                "context": x["context"],
                "query": x["query"],
                "result": self._execute_query(x["query"])
            })
            | RunnableLambda(lambda x: self._generate_response(x["question"], x["result"], x["context"]))
        )

    def run(self, question: str) -> Dict[str, Any]:
        """Run the GraphRAG pipeline for a given question.

        Args:
            question: User's natural language question.

        Returns:
            Dict[str, Any]: Dictionary containing query results and response text.

        Raises:
            ValueError: If pipeline execution fails.
        """
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
    """Main entry point for the Healthcare GraphRAG system."""
    config = Config()
    try:
        config.validate()
        chatbot = HealthcareGraphRAG(config)
        question = "Bệnh nhân nào được điều trị bởi bác sĩ kevin wells?"
        result = chatbot.run(question)
        print(f"🔍 Question: {question}")
        print(f"📝 Response: {result['response']}")
        print(f"🔗 Neo4j Results: {result['query']}")
    except ValueError as e:
        logger.error(f"Startup failed: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

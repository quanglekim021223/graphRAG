#!/usr/bin/env python
# coding: utf-8

import os
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama, OllamaEmbeddings
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from dotenv import load_dotenv
import logging
import re
import streamlit as st

# Thiết lập logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# Kết nối Neo4j
graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)

driver = GraphDatabase.driver(
    uri=os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# Cấu hình embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Tự động lấy schema
schema = graph.get_structured_schema
node_labels = list(schema["node_props"].keys())
logger.info(f"Detected node labels: {node_labels}")

# Lấy thuộc tính kiểu chuỗi cho mỗi nhãn
index_properties = {}
for label, props in schema["node_props"].items():
    string_props = [prop["property"]
                    for prop in props if prop.get("type") in ["STRING", "string"]]
    if string_props:
        index_properties[label] = string_props
logger.info(f"Detected string properties: {index_properties}")

# Tạo vector indices cho tất cả nhãn
vector_indices = {}
for label in node_labels:
    text_properties = index_properties.get(label, [])
    if not text_properties:
        logger.warning(
            f"No string properties found for label {label}. Skipping vector index.")
        continue
    try:
        vector_indices[label] = Neo4jVector.from_existing_graph(
            embeddings,
            search_type="hybrid",
            node_label=label,
            text_node_properties=text_properties,
            embedding_node_property="embedding"
        )
        logger.info(
            f"Vector index created for {label} with properties {text_properties}")
    except Exception as e:
        logger.error(f"Failed to create vector index for {label}: {e}")

vector_retrievers = {label: index.as_retriever()
                     for label, index in vector_indices.items()}

# Tạo full-text index với thuộc tính duy nhất


def create_fulltext_index(tx):
    labels = "|".join(node_labels)
    unique_props = list(
        set([prop for label in index_properties for prop in index_properties[label]]))
    props = ", ".join([f"n.{prop}" for prop in unique_props])
    query = f'''
    CREATE FULLTEXT INDEX `fulltext_entity_id`
    IF NOT EXISTS
    FOR (n:{labels})
    ON EACH [{props}];
    '''
    logger.info(f"Creating fulltext index with query: {query}")
    tx.run(query)


def create_index():
    with driver.session() as session:
        try:
            session.execute_write(create_fulltext_index)
            logger.info("Fulltext index created successfully.")
        except Exception as e:
            logger.error(f"Failed to create fulltext index: {e}")


create_index()

# Định nghĩa Entities


class Entities(BaseModel):
    """Identifying information about entities."""
    names: list[str] = Field(
        description="All the person, organization, or business entities that appear in the text",
    )


# Tạo entity extractor
llm = ChatOllama(model="llama3.2", temperature=0.1)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are extracting organization and person entities from the text."),
        ("human",
         "Use the given format to extract information from the following input: {question}"),
    ]
)
entity_chain = llm.with_structured_output(Entities)

# Hàm tạo truy vấn full-text


def generate_full_text_query(input: str) -> str:
    words = [re.sub(r'[^\w\s]', '', el) for el in input.split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    logger.info(f"Generated full-text query: {full_text_query}")
    return full_text_query.strip()

# Graph retriever


def graph_retriever(question: str) -> str:
    result = set()
    try:
        entities = entity_chain.invoke(question)
        logger.info(f"Extracted entities: {entities}")
        for entity in entities.names:
            full_text_query = generate_full_text_query(entity)
            if not full_text_query:
                continue
            response = graph.query(
                """
                CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit: 5})
                YIELD node, score
                CALL {
                    WITH node
                    MATCH (node)-[r]->(neighbor)
                    RETURN labels(node)[0] + ': ' + coalesce(node.name, 'Unknown') + 
                           ' - ' + type(r) + ' -> ' + labels(neighbor)[0] + ': ' + coalesce(neighbor.name, 'Unknown') AS output
                    UNION ALL
                    WITH node
                    MATCH (node)<-[r]-(neighbor)
                    RETURN labels(neighbor)[0] + ': ' + coalesce(neighbor.name, 'Unknown') + 
                           ' - ' + type(r) + ' -> ' + labels(node)[0] + ': ' + coalesce(node.name, 'Unknown') AS output
                }
                RETURN output LIMIT 10
                """,
                {"query": full_text_query},
            )
            result.update([el["output"] for el in response if el["output"]])
    except Exception as e:
        logger.error(f"Error in graph retrieval: {e}")
        return "I couldn’t retrieve graph data due to an error."

    result_str = "\n".join(sorted(result))
    logger.info(f"Graph data retrieved: {result_str}")
    return result_str or "No relevant relationships found."

# Full retriever


def full_retriever(question: str):
    graph_data = graph_retriever(question)
    vector_data = []
    for label, retriever in vector_retrievers.items():
        docs = retriever.invoke(question)
        vector_data.extend([f"{label}: {doc.page_content}" for doc in docs])
    vector_data_str = "\n".join(vector_data)
    final_data = f"""Graph data:
{graph_data}
Vector data:
{vector_data_str}
    """
    logger.info(f"Final context: {final_data}")
    return final_data


# Template
template = """Answer the question based only on the following context:
{context}

The context includes 'Graph data' (relationships between entities) and 'Vector data' (text from similar documents).
- Use 'Graph data' to describe relationships like '[Entity1] [relationship] [Entity2]' or attributes like '[Entity] has [attribute]'.
- Use 'Vector data' only if 'Graph data' is empty or does not provide enough information, and ignore any Vector data where entities like 'Doctor' or 'Hospital' have names like 'Diabetes' unless explicitly relevant.
- Answer in concise, natural language. Capitalize names properly.
- If the question has multiple parts, address each part separately.
- If more than 5 results are found, list only the first 5 and add "and others" at the end.
- If no relevant data is found, say "I couldn’t find any information about that."

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Chain
chain = (
    {"context": full_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit Chatbot UI


def main():
    st.title("Chatbot Truy vấn Thông tin Bệnh nhân")

    # Khởi tạo lịch sử trò chuyện trong session_state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Chào bạn! Tôi có thể giúp bạn tra cứu thông tin bệnh nhân. Hãy nhập câu hỏi của bạn dưới đây."}
        ]

    # Khu vực hiển thị lịch sử trò chuyện
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Ô nhập liệu ở dưới cùng
    question = st.chat_input(
        "Nhập câu hỏi của bạn (ví dụ: 'bệnh nhân Adrienne Bell có những thông tin gì')")

    # Xử lý khi người dùng gửi câu hỏi
    if question:
        # Thêm câu hỏi của người dùng vào lịch sử
        st.session_state.messages.append({"role": "user", "content": question})

        # Hiển thị tin nhắn người dùng ngay lập tức
        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)

        # Xử lý câu hỏi và nhận câu trả lời
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Đang xử lý..."):
                    try:
                        answer = chain.invoke(question)
                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer})
                        logger.info(f"Question: {question}")
                        logger.info(f"Answer: {answer}")
                    except Exception as e:
                        error_msg = f"Có lỗi xảy ra: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg})
                        logger.error(f"Error in chain execution: {e}")

    # Tự động cuộn xuống dưới cùng
    st.markdown(
        """
        <script>
        var chatContainer = window.parent.document.getElementsByClassName('stChatMessageContainer')[0];
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        </script>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

driver.close()

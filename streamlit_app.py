import streamlit as st
import requests

# URL của API Flask
API_URL = "http://localhost:5000/chat"


def main():
    st.title("Healthcare GraphRAG Chatbot")

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

        # Gửi câu hỏi đến API Flask và nhận câu trả lời
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Đang xử lý..."):
                    try:
                        response = requests.post(
                            API_URL, json={"question": question})
                        response_data = response.json()
                        answer = response_data.get(
                            "response", "Không có câu trả lời.")
                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"Có lỗi xảy ra: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg})

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

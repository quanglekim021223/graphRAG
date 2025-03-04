from flask import Flask, request, jsonify
from test import Config, HealthcareGraphRAG

app = Flask(__name__)

# Load configuration
config = Config()
config.validate()

# Initialize the HealthcareGraphRAG system
chatbot = HealthcareGraphRAG(config)


@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Healthcare GraphRAG chatbot!"


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        result = chatbot.run(question)
        return jsonify({
            "question": question,
            "response": result['response'],
            "query": result['query']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

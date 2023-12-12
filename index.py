from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

import os
from sentence_transformers import SentenceTransformer
import pinecone

app = Flask(__name__)

# Get Pinecone API key, environment, and index name
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_environment = os.environ.get('PINECONE_ENVIRONMENT')
pinecone_index_name = os.environ.get('PINECONE_INDEX_NAME')

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

@app.route('/semanticSearch', methods=['POST'])
def semantic_search():
    request_json = request.get_json(silent=True)

    if request_json and 'topK' in request_json:
        topK = int(request_json['topK'])

    if request_json and 'query' in request_json:
        query = request_json['query']
        query_embedding = model.encode([query])

        search_results = pinecone.query(index_name=pinecone_index_name, query_vector=query_embedding[0], top_k=topK)

        if search_results['matches']:
            # Get the metadata of the matched vectors
            matched_vectors_ids = [match['id'] for match in search_results['matches']]
            metadata = pinecone.retrieve(index_name=pinecone_index_name, ids=matched_vectors_ids)

            # You can return the metadata or perform further processing as needed
            return jsonify({'candidates': metadata})
        else:
            return jsonify({'candidates': []})

if __name__ == '__main__':
    app.run(port=8080, debug=True)
from flask import Flask, request, jsonify, Response
import os
from sentence_transformers import SentenceTransformer
import pinecone
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Get Pinecone API key, environment, and index name
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_environment = os.environ.get('PINECONE_ENVIRONMENT')
pinecone_index_name = os.environ.get('PINECONE_INDEX_NAME')

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
index = pinecone.Index(pinecone_index_name)

model = SentenceTransformer('BAAI/bge-large-en-v1.5')

@app.route('/semanticSearch', methods=['POST'])
def semantic_search():
    request_json = request.get_json(silent=True)
    if request_json and 'topK'  not in request_json or 'query' not in request_json:
        return {'error': "Invalid request"}
    
    topK = int(request_json['topK'])
    query = request_json['query']    

    query_embedding = model.encode([query]).tolist()

    search_results = index.query([], top_k=topK, include_metadata=True, vector=query_embedding[0])
    # search_results = index.query(top_k=topK, include_metadata=True, vector=query_embedding[0])
    print({'candidates': search_results['matches']})

    if search_results and search_results['matches'] is not None: 
        # search_results = jsonify(search_results)
        # print(search_results['matches'])
        return Response(json.dumps({'candidates': search_results['matches']}), mimetype='application/json')
    
    return Response(json.dumps({'candidates': []}, mimetype='application/json'))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
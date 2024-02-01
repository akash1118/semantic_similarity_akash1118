from flask import Flask, request, jsonify
from semantic_similarity import semanticSimilarity

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health():
    return "Server is Up and Running"

@app.route('/semantic-similarity', methods = ['POST'])
def get_semantic_similarity():
    
    try:
        if request.method == 'POST':
            data = request.get_json()
            sent1 = data["text1"]
            sent2 = data["text2"]
            semantic_similarity_obj = semanticSimilarity(sent1, sent2)
            similarity_score = semantic_similarity_obj.score_calculations()
            response_obj = {
                "similarity score": str(similarity_score)
            }
            return jsonify(response_obj), 200
        else:
            return {"message":"Invalid Request Method"},405
    except Exception as e:
        print(e)
        return jsonify({"error": "Error Occured"}), 400
if __name__ == '__main__':
    app.run(debug=True)
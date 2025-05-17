from flask import Flask, request, jsonify, render_template
from model.bert_model import SentimentAnalyzer
from gingerit.gingerit import GingerIt  # Correct import for this specific fork

app = Flask(__name__)
analyzer = SentimentAnalyzer()

def get_corrected_text(text):
    try:
        return GingerIt().parse(text)['result']
    except:
        return text # Fallback to original text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Please enter text"}), 400
        
        corrected = get_corrected_text(text)
        result = analyzer.analyze(text)
        
        return jsonify({
            "text": text,
            "corrected": corrected,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
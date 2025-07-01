"""
Web Interface for Named Entity Linking System
Flask-based UI for visualizing entity linking results
"""

from flask import Flask, render_template_string, request, jsonify
import json
from nel_system import NamedEntityLinker, format_results
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize NEL system (will be done on first request to avoid startup delay)
nel_system = None

def get_nel_system():
    """Lazy initialization of NEL system"""
    global nel_system
    if nel_system is None:
        logger.info("Initializing NEL system...")
        nel_system = NamedEntityLinker()
        logger.info("NEL system ready!")
    return nel_system

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Named Entity Linking System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(45deg, #2196F3, #21CBF3);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 40px;
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        .input-section label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
            color: #333;
        }
        
        #textInput {
            width: 100%;
            min-height: 150px;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        #textInput:focus {
            outline: none;
            border-color: #2196F3;
        }
        
        .button-group {
            margin: 20px 0;
            display: flex;
            gap: 15px;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #2196F3, #21CBF3);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(33, 150, 243, 0.3);
        }
        
        .btn-secondary {
            background: #f5f5f5;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #2196F3;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results-section {
            margin-top: 30px;
        }
        
        .stats-box {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #2196F3;
        }
        
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        
        .entity-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            margin-bottom: 15px;
            overflow: hidden;
            transition: transform 0.2s;
        }
        
        .entity-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }
        
        .entity-header {
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .entity-text {
            font-size: 1.2em;
            font-weight: bold;
            color: #2196F3;
        }
        
        .entity-type {
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 4px 8px;
            border-radius: 15px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        
        .confidence-bar {
            height: 4px;
            background: #e0e0e0;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            transition: width 0.5s ease;
        }
        
        .entity-body {
            padding: 15px;
        }
        
        .linked-info {
            display: grid;
            grid-template-columns: auto 1fr auto;
            gap: 15px;
            align-items: center;
        }
        
        .wikipedia-icon {
            width: 40px;
            height: 40px;
            background: #000;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .linked-details h4 {
            margin-bottom: 5px;
            color: #333;
        }
        
        .linked-details p {
            color: #666;
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        .confidence-score {
            text-align: center;
            font-weight: bold;
            color: #4caf50;
        }
        
        .no-results {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .error-message {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 10px;
            }
            
            .main-content {
                padding: 20px;
            }
            
            .header {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .button-group {
                flex-direction: column;
            }
            
            .linked-info {
                grid-template-columns: 1fr;
                text-align: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔗 Named Entity Linking</h1>
            <p>Intelligent entity detection and Wikipedia linking system</p>
        </div>
        
        <div class="main-content">
            <div class="input-section">
                <label for="textInput">Enter your text to analyze:</label>
                <textarea id="textInput" placeholder="Type or paste your text here... For example: 'Barack Obama was born in Hawaii and served as the 44th President of the United States. He graduated from Harvard Law School and worked in Chicago before entering politics.'"></textarea>
            </div>
            
            <div class="button-group">
                <button class="btn-primary" onclick="processText()">🔍 Analyze Text</button>
                <button class="btn-secondary" onclick="loadExample()">📝 Load Example</button>
                <button class="btn-secondary" onclick="clearAll()">🗑️ Clear</button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing entities...</p>
            </div>
            
            <div id="error" class="error-message" style="display: none;"></div>
            
            <div class="results-section" id="results" style="display: none;">
                <h2>📊 Results</h2>
                <div class="stats-box" id="stats"></div>
                <div id="entities"></div>
            </div>
        </div>
    </div>

    <script>
        function loadExample() {
            const exampleText = `Barack Obama was born in Honolulu, Hawaii. He served as the 44th President of the United States from 2009 to 2017. Obama graduated from Harvard Law School, where he was the first African-American president of the Harvard Law Review. Before his presidency, he worked as a community organizer in Chicago and served as a U.S. Senator from Illinois.

Microsoft Corporation is an American multinational technology corporation headquartered in Redmond, Washington. The company was founded by Bill Gates and Paul Allen on April 4, 1975. Microsoft is best known for its Windows operating systems and Office productivity suite.

The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair. The tower was designed by Gustave Eiffel and stands at 324 meters tall.`;
            
            document.getElementById('textInput').value = exampleText;
        }
        
        function clearAll() {
            document.getElementById('textInput').value = '';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
        }
        
        async function processText() {
            const text = document.getElementById('textInput').value.trim();
            
            if (!text) {
                showError('Please enter some text to analyze.');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
                
            } catch (error) {
                showError('An error occurred while processing the text: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
        
        function displayResults(data) {
            // Display statistics
            const statsHtml = `
                <div class="stat-item">
                    <div class="stat-number">${data.statistics.total_entities}</div>
                    <div class="stat-label">Entities Detected</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">${data.statistics.linked_entities}</div>
                    <div class="stat-label">Successfully Linked</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">${(data.statistics.average_confidence * 100).toFixed(1)}%</div>
                    <div class="stat-label">Average Confidence</div>
                </div>
            `;
            document.getElementById('stats').innerHTML = statsHtml;
            
            // Display entities
            const entitiesDiv = document.getElementById('entities');
            
            if (data.entities.length === 0) {
                entitiesDiv.innerHTML = '<div class="no-results">No entities found in the text.</div>';
            } else {
                let entitiesHtml = '';
                
                data.entities.forEach(entity => {
                    const confidencePercent = (entity.confidence * 100).toFixed(1);
                    const confidenceColor = entity.confidence > 0.8 ? '#4caf50' : 
                                          entity.confidence > 0.6 ? '#ff9800' : '#f44336';
                    
                    entitiesHtml += `
                        <div class="entity-card">
                            <div class="entity-header">
                                <span class="entity-text">${entity.text}</span>
                                <span class="entity-type">${entity.type}</span>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${confidencePercent}%; background: ${confidenceColor}"></div>
                                </div>
                            </div>
                            <div class="entity-body">
                                <div class="linked-info">
                                    <div class="wikipedia-icon">W</div>
                                    <div class="linked-details">
                                        <h4><a href="${entity.linked_to.url}" target="_blank">${entity.linked_to.title}</a></h4>
                                        <p>${entity.linked_to.description}</p>
                                    </div>
                                    <div class="confidence-score">
                                        ${confidencePercent}%<br>
                                        <small>confidence</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                entitiesDiv.innerHTML = entitiesHtml;
            }
            
            document.getElementById('results').style.display = 'block';
        }
        
        // Allow Enter key to process text (Ctrl+Enter for newline)
        document.getElementById('textInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                processText();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/process', methods=['POST'])
def process_text():
    """Process text and return entity linking results"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'})
        
        # Get NEL system
        nel = get_nel_system()
        
        # Process text
        linked_entities = nel.link_entities(text)
        
        # Format results
        results = format_results(linked_entities)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'nel_system_loaded': nel_system is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Named Entity Linking Web Interface...")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
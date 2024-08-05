from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Placeholder functions for process_pdf_query and read_pdf
def process_pdf_query(filename, question, mode):
    return f"Processed {filename} with question '{question}' in mode '{mode}'."

def read_pdf(filename):
    return f"Content of {filename}"

# Global variable to store the PDF text for interactive mode
pdf_text = ""

@app.route('/', methods=['GET', 'POST'])
def index():
    global pdf_text
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        file.save(filename)

        question = request.form['question']
        mode = request.form.get('mode', 'question')  # Default to 'question'

        if mode not in ['question', 'summarize', 'interactive']:
            return jsonify({"error": "Invalid mode selected"}), 400

        if mode == 'interactive':
            pdf_text = read_pdf(filename)
            response = "Interactive mode started. You can now ask questions about the content."
        else:
            response = process_pdf_query(filename, question, mode)

        return render_template('upload.html', response=response)

    return render_template('upload.html')

@app.route('/ask_interactive', methods=['POST'])
def ask_interactive():
    global pdf_text
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Combine PDF text with the question for context
        combined_prompt = f"Based on the following content:\n{pdf_text}\n\nQuestion: {question}\nAnswer:"
        response = process_pdf_query("", combined_prompt, mode='question')  # Adjust if needed
        return jsonify({'answer': response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/summarize', methods=['POST'])
def summarize():
    global pdf_text
    try:
        response = process_pdf_query('','', mode='summarize')  #not tested
        return jsonify({'answer': response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'file_path': file_path})

@app.route('/tutor', methods=['POST'])
def tutor_mode():
    file_path = request.json.get('file_path')
    if not file_path:
        return jsonify({'error': 'File path is required'})
    
    # Get the questions from the tutor mode
    tutor_response = process_pdf_query(file_path, query="", mode='tutor')
    
    return jsonify({'tutor_response': tutor_response})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
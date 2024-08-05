from flask import Flask, render_template
import subprocess
import my_pdf_processor

app = Flask(__name__)

@app.route('/question')
def index():
    # Run the Python script and capture its output
    result = subprocess.run(['python3', 'script.py'], capture_output=True, text=True)
    output = result.stdout.strip()
    
    return render_template('index.html', output=output)

if __name__ == "__main__":
    app.run(debug=True)

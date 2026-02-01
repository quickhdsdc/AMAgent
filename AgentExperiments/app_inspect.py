import os
import sys
import csv
import argparse
import json
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for

sys.path.append(os.getcwd())

try:
    from benchmark_cls_AMagent import _call_llm, _build_supervisor_prompt, _extract_label_from_response
    print("Successfully imported benchmark_cls_AMagent utilities.")
except ImportError as e:
    print(f"Error importing benchmark_cls_AMagent: {e}")
    sys.exit(1)

app = Flask(__name__, template_folder='templates', static_folder='assets')

DATA_FILE = None
CSV_DATA = []
HEADER = []

def load_csv(filepath):
    """Loads the CSV data into memory."""
    global CSV_DATA, HEADER
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            HEADER = reader.fieldnames
            CSV_DATA = list(reader)
        print(f"Loaded {len(CSV_DATA)} rows from {filepath}")
    except Exception as e:
        print(f"Error loading CSV {filepath}: {e}")
        CSV_DATA = []

@app.route('/')
def index():
    if not CSV_DATA:
        return "No data loaded or empty file.", 404
    return redirect(url_for('view_sample', row_idx=0))

@app.route('/view/<int:row_idx>')
def view_sample(row_idx):
    if not CSV_DATA or row_idx < 0 or row_idx >= len(CSV_DATA):
        return "Row index out of range", 404

    row = CSV_DATA[row_idx]
    total_rows = len(CSV_DATA)
    
    prev_idx = row_idx - 1 if row_idx > 0 else None
    next_idx = row_idx + 1 if row_idx < total_rows - 1 else None

    prompts = {}
    if row.get('prompts_debug'):
        try:
            prompts = json.loads(row['prompts_debug'])
        except:
            prompts = {"error": "Could not parse prompts_debug field"}

    return render_template(
        'view_sample.html',
        row=row,
        row_idx=row_idx,
        total_rows=total_rows,
        prev_idx=prev_idx,
        next_idx=next_idx,
        filename=os.path.basename(DATA_FILE),
        prompts=prompts
    )

@app.route('/evaluate', methods=['POST'])
def evaluate_response():
    """
    Endpoint to evaluate a specific response using an LLM.
    Expects JSON payload: { "response_text": "...", "response_type": "...", "gt_label": "..." }
    """
    data = request.json
    response_text = data.get('response_text', '')
    response_type = data.get('response_type', 'Unknown')
    gt_label = data.get('gt_label', 'Unknown')
    
    if not response_text:
        return jsonify({"error": "No response text provided"}), 400

    eval_prompt = (
        f"You are an expert AM Process Engineer. Please evaluate the following {response_type}.\n"
        f"Ground Truth Label: {gt_label}\n\n"
        f"Response to Evaluate:\n{response_text}\n\n"
        "Task:\n"
        "1. Check if the reasoning mechanism is sound based on LPBF physics.\n"
        "2. Verify if the final label matches the provided Ground Truth and if the justification is valid.\n"
        "3. Point out any hallucinations or incorrect assumptions.\n"
        "4. Provide a brief score (0-10) on quality.\n\n"
        "Return your evaluation in markdown format."
    )

    try:
        eval_result = _call_llm(eval_prompt, profile="default")
        return jsonify({"evaluation": eval_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AM Agent Results Viewer")
    parser.add_argument('--file', type=str, required=True, help="Path to the results CSV file")
    parser.add_argument('--port', type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()

    DATA_FILE = args.file
    if not os.path.exists(DATA_FILE):
        print(f"File not found: {DATA_FILE}")
        sys.exit(1)

    load_csv(DATA_FILE)
    
    print(f"Starting server on port {args.port}...")
    app.run(debug=True, port=args.port, host='0.0.0.0')

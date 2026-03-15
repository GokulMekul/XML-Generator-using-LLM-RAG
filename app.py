import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

# Import your modular functions
from SRC.data_loader import load_data, filter_data, create_chunks
from SRC.vector_store import build_vectorstore
from SRC.rag import rag_query
from SRC.XML_generator import generate_xml

# Initialize Flask
app = Flask(__name__)

# Load environment variables
load_dotenv()

# -----------------------------
# Build vectorstore at startup
# -----------------------------
try:
    path = os.path.join("Data", "Learn.pdf")
    documents = load_data(path)
    clean_data = filter_data(documents)
    texts = create_chunks(clean_data)
    vectorstore = build_vectorstore(texts)
except Exception as e:
    print("Error initializing vectorstore:", e)
    vectorstore = None

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    if request.method == "POST":
        user_input = request.form.get("user_input")
        mode = request.form.get("mode")

        if mode == "RAG":
            if vectorstore:
                try:
                    output = rag_query(user_input, vectorstore)
                except Exception as e:
                    output = f"Error running RAG: {e}"
            else:
                output = "Vectorstore not initialized."
        elif mode == "XML":
            try:
                output = generate_xml(user_input)
            except Exception as e:
                output = f"Error generating XML: {e}"
        else:
            output = "Invalid option selected."

    return render_template("index.html", output=output)

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
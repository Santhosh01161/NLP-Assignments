from flask import Flask, request, render_template
import pickle
import numpy as np
import re
import os

app = Flask(__name__)

# --- ROBUST PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')
CORPUS_PATH = os.path.join(MODEL_DIR, 'corpus.pkl')

# Define your available models and their filenames here
# Keys = ID used in HTML, Values = Filename in 'model' folder
MODEL_FILES = {
    'glove': 'embed_glove.pkl',
    'skipgram': 'embed_skipgram.pkl',
    'skipgram_neg': 'embed_skipgram_negative.pkl',
    'gensim': 'model_gensim.pkl'
}

# --- GLOBAL VARIABLES ---
# Structure: { 'model_id': { 'embeddings': ..., 'doc_vectors': [...] } }
model_data = {} 
corpus = []

def get_sentence_vector(text, embedding_model):
    """
    Calculates the vector for a sentence using a SPECIFIC embedding model.
    """
    # 1. Handle list inputs safely
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    text = str(text)
    
    # 2. Tokenize
    tokens = re.findall(r'\b\w+\b', text.lower())
    vectors = []
    
    # 3. Lookup vectors (handles both Dicts and Gensim models)
    for token in tokens:
        if hasattr(embedding_model, '__getitem__') and token in embedding_model:
            # Standard dictionary or KeyedVectors
            vectors.append(embedding_model[token])
        elif hasattr(embedding_model, 'wv') and token in embedding_model.wv:
            # Some Gensim models store vectors in .wv
            vectors.append(embedding_model.wv[token])
            
    if not vectors:
        # If no words found, return a zero vector
        # Try to find dimension size from the model
        try:
            # peek at the first item to get size
            if hasattr(embedding_model, 'vector_size'):
                dim = embedding_model.vector_size
            else:
                # Fallback for dicts: get first value's length
                dim = len(next(iter(embedding_model.values())))
        except:
            dim = 100 # absolute fallback
        return np.zeros(dim)
    
    return np.mean(vectors, axis=0)

def load_data():
    global corpus, model_data
    
    print("--- Loading Data ---")
    
    # 1. Load Corpus
    if not os.path.exists(CORPUS_PATH):
        print(f"❌ ERROR: Corpus not found at {CORPUS_PATH}")
        return
    with open(CORPUS_PATH, 'rb') as f:
        corpus = pickle.load(f)
    print(f"✅ Corpus loaded ({len(corpus)} documents).")

    # 2. Load Models and Pre-compute Vectors
    print("⏳ Loading models and pre-computing vectors... (This may take a moment)")
    
    for model_name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        
        if not os.path.exists(path):
            print(f"⚠️ Warning: Skipped {model_name} (File not found: {filename})")
            continue
            
        try:
            with open(path, 'rb') as f:
                loaded_model = pickle.load(f)
                
            # Pre-compute document vectors for THIS specific model
            doc_vecs = []
            for text in corpus:
                vec = get_sentence_vector(text, loaded_model)
                doc_vecs.append(vec)
                
            # Store everything in our global dictionary
            model_data[model_name] = {
                'embeddings': loaded_model,
                'doc_vectors': doc_vecs
            }
            print(f"   🔹 Loaded {model_name}")
            
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")

    print("🚀 Server is ready!")

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    search_query = ""
    selected_model = "glove" # Default
    
    if request.method == 'POST':
        search_query = request.form.get('query', '')
        selected_model = request.form.get('model_type', 'glove')
        
        # Check if model exists and query is valid
        if search_query and selected_model in model_data:
            
            # Get the specific data for the selected model
            current_data = model_data[selected_model]
            embeddings = current_data['embeddings']
            doc_vectors = current_data['doc_vectors']
            
            # Process Query
            query_vec = get_sentence_vector(search_query, embeddings)
            
            # Calculate Scores (Dot Product)
            scores = []
            for i, p_vec in enumerate(doc_vectors):
                score = np.dot(query_vec, p_vec)
                scores.append((score, corpus[i]))
            
            # Sort and Top 5
            scores.sort(key=lambda x: x[0], reverse=True)
            results = scores[:5]

    return render_template('index.html', 
                           query=search_query, 
                           results=results, 
                           selected_model=selected_model,
                           models=MODEL_FILES.keys()) # Send available models to UI

if __name__ == '__main__':
    load_data()
    app.run(debug=True, port=5000)
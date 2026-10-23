import os
import re
import json
import numpy as np

class MeyRAG:
    def __init__(self, db_path="knowledge_base.json"):
        self.db_path = db_path
        self.documents = {}  # {filename: text}
        self.chunks = []     # List of {"filename": str, "text": str, "id": int}
        self.embeddings = []  # List of TF-IDF vectors (numpy arrays)
        
        self.vocab = {}
        self.idf = {}
        
        self.load()

    def chunk_text(self, text, chunk_size=400, overlap=80):
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # Try to split at a space if not at the end
            if end < len(text):
                space_idx = text.rfind(' ', start, end)
                if space_idx != -1 and space_idx > start + chunk_size // 2:
                    end = space_idx
            chunks.append(text[start:end].strip())
            start = end - overlap
            if overlap >= chunk_size or start >= len(text) - overlap:
                break
        return [c for c in chunks if len(c) > 10]

    def build_tfidf(self):
        """Build TF-IDF weights and generate vectors for chunks"""
        if not self.chunks:
            self.embeddings = []
            self.vocab = {}
            self.idf = {}
            return
        
        token_pattern = re.compile(r'\w+', flags=re.UNICODE)
        
        # 1. Build term counts in document chunks
        doc_counts = {}
        tfs = []
        
        for chunk in self.chunks:
            words = token_pattern.findall(chunk["text"].lower())
            tf = {}
            for w in words:
                tf[w] = tf.get(w, 0) + 1
            tfs.append(tf)
            
            # Count document frequency
            for w in set(words):
                doc_counts[w] = doc_counts.get(w, 0) + 1
                
        # 2. Build vocabulary mapping
        self.vocab = {w: i for i, w in enumerate(doc_counts.keys())}
        num_docs = len(self.chunks)
        
        # 3. Calculate IDF values
        self.idf = {w: np.log((1 + num_docs) / (1 + count)) + 1 for w, count in doc_counts.items()}
        
        # 4. Generate vectors
        self.embeddings = []
        for tf in tfs:
            vec = np.zeros(len(self.vocab))
            for w, val in tf.items():
                if w in self.vocab:
                    vec[self.vocab[w]] = val * self.idf[w]
            # Normalize vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.embeddings.append(vec.tolist())

    def get_tfidf_vector(self, text):
        token_pattern = re.compile(r'\w+', flags=re.UNICODE)
        words = token_pattern.findall(text.lower())
        
        vec = np.zeros(len(self.vocab))
        if not self.vocab:
            return vec
            
        tf = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
            
        for w, val in tf.items():
            if w in self.vocab:
                vec[self.vocab[w]] = val * self.idf[w]
                
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def add_document(self, filename, text):
        # Remove old version if it exists
        self.delete_document(filename)
        
        self.documents[filename] = text
        new_chunks = self.chunk_text(text)
        
        # Assign IDs and add to database
        start_id = len(self.chunks)
        for i, chunk in enumerate(new_chunks):
            self.chunks.append({
                "id": start_id + i,
                "filename": filename,
                "text": chunk
            })

        # Re-build index
        self.build_tfidf()
        self.save()

    def delete_document(self, filename):
        if filename in self.documents:
            del self.documents[filename]
            
        # Re-filter and re-index chunks
        self.chunks = [c for c in self.chunks if c["filename"] != filename]
        
        for i, c in enumerate(self.chunks):
            c["id"] = i
            
        self.build_tfidf()
        self.save()

    def search(self, query, top_k=3):
        if not self.chunks or not self.embeddings:
            return []
            
        q_vec = self.get_tfidf_vector(query)
        similarities = []
        
        for i, emb in enumerate(self.embeddings):
            emb_arr = np.array(emb)
            # cosine similarity
            dot = np.dot(q_vec, emb_arr)
            norm_q = np.linalg.norm(q_vec)
            norm_e = np.linalg.norm(emb_arr)
            sim = dot / (norm_q * norm_e) if (norm_q > 0 and norm_e > 0) else 0.0
            similarities.append((sim, self.chunks[i]))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [{"similarity": float(item[0]), "text": item[1]["text"], "filename": item[1]["filename"]} for item in similarities[:top_k]]

    def save(self):
        data = {
            "documents": self.documents,
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "vocab": self.vocab,
            "idf": self.idf
        }
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving knowledge base: {e}")

    def load(self):
        if not os.path.exists(self.db_path):
            return
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.documents = data.get("documents", {})
            self.chunks = data.get("chunks", [])
            self.embeddings = data.get("embeddings", [])
            self.vocab = data.get("vocab", {})
            self.idf = data.get("idf", {})
        except Exception as e:
            print(f"Error loading knowledge base: {e}")

import os
import clip
import torch
import pickle
from PIL import Image
from datetime import datetime

# Load CLIP model once at module level
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

EMBEDDINGS_PATH = "clip_embeddings.pkl"

def load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_embeddings(embeddings):
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)

def embed_frame(frame_path):
    """Generate and store CLIP embedding for a saved frame."""
    embeddings = load_embeddings()

    if frame_path in embeddings:
        return  # already embedded

    try:
        image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embeddings[frame_path] = embedding.cpu()
        save_embeddings(embeddings)
    except Exception as e:
        print(f"Embedding failed for {frame_path}: {e}")

def embed_all_frames():
    """Embed all frames in the frames/ folder."""
    frames_dir = "frames"
    if not os.path.exists(frames_dir):
        return
    for fname in os.listdir(frames_dir):
        if fname.endswith(".jpg"):
            embed_frame(os.path.join(frames_dir, fname))

def search_frames(query, top_k=5):
    """Search frames using a natural language query."""
    embeddings = load_embeddings()
    if not embeddings:
        return []

    # Encode the text query
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    # Compute similarity scores
    scores = []
    for frame_path, img_embedding in embeddings.items():
        img_embedding = img_embedding.to(device)
        similarity = (text_embedding @ img_embedding.T).item()
        scores.append((frame_path, similarity))

    # Sort by similarity descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
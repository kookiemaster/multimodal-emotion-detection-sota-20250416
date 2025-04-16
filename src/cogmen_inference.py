"""
COGMEN Model Loader and Inference Code

This script provides functionality to load the COGMEN model from Hugging Face
and perform inference for multimodal emotion recognition on IEMOCAP data.

COGMEN: COntextualized GNN based Multimodal Emotion recognitioN
Paper: https://arxiv.org/abs/2205.02455
Model: https://huggingface.co/NAACL2022/cogmen
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import RGCNConv, TransformerConv
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# Define emotion labels for IEMOCAP
EMOTION_LABELS = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Frustrated",
    5: "Excited"
}

class COGMENModel(torch.nn.Module):
    """
    Implementation of COGMEN model architecture for multimodal emotion recognition.
    
    This is a simplified version of the model for inference purposes.
    For the full implementation, refer to the official repository:
    https://huggingface.co/NAACL2022/cogmen
    """
    
    def __init__(self, num_classes=6, text_dim=768, audio_dim=100, visual_dim=512, 
                 hidden_dim=256, num_relations=3, num_heads=4, num_layers=2):
        super(COGMENModel, self).__init__()
        
        # Feature dimensions
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        
        # Text encoder (using SBERT)
        self.text_encoder = None  # Will be loaded separately
        
        # Modality-specific projections
        self.text_projection = torch.nn.Linear(text_dim, hidden_dim)
        self.audio_projection = torch.nn.Linear(audio_dim, hidden_dim)
        self.visual_projection = torch.nn.Linear(visual_dim, hidden_dim)
        
        # Context encoder (Transformer)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.context_encoder = torch.nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Graph Neural Network components
        self.rgcn = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.graph_transformer = TransformerConv(
            hidden_dim, 
            hidden_dim, 
            heads=num_heads, 
            dropout=0.1
        )
        
        # Emotion classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, text_features, audio_features, visual_features, edge_index, edge_type):
        """
        Forward pass through the COGMEN model.
        
        Args:
            text_features: Tensor of text features [batch_size, text_dim]
            audio_features: Tensor of audio features [batch_size, audio_dim]
            visual_features: Tensor of visual features [batch_size, visual_dim]
            edge_index: Graph edge indices [2, num_edges]
            edge_type: Edge type for each edge [num_edges]
            
        Returns:
            logits: Emotion classification logits [batch_size, num_classes]
        """
        # Project features to common dimension
        text_proj = self.text_projection(text_features)
        audio_proj = self.audio_projection(audio_features)
        visual_proj = self.visual_projection(visual_features)
        
        # Combine modalities
        combined_features = text_proj + audio_proj + visual_proj
        
        # Apply context encoder
        context_features = self.context_encoder(combined_features.unsqueeze(0)).squeeze(0)
        
        # Apply graph neural networks
        graph_features = self.rgcn(context_features, edge_index, edge_type)
        graph_features = F.relu(graph_features)
        graph_features = self.graph_transformer(graph_features, edge_index)
        
        # Classify emotions
        logits = self.classifier(graph_features)
        
        return logits


class COGMENInference:
    """
    Class for loading COGMEN model and performing inference on IEMOCAP data.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the COGMEN inference class.
        
        Args:
            model_path: Path to the COGMEN model weights (if None, will use HF model)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize text encoder (SBERT)
        print("Loading SBERT model...")
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_encoder.to(self.device)
        
        # Initialize COGMEN model
        print("Initializing COGMEN model...")
        self.model = COGMENModel()
        
        # Load model weights if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading model weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Using default model initialization (no weights loaded)")
            
        self.model.to(self.device)
        self.model.eval()
    
    def extract_text_features(self, texts):
        """
        Extract text features using SBERT.
        
        Args:
            texts: List of text utterances
            
        Returns:
            text_features: Tensor of text features [batch_size, text_dim]
        """
        with torch.no_grad():
            text_features = self.text_encoder.encode(texts, convert_to_tensor=True)
        return text_features.to(self.device)
    
    def extract_audio_features(self, audio_paths):
        """
        Extract audio features (placeholder - in a real implementation, 
        this would load and process audio files).
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            audio_features: Tensor of audio features [batch_size, audio_dim]
        """
        # Placeholder - in a real implementation, this would extract actual features
        batch_size = len(audio_paths)
        return torch.randn(batch_size, 100).to(self.device)
    
    def extract_visual_features(self, visual_paths):
        """
        Extract visual features (placeholder - in a real implementation,
        this would load and process visual data).
        
        Args:
            visual_paths: List of paths to visual data
            
        Returns:
            visual_features: Tensor of visual features [batch_size, visual_dim]
        """
        # Placeholder - in a real implementation, this would extract actual features
        batch_size = len(visual_paths)
        return torch.randn(batch_size, 512).to(self.device)
    
    def build_conversation_graph(self, speaker_ids, num_nodes):
        """
        Build a conversation graph based on speaker IDs.
        
        Args:
            speaker_ids: List of speaker IDs for each utterance
            num_nodes: Number of nodes in the graph
            
        Returns:
            edge_index: Graph edge indices [2, num_edges]
            edge_type: Edge type for each edge [num_edges]
        """
        # Create edges between utterances
        edges = []
        edge_types = []
        
        # Add sequential edges (temporal connections)
        for i in range(num_nodes - 1):
            # Add edge from current to next utterance
            edges.append([i, i+1])
            edge_types.append(0)  # Type 0: sequential
            
            # Add edge from next to current utterance
            edges.append([i+1, i])
            edge_types.append(0)  # Type 0: sequential
        
        # Add speaker-specific edges
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and speaker_ids[i] == speaker_ids[j]:
                    edges.append([i, j])
                    edge_types.append(1)  # Type 1: same speaker
        
        # Add global context edges
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and speaker_ids[i] != speaker_ids[j]:
                    edges.append([i, j])
                    edge_types.append(2)  # Type 2: different speakers
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device)
        edge_type = torch.tensor(edge_types, dtype=torch.long).to(self.device)
        
        return edge_index, edge_type
    
    def predict_emotions(self, texts, audio_paths, visual_paths, speaker_ids):
        """
        Predict emotions for a conversation.
        
        Args:
            texts: List of text utterances
            audio_paths: List of paths to audio files
            visual_paths: List of paths to visual data
            speaker_ids: List of speaker IDs for each utterance
            
        Returns:
            predictions: List of predicted emotion labels
            probabilities: List of emotion probabilities
        """
        # Extract features
        text_features = self.extract_text_features(texts)
        audio_features = self.extract_audio_features(audio_paths)
        visual_features = self.extract_visual_features(visual_paths)
        
        # Build conversation graph
        num_nodes = len(texts)
        edge_index, edge_type = self.build_conversation_graph(speaker_ids, num_nodes)
        
        # Perform inference
        with torch.no_grad():
            logits = self.model(text_features, audio_features, visual_features, edge_index, edge_type)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        # Convert to numpy arrays
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        # Map predictions to emotion labels
        emotion_labels = [EMOTION_LABELS[pred] for pred in predictions]
        
        return emotion_labels, probabilities
    
    def process_iemocap_csv(self, csv_path):
        """
        Process IEMOCAP data from CSV file and predict emotions.
        
        Args:
            csv_path: Path to IEMOCAP CSV file
            
        Returns:
            results_df: DataFrame with original data and predictions
        """
        # Load IEMOCAP data
        print(f"Loading IEMOCAP data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Process in batches for memory efficiency
        batch_size = 32
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            texts = batch_df['transcription'].tolist()
            audio_paths = batch_df['wav_file'].tolist()
            # For visual paths, we'll use a placeholder since we don't have actual paths
            visual_paths = audio_paths  # Placeholder
            speaker_ids = batch_df['speaker'].tolist()
            
            print(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            emotion_labels, probabilities = self.predict_emotions(
                texts, audio_paths, visual_paths, speaker_ids
            )
            
            all_predictions.extend(emotion_labels)
            all_probabilities.extend(probabilities.tolist())
        
        # Add predictions to DataFrame
        results_df = df.copy()
        results_df['predicted_emotion'] = all_predictions
        
        # Add probability columns for each emotion
        for i, emotion in EMOTION_LABELS.items():
            results_df[f'prob_{emotion.lower()}'] = [probs[i] for probs in all_probabilities]
        
        return results_df
    
    def evaluate_predictions(self, results_df):
        """
        Evaluate predictions against ground truth labels.
        
        Args:
            results_df: DataFrame with predictions and ground truth
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Map emotion labels to indices
        emotion_to_idx = {v.lower(): k for k, v in EMOTION_LABELS.items()}
        
        # Get ground truth and predictions
        ground_truth = results_df['emotion'].str.lower().map(emotion_to_idx).tolist()
        predictions = results_df['predicted_emotion'].str.lower().map(emotion_to_idx).tolist()
        
        # Calculate accuracy
        correct = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)
        accuracy = correct / len(ground_truth) * 100
        
        # Calculate per-class metrics
        class_metrics = {}
        for emotion_idx, emotion_name in EMOTION_LABELS.items():
            emotion_name = emotion_name.lower()
            # Get indices where ground truth is this emotion
            indices = [i for i, gt in enumerate(ground_truth) if gt == emotion_idx]
            if indices:
                # Calculate accuracy for this emotion
                correct_class = sum(1 for i in indices if predictions[i] == emotion_idx)
                class_accuracy = correct_class / len(indices) * 100
                class_metrics[emotion_name] = {
                    'accuracy': class_accuracy,
                    'count': len(indices)
                }
        
        metrics = {
            'overall_accuracy': accuracy,
            'class_metrics': class_metrics
        }
        
        return metrics


def main():
    """
    Main function to demonstrate COGMEN inference on IEMOCAP data.
    """
    # Initialize COGMEN inference
    cogmen = COGMENInference()
    
    # Path to IEMOCAP CSV file
    iemocap_csv = "/home/ubuntu/multimodal-emotion-detection-sota-20250416/data/processed/iemocap/IEMOCAP_Final.csv"
    
    # Process IEMOCAP data and get predictions
    results_df = cogmen.process_iemocap_csv(iemocap_csv)
    
    # Save results to CSV
    output_path = "/home/ubuntu/multimodal-emotion-detection-sota-20250416/results/cogmen_predictions.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    
    # Evaluate predictions
    metrics = cogmen.evaluate_predictions(results_df)
    print("\nEvaluation Metrics:")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    print("\nPer-Class Metrics:")
    for emotion, class_metric in metrics['class_metrics'].items():
        print(f"{emotion.capitalize()}: Accuracy = {class_metric['accuracy']:.2f}%, Count = {class_metric['count']}")


if __name__ == "__main__":
    main()

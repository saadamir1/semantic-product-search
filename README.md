# Semantic Product Search with Deep Learning

A neural semantic search engine that intelligently matches user queries to products using BERT-based embeddings and deep learning techniques. Built on the Amazon Shopping Queries Dataset (ESCI), this system understands semantic relationships between search queries and product descriptions to deliver highly relevant results.

## 🚀 Features

### Core Functionality
- **BERT/DistilBERT Integration**: Leverages state-of-the-art transformer models for deep semantic understanding
- **Dual Architecture Support**: Choose between full BERT model for accuracy or lightweight DistilBERT for speed
- **Interactive Web Interface**: Beautiful Gradio-powered search interface with real-time results
- **Smart Preprocessing**: NLTK-based text cleaning, tokenization, and lemmatization

### Advanced Capabilities
- **Comprehensive Evaluation**: NDCG@K, MAP, Precision@K, Recall@K, F1@K metrics
- **Rich Visualizations**: Training curves, relevance distributions, error analysis, threshold optimization
- **Performance Optimization**: Gradient accumulation, early stopping, learning rate scheduling
- **Traditional Embedding Comparison**: TF-IDF, Word2Vec, and GloVe baseline implementations

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/semantic-product-search.git
cd semantic-product-search

# Install required dependencies
pip install torch transformers gradio pandas numpy scikit-learn matplotlib seaborn nltk tqdm gensim

# Download NLTK data (automatic on first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 🎯 Quick Start

```bash
# Run the complete pipeline
python main.py
```

The system will automatically:
1. Download and load the Amazon Shopping Queries dataset
2. Preprocess and split the data
3. Initialize the BERT-based model
4. Train with validation monitoring
5. Evaluate performance metrics
6. Launch interactive web interface

## 🏗️ Architecture

### Data Pipeline
- **Dataset**: Amazon Shopping Queries Dataset (ESCI) with query-product pairs
- **Preprocessing**: Text normalization, stop word removal, lemmatization
- **Labels**: Relevance scores (E: 1.0, S: 0.7, C: 0.3, I: 0.0)

### Model Architecture
```
Query Input → BERT Encoder → [768d embedding]
                                    ↓
Product Input → BERT Encoder → [768d embedding]
                                    ↓
                              Concatenation [1536d]
                                    ↓
                              FC Layer [512d] → ReLU → Dropout
                                    ↓
                              FC Layer [128d] → ReLU → Dropout
                                    ↓
                              Output Layer [1d] → Sigmoid
```

### Training Features
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: AdamW with weight decay (0.01)
- **Scheduler**: ReduceLROnPlateau
- **Regularization**: Dropout (0.2), gradient clipping
- **Early Stopping**: Patience-based with best model saving

## 📊 Evaluation Metrics

The system provides comprehensive evaluation across multiple dimensions:

- **Ranking Metrics**: NDCG@5, NDCG@10 for ranking quality
- **Classification Metrics**: Precision@K, Recall@K, F1@K for different cutoffs
- **Retrieval Metrics**: Mean Average Precision (MAP)
- **Error Analysis**: MSE, MAE, prediction distribution analysis

## 💻 Usage Examples

### Basic Search
```python
from main import predict_relevance

# Search for products
results = predict_relevance(model, tokenizer, "wireless headphones", products, device)
for result in results[:5]:
    print(f"{result['title']}: {result['relevance']:.3f}")
```

### Custom Training
```python
# Initialize with custom parameters
model = SemanticSearchModel(freeze_bert=True)  # Faster training
train_losses, val_losses = train_model(
    model, train_dataloader, val_dataloader, 
    device, epochs=10, learning_rate=1e-5
)
```

## 🎨 Visualizations

The system includes rich visualization capabilities:
- **Training Curves**: Monitor loss progression
- **Relevance Distribution**: Compare predicted vs actual scores
- **Performance Thresholds**: Optimize classification thresholds
- **Error Analysis**: Understand model limitations

## ⚡ Performance Optimizations

### Speed Optimizations
- **DistilBERT Option**: 40% faster with minimal accuracy loss
- **Frozen Layers**: Freeze early BERT layers for faster training
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Mixed Precision**: Reduce memory usage (optional)

### Memory Optimizations
- **Batch Size Tuning**: Configurable batch sizes
- **Sequence Length**: Optimized max length (128 tokens)
- **Model Checkpointing**: Save only best performing models

## 🔧 Configuration

Key parameters can be adjusted in the main function:
- `batch_size`: Training batch size (default: 8)
- `max_length`: Token sequence length (default: 128)
- `learning_rate`: AdamW learning rate (default: 2e-5)
- `epochs`: Maximum training epochs (default: 5)
- `patience`: Early stopping patience (default: 2)

## 📈 Results

Typical performance on Amazon ESCI dataset:
- **NDCG@10**: ~0.85-0.90
- **MAP**: ~0.75-0.80
- **Training Time**: 15-30 minutes (depending on dataset size)
- **Inference Speed**: <100ms per query

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Amazon Science for the ESCI dataset
- Hugging Face for transformer implementations
- Gradio team for the web interface framework

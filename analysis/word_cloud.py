import json
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
def preprocess_text(text):
    """Clean and normalize text for n-gram analysis"""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    return text.strip()

def get_reasoning_think_data():
    label_dir = "data/label"
    reasoning_models = ["deepseekR1", "Qwen3_32B", "QwQ32B", "Phi4R", "DSQwen32B"]
    
    data = {}
    
    for model in reasoning_models:
        model_path = os.path.join(label_dir, model)
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} does not exist")
            continue

        for file in os.listdir(model_path):
            if not file.endswith('.json'):
                continue

            with open(os.path.join(model_path, file), "r") as f:
                file_data = json.load(f)
            
            for item in file_data:
                if item.get("sentence-type") == "think":
                    cat = item.get("sentence-category")
                    sentence = item.get("sentence", "")
                    
                    if cat not in data:
                        data[cat] = []
                    data[cat].append(sentence)
    return data

def create_word_clouds():
    """Create word clouds for different categories using proper cross-category TF-IDF"""
    
    # Load original data
    # with open("data/inphase/reasoning_think.json", "r") as f:
    #     data = json.load(f)
    print("Loading data from raw files...")
    data = get_reasoning_think_data()
    
    # Create output directory
    output_dir = "analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    categories = list(data.keys())
    
    # Prepare data for cross-category TF-IDF
    category_docs = {}
    for category, sentences in data.items():
        processed_sentences = [preprocess_text(sentence) for sentence in sentences]
        category_docs[category] = ' '.join(processed_sentences)
    

    with open('analysis/rare_words.txt', 'r') as f:
        rare_words = f.read().splitlines()
    # Create TF-IDF vectorizer across all categories
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),  # 1-2 grams
        max_features=1000,
        stop_words = list(set([
            *list(CountVectorizer(stop_words='english').get_stop_words()),
            # Add your custom stopwords here
            *rare_words
        ])),
        lowercase=True,
        token_pattern=r'\b\w+\b',
        min_df=1,
        max_df=0.8  # Ignore terms that appear in more than 95% of categories
    )
    
    # Fit and transform across all categories
    docs = list(category_docs.values())
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    # Use get_feature_names() for older sklearn versions
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = vectorizer.get_feature_names()
    
    # Create a grid of subplots (2x4 for 8 categories)
    # Reduce figure size to avoid memory issues
    # height reduced to bring rows closer
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    # Custom order
    categories = ['Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Answer', 'Monitor']
    
    # Map original category names to document indices
    # Since tfidf_matrix is built from list(category_docs.values()), 
    # and category_docs keys are inserted in data.keys() order.
    original_categories = list(data.keys())
    
    for i, category in enumerate(categories):
        print(f"Processing {category}...")
        
        # Find index of this category in the original data structure
        try:
            doc_idx = original_categories.index(category)
        except ValueError:
            print(f"Warning: Category {category} not found in data.")
            continue
        
        # Get TF-IDF scores for this category using the correct index
        scores = tfidf_matrix[doc_idx].toarray().flatten()
        
        # Create frequency dictionary from TF-IDF scores
        ngram_freq = {}
        for j, score in enumerate(scores):
            if score > 0:
                ngram_freq[feature_names[j]] = score
        
        if ngram_freq:
            # Create word cloud
            # Reduce resolution to avoid memory issues
            wordcloud = WordCloud(
                width=600,
                height=300,
                background_color='white',
                max_words=100,
                colormap='viridis',
                relative_scaling=0.5,
                random_state=42
            ).generate_from_frequencies(ngram_freq)
            
            # Plot in subplot
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(category, fontsize=20, fontweight='bold')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'No data for {category}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(category, fontsize=20, fontweight='bold')
            axes[i].axis('off')
    
    # Hide unused subplots if any
    for i in range(len(categories), 8):
        axes[i].axis('off')
    
    # plt.suptitle('N-gram Analysis Word Clouds by Category (1-1 grams)', 
    #              fontsize=18, fontweight='bold')
    
    # Adjust layout to bring rows closer
    # h_pad controls vertical padding between rows
    # Reducing bottom margin to pull everything down slightly if needed, or just tight layout
    plt.tight_layout(h_pad=-8.0, w_pad=0.5)
    
    # Save the final word cloud
    output_path = os.path.join(output_dir, "word_clouds.png")
    # Reduce DPI
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Word cloud saved: {output_path}")

if __name__ == "__main__":
    create_word_clouds()
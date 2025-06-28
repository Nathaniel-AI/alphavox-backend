import logging
import os
import pickle
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = 'data'
RESEARCH_DIR = os.path.join(DATA_DIR, 'research')
RESEARCH_CACHE = os.path.join(DATA_DIR, 'research_cache.pkl')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESEARCH_DIR, exist_ok=True)

# Load spaCy for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}")
    logger.info("Install spaCy and download model: pip install spacy && python -m spacy download en_core_web_sm")
    # Fallback to None, which will be checked before use
    nlp = None

class AlphaVoxResearchModule:
    """Research module for AlphaVox to study nonverbal autism and neurodivergent therapies."""
    
    def __init__(self, cache_max_age: int = 86400, max_articles: int = 50):
        """Initialize the research module."""
        self.cache_max_age = cache_max_age  # Cache refresh interval (seconds)
        self.max_articles = max_articles
        self.cache = self.load_cache()
        self.search_terms = [
            "nonverbal autism communication",
            "neurodivergent therapies",
            "autism AAC intervention",
            "PECS autism",
            "speech generating device autism",
            "AI assistive technology autism"
        ]
        self.nlp = nlp
        logger.info("AlphaVox Research Module initialized")

    def load_cache(self) -> Dict:
        """Load cached research data."""
        if os.path.exists(RESEARCH_CACHE):
            try:
                with open(RESEARCH_CACHE, 'rb') as f:
                    cache = pickle.load(f)
                if (datetime.now() - cache.get('timestamp', datetime.min)).total_seconds() < self.cache_max_age:
                    logger.info("Loaded valid research cache")
                    return cache
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")
        return {'timestamp': datetime.min, 'articles': []}

    def save_cache(self, articles: List[Dict]):
        """Save research data to cache."""
        try:
            cache = {'timestamp': datetime.now(), 'articles': articles}
            with open(RESEARCH_CACHE, 'wb') as f:
                pickle.dump(cache, f)
            logger.info("Saved research cache")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

    def fetch_research(self) -> List[Dict]:
        """Fetch recent research articles from web sources."""
        if (datetime.now() - self.cache.get('timestamp', datetime.min)).total_seconds() < self.cache_max_age:
            logger.info("Using cached research articles")
            return self.cache['articles']
        
        articles = []
        for term in self.search_terms:
            try:
                # Simulate search using a mock API (replace with real API like PubMed or Google Scholar)
                url = f"https://www.google.com/search?q={term.replace(' ', '+')}+site:*.edu|site:*.gov|site:*.org"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True)[:10]:  # Limit to 10 per term
                    href = link['href']
                    if href.startswith('/url?q='):
                        href = href.split('/url?q=')[1].split('&')[0]
                    if not href.startswith(('http', 'https')):
                        continue
                    try:
                        article_response = requests.get(href, headers=headers, timeout=10)
                        article_soup = BeautifulSoup(article_response.text, 'html.parser')
                        title = article_soup.find('title').text if article_soup.find('title') else 'Untitled'
                        content = ' '.join(p.text for p in article_soup.find_all('p')[:5])  # First 5 paragraphs
                        if len(content) > 100:  # Ensure meaningful content
                            articles.append({
                                'title': title,
                                'url': href,
                                'content': content,
                                'term': term,
                                'timestamp': datetime.now()
                            })
                    except Exception as e:
                        logger.debug(f"Error fetching article {href}: {str(e)}")
            except Exception as e:
                logger.error(f"Error searching term {term}: {str(e)}")
        
        articles = articles[:self.max_articles]
        self.save_cache(articles)
        logger.info(f"Fetched {len(articles)} research articles")
        return articles

    def analyze_research(self, articles: List[Dict]) -> List[Dict]:
        """Analyze research articles to extract insights."""
        try:
            if not articles:
                logger.warning("No articles to analyze")
                return []
                
            # Preprocess articles
            documents = [article['content'] for article in articles]
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            X = vectorizer.fit_transform(documents)
            
            # Cluster articles to identify themes
            n_clusters = min(5, len(articles))  # Avoid more clusters than articles
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            
            # Extract key insights per cluster
            insights = []
            feature_names = vectorizer.get_feature_names_out()
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]
                if not cluster_indices.size:
                    continue
                cluster_docs = [documents[i] for i in cluster_indices]
                cluster_articles = [articles[i] for i in cluster_indices]
                
                # Extract key phrases using TF-IDF
                cluster_X = X[cluster_indices]
                top_terms = np.argsort(cluster_X.sum(axis=0).A1)[-10:][::-1]
                key_phrases = [feature_names[i] for i in top_terms]
                
                # Summarize cluster content
                summary = self._summarize_cluster(cluster_docs)
                
                insights.append({
                    'theme': f"Cluster {cluster_id}",
                    'key_phrases': key_phrases,
                    'summary': summary,
                    'articles': [
                        {'title': a['title'], 'url': a['url'], 'term': a['term']}
                        for a in cluster_articles
                    ]
                })
            
            logger.info(f"Extracted {len(insights)} research insights")
            return insights
        except Exception as e:
            logger.error(f"Error analyzing research: {str(e)}")
            return []

    def _summarize_cluster(self, documents: List[str]) -> str:
        """Generate a summary for a cluster of documents."""
        try:
            if not self.nlp:
                logger.warning("spaCy model not available, returning basic summary")
                return " ".join([doc[:100] for doc in documents[:2]])
                
            combined_text = ' '.join(documents)[:1000]  # Limit length
            doc = self.nlp(combined_text)
            sentences = [sent.text for sent in doc.sents][:3]  # Take first 3 sentences
            return ' '.join(sentences)
        except Exception as e:
            logger.error(f"Error summarizing cluster: {str(e)}")
            return "Unable to generate summary."

    def integrate_insights(self, insights: List[Dict], nlc: Optional[object] = None) -> Dict:
        """Integrate research insights into AlphaVox's Neural Learning Core."""
        try:
            updates = {
                'new_strategies': [],
                'updated_intents': [],
                'therapy_recommendations': []
            }
            
            for insight in insights:
                key_phrases = insight['key_phrases']
                summary = insight['summary']
                
                # Identify new communication strategies
                if any(phrase in key_phrases for phrase in ['aac', 'pecs', 'speech generating', 'visual support']):
                    strategy = {
                        'type': 'communication',
                        'description': f"Adopt {key_phrases[0]} strategy: {summary[:100]}",
                        'source': insight['articles'][0]['url'] if insight['articles'] else ""
                    }
                    updates['new_strategies'].append(strategy)
                
                # Update intent mappings
                if any(phrase in key_phrases for phrase in ['social interaction', 'communication intent']):
                    updates['updated_intents'].append({
                        'intent': key_phrases[0].replace(' ', '_') if key_phrases else "communication",
                        'weight': 1.0,
                        'description': summary[:100]
                    })
                
                # Recommend therapies
                if any(phrase in key_phrases for phrase in ['therapy', 'intervention', 'esdm']):
                    updates['therapy_recommendations'].append({
                        'therapy': key_phrases[0] if key_phrases else "general_therapy",
                        'description': summary[:100],
                        'source': insight['articles'][0]['url'] if insight['articles'] else ""
                    })
            
            # Integrate with Neural Learning Core
            if nlc:
                try:
                    for strategy in updates['new_strategies']:
                        nlc.intent_weights[strategy['type']] = nlc.intent_weights.get(strategy['type'], 0.0) + 0.1
                    for intent in updates['updated_intents']:
                        nlc.intent_weights[intent['intent']] = intent['weight']
                    logger.info("Integrated research insights into Neural Learning Core")
                except Exception as e:
                    logger.error(f"Error updating Neural Learning Core: {str(e)}")
            
            logger.info(f"Integrated {len(updates['new_strategies'])} strategies, {len(updates['updated_intents'])} intents")
            return updates
        except Exception as e:
            logger.error(f"Error integrating insights: {str(e)}")
            return {}

    def update_knowledge_base(self):
        """Update AlphaVox's knowledge base with new research."""
        try:
            articles = self.fetch_research()
            insights = self.analyze_research(articles)
            try:
                from neural_learning_core import get_neural_learning_core
                nlc = get_neural_learning_core()
                updates = self.integrate_insights(insights, nlc)
            except Exception as e:
                logger.error(f"Error connecting to Neural Learning Core: {str(e)}")
                updates = self.integrate_insights(insights)
                
            return {
                'status': 'success',
                'articles_fetched': len(articles),
                'insights_extracted': len(insights),
                'updates_applied': updates
            }
        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}")
            return {'status': 'error', 'message': str(e)}

def main():
    """Demonstrate the AlphaVox Research Module."""
    research_module = AlphaVoxResearchModule()
    
    # Fetch and analyze research
    articles = research_module.fetch_research()
    insights = research_module.analyze_research(articles)
    
    # Integrate insights
    try:
        from neural_learning_core import get_neural_learning_core
        nlc = get_neural_learning_core()
        updates = research_module.integrate_insights(insights, nlc)
    except Exception as e:
        logger.error(f"Error connecting to Neural Learning Core: {str(e)}")
        updates = research_module.integrate_insights(insights)
    
    print(f"Fetched {len(articles)} articles")
    print(f"Extracted {len(insights)} insights:")
    for insight in insights:
        print(f"- Theme: {insight['theme']}, Key Phrases: {insight['key_phrases']}")
    print(f"Applied updates: {updates}")

if __name__ == "__main__":
    main()
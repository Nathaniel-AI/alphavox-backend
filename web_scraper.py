"""
AlphaVox - Web Content Scraper
--------------------------
This module provides web scraping capabilities to help AlphaVox educate itself 
by gathering information from trusted sources. It uses Trafilatura, which is an 
advanced web scraping tool that extracts clean text content from web pages.

The scraper focuses on educational content about:
1. Nonverbal communication
2. Assistive technologies
3. Neurodiversity
4. Communication development

Usage:
    from web_scraper import get_website_text_content, get_content_from_trusted_sources
    
    # Get content from a specific URL
    content = get_website_text_content("https://example.com/article-about-assistive-tech")
    
    # Get content from multiple trusted sources about a topic
    contents = get_content_from_trusted_sources("nonverbal communication")
"""

import os
import json
import logging
import random
import time
from datetime import datetime
import trafilatura
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define trusted sources for educational content
TRUSTED_SOURCES = {
    "academic": [
        "scholar.google.com",
        "ncbi.nlm.nih.gov",
        "researchgate.net",
        "frontiersin.org",
        "academia.edu",
    ],
    "healthcare": [
        "asha.org",  # American Speech-Language-Hearing Association
        "aota.org",  # American Occupational Therapy Association
        "mayoclinic.org",
        "healthline.com",
        "nichd.nih.gov",  # National Institute of Child Health and Human Development
    ],
    "advocacy": [
        "autism.org",
        "autismspeaks.org",
        "specialolympics.org",
        "understood.org",
        "cdc.gov/ncbddd/autism",  # CDC Autism resources
    ],
    "education": [
        "edutopia.org",
        "understood.org",
        "readingrockets.org",
        "teachspeced.com",
    ]
}

# Define base URLs for search queries
SEARCH_URLS = {
    "academic": "https://scholar.google.com/scholar?q=",
    "general": "https://www.google.com/search?q="
}

def get_website_text_content(url: str) -> str:
    """
    Extract clean text content from a website using Trafilatura.
    
    Args:
        url: The URL of the website to scrape
        
    Returns:
        Extracted text content or empty string if extraction fails
    """
    try:
        logger.info(f"Fetching content from: {url}")
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        
        if text:
            logger.info(f"Successfully extracted content from {url} ({len(text)} characters)")
            return text
        else:
            logger.warning(f"Failed to extract content from {url}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}")
        return ""

def is_trusted_domain(url: str) -> bool:
    """
    Check if a URL is from a trusted domain.
    
    Args:
        url: The URL to check
        
    Returns:
        True if the domain is trusted, False otherwise
    """
    try:
        domain = urlparse(url).netloc.lower()
        
        for category in TRUSTED_SOURCES:
            for trusted_domain in TRUSTED_SOURCES[category]:
                if trusted_domain in domain:
                    return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking domain trust: {str(e)}")
        return False

def get_links_from_search(query: str, search_type: str = "general", max_results: int = 5) -> list:
    """
    Get links from a search engine based on a query.
    
    Args:
        query: The search query
        search_type: Type of search ("academic" or "general")
        max_results: Maximum number of results to return
        
    Returns:
        List of URLs
    """
    try:
        search_url = SEARCH_URLS.get(search_type, SEARCH_URLS["general"]) + requests.utils.quote(query)
        
        # For demonstration purposes, we would implement a proper search API here
        # but for now, we'll return a simulated list of results to avoid search engine restrictions
        logger.info(f"Would search for '{query}' using {search_type} search")
        
        # In a real implementation, this would scrape search results
        # Here we simulate finding some trusted sources
        results = []
        if search_type == "academic":
            # Simulated academic results
            results = [
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{random.randint(1000000, 9999999)}/",
                f"https://www.frontiersin.org/articles/{random.randint(100000, 999999)}/full",
                f"https://www.researchgate.net/publication/{random.randint(300000000, 399999999)}",
            ]
        else:
            # Simulated general results
            results = [
                f"https://www.healthline.com/health/{query.replace(' ', '-')}",
                f"https://www.asha.org/articles/{query.replace(' ', '-')}/",
                f"https://www.understood.org/articles/{query.replace(' ', '-')}",
            ]
            
        return results[:max_results]
    
    except Exception as e:
        logger.error(f"Error getting search results: {str(e)}")
        return []

def get_content_from_trusted_sources(topic: str, max_sources: int = 3) -> list:
    """
    Get content about a topic from multiple trusted sources.
    
    Args:
        topic: The topic to search for
        max_sources: Maximum number of sources to use
        
    Returns:
        List of dictionaries with source and content
    """
    try:
        logger.info(f"Searching for information about: {topic}")
        
        # Get links from both academic and general searches
        academic_links = get_links_from_search(topic, "academic", max_sources)
        general_links = get_links_from_search(topic, "general", max_sources)
        
        # Combine and filter for trusted domains
        all_links = academic_links + general_links
        trusted_links = [link for link in all_links if is_trusted_domain(link)]
        
        # Limit to max_sources
        selected_links = trusted_links[:max_sources]
        
        # Extract content from each link
        results = []
        for link in selected_links:
            content = get_website_text_content(link)
            if content:
                results.append({
                    "source": link,
                    "content": content,
                    "domain": urlparse(link).netloc,
                    "timestamp": datetime.now().isoformat()
                })
        
        logger.info(f"Found content from {len(results)} trusted sources for topic: {topic}")
        return results
    
    except Exception as e:
        logger.error(f"Error getting content from trusted sources: {str(e)}")
        return []

def save_scraped_content(topic: str, contents: list, output_dir: str = "data/scraped"):
    """
    Save scraped content to a file.
    
    Args:
        topic: The topic that was searched for
        contents: List of content dictionaries
        output_dir: Directory to save the content
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a filename based on the topic and timestamp
        filename = f"{output_dir}/{topic.replace(' ', '_')}_{int(time.time())}.json"
        
        # Save the content to a JSON file
        with open(filename, "w") as f:
            json.dump({
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "sources": len(contents),
                "contents": contents
            }, f, indent=2)
            
        logger.info(f"Saved scraped content to {filename}")
        return filename
    
    except Exception as e:
        logger.error(f"Error saving scraped content: {str(e)}")
        return None

def educate_on_topic(topic: str, save_results: bool = True) -> dict:
    """
    Educate AlphaVox on a specific topic by gathering information.
    
    Args:
        topic: The topic to learn about
        save_results: Whether to save the results to disk
        
    Returns:
        Dictionary with education results
    """
    try:
        logger.info(f"Educating AlphaVox about: {topic}")
        
        # Get content from trusted sources
        contents = get_content_from_trusted_sources(topic)
        
        # Save the content if requested
        filename = None
        if save_results and contents:
            filename = save_scraped_content(topic, contents)
        
        # Return education results
        return {
            "topic": topic,
            "sources_count": len(contents),
            "sources": [c["source"] for c in contents],
            "file_saved": filename is not None,
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "success": len(contents) > 0
        }
    
    except Exception as e:
        logger.error(f"Error educating on topic: {str(e)}")
        return {
            "topic": topic,
            "sources_count": 0,
            "sources": [],
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # When run as a script, educate on a test topic
    test_topics = [
        "nonverbal communication in autism",
        "assistive technology for communication disorders",
        "neurodiversity in education",
        "communication development milestones"
    ]
    
    for topic in test_topics:
        result = educate_on_topic(topic)
        print(f"Education result for '{topic}': {result['sources_count']} sources found")

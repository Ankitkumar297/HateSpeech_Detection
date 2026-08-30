import re
import string
from typing import List, Dict, Any, Set

# Optional NLTK imports with robust fallback
try:
    from nltk.corpus import stopwords  # type: ignore
    from nltk.stem import SnowballStemmer  # type: ignore
    NLTK_AVAILABLE = True
except ImportError:
    stopwords = None  # type: ignore
    SnowballStemmer = None  # type: ignore
    NLTK_AVAILABLE = False

# Comprehensive English Stopwords fallback
DEFAULT_STOPWORDS: Set[str] = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", 
    "s", "t", "can", "will", "just", "don", "should", "now"
}

class TextPreprocessor:
    """
    Text preprocessing pipeline for Hate Speech Detection:
    1. Lowercasing
    2. URL & User mention removal
    3. Punctuation and special character removal
    4. Stopwords filtering
    5. Snowball Stemming
    """
    def __init__(self, language: str = "english"):
        self.language = language
        if NLTK_AVAILABLE and SnowballStemmer is not None:
            try:
                self.stemmer = SnowballStemmer(language)
            except Exception:
                self.stemmer = None
        else:
            self.stemmer = None

        self.stop_words: Set[str] = set(DEFAULT_STOPWORDS)
        if NLTK_AVAILABLE and stopwords is not None:
            try:
                self.stop_words = set(stopwords.words(language))
            except Exception:
                self.stop_words = set(DEFAULT_STOPWORDS)
        
        # Preserve critical intent words
        custom_removals = {"not", "no", "nor", "never", "against"}
        self.stop_words = self.stop_words - custom_removals

    def clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        
        # 1. Convert to lowercase
        text = text.lower()
        
        # 2. Remove URLs (http, https, www)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 3. Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # 4. Remove User Mentions (@username) and Hashtags (#hashtag)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content as normal word
        
        # 5. Remove numbers / digits
        text = re.sub(r'\d+', '', text)
        
        # 6. Remove punctuation and special symbols
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^\w\s]', '', text)
        
        # 7. Normalize extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize_and_stem(self, text: str) -> str:
        """
        Cleans text, filters stopwords, and applies SnowballStemmer.
        Returns stemmed space-separated string.
        """
        cleaned = self.clean_text(text)
        words = cleaned.split()
        
        # Filter stopwords and apply stemming
        processed_tokens: List[str] = []
        for word in words:
            if word not in self.stop_words and len(word) > 1:
                stem = self.stemmer.stem(word) if self.stemmer is not None else word
                processed_tokens.append(stem)
        
        return " ".join(processed_tokens)

    def get_token_details(self, original_text: str) -> List[Dict[str, Any]]:
        """
        Returns detailed token breakdown for UI explainability.
        """
        cleaned = self.clean_text(original_text)
        words = cleaned.split()
        details: List[Dict[str, Any]] = []
        for word in words:
            is_stop = word in self.stop_words
            stem = word
            if not is_stop and self.stemmer is not None:
                stem = self.stemmer.stem(word)
            details.append({
                "original": word,
                "is_stopword": is_stop,
                "stemmed": stem
            })
        return details

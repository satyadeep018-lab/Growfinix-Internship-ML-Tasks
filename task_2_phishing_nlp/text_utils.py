import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

try:
    STOP = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP = set(stopwords.words("english"))

_tok = RegexpTokenizer(r"[a-z]+")
URL = re.compile(r"https?://\S+|www\.\S+")

def tokenize(text, max_tokens=200):
    text = URL.sub(" urltoken ", str(text)[:3000].lower())
    toks = [t for t in _tok.tokenize(text) if t not in STOP]
    return (toks or ["unk"])[:max_tokens]

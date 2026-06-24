import os
import re
import math
import glob

class BM25Retriever:
    def __init__(self, notes_dir, chunk_size=1000, chunk_overlap=200):
        self.notes_dir = os.path.abspath(notes_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []  # list of dict: {"filename": str, "content": str, "tokens": list[str], "chunk_index": int}
        self.df = {}         # word -> doc count
        self.avgdl = 0.0
        self.N = 0
        self.k1 = 1.5
        self.b = 0.75
        self.last_mtime = -1
        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
            "at", "by", "from", "for", "in", "out", "on", "of", "to", "with",
            "is", "was", "were", "are", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "shall", "should", "can", "could",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"
        }

    def tokenize(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in self.stopwords and len(w) > 1]

    def get_dir_mtime(self):
        if not os.path.exists(self.notes_dir):
            return 0
        try:
            mtimes = [
                os.path.getmtime(os.path.join(self.notes_dir, f))
                for f in os.listdir(self.notes_dir)
                if os.path.isfile(os.path.join(self.notes_dir, f)) and f.lower().endswith(('.md', '.txt'))
            ]
            return max(mtimes) if mtimes else os.path.getmtime(self.notes_dir)
        except Exception:
            return 0

    def rebuild_index(self):
        self.documents = []
        self.df = {}
        
        if not os.path.exists(self.notes_dir):
            self.N = 0
            self.avgdl = 0.0
            return

        filepaths = []
        for ext in ("*.md", "*.txt", "*.MD", "*.TXT"):
            filepaths.extend(glob.glob(os.path.join(self.notes_dir, ext)))

        for fp in filepaths:
            filename = os.path.basename(fp)
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                continue
            
            if not content.strip():
                continue

            chunks = self._chunk_text(content)
            for i, chunk in enumerate(chunks):
                tokens = self.tokenize(chunk)
                self.documents.append({
                    "filename": filename,
                    "content": chunk,
                    "tokens": tokens,
                    "chunk_index": i
                })

        self.N = len(self.documents)
        if self.N == 0:
            self.avgdl = 0.0
            return

        total_length = 0
        for doc in self.documents:
            total_length += len(doc["tokens"])
            unique_tokens = set(doc["tokens"])
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1
        
        self.avgdl = total_length / self.N

    def _chunk_text(self, text):
        # Paragraph-based chunking with paragraph-level boundaries
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = []
        current_len = 0
        
        for p in paragraphs:
            p_len = len(p)
            if p_len > self.chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                # Split large paragraph by lines or sentences
                sentences = re.split(r'(?<=[.!?])\s+', p)
                sub_chunk = []
                sub_len = 0
                for s in sentences:
                    if sub_len + len(s) > self.chunk_size:
                        if sub_chunk:
                            chunks.append(" ".join(sub_chunk))
                        sub_chunk = [s]
                        sub_len = len(s)
                    else:
                        sub_chunk.append(s)
                        sub_len += len(s)
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))
            else:
                if current_len + p_len > self.chunk_size:
                    chunks.append("\n\n".join(current_chunk))
                    if len(current_chunk) > 1:
                        current_chunk = [current_chunk[-1], p]
                        current_len = len(current_chunk[0]) + p_len
                    else:
                        current_chunk = [p]
                        current_len = p_len
                else:
                    current_chunk.append(p)
                    current_len += p_len + 2
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks

    def retrieve(self, query, top_k=3):
        current_mtime = self.get_dir_mtime()
        if current_mtime != self.last_mtime:
            self.rebuild_index()
            self.last_mtime = current_mtime

        if self.N == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for doc in self.documents:
            score = 0.0
            doc_len = len(doc["tokens"])
            tf = {}
            for token in doc["tokens"]:
                tf[token] = tf.get(token, 0) + 1

            for q in query_tokens:
                if q not in self.df:
                    continue
                f_q = tf.get(q, 0)
                df_q = self.df[q]
                idf = math.log((self.N - df_q + 0.5) / (df_q + 0.5) + 1.0)
                
                numerator = f_q * (self.k1 + 1.0)
                denominator = f_q + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl)) if self.avgdl > 0 else 1.0
                score += idf * (numerator / denominator)
            
            if score > 0.0:
                scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in scores[:top_k]:
            results.append({
                "filename": doc["filename"],
                "content": doc["content"],
                "score": score,
                "chunk_index": doc["chunk_index"]
            })
        return results


# ── Web Search RAG Retriever (DuckDuckGo Lite) ──────────────────────────────
from html.parser import HTMLParser
import urllib.request
import urllib.parse

class DDGLiteParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current_result = None
        self.in_link = False
        self.in_snippet = False
        self.temp_text = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")
        
        if tag == "a" and "result-link" in cls:
            self.in_link = True
            self.temp_text = []
            self.current_result = {"url": attrs_dict.get("href", "")}
        elif tag == "td" and "result-snippet" in cls:
            self.in_snippet = True
            self.temp_text = []

    def handle_endtag(self, tag):
        if tag == "a" and self.in_link:
            self.in_link = False
            if self.current_result:
                self.current_result["title"] = "".join(self.temp_text).strip()
        elif tag == "td" and self.in_snippet:
            self.in_snippet = False
            if self.current_result:
                self.current_result["snippet"] = "".join(self.temp_text).strip()
                self.results.append(self.current_result)
                self.current_result = None

    def handle_data(self, data):
        if self.in_link or self.in_snippet:
            self.temp_text.append(data)

def retrieve_web_context(query, top_k=3):
    url = "https://lite.duckduckgo.com/lite/"
    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            html_data = response.read().decode("utf-8", errors="ignore")
            parser = DDGLiteParser()
            parser.feed(html_data)
            
            clean_results = []
            for r in parser.results:
                if r.get("url") and r.get("title") and r.get("snippet"):
                    url_str = r["url"]
                    if url_str.startswith("//"):
                        url_str = "https:" + url_str
                    elif url_str.startswith("/"):
                        url_str = "https://lite.duckduckgo.com" + url_str
                    
                    if "duckduckgo.com/y.js" in url_str:
                        try:
                            parsed_url = urllib.parse.urlparse(url_str)
                            qs = urllib.parse.parse_qs(parsed_url.query)
                            if "uddg" in qs:
                                url_str = qs["uddg"][0]
                        except Exception:
                            pass
                    
                    clean_results.append({
                        "title": r["title"],
                        "url": url_str,
                        "snippet": r["snippet"]
                    })
            return clean_results[:top_k]
    except Exception as e:
        print(f"[RAG WEB ERROR] Live search failed: {e}")
        return []

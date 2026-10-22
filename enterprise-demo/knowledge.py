"""Local-only ingestion and retrieval. Hashing vectors are lexical, NOT learned embeddings."""
from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import threading
import uuid
from pathlib import Path

import numpy as np
from docx import Document as WordDocument
from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

SUPPORTED = {'.txt', '.md', '.pdf', '.docx'}
MAX_BYTES = 10 * 1024 * 1024
PIPELINE_VERSION = 'char500-overlap70-v1'


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse_file(name: str, content: bytes) -> list[tuple[str, str]]:
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError('支持 PDF、DOCX、Markdown 和 TXT；暂不支持旧版 DOC。')
    if not content or len(content) > MAX_BYTES:
        raise ValueError('文件不能为空，且不能超过 10 MB。')
    if suffix in {'.txt', '.md'}:
        try:
            return [('正文', content.decode('utf-8-sig'))]
        except UnicodeDecodeError:
            raise ValueError('文本文件需要使用 UTF-8 编码。') from None
    if suffix == '.pdf':
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise ValueError('请先解密 PDF 后再导入。')
        sections = [(f'第 {i} 页', p.extract_text() or '') for i, p in enumerate(reader.pages, 1)]
        if not any(t.strip() for _, t in sections):
            raise ValueError('PDF 无可提取文字；本演示不含扫描件 OCR。')
        return sections
    doc = WordDocument(io.BytesIO(content))
    # Preserve paragraph/table reading order. Table headers remain in the same section.
    sections = []
    for block in doc.iter_inner_content():
        if hasattr(block, 'rows'):
            sections.append(('表格', '\n'.join(' | '.join(c.text for c in row.cells) for row in block.rows)))
        elif block.text.strip():
            sections.append(('段落', block.text))
    return sections


class KnowledgeBase:
    """Single-process local prototype; SQLite is source of truth, Qdrant is rebuildable."""
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.sql = sqlite3.connect(self.directory / 'catalog.sqlite', check_same_thread=False)
        self.sql.row_factory = sqlite3.Row
        self.sql.executescript('''
        CREATE TABLE IF NOT EXISTS documents(
            doc_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, version TEXT NOT NULL,
            chunk_count INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS chunks(
            chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, name TEXT NOT NULL,
            version TEXT NOT NULL, location TEXT NOT NULL, text TEXT NOT NULL,
            spans TEXT NOT NULL);
        ''')
        self.splitter = SentenceSplitter(chunk_size=500, chunk_overlap=70, tokenizer=list,
                                         secondary_chunking_regex=r'[^。！？；\n]+[。！？；\n]?')
        self.vectorizer = HashingVectorizer(analyzer='char', ngram_range=(2, 4),
                                           n_features=2048, alternate_sign=False, norm='l2')
        self.qdrant = QdrantClient(path=str(self.directory / 'qdrant'), force_disable_check_same_thread=True)
        self._rebuild()

    def _rebuild(self):
        # Safe recovery from a crash between catalogue commit and vector update.
        if self.qdrant.collection_exists('chunks'):
            self.qdrant.delete_collection('chunks')
        self.qdrant.create_collection('chunks', vectors_config=models.VectorParams(size=2048, distance=models.Distance.COSINE))
        rows = self.sql.execute('SELECT * FROM chunks ORDER BY chunk_id').fetchall()
        self.rows = [dict(r) for r in rows]
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            vectors = self.vectorizer.transform([r['text'] for r in batch]).toarray()
            self.qdrant.upsert('chunks', points=[models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, r['chunk_id'])), vector=v.tolist(),
                payload={'chunk_id': r['chunk_id']}) for r, v in zip(batch, vectors)])
        if rows:
            self.lexical = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True)
            self.lexical_matrix = self.lexical.fit_transform([r['text'] for r in rows])
        else:
            self.lexical = None

    def ingest(self, name: str, content: bytes) -> dict:
        name = Path(name.replace('\\', '/')).name
        if not name or len(name) > 200:
            raise ValueError('文件名无效或过长。')
        sections = parse_file(name, content)
        doc_id = digest(name)[:24]
        version = hashlib.sha256(content + PIPELINE_VERSION.encode()).hexdigest()[:24]
        chunks = []
        for section_index, (location, text) in enumerate(sections):
            text = text.replace('\x00', '').strip()
            if not text:
                continue
            for index, chunk in enumerate(self.splitter.split_text(text)):
                chunk_id = f'{doc_id}-{version}-{section_index}-{index}'
                pieces = [s.strip() for s in re.split(r'(?<=[。！？.!?])\s*|\n+', chunk) if s.strip()]
                spans = [{'span_id': f'{chunk_id}:s{i}', 'text': s}
                         for i, s in enumerate(pieces, 1)]
                chunks.append((chunk_id, doc_id, name, version, location, chunk,
                               json.dumps(spans, ensure_ascii=False)))
        if not chunks:
            raise ValueError('文件没有可索引文本。')
        with self.lock:
            current = self.sql.execute('SELECT version FROM documents WHERE doc_id=?', (doc_id,)).fetchone()
            if current and current['version'] == version:
                return {'name': name, 'chunks': len(chunks), 'unchanged': True}
            with self.sql:
                self.sql.execute('DELETE FROM chunks WHERE doc_id=?', (doc_id,))
                self.sql.executemany('INSERT INTO chunks VALUES(?,?,?,?,?,?,?)', chunks)
                self.sql.execute('INSERT OR REPLACE INTO documents VALUES(?,?,?,?)',
                                 (doc_id, name, version, len(chunks)))
            self._rebuild()
        return {'name': name, 'chunks': len(chunks), 'unchanged': False, 'version': version}

    def documents(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.sql.execute('SELECT * FROM documents ORDER BY name')]

    def search(self, query: str, topk: int = 3) -> list[dict]:
        with self.lock:
            if not self.rows or not query.strip() or self.lexical is None:
                return []
            query_vector = self.vectorizer.transform([query]).toarray()[0]
            hits = self.qdrant.query_points('chunks', query=query_vector.tolist(),
                                           limit=min(len(self.rows), max(topk * 4, 12))).points
            dense_ranks = {h.payload['chunk_id']: i for i, h in enumerate(hits, 1) if h.score > 0}
            similarities = (self.lexical_matrix @ self.lexical.transform([query]).T).toarray().ravel()
            ranked = sorted(range(len(self.rows)), key=lambda i: similarities[i], reverse=True)
            lexical_ranks = {self.rows[i]['chunk_id']: rank for rank, i in enumerate(ranked, 1) if similarities[i] > 0}
            output = []
            for i, row in enumerate(self.rows):
                # Avoid fabricating an answer from all-zero hashed collisions.
                if similarities[i] <= 0:
                    continue
                cid = row['chunk_id']
                score = sum(1 / (60 + ranks[cid]) for ranks in (dense_ranks, lexical_ranks) if cid in ranks)
                output.append({**row, 'spans': json.loads(row['spans']), 'score': float(score),
                               'lexical_similarity': float(similarities[i])})
            return sorted(output, key=lambda r: r['score'], reverse=True)[:max(1, min(10, topk))]

    def close(self):
        self.qdrant.close()
        self.sql.close()

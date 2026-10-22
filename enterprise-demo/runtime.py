"""An explicit offline StateFlow demonstration, not an autonomous LLM agent."""
import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from knowledge import KnowledgeBase
from stateflow_gate import parse_and_validate_state


class RunState(TypedDict, total=False):
    question: str
    hits: list
    evidence_state: dict
    gate: dict
    answer: str
    citations: list
    trace: list
    mode: str


class OfflineRuntime:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        graph = StateGraph(RunState)
        graph.add_node('retrieve', self.retrieve)
        graph.add_node('build_evidence', self.build_evidence)
        graph.add_node('gate', self.gate)
        graph.add_node('extract', self.extract)
        graph.add_edge(START, 'retrieve')
        graph.add_edge('retrieve', 'build_evidence')
        graph.add_edge('build_evidence', 'gate')
        graph.add_edge('gate', 'extract')
        graph.add_edge('extract', END)
        self.graph = graph.compile()

    def retrieve(self, state):
        hits = self.kb.search(state['question'])
        return {'hits': hits, 'trace': [{'node': 'retrieve', 'detail': f'召回 {len(hits)} 个片段'}],
                'mode': 'offline_extract_only'}

    def build_evidence(self, state):
        selected = []
        # Full original spans avoid any fabricated synthesis; limit keeps protocol bounded.
        for hit in state['hits']:
            for span in hit['spans']:
                if len(span['text']) <= 600:
                    selected.append({**span, 'role': 'direct'})
                    break
            if len(selected) >= 3:
                break
        # Keep decision=search even with hits: lexical retrieval cannot establish answer sufficiency.
        payload = {'evidence': selected, 'open_questions': ['尚需模型或人工判断证据是否充分并支持答案'],
                   'decision': 'search'}
        return {'evidence_state': payload, 'trace': state['trace'] + [
            {'node': 'build_evidence', 'detail': f'规则选取 {len(selected)} 条原文，未判断回答充分性'}]}

    def gate(self, state):
        visible = {s['span_id']: {'text': s['text']} for h in state['hits'] for s in h['spans']}
        tagged = '<evidence_state>' + json.dumps(state['evidence_state'], ensure_ascii=False) + '</evidence_state>'
        result = parse_and_validate_state(tagged, visible,
                                          require_current_span_ids=set(visible) if state['evidence_state']['evidence'] else set())
        return {'gate': {'ok': result.ok, 'code': result.code, 'message': result.message},
                'trace': state['trace'] + [{'node': 'gate', 'detail': '原文与协议校验通过' if result.ok else result.code}]}

    def extract(self, state):
        citations = []
        if state['gate']['ok']:
            sources = {s['span_id']: h for h in state['hits'] for s in h['spans']}
            for item in state['evidence_state']['evidence']:
                source = sources[item['span_id']]
                citations.append({'file': source['name'], 'location': source['location'],
                                  'version': source['version'], 'chunk_id': source['chunk_id'], **item})
        answer = ('以下是相关原文摘录，尚未经模型判断，不能视为完整答案。' if citations
                  else '未找到可展示的相关原文；请调整关键词或导入相关文档。')
        return {'answer': answer, 'citations': citations, 'trace': state['trace'] + [
            {'node': 'extract', 'detail': '返回可追溯原文；没有调用大模型'}]}

    def ask(self, question: str) -> dict:
        result = self.graph.invoke({'question': question})
        result.pop('hits', None)
        return result

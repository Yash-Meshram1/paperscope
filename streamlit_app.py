"""Run: streamlit run streamlit_app.py"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

import streamlit as st
from paperscope.engine import Config, load_engine
from paperscope.paper_aliases import detect_paper_ids


@st.cache_resource
def initialize(config):
    return load_engine(config)


st.set_page_config(page_title='PaperScope', page_icon='📚')
st.title('PaperScope')
st.caption('Ask questions about your research papers and inspect the evidence.')
with st.form('question_form'):
    question = st.text_area('Question', placeholder='Compare Self-RAG and CRAG.')
    top_k = st.slider('Maximum source passages', 1, 10, 5)
    submitted = st.form_submit_button('Explore')

if submitted:
    st.session_state.pop('response', None)
    if not question.strip():
        st.warning('Enter a question first.')
    else:
        stage = 'initialization'
        try:
            with st.spinner('Loading paper search…'):
                engine = initialize(Config.from_env())
            stage = 'retrieval'
            with st.spinner('Finding evidence…'):
                evidence = engine.retrieve(question, top_k=top_k)
            st.session_state.response = {'question':question, 'evidence':evidence, 'answer':None,
                                         'named':detect_paper_ids(question)}
            stage = 'generation'
            with st.spinner('Writing an answer from the evidence…'):
                st.session_state.response['answer'] = engine.answer(question, evidence)
        except Exception as exc:
            messages = {
                'initialization':'Could not load paper search. Check saved artifacts and cached models. Close any notebook using the same Qdrant storage.',
                'retrieval':'Could not retrieve evidence. Check the search resources and try again.',
                'generation':'Could not generate an answer. Check that Ollama is running and the configured model is installed; a timeout may require retrying.'}
            st.error(messages[stage])
            with st.expander('Error details'):
                st.code(str(exc))

if 'response' in st.session_state:
    response = st.session_state.response
    st.subheader(response['question'])
    if response['answer']:
        st.markdown(response['answer'])
    if not response['named']:
        st.caption('Searching the full corpus. Unrecognized paper names do not apply a paper filter.')
    if not response['evidence']:
        st.info('No usable source passages were found.')
    for index, row in enumerate(response['evidence'], 1):
        c = row['chunk']
        with st.expander(f"[SOURCE {index}] {c['title']}"):
            st.write(f"Paper: {c['paper_id']}")
            st.write(f"Section: {c['section_heading']}")
            st.write(f"Pages: {c['page_start'] if c['page_start'] is not None else '?'}–{c['page_end'] if c['page_end'] is not None else '?'}")
            st.caption(f"Chunk: {c['chunk_id']}")
            st.caption(f"Ranking score: {row.get('rerank_score', 'unavailable')} · RRF: {row.get('rrf_score', 'unavailable')}")
            st.caption('Ranking scores are not confidence probabilities. Page spans are inherited from sections.')
            st.text(c['text'])

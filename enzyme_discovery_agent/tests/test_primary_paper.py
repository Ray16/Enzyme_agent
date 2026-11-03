"""Primary reading preserves provenance and distinguishes access from absence."""
import hashlib
from unittest.mock import patch
from enzyme_discovery_agent.tools.primary_paper import read_primary_paper


def test_primary_passage_has_verifiable_source_and_identity_leads(tmp_path):
    text='Context. Protein P12345 was measured in the reported study. More context. '*30
    with patch('enzyme_discovery_agent.tools.measurement_extraction.resolve_paper',return_value={'doi':'synthetic:paper'}), \
         patch('enzyme_discovery_agent.tools.measurement_extraction.fetch_jats_or_text',return_value=(b'content','xml')), \
         patch('enzyme_discovery_agent.tools.measurement_extraction._jats_to_text',return_value=text):
        result=read_primary_paper(tmp_path,'synthetic:paper',query='P12345',max_characters=500)
    data=result['data']
    assert result['status']=='success' and 'P12345' in data['mentioned_uniprot_accessions']
    assert data['text_sha256']==hashlib.sha256((tmp_path/data['source_path']).read_bytes()).hexdigest()
    assert data['passage']==text[data['start_character']:data['start_character']+500]


def test_unavailable_primary_text_is_not_a_negative_biological_result(tmp_path):
    with patch('enzyme_discovery_agent.tools.measurement_extraction.resolve_paper',return_value={}), \
         patch('enzyme_discovery_agent.tools.measurement_extraction.fetch_jats_or_text',return_value=(None,'none')):
        result=read_primary_paper(tmp_path,'synthetic:paper')
    assert result['status']=='unavailable' and result['data']['text_available'] is False


def test_continuation_advances_even_when_query_is_retained(tmp_path):
    text='query '+('remaining text '*300)
    with patch('enzyme_discovery_agent.tools.measurement_extraction.resolve_paper',return_value={}), \
         patch('enzyme_discovery_agent.tools.measurement_extraction.fetch_jats_or_text',return_value=(b'content','xml')), \
         patch('enzyme_discovery_agent.tools.measurement_extraction._jats_to_text',return_value=text):
        result=read_primary_paper(tmp_path,'synthetic:paper',query='query',offset_characters=500,max_characters=500)
    assert result['data']['start_character']==500
    assert result['data']['passage']==text[500:1000]

"""Saved reasoning remains inspectable without executing text from sources."""
import json
import re

from enzyme_discovery_agent.output.reasoning_viewer import write_reasoning_viewer


def test_viewer_embeds_source_text_as_data_and_keeps_full_records(tmp_path):
    hostile='</script><script>window.untrustedCodeRan=true</script>'
    trace={'events':[{'id':'prediction_registered:synthetic','sequence':1,'type':'prediction_registered',
                     'record':{'hypothesis':hostile,'alternative':'A distinct explanation.'}}],
           'hash_chain_valid':True,'source_files_intact':True,'run_dir':'synthetic campaign'}
    output=write_reasoning_viewer(tmp_path,trace)
    html=output.read_text()
    data=re.search(r'<script id="trace-data" type="application/json">(.*?)</script>',html,re.S).group(1)
    assert json.loads(data)==trace
    assert hostile not in html
    assert '\\u003c/script>' in data
    assert 'https://' not in html and '<script src=' not in html


def test_empty_journal_is_not_presented_as_a_successful_reasoning_run(tmp_path):
    output=write_reasoning_viewer(tmp_path,{'events':[],'hash_chain_valid':True,'source_files_intact':True})
    assert 'No explicit scientific records have been captured yet' in output.read_text()

"""Alignment transport recovery and sequence identity regressions."""
import io
import json
from pathlib import Path
import tarfile
from unittest.mock import Mock

import pytest
import requests

from structural_modeling_agent import msa


def archive_bytes(sequences):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for filename in msa.ALIGNMENTS:
            data=''.join(f'>{101+i}\n{s}\n>hit\n{s[:-1]}A\n\x00' for i,s in enumerate(sequences)).encode()
            member=tarfile.TarInfo(filename);member.size=len(data);tar.addfile(member,io.BytesIO(data))
    return stream.getvalue()


def test_download_failure_resumes_same_ticket_without_resubmitting(tmp_path,monkeypatch):
    monkeypatch.setattr(msa.time,'sleep',lambda _:None)
    session=Mock()
    session.request.side_effect=[Mock(status_code=200,json=lambda:{'id':'SYNTH_TICKET','status':'COMPLETE'})]
    session.get.side_effect=requests.ConnectTimeout('synthetic transient download failure')
    sequences=['ACDEFGHIK','LMNPQRSTV']
    with pytest.raises(requests.ConnectTimeout):msa.fetch_batch(sequences,tmp_path,session=session)
    session.request.assert_called_once()
    session.get.side_effect=None;session.get.return_value=Mock(content=archive_bytes(sequences))
    paths=msa.fetch_batch(sequences,tmp_path,session=session)
    assert session.request.call_count==1
    assert all(msa.valid_csv(path,sequence) for path,sequence in zip(paths,sequences))


def test_ambiguous_submission_is_checkpointed_instead_of_duplicated(tmp_path):
    session=Mock();session.request.side_effect=requests.ReadTimeout('response unavailable')
    with pytest.raises(requests.ReadTimeout):msa.fetch_batch(['ACDEFGHIK'],tmp_path,session=session)
    with pytest.raises(RuntimeError,match='ambiguous'):msa.fetch_batch(['ACDEFGHIK'],tmp_path,session=session)
    assert session.request.call_count==1


def test_preparation_batches_distinct_sequences_and_detects_corrupted_cache(tmp_path):
    fasta=tmp_path/'input.fasta';fasta.write_text('>SYNTH_A\nACDEFGHIK\n>SYNTH_ALIAS\nACDEFGHIK\n>SYNTH_B\nLMNPQRSTV\n')
    cache=tmp_path/'cache';calls=[]
    def fetch(sequences,root):
        calls.append(sequences)
        for sequence in sequences:
            path=root/(msa.digest(sequence.encode())+'.csv');content='key,sequence\n-1,'+sequence+'\n-1,'+sequence[:-1]+'A\n'
            msa.write(path,content);msa.save(path.with_suffix('.json'),{
                'csv_sha256':msa.digest(content.encode()),'sequence_sha256':msa.digest(sequence.encode()),
                'alignment_policy':'full_server_alignment_v1'})
    rows=msa.prepare(fasta,tmp_path/'out',cache,fetcher=fetch)
    assert calls==[['ACDEFGHIK','LMNPQRSTV']]
    assert rows[0]['csv']==rows[1]['csv']
    msa.prepare(fasta,tmp_path/'out',cache,fetcher=fetch)
    assert len(calls)==1
    Path(rows[0]['csv']).write_text('key,sequence\n-1,ACDEFGHIK\n-1,WRONG\n')
    msa.prepare(fasta,tmp_path/'out',cache,fetcher=fetch)
    assert calls[-1]==['ACDEFGHIK']


def test_archive_with_wrong_query_is_rejected(tmp_path):
    archive=tmp_path/'result.tar.gz';archive.write_bytes(archive_bytes(['ACDEFGHIK']))
    with pytest.raises(ValueError,match='exact source sequence'):
        msa.alignment_csvs(archive,['LMNPQRSTV'])

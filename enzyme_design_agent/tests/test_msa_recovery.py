import unittest
from unittest.mock import Mock
from enzyme_design_agent.saturation.msa import request_batch,alignment_csv,transient_service_error


class MsaRecoveryTests(unittest.TestCase):
    def test_dns_failure_retries_same_full_batch(self):
        queries=['A'*20,'C'*20]
        fetch=Mock(side_effect=[ConnectionError('Temporary failure in name resolution'),['alignment_a','alignment_b']])
        sleep=Mock()
        result=request_batch(fetch,queries,'batch_path',retry_seconds=7,sleep=sleep)
        self.assertEqual(result,['alignment_a','alignment_b'])
        self.assertEqual(fetch.call_count,2)
        self.assertEqual(fetch.call_args_list[0],fetch.call_args_list[1])
        self.assertEqual(fetch.call_args.args[0],queries)
        sleep.assert_called_once_with(7)

    def test_retry_budget_is_bounded(self):
        fetch=Mock(side_effect=ConnectionError('network unavailable'));sleep=Mock()
        with self.assertRaises(ConnectionError):request_batch(fetch,['ACD'],'prefix',max_attempts=3,sleep=sleep)
        self.assertEqual(fetch.call_count,3);self.assertEqual(sleep.call_count,2)

    def test_invalid_alignment_is_not_a_retryable_service_error(self):
        with self.assertRaises(ValueError) as ctx:alignment_csv('>query\nACD\n','ACE')
        self.assertFalse(transient_service_error(ctx.exception))
        fetch=Mock(side_effect=ValueError('MSA query does not match variant sequence'));sleep=Mock()
        with self.assertRaises(ValueError):request_batch(fetch,['ACD'],'prefix',sleep=sleep)
        fetch.assert_called_once();sleep.assert_not_called()

    def test_insertions_are_not_counted_as_aligned_columns(self):
        self.assertIn('-1,ACaa-D\n',alignment_csv('>q\nAC-D\n>hit\nACaa-D\n','AC-D'))

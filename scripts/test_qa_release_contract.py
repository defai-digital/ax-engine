#!/usr/bin/env python3
"""Regressions for complete responses and declared structured-answer contracts."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qa'))
from checkers import check_exact_answer, run_all_checks
from client import QaResponse
from prompts import get_prompt_by_id
import run_qa
from reporter import build_results_payload, generate_html_report


class ReleaseContractTests(unittest.TestCase):
    def test_reports_preserve_checker_contract_without_relabeling_legacy_runs(self):
        contract = run_qa.CHECKER_CONTRACT_VERSION
        metadata = {'checker_contract': contract}
        assert build_results_payload([], metadata)['checker_contract'] == contract
        assert contract in generate_html_report([], metadata)
        assert build_results_payload([], {})['checker_contract'] is None

    def test_ordered_comma_lists_match_whole_answers_and_preserve_multiplicity(self):
        letters = get_prompt_by_id('instruction_alphabet_first')
        numbers = get_prompt_by_id('instruction_sort_numbers')
        for text in ('a,b,c,d,e', 'A, B, C, D, E', ' a , b,c, d ,e '):
            assert check_exact_answer(text, letters).passed
        for text in ('a,b,c,d', 'a,b,c,d,e,f', 'a,c,b,d,e', 'a,b,b,d,e',
                     'a,b,c,d,e,', 'a,,b,c,d,e', 'Answer: a,b,c,d,e',
                     'a,b,c,d,e are letters', 'a;b;c;d;e', 'a,b,c,d,e.'):
            assert not check_exact_answer(text, letters).passed, text
        assert check_exact_answer('2, 2,7, 9', numbers).passed
        for text in ('2,7,9', '2,2,2,7,9', '2,7,2,9', '2,2,7,9,10'):
            assert not check_exact_answer(text, numbers).passed, text
        assert not check_exact_answer('A, B, C, D, E', replace(letters, exact_match='auto')).passed
        assert not check_exact_answer('A,B,C,D,E', replace(letters, exact_match='full')).passed


    def test_complete_declared_numeric_pattern_is_not_incoherent_prose(self):
        prompt = get_prompt_by_id('reasoning_set_membership')
        for text in ('2, 3', '3,2'):
            assert run_all_checks(text, prompt).auto_pass
        for text in ('2,4', '2,3,4', '232323', '2 3'):
            assert not run_all_checks(text, prompt).auto_pass, text
        assert not run_all_checks('2, 3', replace(prompt, regex_patterns=[])).auto_pass


    def test_qa_suite_rejects_budget_exhaustion_even_when_gold_is_present(self):
        for stream in (False, True):
            for finish in ('stop', 'length', 'max_output_tokens', 'cancelled', 'error', None):
                response = QaResponse(text='8', finish_reason=finish, stream=stream)
                with patch.object(run_qa, 'send_request', return_value=response):
                    rows, _, _ = run_qa.run_qa_suite(
                        base_url='http://fixture', model_id='fixture', mode_label='direct',
                        streams=[stream], max_tokens=256, temperature=0,
                        repetition_penalty=None, prompt_ids=['knowledge_binary_bit'],
                        timeout=1, seed=1,
                    )
                assert rows[0]['report'].auto_pass is (finish == 'stop'), (stream, finish)


if __name__ == "__main__":
    unittest.main()

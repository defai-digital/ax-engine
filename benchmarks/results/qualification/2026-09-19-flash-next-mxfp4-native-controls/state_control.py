"""Original bounded native state assertions; no model execution."""
import math

def require(ok,message):
    if not ok:raise ValueError(message)

def state_control(raw,runner=False):
    require(raw['qualification'] is False and raw['within_tolerance'] is True
            and raw['identity_until_first_tie'] is True and raw['greedy_identity'] is True
            and raw['tie_divergences']==[], 'Original state/runner diagnostic failed')
    require(math.isclose(raw['tolerance'],0.1,rel_tol=1e-6) and math.isclose(raw['tie_margin'],0.5),
            'Original numeric limits changed')
    if runner:
        require(raw['target_schedule']=='canonical_singleton' and raw['decode_state_compared'] is False,
                'Wrong runner scope')
        require(raw['direct_ids'] and raw['direct_ids']==raw['mtp_ids'],
                'Runner token identity cannot be reconstructed')
    else:
        require(all(raw[key] is True for key in ['prefill_primary_and_draft_state_exact',
                    'forced_acceptance_rejection_budget_and_eos','zero_budget_preserves_primary_and_draft_state',
                    'primary_state_within_tolerance_each_compared_step','draft_state_within_tolerance_each_compared_step']),
                'Incomplete state safety controls')
        require(type(raw['compared_state_steps']) is int and raw['compared_state_steps']>0
                and type(raw['compared_generated_tokens']) is int and raw['compared_generated_tokens']>0
                and raw['max_state_relative_divergence']<=raw['tolerance']
                and raw['max_logit_relative_divergence']<=raw['tolerance']
                and raw['mtp_only_selected_decode_payload_bytes']>0,'Incomplete state comparison coverage')
        require(len(raw['generated_ids'])==raw['compared_generated_tokens'] and raw['state_arrays'],
                'State coverage does not match raw arrays')

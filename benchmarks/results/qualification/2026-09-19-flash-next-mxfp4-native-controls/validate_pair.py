"""Validate trained-head coverage, raw accounting and exact paired output identity."""
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    require(type(value) is int and value >= 0, 'Expected a nonnegative integer')
    return value


def validate_head(raw, permuted, manifest, terminal_ids):
    require(raw['qualification'] is False and raw['route'] == 'flash_next_mtp_trained_head_oracle',
            'Wrong result scope')
    require(raw['head_permuted'] is permuted, 'Wrong head mode')
    require(raw['permute_seed'] == (20260916 if permuted else None), 'Wrong permutation seed')
    require(raw['terminal_ids'] == terminal_ids, 'Wrong terminal IDs')
    require(raw['greedy_identity'] is True and raw['identity_until_first_tie'] is True
            and not raw['tie_divergences'], 'Exact direct/MTP identity is required')
    expected = manifest['requests']
    require([x['id'] for x in raw['requests']] == [x['id'] for x in expected], 'Incomplete/reordered cohort')
    proposed = accepted = agreed = samples = short_count = contributing = 0
    outputs = []
    for item, prompt in zip(raw['requests'], expected, strict=True):
        require(item['prompt_ids'] == prompt['prompt_ids'], 'Prompt IDs changed')
        require(item['greedy_identity'] is True and item['identity_until_first_tie'] is True
                and not item['tie_divergences'], 'Request token identity failed')
        tokens = item['greedy_tokens']
        require(tokens == item['generated_ids'], 'Generated token aliases disagree')
        require(0 < len(tokens) <= manifest['max_new_tokens'], 'Wrong output budget')
        require(all(type(x) is int and x >= 0 for x in tokens), 'Invalid generated tokens')
        terminal_positions = [i for i, token in enumerate(tokens) if token in terminal_ids]
        position = terminal_positions[0] if terminal_positions else None
        require(terminal_positions == ([len(tokens) - 1] if terminal_positions else []),
                'Output continues past terminal token')
        require(item['stopped_at_terminal'] is (position is not None), 'Wrong terminal flag')
        require(item['terminal_position'] == position, 'Wrong terminal position')
        require(position is not None or len(tokens) == manifest['max_new_tokens'], 'Truncated nonterminal output')
        require(integer(item['compared_positions']) == len(tokens), 'Incomplete position coverage')
        too_short = (position if position is not None else len(tokens)) < 2
        require(item['too_short'] is too_short, 'Wrong short-request exclusion')
        p, a = integer(item['proposed']), integer(item['accepted'])
        require(a <= p, 'Accepted exceeds proposed')
        agreement = item['draft_vs_primary_top1_agreement']
        require(all(type(x) is bool for x in agreement), 'Invalid agreement samples')
        # Agreement samples and session proposals are different diagnostics.
        agreement_rate = sum(agreement) / len(agreement) if agreement else 0.0
        require(math.isclose(item['draft_vs_primary_top1_agreement_rate'], agreement_rate,
                             rel_tol=1e-12, abs_tol=1e-12), 'Wrong request agreement rate')
        if too_short:
            short_count += 1
        else:
            require(p > 0, 'Non-short request has no session proposals')
            proposed += p
            accepted += a
            agreed += sum(agreement)
            samples += len(agreement)
            contributing += int(p > 0)
        outputs.append(tokens)
    require(raw['too_short_requests'] == short_count, 'Wrong excluded count')
    require(integer(raw['proposed']) == proposed > 0 and integer(raw['accepted']) == accepted,
            'Wrong aggregate session counters')
    rate = accepted / proposed
    require(math.isclose(raw['acceptance_rate'], rate, rel_tol=1e-12, abs_tol=1e-12), 'Wrong acceptance rate')
    require(raw['draft_vs_primary_top1_agreement_samples'] == samples
            and raw['draft_vs_primary_top1_agreement_matches'] == agreed, 'Wrong agreement counts')
    require(math.isclose(raw['draft_vs_primary_top1_agreement_rate'], agreed / samples if samples else 0.0,
                         rel_tol=1e-12, abs_tol=1e-12), 'Wrong aggregate agreement rate')
    return dict(proposed=proposed, accepted=accepted, acceptance_rate=rate, outputs=outputs,
                requests=len(expected), excluded_short_requests=short_count,
                contributing_requests=contributing)


def validate_pair(real, permuted, manifest, terminal_ids):
    first = validate_head(real, False, manifest, terminal_ids)
    second = validate_head(permuted, True, manifest, terminal_ids)
    require(first['outputs'] == second['outputs'], 'Changing only the draft head changed primary outputs')
    return dict(qualification=False, release_ready=False, greedy_identity=True,
                real_head=first, permuted_head=second,
                real_acceptance_min=0.7, permuted_acceptance_max_exclusive=0.1,
                passed=first['accepted'] * 10 >= first['proposed'] * 7
                and second['accepted'] * 10 < second['proposed'])

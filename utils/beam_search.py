
from typing import List, Tuple, Dict
import argparse

from vllm import LLM, SamplingParams

from prm import RewardModel


def get_scores(
        prm: RewardModel,
        questions: List[str],
        current_texts: List[List[str]],
        score_dict: Dict[str, float],
):
    """
    prm:              RewardModel
    questions:        List[str] of length N
    current_texts:    List[List[str]] of length N
    score_dict:       Dict[str, float]
    returns:          List[List[float]] scores, the last element of prm.score’s output.
    """
    # 1) Pre-fill using score_dict
    N = len(questions)

    scores = []
    done = []
    for i in range(N):
        sc_row = []
        dn_row = []
        for text in current_texts[i]:
            key = (questions[i], text)
            sc_row.append(score_dict.get(key, 0.0))
            dn_row.append(key in score_dict)
        scores.append(sc_row)
        done.append(dn_row)

    # 2) Build a smaller batch of only the undone (i,j) entries
    batch_qs = []
    batch_texts = []
    index_map = []  # will hold tuples (i, [j1, j2, ...]) per question

    for i, (q, texts, dones) in enumerate(zip(questions, current_texts, done)):
        undone_idxs = [j for j, d in enumerate(dones) if not d]
        if undone_idxs:
            batch_qs.append(q)
            batch_texts.append([texts[j] for j in undone_idxs])
            index_map.append((i, undone_idxs))

    # 3) Score only that batch
    if batch_qs:
        raw_batch = prm.score(batch_qs, batch_texts)
        # raw_batch[k] corresponds to question batch_qs[k] and its subset of texts

        # 4) Scatter the real scores back into the pre-filled “scores” array
        for (i, undone_idxs), raw_row in zip(index_map, raw_batch):
            real_scores = [rlist[-1] for rlist in raw_row]
            for j, sc in zip(undone_idxs, real_scores):
                scores[i][j] = sc
                score_dict[(questions[i], current_texts[i][j])] = sc

    return scores


def _process_batch_one_step(
        batch_prompts: List[str],
        n_samples: int,
        llm: LLM,
        stop_ids: List[int],
        temperature: float,
        max_tokens: int,
        seed: int,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Run LLM.generate on a batch, return texts and their stop_reasons."""
    if not batch_prompts:
        return [], []
    params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["\n\n", "<|eot_id|>"],
        stop_token_ids=stop_ids,
        seed=seed,
    )
    results = llm.generate(batch_prompts, params)
    outputs = [[out.text for out in res.outputs] for res in results]
    stop_reasons = [[out.stop_reason for out in res.outputs] for res in results]
    return outputs, stop_reasons


def process_one_step(
        formatted_inputs: List[List[str]],
        n_samples_grid: List[List[int]],
        done: List[List[bool]],
        llm: LLM,
        args: argparse.Namespace
) -> Tuple[List[List[List[str]]], List[List[List[str]]]]:
    """
    For each (question, path) not yet done, generate n_samples;
    skip those already done.
    """
    n_q = len(formatted_inputs)
    # prep output placeholders
    all_outputs: List[List[List[str]]] = [[[] for _ in row] for row in formatted_inputs]
    all_reasons: List[List[List[str]]] = [[[] for _ in row] for row in formatted_inputs]

    # bucket by sample count
    to_batch: Dict[int, List[Tuple[int, int]]] = {}
    for i, (inp_row, ns_row, done_row) in enumerate(zip(formatted_inputs, n_samples_grid, done)):
        for j, (prompt, n_samp, is_done) in enumerate(zip(inp_row, ns_row, done_row)):
            if is_done:
                all_outputs[i][j] = [""]
                all_reasons[i][j] = [None]
            else:
                to_batch.setdefault(n_samp, []).append((i, j))

    # run each batch
    for samp_count, positions in to_batch.items():
        prompts = [formatted_inputs[i][j] for i, j in positions]
        outs, reasons = _process_batch_one_step(
            prompts, samp_count, llm, args.stop_ids, args.temperature,
            args.max_tokens, args.seed + args.trial
        )
        for (i, j), out_list, rsn_list in zip(positions, outs, reasons):
            all_outputs[i][j] = out_list
            all_reasons[i][j] = rsn_list

    return all_outputs, all_reasons


def update_done(
        done: List[List[bool]],
        stop_reasons: List[List[str]],
) -> List[List[bool]]:
    """
    Mark a path done if previously done or if any stop_reason != initial_stop.
    """
    new_done: List[List[bool]] = []
    for prev_row, sr_row in zip(done, stop_reasons):
        row_done: List[bool] = []
        for was_done, reason in zip(prev_row, sr_row):
            row_done.append(was_done or (reason != "\n\n"))
        new_done.append(row_done)
    return new_done


def select_topk(
        output_reasonings: List[List[List[str]]],
        outputs: List[List[str]],
        stop_reasons: List[List[str]],
        scores: List[List[float]],
        done: List[List[bool]],
        k_list: List[int],
        median_idx: int = None,
) -> Tuple[
    List[List[List[str]]],  # kept reasonings
    List[List[str]],  # kept outputs
    List[List[str]],  # kept stop reasons
    List[List[float]],  # kept scores
    List[List[bool]]  # kept done flags
]:
    """
    For each example, pick the top-k samples by score.
    """
    B_or, B_out, B_sr, B_sc, B_dn = [], [], [], [], []
    for or_row, out_row, sr_row, sc_row, dn_row, k in zip(
            output_reasonings, outputs, stop_reasons, scores, done, k_list
    ):
        # get indices of top-k scores (or all if fewer than k)
        if median_idx is None:
            topk_idxs = sorted(
                range(len(sc_row)),
                key=lambda i: sc_row[i],
                reverse=True
            )[:k]
        else:
            topk_idxs = sorted(
                range(len(sc_row)),
                key=lambda i: sc_row[i][median_idx],
                reverse=True
            )[:k]

        # collect the top-k entries
        B_or.append([or_row[i] for i in topk_idxs])
        B_out.append([out_row[i] for i in topk_idxs])
        B_sr.append([sr_row[i] for i in topk_idxs])
        B_sc.append([sc_row[i] for i in topk_idxs])
        B_dn.append([dn_row[i] for i in topk_idxs])

    return B_or, B_out, B_sr, B_sc, B_dn


def aggregate(
        output_reasonings: List[List[List[str]]],
        done: List[List[bool]],
        step_outputs: List[List[List[str]]],
        step_reasons: List[List[List[str]]],
) -> Tuple[List[List[List[str]]], List[List[bool]], List[List[str]]]:
    """
    Take all existing paths and extend each by every new segment
    """
    agg_or, agg_sr, agg_dn = [], [], []
    for prev_paths, prev_dones, new_lists, new_reasons in zip(output_reasonings, done, step_outputs, step_reasons):
        next_paths, next_reas, next_dones = [], [], []
        for path, flag, candidates, reasons in zip(prev_paths, prev_dones, new_lists, new_reasons):
            for seg, rsn in zip(candidates, reasons):
                if """\\boxed""" in seg:
                    rsn = None

                if seg != "":
                    next_paths.append(path + [seg])
                else:
                    next_paths.append(path.copy())

                next_dones.append(flag)  # duplicated & appended
                next_reas.append(rsn)

        agg_or.append(next_paths)
        agg_dn.append(next_dones)
        agg_sr.append(next_reas)
    return agg_or, agg_dn, agg_sr

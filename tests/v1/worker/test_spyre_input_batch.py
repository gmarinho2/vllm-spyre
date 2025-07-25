# # SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.v1.sample.logits_processor import (BatchUpdateBuilder,
                                             init_builtin_logitsprocs)
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_spyre.v1.worker.spyre_input_batch import (CachedRequestState,
                                                    InputBatch)

VOCAB_SIZE = 1024
NUM_OUTPUT_TOKENS = 20
MAX_PROMPT_SIZE = 100
MAX_NUM_PROMPT_TOKENS = 64


def _remove_requests(input_batch: InputBatch, batch_size: int,
                     reqs: list[CachedRequestState]) -> set[str]:
    """
    Remove some requests randomly from the batch and returns a set of 
    request ids removed 
    """

    num_reqs_to_remove = np.random.randint(0, batch_size)
    req_indices_to_remove: set[int] = set()
    for _ in range(num_reqs_to_remove):
        req_index_to_remove = np.random.randint(0, batch_size)
        req_indices_to_remove.add(req_index_to_remove)

    req_ids_to_remove: set[str] = set()
    for index in req_indices_to_remove:
        input_batch.remove_request(reqs[index].req_id)
        req_ids_to_remove.add(reqs[index].req_id)
    return req_ids_to_remove


def _construct_expected_sampling_metadata(
    reqs: list[CachedRequestState],
    req_ids_retained: set[int],
    input_batch: InputBatch,
    device: torch.device,
) -> SamplingMetadata:
    """
    Constructs and returns the expected SamplingMetadata for this
    batch.
    """
    num_reqs = len(req_ids_retained)
    output_token_ids: list[list[int]] = [list() for _ in range(num_reqs)]
    prompt_token_ids: list[list[int]] = [list() for _ in range(num_reqs)]
    presence_penalties = [0.0 for _ in range(num_reqs)]
    frequency_penalties = [0.0 for _ in range(num_reqs)]
    repetition_penalties = [1.0 for _ in range(num_reqs)]
    top_k = [0 for _ in range(num_reqs)]
    top_p = [0.0 for _ in range(num_reqs)]
    temperature = [0.0 for _ in range(num_reqs)]
    allowed_token_ids_mask = torch.zeros(num_reqs,
                                         VOCAB_SIZE,
                                         dtype=torch.bool,
                                         device=device)

    batch_update_builder = BatchUpdateBuilder()
    logitsprocs = init_builtin_logitsprocs(pin_memory_available=False,
                                           max_num_reqs=len(reqs) + 1,
                                           device=device)

    bad_words_token_ids = {}
    for req in reqs:
        if req.req_id not in req_ids_retained:
            continue
        index_in_input_batch = input_batch.req_id_to_dense_index(req.req_id)

        params = req.sampling_params
        batch_update_builder.added.append(
            (index_in_input_batch, params, req.output_token_ids))

        output_token_ids[index_in_input_batch] = req.output_token_ids
        prompt_token_ids[index_in_input_batch] = req.prompt_token_ids
        presence_penalties[
            index_in_input_batch] = req.sampling_params.presence_penalty
        frequency_penalties[index_in_input_batch] = (
            req.sampling_params.frequency_penalty)
        repetition_penalties[index_in_input_batch] = (
            req.sampling_params.repetition_penalty)
        top_k[index_in_input_batch] = req.sampling_params.top_k
        top_p[index_in_input_batch] = req.sampling_params.top_p
        temperature[index_in_input_batch] = req.sampling_params.temperature
        if req.sampling_params.allowed_token_ids:
            allowed_token_ids_mask[index_in_input_batch][
                req.sampling_params.allowed_token_ids] = True
        if req.sampling_params.bad_words_token_ids:
            bad_words_token_ids[
                index_in_input_batch] = req.sampling_params.bad_words_token_ids

    batch_update = batch_update_builder.get_and_reset(num_reqs)
    for logit_proc in logitsprocs.all:
        logit_proc.update_state(batch_update)

    return SamplingMetadata(
        temperature=torch.tensor(temperature, dtype=torch.float,
                                 device=device),
        all_greedy=False,
        all_random=True,
        top_p=None if all(x == 1.0 for x in top_p) else torch.tensor(
            top_p, dtype=torch.float, device=device),
        top_k=None if all(x == 0 for x in top_k) else torch.tensor(
            top_k, dtype=torch.int, device=device),
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=make_tensor_with_pad(
            prompt_token_ids,
            pad=VOCAB_SIZE,
            device=torch.device(device),
            dtype=torch.int64,
        ),
        frequency_penalties=torch.tensor(frequency_penalties,
                                         dtype=torch.float,
                                         device=device),
        presence_penalties=torch.tensor(presence_penalties,
                                        dtype=torch.float,
                                        device=device),
        repetition_penalties=torch.tensor(repetition_penalties,
                                          dtype=torch.float,
                                          device=device),
        output_token_ids=output_token_ids,
        no_penalties=(all(x == 0 for x in presence_penalties)
                      and all(x == 0 for x in frequency_penalties)
                      and all(x == 1 for x in repetition_penalties)),
        allowed_token_ids_mask=allowed_token_ids_mask,
        bad_words_token_ids=bad_words_token_ids,
        logitsprocs=logitsprocs,
    )


def _create_sampling_params():
    return SamplingParams(
        top_k=np.random.randint(1, 10),
        top_p=np.random.uniform(0.0, 1.0),
        presence_penalty=np.random.uniform(-2.0, 2.0),
        repetition_penalty=np.random.uniform(0.0, 2.0),
        frequency_penalty=np.random.uniform(-2.0, 2.0),
        min_tokens=np.random.randint(1, 10),
        stop_token_ids=[
            np.random.randint(0, VOCAB_SIZE)
            for _ in range(np.random.randint(10))
        ],
        logit_bias={0: np.random.uniform(-3.0, 3.0)},
    )


def _construct_cached_request_state(req_id_suffix: int):
    prompt_token_ids = [
        np.random.randint(0, VOCAB_SIZE)
        for _ in range(np.random.randint(0, MAX_PROMPT_SIZE))
    ]
    output_token_ids = [
        np.random.randint(0, VOCAB_SIZE)
        for _ in range(np.random.randint(0, NUM_OUTPUT_TOKENS))
    ]
    return CachedRequestState(
        req_id=f"req_id_{req_id_suffix}",
        prompt_token_ids=prompt_token_ids,
        sampling_params=_create_sampling_params(),
        generator=None,
        output_token_ids=output_token_ids,
        left_padding=0,
    )


def compare_results(sampling_metadata, expected_sampling_metadata):

    def same(t1: Optional[torch.Tensor], t2: Optional[torch.Tensor]) -> bool:
        return (t1 is None
                and t2 is None) or (t1 is not None and t2 is not None
                                    and torch.allclose(t1, t2))

    # Assert the actual and expected output.
    assert torch.allclose(expected_sampling_metadata.temperature,
                          sampling_metadata.temperature)
    assert same(expected_sampling_metadata.top_p, sampling_metadata.top_p)
    assert same(expected_sampling_metadata.top_k, sampling_metadata.top_k)
    assert torch.allclose(
        expected_sampling_metadata.frequency_penalties,
        sampling_metadata.frequency_penalties,
    )
    assert torch.allclose(
        expected_sampling_metadata.presence_penalties,
        sampling_metadata.presence_penalties,
    )
    assert torch.allclose(
        expected_sampling_metadata.repetition_penalties,
        sampling_metadata.repetition_penalties,
    )

    assert torch.allclose(expected_sampling_metadata.prompt_token_ids,
                          sampling_metadata.prompt_token_ids)
    assert (expected_sampling_metadata.output_token_ids ==
            sampling_metadata.output_token_ids)
    assert expected_sampling_metadata.no_penalties == \
           sampling_metadata.no_penalties
    if sampling_metadata.allowed_token_ids_mask:
        assert torch.allclose(
            expected_sampling_metadata.allowed_token_ids_mask,
            sampling_metadata.allowed_token_ids_mask)
    assert expected_sampling_metadata.bad_words_token_ids == \
        sampling_metadata.bad_words_token_ids


@pytest.mark.v1
@pytest.mark.worker
@pytest.mark.parametrize("batch_size", [1, 2, 32, 64])
def test_sampling_metadata_in_input_batch(batch_size: int):
    """
    Tests the logic for managing sampling metadata in the InputBatch.

    This test involves adding a set of requests to the InputBatch,
    followed by removing a subset of them. Afterward, the batch is compacted,
    and the `make_sampling_metadata` method is invoked on the batch. The
    output of `make_sampling_metadata` is then compared against the expected
    results to ensure correctness.
    """

    device = torch.device('cpu')
    input_batch: InputBatch = InputBatch(
        max_num_reqs=batch_size,
        max_model_len=1024,
        device=device,
        pin_memory=is_pin_memory_available(),
        vocab_size=1024,
    )
    reqs: list[CachedRequestState] = []
    req_id_reqs = {}
    # Add requests
    for req_index in range(batch_size):
        req: CachedRequestState = _construct_cached_request_state(req_index)
        input_batch.add_request(req, req_index)
        reqs.append(req)
        req_id_reqs[req.req_id] = req

    # Remove some requests
    req_ids_to_remove = _remove_requests(input_batch, batch_size, reqs)
    req_ids_retained = set(req_id_reqs.keys()) - req_ids_to_remove

    # Generate the sampling metadata
    sampling_metadata = input_batch._make_sampling_metadata()

    # Create expected output.
    expected_sampling_metadata = _construct_expected_sampling_metadata(
        reqs, req_ids_retained, input_batch, device=torch.device(device))

    compare_results(sampling_metadata, expected_sampling_metadata)

    if len(req_ids_to_remove) == 0:
        return

    # Add more requests
    for req_index in range(len(req_ids_to_remove)):
        req: CachedRequestState = _construct_cached_request_state(req_index +
                                                                  batch_size)
        input_batch.add_request(req)
        reqs.append(req)
        req_ids_retained.add(req.req_id)

    sampling_metadata = input_batch._make_sampling_metadata()

    # Create expected output.
    expected_sampling_metadata = _construct_expected_sampling_metadata(
        reqs, req_ids_retained, input_batch, device=torch.device(device))

    compare_results(sampling_metadata, expected_sampling_metadata)

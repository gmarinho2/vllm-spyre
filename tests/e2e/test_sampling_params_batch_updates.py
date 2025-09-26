import torch
from vllm import SamplingParams

from vllm_spyre.v1.worker.spyre_input_batch import (SamplingInputBatch,
                                                    SamplingRequestState)

device = torch.device('cpu')
sampling_params = [
    SamplingParams(max_tokens=100),
    SamplingParams(min_tokens=10),
    SamplingParams(temperature=0.8),
    SamplingParams(presence_penalty=0.5),
    SamplingParams(frequency_penalty=0.5),
    SamplingParams(ignore_eos=True),
    SamplingParams(top_k=1),
    SamplingParams(top_p=0.5),
]

prompt_tokens_ids = [[94, 122, 40, 37, 11], [0, 82, 40, 37, 23],
                     [14, 240, 50, 37, 94], [1, 2, 40, 27, 193],
                     [104, 11, 24, 7, 143], [5, 12, 430, 4, 13],
                     [32, 111, 40, 37, 43], [0, 12, 340, 37, 3]]

ids = ["A", "B", "C", "D", "E", "F", "G", "H"]
requests = []

# build the requests
for params, tokens, req_id in zip(sampling_params, prompt_tokens_ids, ids):
    requests.append(
        SamplingRequestState(req_id=req_id,
                             sampling_params=params,
                             prompt_token_ids=tokens))

batch = SamplingInputBatch(max_model_len=128,
                           max_num_reqs=8,
                           vocab_size=100,
                           device=device,
                           pin_memory=False)

print(f"Dense indices (ocupados quando True): {batch.req_indices_mask}")
print(
    f"Nenhum request adicionado. Proximo vazio é: {batch.get_available_index()}"
)

batch.add_request(request=requests[0])
print(f"Adicionado req A. Proximo vazio: {batch.get_available_index()}")
print(f"Dense indices (ocupados quando True): {batch.req_indices_mask}")

batch.add_request(request=requests[1])
print(f"Adicionado req B. Proximo vazio: {batch.get_available_index()}")
print(f"Dense indices (ocupados quando True): {batch.req_indices_mask}")

batch.add_request(request=requests[2])
print(f"Adicionado req C. Proximo vazio: {batch.get_available_index()}")
print(f"Dense indices (ocupados quando True): {batch.req_indices_mask}")

batch.remove_request("B")
print(f"Removido req B. Proximo vazio: {batch.get_available_index()}")
print(f"Dense indices (ocupados quando True): {batch.req_indices_mask}")

print(batch.get_unpadded_output_indices)
batch.add_request(request=requests[1])
print(batch.get_unpadded_output_indices)

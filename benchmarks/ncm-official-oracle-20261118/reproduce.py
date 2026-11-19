"""Capture official TinyStories CPU inference through the upstream evaluation APIs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import pickletools
import subprocess
import sys

from omegaconf import OmegaConf
from safetensors.torch import save_file
import torch

UPSTREAM = "c70274273f489abc73832b2b98584a8483b6b203"
DATA_OPS = frozenset("""
    APPENDS BINGET BININT1 BININT2 EMPTY_DICT EMPTY_LIST FRAME LONG_BINGET
    MARK MEMOIZE PROTO SETITEMS SHORT_BINBYTES STOP TUPLE2
""".split())


class DataOnlyPickle(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("pickle globals are disabled")

    def persistent_load(self, pid):
        raise ValueError("pickle persistent objects are disabled")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(root, pin):
    for name, expected in pin["files"].items():
        path = root / name
        if path.stat().st_size != expected["size_bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"unexpected bytes for {name}")


def load_data(path):
    raw = path.read_bytes()
    ops = list(pickletools.genops(raw))
    if not ops or ops[-1][0].name != "STOP" or ops[-1][2] + 1 != len(raw):
        raise ValueError("invalid pickle boundary")
    if any(op.name not in DATA_OPS for op, _, _ in ops):
        raise ValueError("unexpected pickle opcode")
    return DataOnlyPickle(io.BytesIO(raw)).load()


def capture(net, codec, prefix, level0, mode, arrays, forced=None):
    active = 0
    samples = 0
    original_logits = net._decode_level_logits
    original_sample = net._sample_level
    original_input = codec.vq.get_next_autoregressive_input

    def put(name, tensor):
        arrays[name] = tensor.detach().to(dtype=torch.float32).contiguous().clone()

    def next_input(index, *args, **kwargs):
        result = original_input(index, *args, **kwargs)
        if mode == "teacher":
            put(f"teacher.pool.{index + 1}", result[1])
        return result

    def logits(x, prefix_embedding, level_idx, *args, **kwargs):
        nonlocal active
        active = level_idx + 1
        if mode == "teacher" and active == 1:
            put("prefix.embedding", prefix_embedding)
        try:
            output = original_logits(x, prefix_embedding, level_idx, *args, **kwargs)
            put(f"{mode}.logits.{active}", output)
            return output
        finally:
            active = 0

    def sample(logits, *args, **kwargs):
        nonlocal samples
        samples += 1
        if forced is not None:
            return forced[samples]
        return original_sample(logits, *args, **kwargs)

    def watch_block(index):
        def hook(_module, args, output):
            if mode == "teacher" and active == 1:
                if index == 0:
                    put("first.input", args[0])
                put(f"first.block.{index}", output)
            if mode == "teacher" and active == 3 and index == 0:
                put("medium.input", args[0])
                put("medium.block.0", output)
        return hook

    hooks = [block.register_forward_hook(watch_block(i))
             for i, block in enumerate(net.transformer.blocks)]
    net._decode_level_logits = logits
    net._sample_level = sample
    codec.vq.get_next_autoregressive_input = next_input
    try:
        codes = net.generate(level0, prefix, temperature=0.0, top_k=0, top_p=1.0,
                             use_kv_cache=False, cfg_scale=0.0)
    finally:
        net._decode_level_logits = original_logits
        net._sample_level = original_sample
        codec.vq.get_next_autoregressive_input = original_input
        for hook in hooks:
            hook.remove()
    if samples != len(codec.vq.levels) - 1:
        raise ValueError("upstream did not predict every concept level")
    chunks = list(codes.split(tuple(codec.vq.levels), dim=1))
    if forced is not None and any(not torch.equal(a, b) for a, b in zip(chunks, forced, strict=True)):
        raise ValueError("upstream did not consume the forced codes")
    if mode == "greedy":
        for level, chunk in enumerate(chunks):
            put(f"greedy.codes.{level}", chunk)
    _, latents = codec.vq.idx_to_zhats(codes, return_zlevels=True)
    decoder = codec.decoder(latents[-1])
    tokens = decoder.argmax(-1)
    put(f"{mode}.decoder", decoder)
    put(f"{mode}.tokens", tokens)
    return chunks, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for param in ("upstream", "official_files", "official_lock", "mlx_files", "mlx_lock", "output"):
        parser.add_argument(param, type=Path)
    parser.add_argument("forced_trace", type=Path, nargs="*")
    args = parser.parse_args()

    git = lambda *cmd: subprocess.check_output(["git", "-C", args.upstream, *cmd], text=True).strip()
    if git("rev-parse", "HEAD") != UPSTREAM or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("unexpected upstream source revision or changes")
    official_lock = json.loads(args.official_lock.read_text())
    mlx_lock = json.loads(args.mlx_lock.read_text())
    verified(args.official_files, official_lock)
    verified(args.mlx_files, mlx_lock)
    checkpoint = args.official_files / "tinystories"
    if (checkpoint / "config.yaml").read_bytes() != (args.upstream / "config/eval/tinystories.yaml").read_bytes():
        raise ValueError("upstream source and checkpoint configs differ")

    sys.path.insert(0, str(args.upstream.resolve()))
    from models.next_concept import NextConceptTransformer
    from models.vqvae import VQVAE
    from tokenization.bpe_tokenizer import BPETokenizer

    config = OmegaConf.load(checkpoint / "config.yaml")
    vocab = load_data(checkpoint / "bpe_vocab_8192.pickle")
    merges = load_data(checkpoint / "bpe_merges_8192.pickle")
    bpe = json.loads((args.mlx_files / "bpe.json").read_text())
    if type(vocab) is not dict or set(vocab) != set(range(8192)):
        raise ValueError("unexpected upstream vocabulary shape")
    if [vocab[i] for i in range(8192)] != [bytes.fromhex(s) for s in bpe["vocab"]]:
        raise ValueError("upstream vocabulary differs from converted bytes")
    if list(merges) != [tuple(bytes.fromhex(s) for s in pair) for pair in bpe["merges"]]:
        raise ValueError("upstream merge ranks differ")
    tok = BPETokenizer(vocab, merges, config.tokenizer.pre_tokenizer.special_tokens)
    if [tok.eos_token_id, tok.pad_token_id] != [0, 1]:
        raise ValueError("unexpected upstream special IDs")

    torch.manual_seed(0)
    torch.set_num_threads(6)
    codec = VQVAE(**config.tokenizer.vqvae)
    codec.load_state_dict(torch.load(checkpoint / "tokenizer.pt", weights_only=True, map_location="cpu")["model"], strict=True)
    codec.eval()
    net = NextConceptTransformer(vqvae_local=codec, config=config.model)
    sd = torch.load(checkpoint / "generator.pt", weights_only=True, map_location="cpu")
    buffers = {
        name: torch.equal(getattr(net, name), sd["model"][name])
        for name in ("lvl_1L", "attn_mask_for_masking")
    }
    if not all(buffers.values()):
        raise ValueError(f"published generator buffers differ from upstream construction: {buffers}")
    net.load_state_dict(sd["model"], strict=True)
    net.eval()

    def decode(raw):
        visible = raw[:raw.index(0)] if 0 in raw else raw
        return tok.decode([v for v in visible if v != 1], skip_special_tokens=True)

    args.output.mkdir(parents=True, exist_ok=False)
    cases = {}
    default_l0 = sd["inference"]["level0_unigram"].argmax().reshape(1, 1)
    prefix = torch.tensor([[tok.eos_token_id]])
    with torch.inference_mode():
        for label, forced in [("unigram-greedy", None), *(
            (trace.stem, [torch.tensor(row, dtype=torch.long) for row in json.loads(trace.read_text())["codes"]])
            for trace in args.forced_trace
        )]:
            arrays = {}
            level0 = default_l0 if forced is None else forced[0]
            greedy_codes, greedy_tokens = capture(net, codec, prefix, level0, "greedy", arrays)
            teacher_codes, teacher_tokens = capture(net, codec, prefix, level0, "teacher", arrays,
                                                   greedy_codes if forced is None else forced)
            if len(arrays) != 65:
                raise ValueError(f"incomplete oracle tensor set: {len(arrays)}")
            trace = {"prefix_tokens": prefix.tolist(), "codes": [codes.tolist() for codes in teacher_codes],
                     "tokens": teacher_tokens.tolist(), "text": decode(teacher_tokens[0].tolist())}
            trace_path = args.output / f"{label}.json"
            trace_path.write_text(json.dumps(trace, indent=2) + "\n")
            arrays["trace.sha256"] = torch.tensor(list(bytes.fromhex(digest(trace_path))), dtype=torch.float32)
            references = args.output / f"{label}.safetensors"
            save_file(arrays, references)
            cases[label] = {"level0": int(level0.item()), "reference_sha256": digest(references),
                            "trace_sha256": digest(trace_path),
                            "greedy_tokens": greedy_tokens.tolist(),
                            "greedy_text": decode(greedy_tokens[0].tolist())}

    receipt = {"upstream_revision": UPSTREAM, "upstream_pack": official_lock["revision"],
               "mlx_pack": mlx_lock["revision"], "torch": torch.__version__, "dtype": "F32",
               "device": "CPU", "torch_threads": 6, "cache": False, "strict_weights_only": True,
               "vocabulary_matches": len(vocab), "merge_ranks_match": len(merges),
               "regenerated_buffers_match": buffers, "cases": cases}
    (args.output / "oracle-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Upstream oracle captured:", ", ".join(cases), flush=True)


if __name__ == "__main__":
    main()

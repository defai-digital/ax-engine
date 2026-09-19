"""Bounded M5-only co-residency probe; not an OOM or endurance test."""
import argparse
import gc
import hashlib
import json
import mmap
import os
from pathlib import Path
import queue
import random
import re
import select
import subprocess
import sys
import threading
import time

GIB = 1024**3


def snapshot():
    raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    page_size = int(re.search(r"page size of (\d+)", raw)[1])
    pages = {k.strip('"'): int(v) for k, v in re.findall(r'^([^:\n]+):\s+(\d+)', raw, re.M)}
    swap = subprocess.check_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True)
    amount, unit = re.search(r"used = ([\d.]+)([MG])", swap).groups()
    return dict(time=time.monotonic(),
        available=(pages["Pages free"] + pages["Pages inactive"] + pages["Pages speculative"]) * page_size,
        compressed=pages["Pages occupied by compressor"] * page_size,
        swap=int(float(amount) * (1024**2 if unit == "M" else GIB)),
        pressure=int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True)),
        pages=pages,
        rss=int(subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())], text=True)) * 1024)


def guard(current, baseline, reserve=0):
    if current["available"] - reserve < 16 * GIB:
        return "headroom"
    if current["swap"] - baseline["swap"] > 128 * 1024**2:
        return "swap_growth"
    if current["compressed"] - baseline["compressed"] > GIB:
        return "compressor_growth"
    if current["pressure"] >= 4:
        return "critical_pressure"
    return None


def pressure_completed(record, last_request_end):
    released = record.get("released", {})
    return (record.get("exit_code") == 0 and released.get("reason") == "released"
            and last_request_end <= released.get("release_started", 0))


def pressure_worker(gib):
    assert 0 <= gib <= 48, "bounded worker accepts at most 48 GiB"
    baseline = snapshot()
    chunks = []
    samples = [baseline]
    deadline = time.monotonic() + 180
    block = random.Random(731).randbytes(1024**2)
    reason = "released"
    try:
        for _ in range(gib):
            if select.select([sys.stdin], [], [], 0)[0]:
                raise RuntimeError("parent_closed_or_release")
            current = snapshot()
            samples.append(current)
            reason = guard(current, baseline, GIB)
            if reason or time.monotonic() > deadline:
                raise RuntimeError(reason or "deadline")
            chunk = mmap.mmap(-1, GIB)
            chunks.append(chunk)
            for offset in range(0, GIB, len(block)):
                chunk[offset:offset + len(block)] = block
        committed = snapshot()
        assert committed["rss"] >= gib * GIB * 0.9, "allocation is not resident"
        reason = guard(committed, baseline)
        if reason:
            raise RuntimeError(reason)
        print(json.dumps(dict(event="ready", gib=gib, snapshot=committed)), flush=True)
        while True:
            if select.select([sys.stdin], [], [], 0.25)[0]:
                sys.stdin.read(1)
                reason = "released"
                break
            current = snapshot()
            samples.append(current)
            reason = guard(current, baseline)
            if reason or time.monotonic() > deadline:
                reason = reason or "deadline"
                break
    except BaseException as error:
        reason = str(error)
    finally:
        release_started = time.monotonic()
        for chunk in chunks:
            chunk.close()
        print(json.dumps(dict(event="released", reason=reason, samples=samples,
                              release_started=release_started, after=snapshot())), flush=True)


class Worker:
    def __init__(self, name, model, tokens):
        self.name, self.model, self.tokens = name, model, tokens
        self.inbox, self.outbox = queue.Queue(), queue.Queue()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        self.receive()

    def receive(self):
        result = self.outbox.get(timeout=180)
        if "error" in result:
            raise RuntimeError(result)
        return result

    def generate(self, session, barrier=None):
        if barrier:
            barrier.wait(timeout=30)
        start = time.monotonic()
        events, tokens, response = [], [], None
        for event in session.stream_generate(input_tokens=self.tokens, max_output_tokens=128,
                temperature=0, top_p=1, top_k=0, seed=0, ignore_eos=True):
            if event["event"] == "step" and event.get("delta_tokens"):
                batch = event["delta_tokens"]
                events.append((time.monotonic(), len(batch)))
                tokens.extend(batch)
                assert len(tokens) <= 128
            if event["event"] == "response":
                assert response is None
                response = event["response"]
        end = time.monotonic()
        assert response and response["output_tokens"] == tokens and len(tokens) == 128
        assert response["finish_reason"] == "max_output_tokens"
        first, first_n = events[0]
        last = events[-1][0]
        counts = response["route"]["crossover_decisions"]
        return dict(model=self.name, started=start, ended=end, tokens=tokens,
            emissions=events, finish_reason=response["finish_reason"],
            ttft_s=first-start, completion_tok_s=128/(last-start),
            decode_tok_s=(128-first_n)/(last-first),
            mtp={k: v for k, v in counts.items() if k.startswith("ax_mtp_source_mtp_")
                 or k in ("ax_mtp_requested", "ax_mtp_drafted_depth2")})

    def run(self):
        import _ax_engine
        session = None
        try:
            session = _ax_engine.Session(model_id=self.name, mlx=True,
                                         mlx_model_artifacts_dir=self.model)
            self.outbox.put({"ready": True})
            while True:
                command, barrier = self.inbox.get()
                if command == "close":
                    session.close()
                    session = None
                    gc.collect()
                    self.outbox.put({"closed": True})
                    return
                if command == "cancel":
                    iterator = session.stream_generate(input_tokens=self.tokens, max_output_tokens=128,
                        temperature=0, top_p=1, top_k=0, seed=0, ignore_eos=True)
                    for event in iterator:
                        if event["event"] == "step" and event.get("delta_tokens"):
                            break
                    del iterator
                    gc.collect()
                self.outbox.put(self.generate(session, barrier))
        except BaseException as error:
            self.outbox.put({"error": repr(error), "model": self.name})
        finally:
            if session is not None:
                session.close()

    def close(self):
        if self.thread.is_alive():
            self.inbox.put(("close", None))
            self.receive()
        self.thread.join(timeout=30)
        assert not self.thread.is_alive()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--order", choices=["tiel", "cyber"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pressure", action="store_true")
    args = parser.parse_args()
    chip = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    memory = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True))
    assert chip == "Apple M5 Max" and memory == 128 * GIB
    os.environ.update({"AX_MLX_NATIVE_CONFIRM": "1", "AX_MLX_MTP_FORCE_REQUESTED": "1",
        "AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP": "1", "AX_MLX_MTP_CONSERVATIVE_DEPTH": "0",
        "AX_ENGINE_PREFIX_REUSE_DISABLED": "1", "AX_MLX_PREFIX_CACHE_MAX_BYTES": "0",
        "AX_MLX_PREFIX_CACHE_MAX_ENTRIES": "0", "AX_MLX_PREFIX_CACHE_DISK_DISABLED": "1"})
    import _ax_engine
    import mlx.core as mx
    doc = dict(hardware=dict(chip=chip, memory_bytes=memory), order=args.order,
        extension_sha256=hashlib.sha256(Path(_ax_engine.__file__).read_bytes()).hexdigest(),
        phases=[], pressure=[], inputs={}, started=time.time())
    output = Path(args.output)
    workers = []
    golden = {}

    def save():
        output.write_text(json.dumps(doc, indent=2) + "\n")

    def state():
        # No getter exists for wired residency. At quiescent boundaries this
        # is an idempotent assertion of the expected zero; restore on failure.
        wired = mx.set_wired_limit(0)
        if wired:
            mx.set_wired_limit(wired)
            raise AssertionError(f"unexpected process wired limit: {wired}")
        return dict(vm=snapshot(), wired_limit=wired,
                    memory_limit=None, memory_limit_note="no getter in admitted MLX wheel",
                    active=mx.get_active_memory(), cache=mx.get_cache_memory(), peak=mx.get_peak_memory())

    def phase(name, concurrent=False, repeats=3, command="generate", pressure_child=None):
        entry = dict(name=name, before=state(), rows=[])
        doc["phases"].append(entry)
        save()
        for rep in range(repeats):
            barrier = threading.Barrier(len(workers)) if concurrent else None
            if concurrent:
                time.sleep(3)
                if pressure_child is not None and pressure_child.poll() is not None:
                    entry["pressure_ended_early"] = True
                    break
                for worker in workers:
                    worker.inbox.put((command, barrier))
                rows = [worker.receive() for worker in workers]
            else:
                rows = []
                for worker in workers:
                    time.sleep(3)
                    worker.inbox.put((command, None))
                    rows.append(worker.receive())
            for row in rows:
                row["rep"] = rep
                entry["rows"].append(row)
                assert row["tokens"] == golden.setdefault(row["model"], row["tokens"]), "token divergence"
            if concurrent and len(rows) == 2:
                assert max(r["started"] for r in rows) < min(r["ended"] for r in rows)
            save()
            if pressure_child is not None and pressure_child.poll() is not None:
                entry["pressure_ended_early"] = True
                break
        entry["after"] = state()
        save()
        print("STOP" if entry.get("pressure_ended_early") else "PASS", name, flush=True)
        return entry

    try:
        order = [args.order, "cyber" if args.order == "tiel" else "tiel"]
        for index, model in enumerate(order):
            tokens = json.loads((Path(args.cases)/(model+"-cases.json")).read_text())[0]["token_ids"]
            doc["inputs"][model] = tokens
            pack = "AX-" + ("Cyber-" if model == "cyber" else "") + "Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP"
            workers.append(Worker(model, str(Path(args.models)/pack), tokens))
            phase("isolated-warmup" if index == 0 else "co-resident-warmup", repeats=2)
            if index == 0:
                phase("isolated")
        phase("sequential")
        phase("concurrent", concurrent=True)
        if args.pressure:
            for gib in (16, 32, 48):
                # A regular file prevents a large final telemetry record from
                # keeping an already-released child alive on a full stdout pipe.
                pressure_log = output.with_name(output.stem + f"-pressure-{gib}.jsonl")
                log = pressure_log.open("w")
                child = subprocess.Popen([sys.executable, __file__, "--pressure-worker", str(gib)],
                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.PIPE, text=True)
                record = {"requested_gib": gib}
                doc["pressure"].append(record)
                entry = None
                try:
                    deadline = time.monotonic() + 90
                    while True:
                        with pressure_log.open() as reader:
                            first = reader.readline()
                        if first.endswith("\n"):
                            record["ready"] = json.loads(first)
                            break
                        if child.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError("pressure worker did not report readiness")
                        time.sleep(0.05)
                    if record["ready"]["event"] != "ready":
                        break
                    entry = phase(f"competing-{gib}gib", concurrent=True, pressure_child=child)
                finally:
                    try:
                        _, errors = child.communicate(input="x", timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        _, errors = child.communicate()
                    log.close()
                    events = [json.loads(line) for line in pressure_log.read_text().splitlines()]
                    record.update(exit_code=child.returncode, released=events[-1] if events else {}, stderr=errors)
                    last_end = max((r["ended"] for r in entry["rows"]), default=0) if entry else 0
                    record["held_through_requests"] = bool(entry and len(entry["rows"]) == 6) and pressure_completed(record, last_end)
                    save()
                phase(f"released-{gib}gib", concurrent=True, repeats=1)
                if not record["held_through_requests"]:
                    break
        phase("cancel-and-reuse", command="cancel", repeats=1)
        workers[-1].close()
        workers.pop()
        phase("survivor")
        doc["complete"] = True
    finally:
        for worker in workers:
            worker.close()
        doc["ended"] = time.time()
        save()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--pressure-worker":
        pressure_worker(int(sys.argv[2]))
    else:
        main()

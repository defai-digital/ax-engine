"""Verify stable HTTP output and early Flash Next MTP rejection receipts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    direct = load("direct-and-required.json")
    build = load("preflight-build.json")
    attached, bypassed, rejected = (direct[name] for name in
                                    ("attached", "bypassed", "required_bypassed"))
    require(attached["depth"] == 1 and attached["route_safe"]
            and bypassed["depth"] == rejected["depth"] == 0
            and not bypassed["route_safe"] and not rejected["route_safe"]
            and all(not arm["mtp_requested"] for arm in (attached, bypassed, rejected)),
            "Flash Next drafter admission does not reflect attachment")
    require(len(attached["responses"]) == 7 and attached["responses"] == bypassed["responses"],
            "Stable direct HTTP response fields differ")
    for row in attached["responses"]:
        require(row["activation_delta"]["ax_engine_flash_next_mtp_verified_steps_total"] == 0,
                "Unexpected MTP verification in direct requests")

    before = (ROOT / "required-before.log").read_text()
    require(rejected["exit_code"] == 1 and rejected["elapsed_seconds"] > 0
            and 'mtp_depth=0' in before and 'mtp_requested=false' in before
            and 'MlxMtpRequiredButUnavailable' in before,
            "Required did not reject the bypassed drafter in the runtime check")
    require(direct["build"]["commit"] != build["commit"], "Preflight and baseline build are identical")
    require(build["profile"] == direct["build"]["profile"] == "release-server"
            and build["mlx"] == direct["build"]["mlx"] == "0.32.4",
            "Build setup changed")
    for name, switch in (("skip-preflight", "AX_MLX_SKIP_MTP_SIDECAR"), ("kill-switch", "AX_NO_SPEC")):
        result = load(name + ".json")
        output = (ROOT / (name + ".log")).read_text()
        require(result["pack_revision"] == direct["pack_revision"]
                and result["manifest_sha256"] == direct["manifest_sha256"]
                and result["inventory_sha256"] == direct["inventory_sha256"]
                and result["server_sha256"] == build["server_sha256"],
                "Preflight build or artifact identity differs")
        require(result["environment"] == {switch: "1"} and result["policy"] == "required"
                and result["exit_code"] == 1 and "MlxMtpRequiredButUnavailable" in output
                and result["inventory_verified_members"] == 49
                and result["process_elapsed_seconds"] > 0,
                "Preflight did not reject the explicit policy conflict")
        markers = result["payload_startup_markers"]
        require(len(markers) == 5 and all(count == output.count(marker) == 0
                                         for marker, count in markers.items()),
                "Native payload startup ran before preflight rejection")
    print("Seven stable direct responses and both Required MTP preflight cases passed.")


if __name__ == "__main__":
    main()

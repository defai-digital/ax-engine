# Flash Next `38c90b27a` native release board on M5 Max

The MacBook Pro M5 Max 128 GB used internal NVMe storage and macOS 27.2.1.
The [fresh delivery](../flash-next-fresh-delivery-38c90b27a-m5-20261122/README.md),
[native QA](../flash-next-native-qa-38c90b27a-m5-20261122/README.md),
[lifecycle](../flash-next-native-lifecycle-38c90b27a-m5-20261122/README.md)
and matrix all use one release-server binary and publisher inventory.

Both direct modes match 31 of 31 output arrays within their two
independent processes. All 124 arrays match the signed native output
oracle. The matrix records 99.53% CPU idle before and 96.12% after.
The two selected matrix processes reach HTTP readiness in 119.798 and
118.950 seconds. The staged native-support repeat passes all 21
checks. The explicit Required MTP and sidecar bypass conflict fails
before the observed payload startup phases in 0.231 seconds.

Five of the nine frozen gates pass. Delivered Required and default
readiness take 126.269 and 125.260 seconds, and staged default lifecycle
takes 124.249 seconds. These observations miss the 120-second gate.
The ordinary independent reference remains incomplete. The integrity,
independent quality and reference-relative timing gates remain open.

The [archive](evidence.tar.xz) contains all native matrix repeats and
the terminal native receipt. Replay binds each input receipt, validates
the signed token trajectories and evaluates the original release gates.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```

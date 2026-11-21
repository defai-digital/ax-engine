# Flash Next lazy CPU loader correctness controls

The [existing probe](../flash-next-weight-loader-lab-20261121/README.md)
compared the name-ordered and offset-ordered mmap readers and MLX's lazy
CPU reader on an M2 Ultra 192 GB Mac Studio with MLX 0.32.4.

The pinned MXFP4 and MXFP8 snapshots each supplied one real tensor per
stored dtype. The probe checked the selected shapes, dtypes and byte
contents for BF16, I64, U32 and U8. Every digest matched across the three
paths. The selections totalled 6,520 bytes for MXFP4 and 2,760 for MXFP8.
This covers stored dtypes before the full-weight and token comparisons.

An independent sparse fixture put a 512 MiB expert-shaped tensor more
than 16 MiB from a selected 16 KiB tensor. Each reader loaded the same
selected bytes. `mincore` showed zero expert pages resident before and
after every load. A separate positive control evaluated a 16 MiB backed
sentinel with the lazy reader and moved its resident-page count from
0 to 1,025. The control uses fresh APFS clones on the internal SSD.

[Rerun the controls](verify_lazy_selection.py) with the probe binary,
the two verified model roots and an output path. Put the loader lab
directory on `PYTHONPATH` to share its APFS and cache-state helpers.
[Raw results and bindings](result.json) include the exact file and
selected tensor names, all digests and the page observations.

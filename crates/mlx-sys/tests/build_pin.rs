#[path = "../build_pin.rs"]
mod build_pin;

use build_pin::{parse_mlx_pin, read_mlx_pin};

#[test]
fn accepts_identified_wheel_and_source_pins() {
    let wheel = parse_mlx_pin(" 0.32.2\n").unwrap();
    assert_eq!(wheel.expected_header_version(), "0.32.2");
    assert_eq!(wheel.describe(), "0.32.2");
    assert!(wheel.install_hint().contains("mlx==0.32.2"));
    let source = parse_mlx_pin("git:abcdef1234567@0.32.2").unwrap();
    assert_eq!(source.expected_header_version(), "0.32.2");
    assert!(source.describe().contains("abcdef1234567"));
    assert!(source.install_hint().contains("abcdef1234567"));
}

#[test]
fn rejects_empty_malformed_or_unidentified_pins() {
    for raw in [
        "",
        "\n",
        "latest",
        "0.32",
        "0.32.2\n0.33.0",
        "git:@0.32.2",
        "git:abcdef1234567@",
        "git:main@0.32.2",
        "git:abcdefg@0.32.2",
        "0.32.4294967296",
    ] {
        assert!(parse_mlx_pin(raw).is_none(), "accepted invalid pin {raw:?}");
    }
}

#[test]
fn requires_pin_file_instead_of_disabling_version_enforcement() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("ax-mlx-pin-{}-{unique}", std::process::id()));
    assert!(read_mlx_pin(&path).is_err());
    std::fs::write(&path, "\n").unwrap();
    assert!(read_mlx_pin(&path).is_err());
    std::fs::write(&path, "0.32.2\n").unwrap();
    assert_eq!(
        read_mlx_pin(&path).unwrap().expected_header_version(),
        "0.32.2"
    );
    std::fs::remove_file(path).unwrap();
}

const FULL_COMMIT: &str = "56c2bf9236d929097f61e2c6e55be4e4bb6da249";

const WHEEL_HEADER: &str = r#"
#define MLX_VERSION_MAJOR 0
#define MLX_VERSION_MINOR 32
#define MLX_VERSION_PATCH 4
/* For dev builds, the version will include the suffix ".devYYYYMMDD+hash" */
"#;

fn dev_header() -> String {
    format!(
        r#"
#define MLX_VERSION_MAJOR 0
#define MLX_VERSION_MINOR 32
#define MLX_VERSION_PATCH 5
/* ax-engine experiment provenance: "0.32.5.dev20261024+{FULL_COMMIT}" */
"#
    )
}

#[test]
fn pinned_path_unchanged() {
    let numeric = build_pin::header_numeric_version(WHEEL_HEADER).unwrap();
    assert_eq!(numeric, "0.32.4");
    assert_eq!(
        build_pin::classify_dev_header(WHEEL_HEADER.as_bytes()),
        build_pin::DevHeader::Absent
    );
    let stray = "0123456789abcdef0123456789abcdef01234567";
    for experiment in [None, Some(stray), Some(FULL_COMMIT)] {
        assert_eq!(
            build_pin::mlx_link_gate("0.32.4", &numeric, None, Some(b"not-read"), experiment),
            Ok(build_pin::MlxLinkGate::PinnedRelease),
            "experiment={experiment:?}"
        );
    }
    // A different release build stays on the historical override path.
    assert_eq!(
        build_pin::mlx_link_gate("0.32.4", "0.32.3", None, None, Some(FULL_COMMIT)),
        Ok(build_pin::MlxLinkGate::NumericMismatch)
    );
}

#[test]
fn dev_pin_requires_exact_commit() {
    let header = dev_header();
    let numeric = build_pin::header_numeric_version(&header).unwrap();
    assert_eq!(numeric, "0.32.5");
    let dev = match build_pin::classify_dev_header(header.as_bytes()) {
        build_pin::DevHeader::One(dev) => dev,
        other => panic!("expected one dev token, got {other:?}"),
    };
    assert_eq!(dev.commit, FULL_COMMIT);
    assert_eq!(dev.token(), format!("0.32.5.dev20261024+{FULL_COMMIT}"));

    let library = format!("build-id\0{}\0", dev.token());
    let library_bytes = library.as_bytes();
    assert!(build_pin::library_contains_dev_version(library_bytes, &dev));

    let admitted = build_pin::mlx_link_gate(
        "0.32.4",
        &numeric,
        Some(&dev),
        Some(library_bytes),
        Some(FULL_COMMIT),
    )
    .unwrap();
    assert_eq!(admitted, build_pin::MlxLinkGate::Experiment);

    let uppercase = FULL_COMMIT.to_ascii_uppercase();
    assert_eq!(
        build_pin::mlx_link_gate(
            "0.32.4",
            &numeric,
            Some(&dev),
            Some(library_bytes),
            Some(uppercase.as_str()),
        )
        .unwrap(),
        build_pin::MlxLinkGate::Experiment
    );

    for experiment in [
        None,
        Some(""),
        Some("56c2bf9236d929097f61e2c6e55be4e4bb6da24"),
        Some("56c2bf9236d929097f61e2c6e55be4e4bb6da2490"),
        Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        Some("56c2bf9"),
    ] {
        assert!(
            build_pin::mlx_link_gate(
                "0.32.4",
                &numeric,
                Some(&dev),
                Some(library_bytes),
                experiment
            )
            .is_err(),
            "accepted experiment={experiment:?}"
        );
    }
    assert!(
        build_pin::mlx_link_gate(
            "0.32.4",
            &numeric,
            Some(&dev),
            Some(b"no provenance"),
            Some(FULL_COMMIT)
        )
        .is_err()
    );
    assert!(
        build_pin::mlx_link_gate("0.32.4", &numeric, Some(&dev), None, Some(FULL_COMMIT)).is_err()
    );
}

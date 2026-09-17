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

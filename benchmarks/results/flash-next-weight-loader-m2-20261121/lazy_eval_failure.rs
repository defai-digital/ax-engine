use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

fn evaluate_selected(first: &Path, second: &Path, truncate: bool) -> Result<(), String> {
    let first = mlx_sys::load_safetensors(first, None)?;
    mlx_sys::try_eval(&[&first["selected"]])?;
    println!("first shard evaluated");

    let second_arrays = mlx_sys::load_safetensors(second, None)?;
    if truncate {
        let source = OpenOptions::new()
            .write(true)
            .open(second)
            .map_err(|error| format!("open fixture: {error}"))?;
        source
            .set_len(4096)
            .and_then(|()| source.sync_all())
            .map_err(|error| format!("truncate fixture: {error}"))?;
    }
    mlx_sys::try_eval(&[&second_arrays["selected"]])?;
    println!("both shards evaluated");
    Ok(())
}

fn write_fixture(path: &Path) -> std::io::Result<()> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    let mut header =
        br#"{"selected":{"dtype":"I32","shape":[262144],"data_offsets":[0,1048576]}}"#.to_vec();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&vec![0; 1048576])?;
    file.sync_all()
}

fn main() -> Result<(), String> {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("missing new fixture directory")?;
    fs::create_dir(&root).map_err(|error| format!("create fixture directory: {error}"))?;
    let first = root.join("initial.safetensors");
    let damaged = root.join("fault.safetensors");
    let intact = root.join("positive.safetensors");
    for path in [&first, &damaged, &intact] {
        write_fixture(path).map_err(|error| format!("write fixture: {error}"))?;
    }

    let failure = evaluate_selected(&first, &damaged, true);
    match failure {
        Err(message) if message.contains("[read] Unable to read from file.") => {
            println!("read error propagated through mlx_sys::try_eval: {message}");
        }
        outcome => return Err(format!("unexpected fault outcome: {outcome:?}")),
    }
    evaluate_selected(&first, &intact, false)?;
    println!("intact retry evaluated after failure in the same process");
    Ok(())
}

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const MIN_SLANG_YEAR: u32 = 2025;
const MIN_SLANG_RELEASE: u32 = 13;

fn main() {
    println!("cargo:rerun-if-changed=shaders/none_drt.slang");
    println!("cargo:rerun-if-changed=shaders/oklab_drt.slang");
    println!("cargo:rerun-if-changed=shaders/agx_s2o3.slang");
    println!("cargo:rerun-if-changed=shaders/agx_hsv.slang");
    println!("cargo:rerun-if-changed=shaders/reinhard_gamut.slang");
    println!("cargo:rerun-if-env-changed=SLANGC");

    let slangc = find_slangc();
    check_version(&slangc);

    let output_directory = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    for (source, output) in [
        ("shaders/none_drt.slang", "none_drt.spv"),
        ("shaders/oklab_drt.slang", "oklab_drt.spv"),
        ("shaders/agx_s2o3.slang", "agx_s2o3.spv"),
        ("shaders/agx_hsv.slang", "agx_hsv.spv"),
        ("shaders/reinhard_gamut.slang", "reinhard_gamut.spv"),
    ] {
        compile(&slangc, Path::new(source), &output_directory.join(output));
    }
}

fn find_slangc() -> PathBuf {
    if let Some(path) = env::var_os("SLANGC") {
        return PathBuf::from(path);
    }

    let executable = if cfg!(windows) {
        "slangc.exe"
    } else {
        "slangc"
    };
    if let Some(path) = find_on_path(executable) {
        return path;
    }

    if let Some(sdk) = env::var_os("VULKAN_SDK") {
        let bin = if cfg!(windows) { "Bin" } else { "bin" };
        let candidate = PathBuf::from(sdk).join(bin).join(executable);
        if candidate.is_file() {
            return candidate;
        }
    }

    panic!(
        "Slang compiler not found. Install Slang v2025.13+ or set SLANGC to the slangc executable"
    );
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    env::split_paths(&env::var_os("PATH")?).find_map(|directory| {
        let candidate = directory.join(name);
        candidate.is_file().then_some(candidate)
    })
}

fn check_version(slangc: &Path) {
    let output = Command::new(slangc)
        .arg("-version")
        .output()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", slangc.display()));
    if !output.status.success() {
        panic!("{} -version failed", slangc.display());
    }

    let version = if output.stdout.is_empty() {
        String::from_utf8_lossy(&output.stderr)
    } else {
        String::from_utf8_lossy(&output.stdout)
    };
    let numbers: Vec<u32> = version
        .trim()
        .split(|character: char| !character.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse().ok())
        .collect();
    let supported = matches!(numbers.as_slice(), [year, release, ..]
        if (*year, *release) >= (MIN_SLANG_YEAR, MIN_SLANG_RELEASE));
    if !supported {
        panic!(
            "Slang v2025.13+ is required, but {} reported {:?}",
            slangc.display(),
            version.trim()
        );
    }
    println!("cargo:warning=using Slang {}", version.trim());
}

fn compile(slangc: &Path, source: &Path, output: &Path) {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).expect("failed to create shader output directory");
    }

    let result = Command::new(slangc)
        .args([
            source.as_os_str(),
            "-entry".as_ref(),
            "main".as_ref(),
            "-stage".as_ref(),
            "compute".as_ref(),
            "-target".as_ref(),
            "spirv".as_ref(),
            "-profile".as_ref(),
            "glsl_460".as_ref(),
            "-capability".as_ref(),
            "SPIRV_1_3".as_ref(),
            "-matrix-layout-row-major".as_ref(),
            "-O2".as_ref(),
            "-o".as_ref(),
            output.as_os_str(),
        ])
        .output()
        .unwrap_or_else(|error| panic!("failed to launch {}: {error}", slangc.display()));

    if !result.status.success() {
        let stdout = String::from_utf8_lossy(&result.stdout);
        let stderr = String::from_utf8_lossy(&result.stderr);
        panic!("Slang compilation failed:\n{stdout}{stderr}");
    }
}

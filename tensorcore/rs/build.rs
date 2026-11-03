// Compiles the tensor-core kernel (cuda/tch.cu) with nvcc against CUTLASS.
//   CUDA_HOME    default /usr/local/cuda
//   CUTLASS_DIR  default ../third_party/cutlass  (git clone --depth 1 https://github.com/NVIDIA/cutlass.git)
//   CUDA_ARCH    default sm_89 (1-bit mma needs sm_80+)
use std::{env, path::PathBuf, process::Command};

fn main() {
    for v in ["CUDA_HOME", "CUTLASS_DIR", "CUDA_ARCH"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    println!("cargo:rerun-if-changed=cuda/tch.cu");
    let cuda = PathBuf::from(env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into()));
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cutlass = env::var("CUTLASS_DIR").map(PathBuf::from).unwrap_or_else(|_| manifest.join("../third_party/cutlass"));
    assert!(cutlass.join("include/cutlass/cutlass.h").exists(), "CUTLASS not found at {} (set CUTLASS_DIR)", cutlass.display());
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".into());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj = out.join("tch.o");
    let st = Command::new(cuda.join("bin/nvcc"))
        .args(["-O3", "-std=c++17", "-c", "-Xcompiler", "-fPIC", &format!("-arch={arch}")])
        .arg(format!("-I{}", cutlass.join("include").display()))
        .arg(format!("-I{}", cutlass.join("tools/util/include").display()))
        .arg("cuda/tch.cu").arg("-o").arg(&obj)
        .status().expect("nvcc not found (set CUDA_HOME)");
    assert!(st.success(), "nvcc failed");
    assert!(Command::new("ar").arg("crs").arg(out.join("libtch.a")).arg(&obj).status().unwrap().success());
    let lib = cuda.join("lib64");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=tch");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}

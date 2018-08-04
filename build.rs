// build.rs

extern crate cmake;
extern crate pkg_config;
#[cfg(unix)]
#[cfg(feature = "decode_test")]
extern crate bindgen;

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    if cfg!(windows) && cfg!(feature = "decode_test") {
        panic!("Unsupported feature on this platform!");
    }

    let cargo_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let build_path = Path::new(&cargo_dir).join("aom_build/aom");
    let debug = if let Some(v) = env::var("PROFILE").ok() {
        match v.as_str() {
            "bench" | "release" => false,
            _ => true,
        }
    } else {
        false
    };

    let dst = cmake::Config::new(build_path)
        .define("CONFIG_DEBUG", (debug as u8).to_string())
        .define("CONFIG_ANALYZER", "0")
        .define("ENABLE_DOCS", "0")
        .define("ENABLE_TESTS", "0")
        .no_build_target(cfg!(windows))
        .build();

    // Dirty hack to force a rebuild whenever the defaults are changed upstream
    let _ = fs::remove_file(dst.join("build/CMakeCache.txt"));

    #[cfg(windows)] {
        if dst.join("lib/pkgconfig").join("aom.pc").exists() {
            env::set_var("PKG_CONFIG_PATH", dst.join("lib/pkgconfig"));
            pkg_config::Config::new().statik(true).probe("aom").unwrap();
        } else { // MSVC
            let bin_dir = if debug {
                "Debug"
            } else {
                "Release"
            };
            println!("cargo:rustc-link-search=native={}", dst.join("build").join(bin_dir).to_str().unwrap());
            println!("cargo:rustc-link-lib=static=aom");
        }
    }
    
    #[cfg(unix)] {
        env::set_var("PKG_CONFIG_PATH", dst.join("lib/pkgconfig"));
        let _libs = pkg_config::Config::new().statik(true).probe("aom").unwrap();

        #[cfg(feature = "decode_test")] {
            use std::io::Write;

            let out_dir = env::var("OUT_DIR").unwrap();

            let headers = _libs.include_paths.clone();

            let mut builder = bindgen::builder()
                .blacklist_type("max_align_t")
                .rustfmt_bindings(false)
                .header("data/aom.h");

            for header in headers {
                builder = builder.clang_arg("-I").clang_arg(header.to_str().unwrap());
            }

            // Manually fix the comment so rustdoc won't try to pick them
            let s = builder
                .generate()
                .unwrap()
                .to_string()
                .replace("/**", "/*")
                .replace("/*!", "/*");

            let dest_path = Path::new(&out_dir).join("aom.rs");

            let mut file = fs::File::create(dest_path).unwrap();

            let _ = file.write(s.as_bytes());
        }
    }

    fn rerun_dir<P: AsRef<Path>>(dir: P) {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            println!("cargo:rerun-if-changed={}", path.to_string_lossy());

            if path.is_dir() {
                rerun_dir(path);
            }
        }
    }

    rerun_dir("aom_build");
}

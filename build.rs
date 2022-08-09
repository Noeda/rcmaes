extern crate cc;

fn main() {
    cc::Build::new()
        .cpp(true)
        .file("src/c/cmaes.cc")
        .flag("-I/usr/include/eigen3")
        .flag("-I/usr/include/libcmaes")
        .flag("-I/usr/local/include/eigen3")
        .flag("-I/usr/local/include/libcmaes")
        .flag("-I/opt/homebrew/Cellar/eigen/3.3.9/include/eigen3")
        .flag("-I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3")
        .flag("-std=c++11")
        .compile("librcmaesglue.a");
    // Hack to convince rust to link to cmaes with Cargo.
    println!("cargo:rustc-link-lib=dylib=cmaes");
}

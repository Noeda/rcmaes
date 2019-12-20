extern crate cc;

fn main() {
    cc::Build::new()
        .cpp(true)
        .file("src/c/cmaes.cc")
        .flag("-I/usr/include/eigen3")
        .flag("-I/usr/include/libcmaes")
        .flag("-I/usr/local/include/eigen3")
        .flag("-I/usr/local/include/libcmaes")
        .compile("librcmaesglue.a");
    // Hack to convince rust to link to cmaes with Cargo.
    println!("cargo:rustc-link-lib=dylib=cmaes");
}

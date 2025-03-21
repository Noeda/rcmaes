extern crate cc;

fn main() {
    let mut cc_build = cc::Build::new();
    cc_build
        .cpp(true)
        .file("src/c/cmaes.cc")
        .flag("-I/usr/include/eigen3")
        .flag("-I/usr/include/libcmaes")
        .flag("-I/usr/local/include/eigen3")
        .flag("-I/usr/local/include/libcmaes")
        .flag("-I/usr/local/include")
        .flag("-I/opt/homebrew/opt/libomp/include")
        .flag("-I/opt/homebrew/Cellar/eigen/3.3.9/include/eigen3")
        .flag("-I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3")
        .flag("-std=c++11");

    cc_build.compile("librcmaesglue.a");
    println!("cargo:rustc-link-lib=dylib=cmaes");
}

extern crate cc;
extern crate pkg_config;

fn main() {
    let mut cc_build = cc::Build::new();

    // Just guessing so many paths is kinda messed up.
    // One day if I re-implement this library (rcmaes) in Rust,
    // I'd probably go and do this more properly. This is a result
    // of laziness, and then adding more paths as time marches on
    // and the code breaks.

    let cmaes = pkg_config::probe_library("libcmaes").unwrap();
    pkg_config::probe_library("eigen3").unwrap();

    cc_build.cpp(true).file("src/c/cmaes.cc").flag("-std=c++11");

    // 7 Oct 2025: When I was testing the pkg-config on MacOS, I noticed this:
    //
    // The pkg-config for libcmaes seems to give a directory without "libcmaes/" root,
    // but libcmaes' own headers are anticipating "libcmaes/" to be present.
    //
    // We "solve" this by unceremoniously slapping "../" to our path we get from pkg-config.
    // I added both the path as given by pkg-config, and also the ../
    // This may or may not cause horrible problems down the line, but considering how the headers
    // were found before I did pkg-config at all, it's _maybe_ an improvement. Maybe.
    for incl_path in cmaes.include_paths.iter() {
        cc_build.flag(format!("-I{}/..", incl_path.display()));
        cc_build.flag(format!("-I{}", incl_path.display()));
    }

    cc_build.compile("librcmaesglue.a");
}

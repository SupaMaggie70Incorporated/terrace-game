fn main() {
    println!("cargo::rerun-if-changed=gui/main.slint");
    slint_build::compile("gui/main.slint").unwrap();
}
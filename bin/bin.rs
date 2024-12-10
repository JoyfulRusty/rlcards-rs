use std::intrinsics::mir::place;

fn main() {
    let a = 10;
    place!(a);
}
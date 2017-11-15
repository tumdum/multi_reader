extern crate multi_read;

use multi_read::{MultiRead,Lines};
use std::fs::File;

fn run() -> multi_read::LineResult<usize> {
    let inputs : std::result::Result<Vec<File>, _> = std::env::args()
        .skip(1)
        .map(|p| File::open(p))
        .collect();
    Ok(Lines::from_multiread(MultiRead::new(inputs?)?)?.len())
}

// Running this program on set of paths is 'equivalent' to counting non empty
// lines from files passed as parameters.
//
// So those two commands will both produce total number of non empty lines:
// $ ./this a/b/c.txt d/e.rs f.cc
// $ cat a/b/c.txt d/e.rs f.cc | sed '/^\s*$/d' | wc -l

fn main() {
    match run() {
        Ok(lines) => println!("{}", lines),
        Err(e) => eprintln!("error: {:?}", e)
    }
}

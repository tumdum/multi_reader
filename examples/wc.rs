extern crate multi_read;

use multi_read::MultiRead;
use std::io::{BufRead,BufReader,Result};
use std::fs::File;

fn run() -> Result<usize> {
    let inputs : std::result::Result<Vec<File>, _> = std::env::args()
        .skip(1)
        .map(|p| File::open(p))
        .collect();

    Ok(BufReader::new(MultiRead::new(inputs?)?).lines().count())
}

// Running this program on set of paths is 'equivalent' to cating all of those
// paths and piping result to `wc -l`.
//
// So those two commands will both produce total number of lines:
// $ ./this a/b/c.txt d/e.rs f.cc
// $ cat a/b/c.txt d/e.rs f.cc | wc -l

fn main() {
    match run() {
        Ok(lines) => println!("{}", lines),
        Err(e) => eprintln!("error: {}", e)
    }
}

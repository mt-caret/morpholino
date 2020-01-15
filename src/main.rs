extern crate bincode;
extern crate serde;
extern crate structopt;

use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
struct Opt {
    #[structopt(short, long)]
    path: PathBuf,
    out_path: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();
    let buffer = fs::read_to_string(opt.path)?;

    let out_file = File::create(opt.out_path)?;

    let first_line_offset = buffer.find('\n').unwrap();
    let tokens: Vec<_> = buffer[0..first_line_offset].trim().split(' ').collect();
    let n = tokens[0].parse::<usize>()?;
    let d = tokens[1].parse::<usize>()?;
    println!("n = {}, d = {}", n, d);

    let embeddings: HashMap<String, Vec<f32>> = buffer[first_line_offset + 1..]
        .par_lines()
        .map(|line| {
            let mut tokens = line.trim().split(' ');
            let word = tokens.next().expect("No word found").to_string();

            let vector: Result<Vec<f32>, _> = tokens.map(|val| val.parse::<f32>()).collect();
            let vector = vector.expect("Parse failure");
            let mut m = HashMap::new();
            m.insert(word, vector);
            m
        })
        .reduce(|| HashMap::new(), |a, b| a.into_iter().chain(b).collect());
    println!("Done reading.");

    bincode::serialize_into(out_file, &embeddings)?;
    println!("Done writing.");

    Ok(())
}

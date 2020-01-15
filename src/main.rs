extern crate bincode;
extern crate serde;
extern crate structopt;

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
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
    let file = File::open(opt.path)?;
    let mut lines = BufReader::new(file).lines();

    let mut embeddings = HashMap::new();

    let out_file = File::create(opt.out_path)?;

    let first_line = lines.next().ok_or("Empty file")??;
    let tokens: Vec<_> = first_line.trim().split(' ').collect();
    let n = tokens[0].parse::<usize>()?;
    let d = tokens[1].parse::<usize>()?;
    println!("n = {}, d = {}", n, d);

    for line in lines {
        let line = line.expect("Line parsing error");
        let mut tokens = line.trim().split(' ');
        let word = tokens.next().ok_or("No word found")?.to_string();

        let vector: Result<Vec<f32>, _> = tokens.map(|val| val.parse::<f32>()).collect();
        let vector = vector?;
        assert!(embeddings.insert(word, vector).is_none());
    }
    println!("Done reading.");

    bincode::serialize_into(out_file, &embeddings)?;
    println!("Done writing.");

    Ok(())
}

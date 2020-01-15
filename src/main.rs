extern crate bincode;
extern crate serde;
extern crate structopt;

use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
struct Opt {
    #[structopt(short, long)]
    path: PathBuf,

    #[structopt(short, long)]
    number: usize,
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
        / a.iter().map(|x| x * x).sum::<f64>().sqrt()
        / b.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn detect_morpheme_boundaries(
    word: &str,
    embeddings: &HashMap<String, Vec<f64>>,
    threshold: f64,
) -> Option<Vec<usize>> {
    let mut current = embeddings.get(word)?;
    let mut boundary_indices = Vec::new();
    for boundary_index in (1..word.len()).rev() {
        if !word.is_char_boundary(boundary_index) {
            continue;
        }

        if let Some(new) = embeddings.get(&word[0..boundary_index]) {
            if cosine_similarity(current, new) > threshold {
                boundary_indices.push(boundary_index);
                current = new;
            }
        }
    }
    boundary_indices.reverse();
    Some(boundary_indices)
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();
    let buffer = fs::read_to_string(opt.path)?;

    let first_line_offset = buffer.find('\n').unwrap();
    let tokens: Vec<_> = buffer[0..first_line_offset].trim().split(' ').collect();
    let n = tokens[0].parse::<usize>()?;
    let d = tokens[1].parse::<usize>()?;
    println!("n = {}, d = {}", n, d);

    let embeddings: HashMap<String, Vec<f64>> = buffer[first_line_offset + 1..]
        .par_lines()
        .map(|line| {
            let mut tokens = line.trim().split(' ');
            let word = tokens.next().expect("No word found").to_string();

            let vector: Result<Vec<f64>, _> = tokens.map(|val| val.parse::<f64>()).collect();
            let vector = vector.expect("Parse failure");
            assert_eq!(vector.len(), d);
            let mut m = HashMap::new();
            m.insert(word, vector);
            m
        })
        .reduce(|| HashMap::new(), |a, b| a.into_iter().chain(b).collect());
    assert_eq!(embeddings.len(), n);
    println!("Done reading.");

    for key in embeddings.keys().take(opt.number) {
        let boundaries = detect_morpheme_boundaries(key, &embeddings, 0.25).unwrap();
        println!("{}: {:?}", key, boundaries);
    }

    Ok(())
}

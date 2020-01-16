extern crate bincode;
extern crate hashbrown;
extern crate serde;
extern crate structopt;

use hashbrown::HashMap;
use rayon::prelude::*;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use structopt::StructOpt;

// TODO: implement loading from bincode?
#[derive(StructOpt, Debug)]
struct Opt {
    #[structopt(short, long)]
    embeddings_path: PathBuf,

    #[structopt(short, long)]
    corpus_path: PathBuf,

    #[structopt(short, long)]
    number: usize,

    #[structopt(short, long)]
    boundary_threshold: f64,
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
    boundary_indices.push(word.len());
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
    boundary_indices.push(0);
    boundary_indices.reverse();
    Some(boundary_indices)
}

fn generate_counts<'a>(
    word: &'a str,
    embeddings: &'a HashMap<String, Vec<f64>>,
    boundary_threshold: f64,
) -> Option<HashMap<(&'a str, &'a str), usize>> {
    let boundaries = detect_morpheme_boundaries(word, embeddings, boundary_threshold)?;

    let mut counts = HashMap::new();
    for i in 1..boundaries.len() {
        let split_index = boundaries[i];
        for start in 0..i {
            let first_morpheme = if boundaries[start] == 0 {
                "^"
            } else {
                &word[boundaries[start]..split_index]
            };

            if boundaries[i] == word.len() {
                assert!(counts.insert((first_morpheme, "$"), 1).is_none());
                continue;
            }

            for end in (i + 1)..boundaries.len() {
                assert!(counts
                    .insert((first_morpheme, &word[split_index..end]), 1)
                    .is_none());
            }
        }
    }
    Some(counts)
}

fn calculate_morpheme_frequencies<'a>(
    word_frequency: &'a HashMap<&str, usize>,
    embeddings: &'a HashMap<String, Vec<f64>>,
    boundary_threshold: f64,
) -> HashMap<(&'a str, &'a str), usize> {
    word_frequency
        .par_iter()
        .map(|(word, count)| {
            let mut counts = generate_counts(word, &embeddings, boundary_threshold).unwrap();
            counts.values_mut().for_each(|val| *val *= count);
            counts
        })
        .reduce(
            || HashMap::new(),
            |a, b| {
                let (mut a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
                for (&word, count) in b.iter() {
                    let entry = a.entry(word).or_insert(0);
                    *entry += count;
                }
                a
            },
        )
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();
    let buffer = fs::read_to_string(&opt.embeddings_path)?;

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
    println!("Done reading embeddings.");

    for key in embeddings.keys().take(opt.number) {
        let boundaries =
            detect_morpheme_boundaries(key, &embeddings, opt.boundary_threshold).unwrap();
        println!("{}: {:?}", key, boundaries);
    }

    let buffer = fs::read_to_string(&opt.corpus_path)?;
    let mut words: Vec<&str> = buffer.split_whitespace().collect();
    words.sort_unstable();
    let mut word_frequency = HashMap::new();
    for &word in words.iter() {
        let entry = word_frequency.entry(word).or_insert(0);
        *entry += 1;
    }
    println!("Done reading corpus.");

    let morphotactic_frequency =
        calculate_morpheme_frequencies(&word_frequency, &embeddings, opt.boundary_threshold);

    for ((a, b), value) in morphotactic_frequency.iter().take(opt.number) {
        println!("{} -> {}, {}", a, b, value);
    }

    Ok(())
}

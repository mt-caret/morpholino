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
) -> Vec<usize> {
    let mut current = embeddings
        .get(word)
        .ok_or_else(|| format!("word {} not found", word))
        .unwrap();

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
    boundary_indices
}

fn generate_counts<'a>(
    word: &'a str,
    embeddings: &'a HashMap<String, Vec<f64>>,
    boundary_threshold: f64,
) -> HashMap<&'a str, HashMap<&'a str, usize>> {
    let boundaries = detect_morpheme_boundaries(word, embeddings, boundary_threshold);
    let mut counts = HashMap::new();

    for i in 0..boundaries.len() {
        if boundaries[i] == 0 {
            let morpheme_entry = counts.entry("^").or_insert(HashMap::new());
            for end in 1..boundaries.len() {
                let entry = morpheme_entry.entry(&word[0..boundaries[end]]).or_insert(0);
                *entry += 1;
            }
            continue;
        }

        for start in 0..i {
            let first_morpheme = if boundaries[start] == 0 {
                "<root>"
            } else {
                &word[boundaries[start]..boundaries[i]]
            };

            let morpheme_entry = counts.entry(first_morpheme).or_insert(HashMap::new());
            if boundaries[i] == word.len() {
                assert!(morpheme_entry.insert("$", 1).is_none());
                continue;
            }

            for end in (i + 1)..boundaries.len() {
                let entry = morpheme_entry
                    .entry(&word[boundaries[i]..boundaries[end]])
                    .or_insert(0);
                *entry += 1;
            }
        }
    }
    counts
}

fn calculate_morpheme_frequencies<'a>(
    word_frequency: &'a HashMap<String, usize>,
    embeddings: &'a HashMap<String, Vec<f64>>,
    boundary_threshold: f64,
) -> HashMap<&'a str, HashMap<&'a str, usize>> {
    word_frequency
        .par_iter()
        .map(|(word, count)| {
            if embeddings.contains_key(word) {
                let mut counts = generate_counts(word, &embeddings, boundary_threshold);
                counts.values_mut().for_each(|morpheme_counts| {
                    morpheme_counts.values_mut().for_each(|val| *val *= count);
                });
                counts
            } else {
                HashMap::new()
            }
        })
        .reduce(
            || HashMap::new(),
            |a, b| {
                let (mut a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
                for (&first_morpheme, morpheme_counts) in b.iter() {
                    let morpheme_entry = a.entry(first_morpheme).or_insert(HashMap::new());
                    for (&second_morpheme, count) in morpheme_counts.iter() {
                        let entry = morpheme_entry.entry(second_morpheme).or_insert(0);
                        *entry += count;
                    }
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
        let boundaries = detect_morpheme_boundaries(key, &embeddings, opt.boundary_threshold);
        let counts = generate_counts(key, &embeddings, opt.boundary_threshold);
        println!("{}: {:?}, {:?}", key, boundaries, counts);
    }

    let buffer = fs::read_to_string(&opt.corpus_path)?;
    let mut words: Vec<String> = buffer
        .split_whitespace()
        .map(|s| {
            let mut s = s.to_string();
            s.retain(|c| !c.is_ascii_punctuation());
            s
        })
        .collect();
    words.sort_unstable();
    let mut word_frequency = HashMap::new();
    for word in words.into_iter() {
        let entry = word_frequency.entry(word).or_insert(0);
        *entry += 1;
    }
    println!("Done reading corpus.");

    let morphotactic_frequency =
        calculate_morpheme_frequencies(&word_frequency, &embeddings, opt.boundary_threshold);

    let morphotactic_frequency: HashMap<&str, (usize, HashMap<&str, usize>)> =
        morphotactic_frequency
            .into_par_iter()
            .map(|(first_morpheme, morpheme_counts)| {
                let sum = morpheme_counts.values().sum();
                (first_morpheme, (sum, morpheme_counts))
            })
            .collect();

    for (first_morpheme, (sum, morpheme_counts)) in morphotactic_frequency.iter().take(opt.number) {
        for (second_morpheme, count) in morpheme_counts.iter() {
            println!(
                "{} -> {}, {}/{}",
                first_morpheme, second_morpheme, count, sum
            );
        }
    }

    Ok(())
}

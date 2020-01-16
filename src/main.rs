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

    #[structopt(short, long)]
    bidirectional_detection: bool,

    #[structopt(short, long)]
    no_header: bool,

    #[structopt(short, long)]
    show_boundaries: bool,

    #[structopt(short, long)]
    show_morphemes: bool,

    #[structopt(short, long)]
    show_results: bool,
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
    bidirectional: bool,
) -> Vec<usize> {
    let original = embeddings
        .get(word)
        .ok_or_else(|| format!("word {} not found", word))
        .unwrap();

    let mut boundary_indices = Vec::new();

    let mut current = original;
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

    if bidirectional {
        current = original;
        for boundary_index in 1..word.len() {
            if !word.is_char_boundary(boundary_index) {
                continue;
            }
            if let Some(new) = embeddings.get(&word[boundary_index..word.len()]) {
                if cosine_similarity(current, new) > threshold {
                    boundary_indices.push(boundary_index);
                    current = new;
                }
            }
        }
    }

    boundary_indices.push(0);
    boundary_indices.push(word.len());
    boundary_indices.sort_unstable();
    boundary_indices.dedup();
    boundary_indices
}

fn generate_counts<'a>(
    word: &'a str,
    embeddings: &'a HashMap<String, Vec<f64>>,
    boundary_threshold: f64,
    bidirectional_detection: bool,
) -> HashMap<&'a str, HashMap<&'a str, usize>> {
    let boundaries = detect_morpheme_boundaries(
        word,
        embeddings,
        boundary_threshold,
        bidirectional_detection,
    );
    let mut counts = HashMap::new();

    for i in 0..boundaries.len() {
        if boundaries[i] == 0 {
            let morpheme_entry = counts.entry("<start>").or_insert(HashMap::new());
            for end in 1..boundaries.len() {
                let entry = morpheme_entry.entry(&word[0..boundaries[end]]).or_insert(0);
                *entry += 1;
            }
        }

        for start in 0..i {
            let first_morpheme = if boundaries[start] == 0 {
                "<root>"
            } else {
                &word[boundaries[start]..boundaries[i]]
            };

            let morpheme_entry = counts.entry(first_morpheme).or_insert(HashMap::new());
            if boundaries[i] == word.len() {
                assert!(morpheme_entry.insert("<end>", 1).is_none());
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
    bidirectional_detection: bool,
) -> HashMap<&'a str, HashMap<&'a str, usize>> {
    word_frequency
        .par_iter()
        .map(|(word, count)| {
            if embeddings.contains_key(word) {
                let mut counts = generate_counts(
                    word,
                    &embeddings,
                    boundary_threshold,
                    bidirectional_detection,
                );
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

// dfs returns the path with max prob ending at index with morpheme
fn dfs<'a>(
    word: &str,
    index: usize,
    morpheme: &str,
    boundaries: &Vec<usize>,
    morph_frequency: &HashMap<&'a str, (usize, HashMap<&'a str, usize>)>,
) -> (Vec<usize>, f64, bool) {
    if index == 0 {
        let mut path = Vec::new();
        path.push(0);

        return (path, 1f64, false);
    }

    (0..index)
        .map(|old_index| {
            let is_root_morpheme = old_index == 0;
            let mut last_morpheme = if is_root_morpheme {
                "<start>"
            } else {
                &word[boundaries[old_index]..boundaries[index]]
            };

            let (mut path, old_prob, was_root_morpheme) =
                dfs(word, old_index, &last_morpheme, boundaries, morph_frequency);
            if was_root_morpheme {
                last_morpheme = "<root>";
            }
            path.push(old_index);

            let prob = old_prob
                * morph_frequency
                    .get(&last_morpheme)
                    .and_then(|(sum, morph_count)| {
                        morph_count.get(morpheme).map(|count| {
                            ((*count + 1) as f64) / ((*sum + morph_frequency.len()) as f64)
                        })
                    })
                    .unwrap_or(1f64 / morph_frequency.len() as f64);

            (path, prob, is_root_morpheme)
        })
        .max_by(|(_, prob1, _), (_, prob2, _)| {
            prob1
                .partial_cmp(prob2)
                .ok_or_else(|| format!("compare failed between {} and {}", prob1, prob2))
                .unwrap()
        })
        .expect("should not be empty")
}

// TODO: this isn't really viterbi, as it naively uses recursion without memoization
fn viterbi_split<'a>(
    word: &str,
    embeddings: &'a HashMap<String, Vec<f64>>,
    boundary_threshold: f64,
    morph_frequency: &HashMap<&'a str, (usize, HashMap<&'a str, usize>)>,
    bidirectional_detection: bool,
) -> Option<Vec<usize>> {
    if embeddings.contains_key(word) {
        let boundaries = detect_morpheme_boundaries(
            word,
            embeddings,
            boundary_threshold,
            bidirectional_detection,
        );
        let (mut path, _prob, _) = dfs(
            word,
            boundaries.len() - 1,
            "<end>",
            &boundaries,
            morph_frequency,
        );
        path.push(word.len());
        Some(path)
    } else {
        None
    }
}

fn print_word_with_splits(word: &str, boundaries: &[usize]) {
    for i in 0..(boundaries.len() - 1) {
        print!("{} ", word[boundaries[i]..boundaries[i + 1]].to_string());
    }
}

fn parse_to_vecs(buffer: &str, opt_dim: Option<usize>) -> HashMap<String, Vec<f64>> {
    buffer
        .par_lines()
        .map(|line| {
            let mut tokens = line.trim().split(' ');
            let word = tokens.next().expect("No word found").to_string();

            let vector: Result<Vec<f64>, _> = tokens.map(|val| val.parse::<f64>()).collect();
            let vector = vector.expect("Parse failure");
            if let Some(d) = opt_dim {
                assert_eq!(vector.len(), d);
            }
            let mut m = HashMap::new();
            m.insert(word, vector);
            m
        })
        .reduce(|| HashMap::new(), |a, b| a.into_iter().chain(b).collect())
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();
    let buffer = fs::read_to_string(&opt.embeddings_path)?;

    let embeddings = if opt.no_header {
        println!("no header");
        parse_to_vecs(&buffer, None)
    } else {
        let first_line_offset = buffer.find('\n').unwrap();
        let tokens: Vec<_> = buffer[0..first_line_offset].trim().split(' ').collect();
        let n = tokens[0].parse::<usize>()?;
        let d = tokens[1].parse::<usize>()?;
        println!("n = {}, d = {}", n, d);

        let embeddings = parse_to_vecs(&buffer[first_line_offset + 1..], Some(d));
        assert_eq!(embeddings.len(), n);
        embeddings
    };
    println!("Done reading embeddings.");

    if opt.show_boundaries {
        for key in embeddings.keys().take(opt.number) {
            let boundaries = detect_morpheme_boundaries(
                key,
                &embeddings,
                opt.boundary_threshold,
                opt.bidirectional_detection,
            );
            let counts = generate_counts(
                key,
                &embeddings,
                opt.boundary_threshold,
                opt.bidirectional_detection,
            );
            print!("{} -> ", key);
            print_word_with_splits(key, &boundaries);
            println!("{:?}", counts);
        }
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

    let morphotactic_frequency = calculate_morpheme_frequencies(
        &word_frequency,
        &embeddings,
        opt.boundary_threshold,
        opt.bidirectional_detection,
    );
    println!("Done calculating morpheme frequencies.");

    let morphotactic_frequency: HashMap<&str, (usize, HashMap<&str, usize>)> =
        morphotactic_frequency
            .into_par_iter()
            .map(|(first_morpheme, morpheme_counts)| {
                let sum = morpheme_counts.values().sum();
                (first_morpheme, (sum, morpheme_counts))
            })
            .collect();
    println!("Done summing morpheme frequencies.");

    if opt.show_morphemes {
        for (first_morpheme, (sum, morpheme_counts)) in
            morphotactic_frequency.iter().take(opt.number)
        {
            for (second_morpheme, count) in morpheme_counts.iter() {
                println!(
                    "{} -> {}, {}/{}",
                    first_morpheme, second_morpheme, count, sum
                );
            }
        }
    }

    if opt.show_results {
        for (word, frequency) in word_frequency.iter().take(opt.number) {
            let boundaries = viterbi_split(
                word,
                &embeddings,
                opt.boundary_threshold,
                &morphotactic_frequency,
                opt.bidirectional_detection,
            );
            if let Some(boundaries) = boundaries {
                print!("{} -> ", word);
                print_word_with_splits(word, &boundaries);
                println!(", {}", frequency);
            }
        }
    }
    Ok(())
}

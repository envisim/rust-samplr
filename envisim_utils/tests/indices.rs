use envisim_utils::indices::*;
use rand::{rngs::SmallRng, SeedableRng};

#[test]
fn clear() {
    let mut il = Indices::with_fill(4);
    il.clear();
    assert_eq!(il.len(), 0);
}

#[test]
fn list() {
    let il = Indices::with_fill(4);
    assert_eq!(il.list(), vec![0, 1, 2, 3]);
}

#[test]
fn get() {
    let il = Indices::with_fill(4);
    assert_eq!(il.get(3).unwrap(), &3);
    assert_eq!(il.get(10), None);
}

#[test]
fn first() {
    let il = Indices::with_fill(4);
    assert_eq!(il.first().unwrap(), &0);
}

#[test]
fn last() {
    let il = Indices::with_fill(4);
    assert_eq!(il.last().unwrap(), &3);
}

#[test]
fn draw() {
    let mut rng = SmallRng::seed_from_u64(4242);
    let il = Indices::with_fill(4);
    assert!(il.draw(&mut rng).is_some());
    assert!(il.draw(&mut rng).is_some());
    assert!(il.draw(&mut rng).is_some());
}

#[test]
fn contains() {
    let il = Indices::with_fill(4);
    assert!(il.contains(3));
    assert!(!il.contains(4));
}

#[test]
fn len() {
    let il = Indices::with_fill(4);
    assert_eq!(il.len(), 4);
}

#[test]
fn is_empty() {
    let mut il = Indices::with_fill(4);
    assert!(!il.is_empty());
    il.clear();
    assert!(il.is_empty());
}

#[test]
fn remove() {
    let mut il = Indices::with_fill(4);
    il.remove(1).unwrap();
    il.remove(3).unwrap();
    assert!(il.contains(0));
    assert!(!il.contains(1));
    assert!(il.contains(2));
    assert!(!il.contains(3));
}

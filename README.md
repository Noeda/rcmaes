# Rust libcmaes bindings

This is a simple library that lets you optimize some model. If you can
implement a trait that turns a struct into a `Vec<f64>` and back (i.e. there is
an isomorphism between them), then you can pass it to CMA-ES library.

The C++ CMA-ES project is located here: https://github.com/beniz/libcmaes

I'm using this mostly for some personal projects so it's not all that
comprehensive. The bindings implement just enough that you can adjust most
important parameters and run heavy optimization concurrently and with
reasonable efficiency.

# Example

```rust
extern crate rcmaes;

use rcmaes::{optimize, CMAESParameters, Vectorizable};

// This example "trains" a very simple model to go to point (2, 8)
#[derive(Clone)]
struct TwoPoly {
    x: f64,
    y: f64,
}

// Vectorizable is essential to specify how to turn something into a Vec<f64>
// and back.
impl Vectorizable for TwoPoly {
    fn to_vec(&self) -> Vec<f64> {
        vec![self.x, self.y]
    }

    fn from_vec(vec: &[f64]) -> Self {
        TwoPoly {
            x: vec[0],
            y: vec[1],
        }
    }
}

fn train_model()
{
    let optimized = optimize(
        &TwoPoly { x: 5.0, y: 6.0 },
        &CMAESParameters::default(),
        |twopoly| (twopoly.x - 2.0).abs() + (twopoly.y - 8.0).abs(),
    ).unwrap();

    let model = optimized.0;
    assert!((model.x - 2.0).abs() < 0.00001);
    assert!((model.y - 8.0).abs() < 0.00001);
}
```

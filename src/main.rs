use std::cmp::Ordering;
use rand;
use std::io::{stdin};
use rand::Rng;

fn main() {
    loop {
        let secret_number = rand::thread_rng().gen_range(1..100000000);
        println!("secret number is {secret_number}");
        println!("please guess number.");
        let mut guess = String::new();
        stdin().read_line(&mut guess).expect("Failed to read line");
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };
        let cmp_res = match guess.cmp(&secret_number) {
            Ordering::Less => "too small",
            Ordering::Greater => "too big",
            Ordering::Equal => {
                println!("you win!");
                break;
            },
        };
        println!("{}", cmp_res);
        println!()
    }
}

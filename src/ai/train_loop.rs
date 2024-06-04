use std::{fmt::Debug, fs, path::{Path, PathBuf}};

use burn::{module::AutodiffModule, optim::{momentum::MomentumConfig, SgdConfig}, record::{CompactRecorder, DefaultFileRecorder}, tensor::backend::AutodiffBackend};
use indicatif::{ProgressBar, ProgressStyle};

use super::{data::load_data_from_file, eval::{compare, elo_comparison}, net::Network, plot::{plot_elo_graphs, plot_loss_graph}, train::{train_on_data, DataLoader}};

pub fn train_networks<P: AsRef<Path>, B: AutodiffBackend, N: Network<B> + AutodiffModule<B>, F: Fn() -> N>(data_dir: P, graph_dir: P, progress_bar: bool, num_networks: usize, dev: &B::Device, create_network: F, learning_rate: f64, momentum: f64) {
    let initial_net = create_network().no_grad();
    let mut graph_dir: PathBuf = graph_dir.as_ref().to_owned();
    let mut networks = Vec::new();
    let mut network_histories: Vec<(Vec<(f32, f32, f32)>, Vec<f32>)> = Vec::new();
    let data_dir: PathBuf = data_dir.as_ref().to_owned();
    let (data_loader, set_count) = DataLoader::<B>::new(dev, data_dir);
    for i in 0..num_networks {
        networks.push((create_network(), SgdConfig::new().with_momentum(Some(MomentumConfig::new().with_momentum(momentum))).init()));
        network_histories.push((Vec::new(), Vec::new()));
    }
    let bar = if progress_bar {
        ProgressBar::new(set_count as u64).with_style(ProgressStyle::with_template("[{elapsed_precise}] {bar:100.cyan/blue} {pos:>7}/{len:7} {msg}").unwrap())
    } else {ProgressBar::hidden()};
    for (iter, (inputs, targets)) in data_loader.enumerate() {
        for (i, (net, optim)) in networks.iter_mut().enumerate() {
            let (n, loss) = train_on_data(dev, net.clone(), inputs.clone(), targets.clone(), optim, learning_rate);
            *net = n;
            network_histories[i].1.push(loss);
            graph_dir.push(format!("loss-net{}.svg", i));
            plot_loss_graph(&graph_dir, &network_histories[i].1);
            graph_dir.pop();
            if iter % 32 == 0 {
                net.clone().save_file(format!("nets/net{}.bin", i), &CompactRecorder::new()).unwrap();
            }
        }
        bar.inc(1);
    }
    bar.finish();
    for (i, (net, optim)) in networks.iter_mut().enumerate() {
        let results = compare(dev, &net.clone().no_grad(), &initial_net, 64, 8);
        let (elo_diff, elo_range) = elo_comparison(results, super::eval::EloComparisonMode::Games, 0.90);
        /*network_histories[i].0.push((elo_diff, elo_diff - elo_range, elo_diff + elo_range));
        graph_dir.push(format!("elo-net{}.svg", i));
        plot_elo_graphs(&graph_dir, &network_histories[i].0);
        graph_dir.pop();*/
        println!("Results: {:?}\nElo: {} +/- {}", results, elo_diff, elo_range);
    }
}
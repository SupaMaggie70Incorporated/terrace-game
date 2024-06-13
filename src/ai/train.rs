use std::{fmt::Debug, fs::{self, File}, io::{BufReader, Read}, path::{Path, PathBuf}, sync::mpsc::{channel, Receiver, Sender}, thread};

use burn::{module::AutodiffModule, optim::{GradientsParams, Optimizer}, tensor::backend::{AutodiffBackend, Backend}};

use super::{net::{Network, NetworkInput, NetworkOutput, NetworkTarget}};



pub fn train_on_data<B: AutodiffBackend, N: Network<B> + AutodiffModule<B>, O: Optimizer<N, B>>(dev: &B::Device, mut net: N, inputs: NetworkInput<B>, targets: NetworkTarget<B>, optim: &mut O, learning_rate: f64) -> (N, f32, f32) {
    let outputs = net.forward(inputs);
    let loss = targets.loss(outputs);
    let selected_loss = loss.value_loss.clone() + loss.policy_loss.clone();
    let grads = selected_loss.backward();
    let grads = GradientsParams::from_grads(grads, &net);
    let value_loss_value = loss.value_loss.into_data().convert().value[0];
    let policy_loss_value = loss.policy_loss.into_data().convert().value[0];
    net = optim.step(learning_rate, net, grads);
    (net, value_loss_value, policy_loss_value)
}
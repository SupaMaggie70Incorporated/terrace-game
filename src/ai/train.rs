use std::{fmt::Debug, fs::{self, File}, io::{BufReader, Read}, path::{Path, PathBuf}, sync::mpsc::{channel, Receiver, Sender}, thread};

use burn::{module::AutodiffModule, optim::{GradientsParams, Optimizer}, tensor::backend::{AutodiffBackend, Backend}};

use super::{data::{self, load_data_from_file, PositionInfo}, net::{Network, NetworkInput, NetworkOutput, NetworkTarget, TrainableData}};


pub struct DataLoader<B: Backend> {
    next_sender: Sender<bool>,
    data_receiver: Receiver<Option<(NetworkInput<B>, NetworkTarget<B>)>>
}
impl<B: Backend> DataLoader<B> {
        const MAX_NUM_DATAPOINTS: usize = 100_000;
    fn loader_handler(dev: &B::Device, dir: PathBuf, sender: Sender<Option<(NetworkInput<B>, NetworkTarget<B>)>>, receiver: Receiver<bool>) {
        let mut files = fs::read_dir(dir).unwrap();
        let mut data = Vec::new();
        loop {
            if let Some(Ok(entry)) = files.next() {
                let file = File::open(entry.path()).unwrap();
                let num_datapoints = file.metadata().unwrap().len() as usize/ PositionInfo::SIZE;
                let mut reader = BufReader::new(file);
                let num_sets = num_datapoints / Self::MAX_NUM_DATAPOINTS;
                for _ in 0..num_sets {
                    data.clear();
                    for _ in 0..Self::MAX_NUM_DATAPOINTS {
                        let mut datapoint = PositionInfo {bytes: [0; PositionInfo::SIZE]};
                        reader.read_exact(&mut datapoint.bytes).unwrap();
                        data.push(datapoint);
                    }
                    let data_to_send = data.as_slice().to_training_values(dev);
                    if sender.send(Some(data_to_send)).is_err() {return}
                    if let Ok(v) = receiver.recv() {
                        if !v {
                            return;
                        }
                    } else {
                        return;
                    }
                };
            } else {
                if sender.send(None).is_err() {return};
                return;
            }
        }
    }
    pub fn new<P: AsRef<Path>>(dev: &B::Device, dir: P) -> (Self, usize) {
        let dir = dir.as_ref().to_owned();
        let file_count = fs::read_dir(&dir).unwrap().count();
        let sets_per_file = fs::read_dir(&dir).unwrap().next().unwrap().unwrap().metadata().unwrap().len() as usize / Self::MAX_NUM_DATAPOINTS;
        let (next_sender, next_receiver) = channel();
        let (data_sender, data_receiver) = channel();
        let other_dev = dev.clone();
        thread::spawn(move || {
            Self::loader_handler(&other_dev, dir, data_sender, next_receiver)
        });
        (Self {
            next_sender,
            data_receiver
        }, file_count * sets_per_file)
    }
    pub fn get_next(&self) -> Option<(NetworkInput<B>, NetworkTarget<B>)> {
        let value = self.data_receiver.recv();
        if let Ok(Some(v)) = value {
            let _ = self.next_sender.send(true);
            Some(v)
        } else {
            None
        }
    }
}
impl<B: Backend> Drop for DataLoader<B> {
    fn drop(&mut self) {
        let _ = self.next_sender.send(false);
    }
}
impl<B: Backend> Iterator for DataLoader<B> {
    type Item = (NetworkInput<B>, NetworkTarget<B>);
    fn next(&mut self) -> Option<Self::Item> {
        self.get_next()
    }
}
pub fn train_on_data<B: AutodiffBackend, N: Network<B> + AutodiffModule<B>, O: Optimizer<N, B>>(dev: &B::Device, mut net: N, inputs: NetworkInput<B>, targets: NetworkTarget<B>, optim: &mut O, learning_rate: f64) -> (N, f32) {
    let outputs = net.forward(inputs);
    let loss = targets.loss(outputs);
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &net);
    let loss_value = loss.into_data().convert().value[0];
    net = optim.step(learning_rate, net, grads);
    (net, loss_value)
}
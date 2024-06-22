use std::path::Path;
use plotters::prelude::*;

pub fn plot_elo_graphs<P: AsRef<Path>>(file: P, steps: &[(f32, f32, f32)]) {
    let mut min = steps[0].0;
    let mut max = steps[0].0;
    for &(a, b, c) in steps {
        min = min.min(a).min(b).min(c);
        max = max.max(a).max(b).max(c);
    }
    let root = SVGBackend::new(file.as_ref(), (2048, 1024)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Elo by iteration", FontDesc::new(FontFamily::SansSerif, 64.0, FontStyle::Normal))
        .x_label_area_size(32)
        .y_label_area_size(64)
        .build_cartesian_2d(0..steps.len(), min..max)
        .unwrap();
    let label_style = ("sans-serif", 24).into_font();
    chart.configure_mesh()
        .x_labels(16)
        .y_labels(16)
        .x_label_style(label_style.clone())
        .y_label_style(label_style)
        .draw()
        .unwrap();
    // Important to put the blue line last so that it isnt overwritten when they overlap
    chart.draw_series(LineSeries::new(steps.iter().enumerate().map(|(i, v)| {(i, v.1)}),
    ShapeStyle::from(&RED).stroke_width(2)
    )).unwrap();
    chart.draw_series(LineSeries::new(steps.iter().enumerate().map(|(i, v)| {(i, v.2)}),
    ShapeStyle::from(&RED).stroke_width(2)
    )).unwrap();
    chart.draw_series(LineSeries::new(steps.iter().enumerate().map(|(i, v)| {(i, v.0)}),
    ShapeStyle::from(&BLUE).stroke_width(2)
    )).unwrap();
    root.present().unwrap();
}
pub fn plot_loss_graph<P: AsRef<Path>>(file: P, losses: &[(f32, f32)]) {
    let mut value_min = losses[0].0;
    let mut value_max = losses[0].0;
    let mut policy_min = losses[0].1;
    let mut policy_max = losses[0].1;
    for &(value, policy) in losses {
        value_min = value_min.min(value);
        value_max = value_max.max(value);
        policy_min = policy_min.min(value);
        policy_max = policy_max.max(value);
    }
    let policy_scaling = (value_max - value_min) / (policy_max - policy_min);
    let root = SVGBackend::new(file.as_ref(), (2048, 1024)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Loss by iteration", FontDesc::new(FontFamily::SansSerif, 64.0, FontStyle::Normal))
        .x_label_area_size(32)
        .y_label_area_size(64)
        .build_cartesian_2d(0..losses.len(), value_min..value_max)
        .unwrap();
    let label_style = ("sans-serif", 24).into_font();
    chart.configure_mesh()
        .x_labels(16)
        .y_labels(16)
        .x_label_style(label_style.clone())
        .y_label_style(label_style)
        .draw()
        .unwrap();
    chart.draw_series(LineSeries::new(losses.iter().enumerate().map(|(i, v)| {(i, v.0)}),
                                      ShapeStyle::from(&BLUE).stroke_width(2)
    )).unwrap();
    chart.draw_series(LineSeries::new(losses.iter().enumerate().map(|(i, v)| {(i, (v.1 - policy_min) * policy_scaling + value_min)}),
                                      ShapeStyle::from(&GREEN).stroke_width(2)
    )).unwrap();
    root.present().unwrap();
}
pub fn plot_net_perf<P: AsRef<Path>>(file: P, perfs: Vec<(&str, f32, f32)>) {
    
}
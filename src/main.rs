use std::borrow::Borrow;
use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::{BufWriter, Read};
use std::string::String;

use image::{GenericImage, ImageBuffer, ImageOutputFormat, Rgb, RgbImage};
use image::io::Reader as ImageReader;
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;
use tract_onnx::prelude::*;
use tract_onnx::prelude::tract_itertools::Itertools;
use tract_onnx::prelude::tract_num_traits::abs;

use crate::benchmark::Timer;
use crate::dot2rect::{Point, wrapped_rect};

mod dot2rect;
mod benchmark;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;


fn main() -> TractResult<()> {
    let mut timer = Timer::new();
    let image_path = "example/40a4bae6f5a3ef025778196b2d0cc708.jpeg";
    let rec_model = onnx()
        // load the model
        .model_for_path("model/rec.onnx")?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;
    let model = onnx()
        // load the model
        .model_for_path("model/model.onnx")?
        // optimize the model
        .into_compact()? //todo: it will panic.
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;
    timer.tick();
    println!("model load successfully");

    let mut buf = String::new();
    File::open("model/label_list.txt").unwrap().read_to_string(&mut buf).expect("Unable to read label");
    let label: Vec<_> = buf.split("\n").collect();
    timer.tick();
    println!("Label loaded");

    let image = ImageReader::open(image_path)?.decode()?.to_rgb8();
    timer.tick();
    println!("Image loaded!");

    let resized =
        image::imageops::resize(&image, WIDTH, HEIGHT, image::imageops::FilterType::Triangle);
    timer.tick();
    println!("Image resized({})!", resized.len());

    let image_tensor: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, HEIGHT as usize, WIDTH as usize), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    })
        .into();
    timer.tick();
    println!("Image transformed!");

    // run the model on the input
    let result = model.run(tvec!(image_tensor.into()))?;
    timer.tick();
    println!("Image Detected!");

    // // find and display the max value with its index
    let binding: &[f32] = result[0].as_slice()?;
    let iter = binding.iter()
        .cloned();
    let (mut x, mut y) = (vec![], vec![]);
    let mut count = HEIGHT;
    for chunk in &iter.chunks(WIDTH as usize) {
        let best = chunk
            .collect::<Vec<_>>();
        for (i, v) in best.iter().enumerate() {
            if *v > 0.5 {
                // println!("({count},{id}) -> ({v})");
                y.push(count);
                x.push(i as u32);
            }
        }
        count -= 1;
    }
    let points: Vec<_> = x.iter().map(|&v| v).zip(y.clone()).map(|(x, y)| Point { x, y }).collect();
    let connected_components = dot2rect::connected_components(points.clone(), 3);
    let mut output_image = ImageBuffer::new(image.width(), image.height());
    output_image.copy_from(&image, 0, 0).expect("Failed to copy background image");
    for component in connected_components {
        if let Some(mut rect) = wrapped_rect(component) {
            rect.remap(image.width(), image.height());
            let rect_sr = Rect::at(rect.x as i32, rect.y as i32).of_size(rect.width, rect.height);
            let fill_color = Rgb([255, 0, 0]); // 填充颜色为红色，透明度为 128
            draw_filled_rect_mut(&mut output_image, rect_sr, fill_color);
            let sub_image = image::imageops::crop_imm(&image, rect.x, rect.y, rect.width, rect.height).to_image();
            let mut buf = BufWriter::new(File::create(format!("output/{}-{}.png", rect.x, rect.y)).unwrap());
            sub_image.write_to(&mut buf, ImageOutputFormat::Png).expect("TODO: panic message");
            rec(&rec_model, &sub_image, &label).unwrap();
        }
    }
    output_image.save("render.png").expect("Failed to save output image");
    timer.tick();
    println!("\n");
    timer.all();
    Ok(())
}

fn rec<F, O, M>(model: &RunnableModel<F, O, M>, image: &RgbImage, label: &Vec<&str>) -> TractResult<String>
    where
        F: Fact + Clone + 'static,
        O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
        M: Borrow<Graph<F, O>>,
{
    let scale = 48f64 / (image.height() as f64);
    let width = image.width() as f64 * scale;
    let resized =
        image::imageops::resize(image, width as u32, 48, image::imageops::FilterType::Triangle);
    // println!("Image resized({})!", resized.len());
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 48, width as usize), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    })
        .into();
    // println!("Image transformed:{}!", width);
    // run the model on the input
    let result = model.run(tvec!(image.into()))?;
    // find and display the max value with its index
    let binding: &[f32] = result[0].as_slice()?;
    let iter = binding.iter()
        .cloned();
    let mut rec = "".to_string();
    let mut last_text = String::new();
    let mut last_probability = 0f32;
    for chunk in &iter.chunks(6625) {
        let best = chunk
            .collect::<Vec<_>>();
        let (mut id, mut value) = (0, 0f32);
        for (i, v) in best.iter().enumerate() {
            if *v > value {
                id = i;
                value = *v;
            }
        }
        let text = if id > 0 && id <= label.len() { format!("{}", label[id - 1]) } else { "".to_string() };
        if text != "" {
            let min = if value < last_probability { value } else { last_probability };
            let repeat_not_remove = text == last_text && abs((value - last_probability) / min) < 0.07 && value > 0.6 && last_probability > 0.6;
            // if repeat_not_remove {
            //     print!("({})({},{}->{})", text, last_probability, value, abs(value - last_probability) / min)
            // }
            if text != last_text || repeat_not_remove || last_probability == 0f32 {
                rec += text.as_str();
            }
            last_text = text;
            last_probability = value;
        }
    }
    println!("\t{:?}", rec);
    Ok(rec)
}

#[test]
fn test() {
    use plt::*;

    let xs: Vec<f64> = (0..=100).map(|n: u32| n as f64 * 0.1).collect();
    let ys: Vec<f64> = xs.iter().map(|x| x.powi(3)).collect();

    let mut sp = Subplot::builder()
        .label(Axes::X, "x data")
        .label(Axes::Y, "y data")
        .build();

    sp.plot(&xs, &ys).unwrap();

    let mut fig = <Figure>::default();
    fig.set_layout(SingleLayout::new(sp)).unwrap();

    fig.draw_file(FileFormat::Png, "example.png").unwrap();
}
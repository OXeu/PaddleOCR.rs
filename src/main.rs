use std::{cmp, fs};
use std::borrow::Borrow;
use std::fmt::{Debug, Display};
use std::fs::File;
#[cfg(feature = "dbg")]
use std::io::BufWriter;
use std::io::Read;
use std::string::String;
use std::sync::{Mutex, RwLock};

#[cfg(feature = "dbg")]
use image::{GenericImage, ImageBuffer, ImageOutputFormat, Rgb};
use image::io::Reader as ImageReader;
use image::RgbImage;
#[cfg(feature = "dbg")]
use imageproc::drawing::draw_filled_rect_mut;
use tract_onnx::prelude::*;
use tract_onnx::prelude::tract_itertools::Itertools;
use tract_onnx::prelude::tract_num_traits::abs;

use crate::benchmark::Timer;
use crate::dot2rect::{Point, wrapped_rect};

mod dot2rect;
mod benchmark;

const WIDTH: u32 = 96 * 2;
const HEIGHT: u32 = WIDTH;
const MAX_H_PARTS: u32 = 3;
// Actual parts: MAX_H_PARTS * MAX_V_PARTS
const MAX_V_PARTS: u32 = MAX_H_PARTS;

const PADDING_X: u32 = 8;
const PADDING_Y: u32 = PADDING_X;
const NEIGHBOR_STEP: i32 = 1;
const DETECT_AREA_EXPAND_PADDING_X: u32 = 3;
const DETECT_AREA_EXPAND_PADDING_Y: u32 = 1;

#[tokio::main]
async fn main() -> TractResult<()> {
    let mut timer = Timer::new();
    let image_path = "example/test.jpeg";
    let rec_model = onnx()
        // load the model
        .model_for_path("model/rec.onnx")?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;
    let rec_model = Arc::new(rec_model);
    let det_model = onnx()
        // load the model
        .model_for_path("model/model.onnx")?
        // optimize the model
        .into_compact()? //todo: it will panic.
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;
    let det_model = Arc::new(det_model);
    timer.tick();
    println!("Model loaded");

    let _ = fs::remove_dir_all("output");
    let _ = fs::create_dir("output");
    timer.tick();
    println!("Cleaned output folder");

    let mut buf = String::new();
    File::open("model/label_list.txt").unwrap().read_to_string(&mut buf).expect("Unable to read label");
    let label: Vec<String> = buf.split("\n").map(|v| v.to_string()).collect();
    let label = Arc::new(label);
    timer.tick();
    println!("Label loaded");

    let image = Arc::new(ImageReader::open(image_path)?.decode()?.to_rgb8());
    timer.tick();
    println!("Image loaded!");

    let (width, height) = (image.width(), image.height());
    let real_padding_x = PADDING_X as f32 / WIDTH as f32 * width as f32;
    let real_padding_y = PADDING_Y as f32 / HEIGHT as f32 * height as f32;
    let (w_parts, h_parts) = (cmp::min(MAX_H_PARTS, (width + WIDTH - 1) / WIDTH), cmp::min(MAX_V_PARTS, (height + HEIGHT - 1) / HEIGHT));
    let points: Arc<RwLock<Vec<(u32, u32)>>> = Arc::new(RwLock::new(Vec::new()));
    let mut handlers = vec![];
    for w_part_i in 0..w_parts {
        for h_part_i in 0..h_parts {
            let points = points.clone();
            let image = image.clone();
            let det_model = det_model.clone();
            let handler = tokio::spawn(async move {
                let w = width as f32 / w_parts as f32;
                let h = height as f32 / h_parts as f32;
                let (mut w_pad,mut h_pad) = (w + 2f32 * real_padding_x, h + 2f32 * real_padding_y);
                let x = w_part_i as f32 * w;
                let y = h_part_i as f32 * h;
                let left_pad = if x < real_padding_x { 0f32 } else { real_padding_x };
                let top_pad = if y < real_padding_y { 0f32 } else { real_padding_y };
                let x_pad = (x - real_padding_x).max(0f32);
                let y_pad = (y - real_padding_y).max(0f32);
                if x_pad + w_pad > width as f32 {
                    w_pad = width as f32 - x_pad;
                }
                if y_pad + h_pad > height as f32 {
                    h_pad = height as f32 - y_pad;
                }
                let w_scale = WIDTH as f32 / width as f32;
                let h_scale = HEIGHT as f32 / height as f32;
                let image_cropped = image::imageops::crop_imm(image.as_ref(), x_pad as u32, y_pad as u32, w_pad as u32, h_pad as u32).to_image();
                let resized =
                    image::imageops::resize(&image_cropped, WIDTH, HEIGHT, image::imageops::FilterType::Triangle);
                let image_tensor: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, HEIGHT as usize, HEIGHT as usize), |(_, c, y, x)| {
                    let mean = [0.485, 0.456, 0.406][c];
                    let std = [0.229, 0.224, 0.225][c];
                    (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
                })
                    .into();
                let result = det_model.run(tvec!(image_tensor.into())).unwrap();
                let binding: &[f32] = result[0].as_slice().unwrap();
                let iter = binding.iter()
                    .cloned();
                let mut count = 0u32;
                for chunk in &iter.chunks(WIDTH as usize) {
                    let best = chunk
                        .collect::<Vec<_>>();
                    for (i, v) in best.iter().enumerate() {
                        if *v > 0.5 {
                            let mut point = points.write().unwrap();
                            let p_x = ((i as f32 * w_pad / width as f32 + (x - left_pad) * w_scale) as u32).min(WIDTH - 1);
                            let p_y = ((count as f32 * h_pad / height as f32 + (y - top_pad) * h_scale) as u32).min(HEIGHT - 1);
                            point.push((p_x, p_y))
                        }
                    }
                    count += 1;
                }
            });
            handlers.push(handler);
        }
    }
    for handle in handlers {
        handle.await.unwrap();
    }
    timer.tick();
    println!("Text Detected!");

    let points_read = points.read().unwrap();
    let points: Vec<_> = points_read.iter().map(|(x, y)| Point { x: *x, y: *y }).collect();
    let connected_components = dot2rect::connected_components(points.clone(), NEIGHBOR_STEP);
    timer.tick();
    println!("Points to Rects");


    #[cfg(feature = "dbg")] let mut output_image = ImageBuffer::new(width, image.height());
    #[cfg(feature = "dbg")]
    output_image.copy_from(image.as_ref(), 0, 0).expect("Failed to copy background image");
    #[cfg(feature = "dbg")]
        let rect = Arc::new(Mutex::new(Vec::new()));

    let mut handlers = vec![];
    let mut counter = 0;
    for component in connected_components {
        let image_cloned = image.clone();
        let label_cloned = label.clone();
        let rec_model_cloned = rec_model.clone();
        #[cfg(feature = "dbg")]
            let rect_clone = rect.clone();
        counter += 1;
        let handler = tokio::spawn(async move {
            if let Some(mut rect) = wrapped_rect(component) {
                let rect_c = rect.clone();
                rect.remap(width, height);
                let sub_image = image::imageops::crop_imm(image_cloned.as_ref(), rect.x, rect.y, rect.width, rect.height).to_image();
                #[cfg(feature = "dbg")]{
                    let mut rect_mut = rect_clone.lock().unwrap();
                    rect_mut.push(rect.clone());
                    let mut buf = BufWriter::new(File::create(format!("output/{counter}.png")).unwrap());
                    sub_image.write_to(&mut buf, ImageOutputFormat::Png).expect(format!("{rect_c:?},{rect:?}").as_str());
                }
                rec(&rec_model_cloned, &sub_image, &label_cloned).unwrap();
            }
        });
        handlers.push(handler);
    }
    for handle in handlers {
        handle.await.unwrap();
    }
    timer.tick();
    println!("Recognized");

    #[cfg(feature = "dbg")]{
        let rect_read = rect.lock().unwrap();
        for rect in rect_read.iter() {
            let rect_sr = imageproc::rect::Rect::at(rect.x as i32, rect.y as i32).of_size(rect.width, rect.height);
            let fill_color = Rgb([255, 0, 0]); // 填充颜色为红色
            draw_filled_rect_mut(&mut output_image, rect_sr, fill_color);
        }
        output_image.save("render.png").expect("Failed to save output image");
        timer.tick();
        println!("Detected Area Rendered");
    }
    timer.all();
    Ok(())
}

fn rec<F, O, M>(model: &RunnableModel<F, O, M>, image: &RgbImage, label: &Vec<String>) -> TractResult<String>
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
            let repeat_not_remove = text == last_text && abs((value - last_probability) / min) < 0.07 && value > 0.85 && last_probability > 0.85;
            #[cfg(feature = "dbg")]
            if repeat_not_remove {
                print!("({})({},{}->{})", text, last_probability, value, abs(value - last_probability) / min)
            }
            if text != last_text || repeat_not_remove || last_probability == 0f32 {
                rec += text.as_str();
            }
            last_text = text;
            last_probability = value;
        }
    }
    #[cfg(feature = "dbg")]
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
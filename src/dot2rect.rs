use std::cmp;

use crate::{DETECT_AREA_EXPAND_PADDING_X, DETECT_AREA_EXPAND_PADDING_Y, HEIGHT, WIDTH};

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub(crate) x: u32,
    pub(crate) y: u32,
}

#[derive(Debug, Clone)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub fn remap(&mut self, width: u32, height: u32) {
        assert_ne!(width, 0);
        assert_ne!(height, 0);
        let w_scale = width as f64 / WIDTH as f64;
        let h_scale = height as f64 / HEIGHT as f64;
        self.x = (self.x as f64 * w_scale) as u32;
        self.y = (self.y as f64 * h_scale) as u32;
        // self.y = height - self.y;
        self.width = (self.width as f64 * w_scale) as u32;
        self.height = (self.height as f64 * h_scale) as u32;
    }
}

fn is_valid_point(point: Point, width: u32, height: u32) -> bool {
    point.x < width && point.y < height
}

fn get_neighbors(point: Point, step: i32) -> Vec<Point> {
    let mut neighbors = vec![];
    for i in -step..step + 1 {
        for j in -step..step + 1 {
            if i != 0 || j != 0 {
                let x = point.x as i32 + i;
                let y = point.y as i32 + j;
                if x > 0 && y > 0 {
                    neighbors.push(Point { x: x as u32, y: y as u32 });
                }
            }
        }
    }
    neighbors
}

fn dfs(grid: &mut Vec<Vec<u32>>, point: Point, connected: &mut Vec<Point>, step: i32) {
    let width = grid[0].len() as u32;
    let height = grid.len() as u32;

    if !is_valid_point(point, width, height) || grid[point.y as usize][point.x as usize] != 1 {
        return;
    }

    connected.push(point);
    grid[point.y as usize][point.x as usize] = 0;

    let neighbors = get_neighbors(point, step);
    for neighbor in neighbors {
        dfs(grid, neighbor, connected, step);
    }
}

pub fn connected_components(points: Vec<Point>, step: i32) -> Vec<Vec<Point>> {
    let mut grid = vec![vec![0; WIDTH as usize]; HEIGHT as usize]; // 假设平面大小为 10x10，可以根据实际情况调整
    let mut result = Vec::new();

    for point in points {
        grid[point.y as usize][point.x as usize] = 1;
    }

    for y in 0..grid.len() {
        for x in 0..grid[y].len() {
            if grid[y][x] == 1 {
                let mut connected = Vec::new();
                dfs(&mut grid, Point { x: x as u32, y: y as u32 }, &mut connected, step);
                result.push(connected);
            }
        }
    }

    result
}


pub fn wrapped_rect(points: Vec<Point>) -> Option<Rect> {
    let (mut max_x, mut max_y) = (u32::MIN, u32::MIN);
    let (mut min_x, mut min_y) = (u32::MAX, u32::MAX);
    for p in points {
        if p.x < min_x {
            min_x = p.x;
        }
        if p.x > max_x {
            max_x = p.x;
        }
        if p.y < min_y {
            min_y = p.y;
        }
        if p.y > max_y {
            max_y = p.y;
        }
    }
    let (w, h) = (max_x - min_x, max_y - min_y);
    if w <= 0 || h <= 0 {
        None
    } else {
        let x = (min_x - DETECT_AREA_EXPAND_PADDING_X).max(0);
        let y = (min_y - DETECT_AREA_EXPAND_PADDING_Y).max(0);
        let w = cmp::min(WIDTH, w + 2 * DETECT_AREA_EXPAND_PADDING_X);
        let h = cmp::min(HEIGHT, h + 2 * DETECT_AREA_EXPAND_PADDING_Y);
        Some(
            Rect {
                x,
                y,
                width: w,
                height: h,
            }
        )
    }
}

#[test]
fn test() {
    let points = vec![
        Point { x: 1, y: 1 },
        Point { x: 2, y: 1 },
        Point { x: 2, y: 2 },
        Point { x: 3, y: 4 },
        Point { x: 4, y: 4 },
    ];

    let connected_components = connected_components(points, 2);
    for component in connected_components {
        println!("{:?}", component);
    }
}


#[test]
fn num_test() {
    let height = 4444u32;
    let small_height: u32 = 224;
    let h_scale = height as f64 / small_height as f64;
    let y = (5u32 as f64 * h_scale) as u32;
    println!("{y}")
}
use chrono::Local;

pub struct Timer {
    pub start: i64,
    pub time: i64,
}

impl Timer {
    pub fn new() -> Timer {
        let now = Local::now().timestamp_millis();
        Timer {
            start: now,
            time: now,
        }
    }
    pub fn tick(&mut self) {
        let now = Local::now().timestamp_millis();
        let diff = now - self.time;
        print!("[+] Spent: \t{diff} ms\t\t");
        self.time = now;
    }

    pub fn all(&self) {
        let now = Local::now().timestamp_millis();
        let diff = now - self.start;
        println!("[+] All Task Spent: \t\t{diff} ms");
    }
}
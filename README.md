# PaddleOCR.rs

PaddleOCR's Rust inference framework

## Features

- Not depend on OpenCV
- Small binary size(Only 8.3MB on Linux)
- Pure CPU inference
- Multi-threads text detection & recognitions

```shell
[+] Spent:      276 ms          Model loaded
[+] Spent:      2 ms            Cleaned output folder
[+] Spent:      0 ms            Label loaded
[+] Spent:      147 ms          Image loaded!
[+] Spent:      1333 ms         Text Detected!
[+] Spent:      2 ms            Points to Rects
[+] Spent:      1368 ms         Recognized
[+] All Task Spent:             3128 ms
```
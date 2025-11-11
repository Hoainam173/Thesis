Absolutely! I can reformat your text into a **well-structured, clean, and visually appealing README** suitable for GitHub. Here’s a polished version ready to paste into your `README.md`:

---

# 🔬 Polymer Fiber Quality Diagnosis Using Computer Vision

This project implements an **automated system for monitoring polymer fiber quality** using high-resolution camera images and advanced image processing techniques. The system measures fiber diameter and length, generates real-time statistics, and visualizes fiber uniformity to support **quality control in polymer production**.

---

## 1. Overview

Consistent fiber diameter and length are crucial parameters in polymer production. Traditional manual measurements are **labor-intensive, time-consuming, and prone to error**.

This system leverages **computer vision** and **real-time data processing** to automatically:

* Capture high-resolution images of fibers at regular intervals
* Detect and measure individual fibers’ **diameter** and **length**
* Generate **statistical summaries** and **visualizations**
* Log results for **traceability** and analysis

---

## 2. Key Features

* **Automated Image Capture:** Configurable intervals (default 30 seconds)
* **Camera Calibration:** Chessboard-based calibration to correct lens distortion and calculate real-world scale (mm/pixel)
* **Image Preprocessing Pipeline:**

  * Grayscale conversion
  * Gaussian blur
  * Adaptive thresholding
  * Morphological operations
* **Fiber Detection & Measurement:**

  * Contour-based detection
  * Diameter estimation via distance transform
  * Length estimation from contour perimeter
* **Data Logging:**

  * Annotated images
  * Zoomed ROI images of each fiber
  * CSV files with timestamps
* **Graphical User Interface (GUI):**

  * Displays average, minimum, and maximum fiber diameter
  * Shows fiber count and quality assessment
  * Updates a cumulative histogram of fiber diameters

---

## 3. Methodology

The workflow is divided into several steps:

### 3.1 Camera Calibration

* Capture chessboard images for **distortion correction**
* Compute **camera matrix** and **distortion coefficients**
* Derive **scale factor** (mm per pixel) for accurate measurement

### 3.2 Image Acquisition

* Capture high-resolution images of the fiber stream at fixed intervals
* Save **raw images** for reference and validation

### 3.3 Image Preprocessing

* Correct lens distortion using calibration data
* Convert image to grayscale and apply Gaussian blur
* Use adaptive thresholding to segment fibers
* Apply morphological closing to remove small gaps

### 3.4 Fiber Detection & Measurement

* Extract contours from binary images
* Calculate fiber **length** using contour perimeter
* Compute fiber **diameter** using distance transform
* Filter out fibers that do not meet **minimum length**, **maximum diameter**, or **area thresholds**

### 3.5 Data Logging & Visualization

* Save annotated images and ROI images of each fiber
* Record measurements in **CSV files with timestamps**
* Update GUI in real-time: average, min, max fiber diameter and fiber count
* Plot **cumulative histogram** showing fiber diameter distribution over all captures

### 3.6 Quality Assessment

Evaluate fiber uniformity based on **diameter deviation**:

| Deviation | Assessment                          |
| --------- | ----------------------------------- |
| Low       | Fibers are uniform and high quality |
| Moderate  | Acceptable, monitor production      |
| High      | Potential issues in fiber extrusion |

### 3.7 Automation & Multithreading

* Continuous monitoring with **separate thread** for image capture and processing
* GUI remains **responsive** for real-time updates
* Program can be terminated using GUI or keyboard input

---

## 4. Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/polymer-fiber-quality.git
cd polymer-fiber-quality
```

2. Install required dependencies:

```bash
pip install opencv-python numpy pandas matplotlib
```

3. Prepare chessboard images for calibration:

* Place `.jpg` or `.png` images in the `chessboard_images/` folder

---

## 5. Usage

Run the main program:

```bash
python test2.py
```

* GUI will display **real-time fiber statistics**
* Press `q` in the OpenCV window to **stop the program**
* All results, images, and logs are saved in the `fiber_diameter/` folder

---

## 6. Results

* **Real-Time Statistics:** Average, minimum, maximum diameter, fiber count, and quality assessment
* **Annotated Images:** Visualizes detected fibers with IDs, length, and diameter
* **Zoomed ROI Images:** Saved for detailed fiber inspection
* **Cumulative Histogram:** Displays the distribution of fiber diameters across all captures, enabling quick evaluation of production consistency

---

## 7. License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---




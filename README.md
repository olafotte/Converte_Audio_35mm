# Converte_Audio_35mm

This project is a complete toolset for scanning 35mm film, processing images, and extracting optical sound tracks. It features a graphical user interface (GUI) built with `customtkinter` to orchestrate the entire workflow from raw images to a final video with synchronized audio.

## Features

- **GUI Control Panel:** A user-friendly interface (`Scanner 35mm Ultra Sound V4.py`) to manage the scanning and conversion process.
- **ROI Setup:** Define Regions of Interest for tracking and audio extraction (`setup_rois.py`).
- **Geometric Tracking:** Image stabilization and tracking (`tracking.py`).
- **Audio Generation:** Extraction of optical soundtracks from the scanned film images (`Sound_generation.py`).
- **Frequency Analysis & Noise Detection:** Tools to analyze and clean up the audio signal (`analizador_de_frequencias.py`, `detect_transition.py`).
- **Image Colorization:** Tools for converting black and white film scans to color using a deep learning Caffe model (`converte_bw_to_color.py`).
- **Video & Audio Merging:** Combines the processed image sequence and generated audio into a final movie file (`Merge_audio_and_video.py`).

## Requirements

- Python 3.x
- `customtkinter`
- `tkinter` (usually included with Python)
- `ffmpeg` (Required for video generation. Make sure the path is configured correctly in the GUI or system PATH).
- Other computer vision and audio processing libraries (e.g., `opencv-python`, `numpy`, `scipy` - check imports in the scripts for exact requirements).

Additonal packaged needed
colorization_release_v2.caffemodel
colorization_deploy_v2.prototxt
pts_in_hull.npy

## Usage

1. **Install dependencies:**
   Make sure to install the required Python packages. You can typically do this using pip:
   ```bash
   pip install customtkinter opencv-python numpy scipy
   ```
2. **Configure FFmpeg:**
   Ensure FFmpeg is installed on your system. The GUI allows you to specify the path to your `ffmpeg.exe` if it's not in your system PATH.
3. **Run the Application:**
   Start the main GUI by running:
   ```bash
   python "Scanner 35mm Ultra Sound V4.py"
   ```
4. **Workflow:**
   - Select the folder containing your scanned 35mm image sequence.
   - Adjust FPS, rotation, and other settings in the panel.
   - Execute the steps in order: ROI setup, Tracking, Sound Generation, Colorization (optional), and Merging.

## Project Structure

- `Scanner 35mm Ultra Sound V4.py`: Main application GUI.
- `setup_rois.py`: Script for configuring regions of interest.
- `tracking.py`: Logic for image tracking and stabilization.
- `Sound_generation.py`: Core logic for optical sound synthesis.
- `analizador_de_frequencias.py`: Audio frequency analysis.
- `detect_transition.py`: Jump/transition detection.
- `converte_bw_to_color.py`: Script for colorizing black and white images.
- `colorization_deploy_v2.prototxt`: Network configuration for the colorization model.
- `colorization_release_v2.caffemodel`: Pre-trained weights for the colorization model.
- `pts_in_hull.npy`: Cluster centers for the colorization model.
- `Merge_audio_and_video.py`: Final multiplexing of video and audio.
- `config_rois.json` & `tracking_refined.json`: Configuration data files.

## License
This project is proprietary and confidential. All rights are reserved. You may not copy, distribute, modify, or use this software without explicit permission from the author.
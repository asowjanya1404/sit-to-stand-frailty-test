# 30-Second Chair Stand Frailty Test

An automated computer vision system for conducting the CDC STEADI 30-Second Chair Stand Test using pose estimation.

## Overview

This project automates the clinical assessment of lower body strength and fall risk in older adults using MediaPipe pose estimation and OpenCV.

## Features

- ✅ Automated sit-to-stand counting
- ✅ Real-time visual feedback
- ✅ Arm usage violation detection
- ✅ CDC STEADI compliant scoring
- ✅ Instant risk assessment reports
- ✅ Works with standard webcam
- ✅ Automatic test start after proper seating  
- ✅ On-screen cues and instructions before and during the test  
- ✅ Countdown timers to guide the user  


## Requirements

- Python 3.7+
- Webcam
- See `requirements.txt` for dependencies

## How to Run

1. Open a terminal in your project folder.  
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the test:
   ```bash
   python sit_to_stand_test.py
   ```

## User Instructions

1. Sit in the middle of a sturdy chair.
2. Cross your arms on your chest (hands on opposite shoulders).
3. Keep your feet flat on the floor.
4. The test will start automatically once seated correctly for a few seconds.
5. Stand up and sit down 

**without using your arms**.

6. If you use your arms, the test will stop and record a score of 0.
7. Press 'Q' at any time to quit.

### Key Enhancements in This Version

- Automatic test start after detecting proper seated posture.
- On-screen countdown before test begins.
- Visual prompts guiding correct positioning (arms crossed, feet flat, etc.).
- Automatic detection of arm usage and test stoppage if violated.
- Clean display without body landmarks overlay (optional for better visuals).

## Notes

- The test is based on the CDC STEADI 30-Second Chair Stand Test.
- Below-average scores indicate increased fall risk.
- Ensure proper lighting and webcam positioning for accurate tracking.

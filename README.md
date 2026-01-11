# 30-Second Chair Stand Frailty Test

An automated computer vision system for conducting the CDC STEADI 30-Second Chair Stand Test using pose estimation.

## Overview

This project automates the clinical assessment of lower body strength and fall risk in older adults using MediaPipe pose estimation and OpenCV.

## Features

- ✅ Automated sit-to-stand counting
- ✅ Real-time visual feedback
- ✅ Arm usage violation detection
- ✅ CDC STEADI compliant scoring
- ✅ Risk assessment generated only for successfully completed tests
- ✅ Works with standard webcam
- ✅ Automatic test start after proper seating  
- ✅ On-screen cues and instructions before and during the test  
- ✅ Countdown timers to guide the user 

## Metrics Used

- Number of valid sit-to-stand repetitions completed within 30 seconds
- Total test duration (fixed at 30 seconds)
- Posture detection using pose landmarks (seated vs standing)
- Arm usage detection (arms crossed vs arms used for support)
- Test start validation based on correct seated posture

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
5. Stand up and sit down repeatedly for 30 seconds **without using your arms**.
6. If you use your arms, the test will stop and be marked as **Test Failed** (no risk assessment is generated).
7. Press 'Q' at any time to quit.

## Passing and Failing Conditions

### Passing Conditions
- User completes one or more valid sit-to-stand repetitions
- Arms remain crossed throughout the test
- User stays within the camera frame
- Test runs uninterrupted for the full duration

### Failing / Interrupted Conditions
- User uses arms for support at any point during the test
- User leaves the camera frame during the test
- Test is manually exited using the 'Q' key
- Test is interrupted before completion

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

## Test Flow & Result Logic

The application follows CDC STEADI guidelines for administering the 30-Second Chair Stand Test.

### Test Start
- The test begins only when the user is seated correctly with arms crossed.
- A 3–2–1 countdown is displayed before the timer starts.

### Test Completion
- The test runs for a fixed duration of 30 seconds.
- If the user completes the full duration without protocol violations, the test is marked as **Test Complete**.

### Test Failure / Interruption
- If the user uses their arms for support at any point, the test is immediately stopped.
- Interrupted tests are marked as **Test Failed**.

### Risk Assessment Logic
- Fall risk assessment is calculated **only when the test completes the full 30 seconds**.
- If the test fails or is interrupted, no risk level is calculated or displayed.
- Failed tests show **“Risk Level: N/A (Test not completed)”**.

This prevents incorrect fall-risk interpretation from incomplete or invalid test results.


import cv2
import mediapipe as mp
import time
import numpy as np

class SitToStandCounter:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose  
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,   # process video, not images
            model_complexity=0,        # lightweight model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        
        # Counter variables
        self.stand_count = 0
        self.current_state = "sitting"  # Start with sitting position
        self.test_duration = 30  # seconds
        self.start_time = None
        self.test_started = False
        self.test_stopped = False  # Flag for arm violation
        self.last_change_time = 0
        self.COOLDOWN = 0.8  # seconds to prevent double-counting
        
        # Arm position tracking
        self.arm_violation_count = 0
        self.arm_violation_threshold = 15  # Number of consecutive frames to confirm violation
        
        # CDC STEADI scoring norms (below average scores)
        self.scoring_norms = {
            'men': {
                '60-64': 14,
                '65-69': 12,
                '70-74': 12,
                '75-79': 11,
                '80-84': 10,
                '85-89': 8,
                '90-94': 7
            },
            'women': {
                '60-64': 12,
                '65-69': 11,
                '70-74': 10,
                '75-79': 10,
                '80-84': 9,
                '85-89': 8,
                '90-94': 4
            }
        }
        # Auto-start variables
        self.seated_time = None
        self.auto_start_enabled = True
        self.countdown_done = False
        self.countdown_start = None
        self.countdown_duration = 3  # seconds
        self.state_buffer = []
        self.required_frames = 10
        self.get_ready_start = None
        self.full_stand_reached = False

        # -------- SESSION DATA STORAGE --------
        self.session_history = []
        self.session_id = 1

    def is_user_seated(self, landmarks):
        """Detect if user is seated based on hip and knee positions."""
        try:
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y

            # Average hip & knee positions
            hip_y = (left_hip + right_hip) / 2
            knee_y = (left_knee + right_knee) / 2

            # Sitting posture → knees LOWER than hips by a margin
            return knee_y > hip_y + 0.03

        except:
            return False
        
    def get_age_range(self, age):
        """Determine age range category"""
        if 60 <= age <= 64:
            return '60-64'
        elif 65 <= age <= 69:
            return '65-69'
        elif 70 <= age <= 74:
            return '70-74'
        elif 75 <= age <= 79:
            return '75-79'
        elif 80 <= age <= 84:
            return '80-84'
        elif 85 <= age <= 89:
            return '85-89'
        elif 90 <= age <= 94:
            return '90-94'
        else:
            return None
    
    def evaluate_score(self, count, age, gender):
        """
        Evaluate the score based on CDC STEADI norms
        Returns assessment and risk level
        """
        age_range = self.get_age_range(age)
        
        if age_range is None:
            return {
                'score': count,
                'age_range': 'Outside 60-94 range',
                'assessment': 'No norms available for this age',
                'risk_level': 'N/A'
            }
        
        gender_lower = gender.lower()
        if gender_lower not in ['men', 'women', 'male', 'female']:
            return {
                'score': count,
                'age_range': age_range,
                'assessment': 'Invalid gender specified',
                'risk_level': 'N/A'
            }
        
        # Normalize gender input
        if gender_lower in ['male', 'men']:
            gender_key = 'men'
        else:
            gender_key = 'women'
        
        threshold = self.scoring_norms[gender_key][age_range]
        
        if count < threshold:
            assessment = "BELOW AVERAGE - Indicates risk for falls"
            risk_level = "HIGH RISK"
        else:
            assessment = "AVERAGE or ABOVE - Good leg strength and endurance"
            risk_level = "LOW RISK"
        
        return {
            'score': count,
            'age_range': age_range,
            'threshold': threshold,
            'assessment': assessment,
            'risk_level': risk_level,
            'gender': gender_key.capitalize()
        }
    
    def print_final_report(self, count, age, gender, arm_violation=False):
        """Print formatted final report with scoring"""
        print("\n" + "="*70)
        print(" "*20 + "30-SECOND CHAIR STAND TEST")
        print(" "*25 + "FINAL RESULTS")
        print("="*70)
        
        result = self.evaluate_score(count, age, gender)
        
        print(f"\nPatient Information:")
        print(f"  Age: {age} years")
        print(f"  Gender: {result.get('gender', gender)}")
        print(f"  Age Range: {result['age_range']}")
        
        print(f"\nTest Results:")
        if arm_violation:
            print(f"  TEST STOPPED - Patient used arms to stand")
            print(f"  Total Stands: 0 (Protocol Violation)")
            print(f"\n  ⚠️  According to CDC STEADI protocol:")
            print(f"     'If the patient must use his/her arms to stand,")
            print(f"      stop the test. Record 0 for the number and score.'")
        else:
            print(f"  Total Stands in 30 seconds: {result['score']}")
            
            if 'threshold' in result:
                print(f"  Below Average Threshold: < {result['threshold']}")
            
            print(f"\nAssessment:")
            print(f"  {result['assessment']}")
            print(f"\nRisk Level: {result['risk_level']}")
        
        print("\n" + "-"*70)
        print("Reference: CDC STEADI - Stopping Elderly Accidents, Deaths & Injuries")
        print("Note: A below average score indicates a risk for falls.")
        print("="*70 + "\n")
        
        return result
    
    def record_session(self, age, gender, count, result, arm_violation):
        session_data = {
            "session_id": self.session_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "age": age,
            "gender": gender,
            "count": count,
            "arm_violation": arm_violation,
            "assessment": result.get("assessment", "N/A"),
            "risk_level": result.get("risk_level", "N/A")
        }
        self.session_history.append(session_data)
        self.session_id += 1


    def display_session_history(self):
        print("\n" + "=" * 70)
        print("SESSION HISTORY")
        print("=" * 70)

        for s in self.session_history:
            print(
                f"Session {s['session_id']} | {s['timestamp']} | "
                f"Age: {s['age']} | Gender: {s['gender']} | "
                f"Count: {s['count']} | "
                f"Arm Violation: {'YES' if s['arm_violation'] else 'NO'} | "
                f"Risk: {s['risk_level']}"
            )

        print("=" * 70 + "\n")


    def prompt_retry_or_quit(self):
        while True:
            choice = input("Do you want to Retry (R) or Quit (Q)? ").strip().lower()
            if choice == 'r':
                return True
            elif choice == 'q':
                return False
            else:
                print("Invalid input. Enter R or Q.")


    def reset_test_state(self):
        self.stand_count = 0
        self.current_state = "sitting"
        self.start_time = None
        self.test_started = False
        self.test_stopped = False
        self.arm_violation_count = 0
        self.last_change_time = 0
        self.seated_time = None
        self.state_buffer.clear()
        self.full_stand_reached = False

    
    def check_arm_usage(self, landmarks):
        """
        Check if patient is using arms to help stand up
        Arms should be crossed on chest - check if hands move away from shoulders
        Returns: True if arm violation detected, False otherwise
        """
        try:
            # Get key landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Calculate distances
            # In correct position, wrists should be near opposite shoulders (arms crossed)
            left_wrist_to_right_shoulder = np.sqrt(
                (left_wrist.x - right_shoulder.x)**2 + 
                (left_wrist.y - right_shoulder.y)**2
            )
            
            right_wrist_to_left_shoulder = np.sqrt(
                (right_wrist.x - left_shoulder.x)**2 + 
                (right_wrist.y - left_shoulder.y)**2
            )
            
            # Check if wrists are below hips (indicating pushing off chair or using armrests)
            left_wrist_below_hip = left_wrist.y > left_hip.y
            right_wrist_below_hip = right_wrist.y > right_hip.y
            
            # Calculate shoulder width for normalization
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            
            # Normalized thresholds
            max_wrist_distance = shoulder_width * 1.8  # Wrists should stay relatively close to chest
            
            # Violation occurs if:
            # 1. Wrists are too far from shoulders (arms uncrossed)
            # 2. OR wrists are below hips (pushing off)
            arm_violation = (
                (left_wrist_to_right_shoulder > max_wrist_distance or 
                 right_wrist_to_left_shoulder > max_wrist_distance) or
                (left_wrist_below_hip or right_wrist_below_hip)
            )
            
            return arm_violation
            
        except Exception as e:
            print(f"Error checking arm usage: {e}")
            return False
        
    def are_arms_crossed(self, landmarks):
        """
          Strict check: wrists must be close to opposite shoulders
        """
        try:
           left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
           right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
           left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
           right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

           # Shoulder width for normalization
           shoulder_width = abs(left_shoulder.x - right_shoulder.x)

           # Distances
           lw_to_rs = np.sqrt(
              (left_wrist.x - right_shoulder.x)**2 +
              (left_wrist.y - right_shoulder.y)**2
           )

           rw_to_ls = np.sqrt(
               (right_wrist.x - left_shoulder.x)**2 +
               (right_wrist.y - left_shoulder.y)**2
            )

           # Threshold: wrists must be close to opposite shoulders
           max_dist = shoulder_width * 1.2

           return lw_to_rs < max_dist and rw_to_ls < max_dist

        except:
            return False

    def calculate_body_posture(self, landmarks):
        try:
            shoulder_y = (
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ) / 2

            hip_y = (
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y +
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ) / 2

            knee_y = (
                 landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y +
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            ) / 2

            # Normalized distances
            torso_len = abs(shoulder_y - hip_y)
            hip_to_knee = abs(hip_y - knee_y)

            if torso_len == 0:
                return self.current_state

            posture_score = hip_to_knee / torso_len

            # DEBUG print(f"[DEBUG] posture_score={posture_score:.2f}")

            if posture_score < 0.55:
                return "sitting"
            elif posture_score > 0.70:
                return "standing"
            else:
                return "transition"

        except:
            return self.current_state
        
    def is_fully_standing(self, landmarks):
        hip_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y +
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
        knee_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y +
                  landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y) / 2
        shoulder_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2

        torso_len = abs(shoulder_y - hip_y)
        return hip_y < knee_y - (torso_len * 0.3)  # Adjust threshold based on torso

    def update_count(self, new_state, landmarks):
        current_time = time.time()

        # Skip transition states
        if new_state == "transition":
            return

        # ---------- Detect full stand ----------
        if new_state == "standing" and self.current_state != "standing":
            if self.is_fully_standing(landmarks) and current_time - self.last_change_time > self.COOLDOWN:
                self.current_state = "standing"
                if not self.full_stand_reached:   # Only mark first time
                    self.full_stand_reached = True
                    print("[DEBUG] Full stand confirmed")
                self.last_change_time = current_time

        # ---------- Detect sitting after standing ----------
        if new_state == "sitting" and self.current_state == "standing" and self.full_stand_reached:
            if current_time - self.last_change_time > self.COOLDOWN:
                self.stand_count += 1
                self.current_state = "sitting"
                self.full_stand_reached = False
                self.last_change_time = current_time
                print(f"✅ Rep completed! Count = {self.stand_count}")


    def get_patient_info(self):
        """Get patient age and gender for scoring"""
        print("\n" + "="*70)
        print("Please enter patient information for scoring:")
        print("="*70)
        
        while True:
            try:
                age = int(input("Enter patient age (60-94): "))
                if age < 0:
                    print("Age must be positive. Please try again.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        while True:
            gender = input("Enter patient gender (Male/Female or M/F): ").strip().lower()
            if gender in ['male', 'female', 'm', 'f', 'men', 'women']:
                if gender == 'm':
                    gender = 'male'
                elif gender == 'f':
                    gender = 'female'
                break
            else:
                print("Invalid input. Please enter Male/Female or M/F.")
        
        return age, gender
    
    def run_test(self):
        """Main function to run the 30-second chair stand test with retry loop"""
    
        age, gender = self.get_patient_info()

        while True:  # Retry loop
             
            frame_skip = 2
            frame_count = 0

            # Auto Select Camera
            for i in range(5):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    print(f"Camera found at index {i}")
                    break
                cap.release()
            else:
                print("No camera found in indices 0-4")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            cv2.namedWindow("30-Second Chair Stand Test", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("30-Second Chair Stand Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # -------- WAIT FOR USER TO SIT PROPERLY --------
            start_auto = False
            self.reset_test_state()
            while not start_auto:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                seated = arms_crossed = knees_visible = False
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    seated = self.is_user_seated(landmarks)
                    arms_crossed = self.are_arms_crossed(landmarks)
                    try:
                       lk = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
                       rk = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
                       knees_visible = lk > 0.6 and rk > 0.6
                    except:
                        knees_visible = False

                # Instructions
                cv2.putText(frame, "30-SECOND CHAIR STAND TEST", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, "Get into starting position:", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                y = 140
                if not seated:
                    cv2.putText(frame, "• Please sit properly on the chair", (30, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y += 40
                if seated and not arms_crossed:
                    cv2.putText(frame, "• Please cross your arms on your chest", (30, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y += 40
                if seated and not knees_visible:
                    cv2.putText(frame, "• Ensure knees are visible to camera", (30, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y += 40
                if seated and arms_crossed and knees_visible:
                    cv2.putText(frame, "✓ Good position detected. Starting test...", (30, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                    start_auto = True
                    self.countdown_start = time.time()

                cv2.imshow("30-Second Chair Stand Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    self.pose.close()
                    return

            # -------- COUNTDOWN BEFORE TEST --------
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                countdown_elapsed = time.time() - self.countdown_start
                remaining_count = max(0, self.countdown_duration - int(countdown_elapsed) - 1)  # FIX
    
                cv2.putText(frame, f"Get Ready: {remaining_count+1}", (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

                cv2.imshow("30-Second Chair Stand Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    self.pose.close()
                    return
                
                if countdown_elapsed >= self.countdown_duration:
                    # Countdown done → start the test
                    self.start_time = time.time()   # <-- start 30s timer here
                    self.test_started = True
                    break

             # -------- MAIN TEST LOOP --------
            while self.test_started and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                   continue
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                if not results.pose_landmarks:
                    cv2.imshow("30-Second Chair Stand Test", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                landmarks = results.pose_landmarks.landmark

                # Always check arm violation during the test
                arm_violation = self.check_arm_usage(landmarks)
                if arm_violation:
                    self.arm_violation_count += 1
                else:
                    self.arm_violation_count = 0

                # Stop test if violation is persistent
                if self.arm_violation_count >= self.arm_violation_threshold:
                    self.test_stopped = True
                    self.test_started = False

                    result = self.evaluate_score(0, age, gender)
                    self.record_session(age, gender, 0, result, arm_violation=True)

                    retry = self.show_result_overlay(cap, result, test_status="failed")
                    if retry:
                        self.reset_test_state()
                        break
                    else:
                        self.pose.close()
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # Update count
                new_state = self.calculate_body_posture(landmarks)
                self.update_count(new_state, landmarks)

                # Timer
                elapsed_time = time.time() - self.start_time
                remaining_time = max(0, self.test_duration - elapsed_time)

                if remaining_time <= 0:
                    self.test_started = False
                    result = self.evaluate_score(self.stand_count, age, gender)
                    self.record_session(age, gender, self.stand_count, result, arm_violation=False)
                    retry = self.show_result_overlay(cap, result, test_status="completed")
                    if retry:
                        self.reset_test_state()
                        break
                    else:
                        self.pose.close()
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    
                # Display info
                cv2.putText(frame, f"Count: {self.stand_count}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"Time: {remaining_time:.1f}s", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                if self.test_stopped:
                    cv2.putText(frame, "TEST STOPPED", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("30-Second Chair Stand Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def show_result_overlay(self, cap, final_result, test_status):

        while True:
           ret, frame = cap.read()
           if not ret:
               continue
 
           frame = cv2.flip(frame, 1)

           if test_status == "completed":
               title_text = "TEST COMPLETE"
               title_color = (0, 255, 0)   # Green
           else:
               title_text = "TEST FAILED"
               title_color = (0, 0, 255)   # Red

           cv2.putText(frame, title_text, (40, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.6, title_color, 3)


           cv2.putText(frame, f"Total Stands: {self.stand_count}", (40, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)

           cv2.putText(frame, f"Risk Level: {final_result['risk_level']}", (40, 165),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

           cv2.putText(frame, "SESSION HISTORY", (40, 230),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

           y_offset = 270
           s = self.session_history[-1]

           text = (
               f"Session {len(self.session_history)} | {s['timestamp']} | "
               f"Age: {s['age']} | Gender: {s['gender']} | "
               f"Count: {s['count']} | "
               f"Arm Violation: {'YES' if s['arm_violation'] else 'NO'} | "
               f"Risk: {s['risk_level']}"
           )

           cv2.putText(frame, text, (40, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

           cv2.putText(frame, "Press R to Retry or Q to Quit", (40, y_offset + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

           cv2.imshow("30-Second Chair Stand Test", frame)
           key = cv2.waitKey(1) & 0xFF

           if key in (ord('r'), ord('R')):
               return True
           elif key in (ord('q'), ord('Q')):
               return False

# Run the test
if __name__ == "__main__":
    counter = SitToStandCounter()
    counter.run_test()
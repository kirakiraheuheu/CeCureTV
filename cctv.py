# cctv.py — CeCureCam (WhatsApp + watermark)
# pip install opencv-python mediapipe numpy
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# (optional snapshot attachment)
from email.mime.image import MIMEImage
import certifi

import time, math, webbrowser
import cv2
import mediapipe as mp
import numpy as np
from urllib.parse import urlencode
from collections import deque

# ------------- SETTINGS -------------
PHONE_NUMBER   = "917044666982"      # digits only, with country code
LOCATION_NAME  = "Home (Demo)"

# Gestures / timing
HANDS_UP_HOLD            = 1.5
CONFIRM_GESTURE_HOLD     = 1.0
PENDING_TIMEOUT_SECONDS  = 8.0

# Core fall heuristics
FALL_HOLD                = 0.7
TORSO_HORIZONTAL_DEG_MAX = 45.0
LOW_BODY_Y_FRAC          = 0.58
WIDE_ASPECT_MIN          = 1.08
MOTION_STILL_FRAC_MAX    = 0.012

# Bottom-exit fall detector
EXITFALL_ENABLED         = True
MISS_FRAMES_FOR_EXIT     = 8
EXIT_DROP_MIN            = 0.06
EXIT_LAST_Y_MIN          = 0.75
EXIT_LAST_X_CENTER_MIN   = 0.25
EXIT_LAST_X_CENTER_MAX   = 0.75

# Smoothing / windows
FPS_ESTIMATE             = 24
WINDOW_SEC               = 0.6
DROP_WINDOW_SEC          = 0.4

# Camera flips
FLIP_HORIZONTAL = True
FLIP_VERTICAL   = False
ROTATE_180      = False
# ------------------------------------
# -------- EMAIL ALERT SETTINGS --------
DEMO_MODE = False           # True = print what would be sent, don't send
SMTP_SERVER = "smtp.gmail.com"  # Gmail: "smtp.gmail.com" (SSL=465) | Outlook: "smtp.office365.com" (TLS=587)
SMTP_PORT   = 465               # Gmail SSL: 465 | Outlook TLS: 587
EMAIL_ADDRESS      = "cecuretv.emergency@gmail.com"
EMAIL_APP_PASSWORD = "qltw qxfw dcox gsna"   # Gmail App Password or Outlook password
ALERT_RECIPIENTS   = [
    "kirahadropped@gmail.com",
]  # add more if you like

# MediaPipe models
mp_pose  = mp.solutions.pose
mp_hands = mp.solutions.hands
pose  = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.4, min_tracking_confidence=0.4)
#hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
# ----- Load the CeCureCam logo (watermark) -----
logo = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)
if logo is None:
    print("⚠️  Logo not found! Make sure 'logo.png' is in the same folder as cctv.py.")

def add_watermark(frame, logo, alpha=0.80, scale=0.22, pos="bottom-right"):
    """Blend the logo with transparency into a corner of the frame."""
    if logo is None:
        return frame
    h, w = frame.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(logo.shape[0] * new_w / logo.shape[1]))
    wm = cv2.resize(logo, (new_w, new_h))
    if wm.shape[2] == 4:
        logo_rgb = wm[:, :, :3]
        mask = wm[:, :, 3] / 255.0
    else:
        logo_rgb = wm
        mask = np.ones((new_h, new_w), dtype=np.float32)

    if pos == "top-left":      x, y = 10, 10
    elif pos == "top-right":   x, y = w - new_w - 10, 10
    elif pos == "bottom-left": x, y = 10, h - new_h - 10
    else:                      x, y = w - new_w - 10, h - new_h - 10  # bottom-right

    roi = frame[y:y+new_h, x:x+new_w]
    # Blend
    for c in range(3):
        roi[:, :, c] = (1 - mask * alpha) * roi[:, :, c] + (mask * alpha) * logo_rgb[:, :, c]
    frame[y:y+new_h, x:x+new_w] = roi
    return frame

# ----- Utility -----
def flip_frame(frame):
    if ROTATE_180:     frame = cv2.rotate(frame, cv2.ROTATE_180)
    if FLIP_HORIZONTAL: frame = cv2.flip(frame, 1)
    if FLIP_VERTICAL:   frame = cv2.flip(frame, 0)
    return frame


def draw_banner(frame, line1, line2="", color=(20,40,160)):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, int(h*0.95)), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, line1, (16, h-36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)
    if line2:
        cv2.putText(frame, line2, (16, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240,240,240), 1, cv2.LINE_AA)

# ----- Gestures -----
def open_palm(hand_landmarks):
    lm = hand_landmarks.landmark
    tips = [8,12,16,20]; pips = [6,10,14,18]
    ext = 0
    for t,p in zip(tips,pips):
        if lm[t].y < lm[p].y:  # tip above PIP
            ext += 1
    return ext >= 4

# ----- Motion Estimator -----
class MotionEstimator:
    def __init__(self, history=300, varThresh=12):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThresh, detectShadows=True)
        self.motion_hist = deque(maxlen=120)
    def update(self, gray):
        mask = self.bg.apply(gray)
        mask = cv2.medianBlur(mask, 5)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        frac = float(np.count_nonzero(mask)) / mask.size
        self.motion_hist.append(frac)
        return frac, mask

# ----- Main -----
def send_email(kind, frame=None):
    """Send an email alert; optionally attach a snapshot."""
    if not ALERT_RECIPIENTS:
        print("No ALERT_RECIPIENTS configured.")
        return

    subject = f"[CeCureCam] {kind} detected at {LOCATION_NAME}"
    body = (
        f"{kind} detected at {LOCATION_NAME}.\n\n"
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"This is an automated alert from CeCureCam."
    )

    print(f"\n--- EMAIL ALERT ---\nTo: {', '.join(ALERT_RECIPIENTS)}\nSubject: {subject}\n{body}\n")
    if DEMO_MODE:
        print("[DEMO_MODE] Not sending email.")
        return

    # Build message
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = ", ".join(ALERT_RECIPIENTS)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Optional snapshot
    if frame is not None:
        try:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                img_part = MIMEImage(buf.tobytes(), _subtype="jpeg", name="snapshot.jpg")
                img_part.add_header("Content-Disposition", "attachment", filename="snapshot.jpg")
                msg.attach(img_part)
        except Exception as e:
            print("Snapshot attach failed:", e)

    # Use certifi’s CA bundle for TLS verification
    context = ssl.create_default_context(cafile=certifi.where())

    try:
        if SMTP_PORT == 465:
            # Gmail SSL
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
                server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
                server.sendmail(EMAIL_ADDRESS, ALERT_RECIPIENTS, msg.as_string())
        else:
            # STARTTLS (e.g., Outlook 587)
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.ehlo()
                server.starttls(context=context)
                server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
                server.sendmail(EMAIL_ADDRESS, ALERT_RECIPIENTS, msg.as_string())
        print("✅ Email(s) sent.")
    except Exception as e:
        print("❌ Email send failed:", e)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_ESTIMATE
    smooth_len = max(5, int(WINDOW_SEC * fps))
    drop_len   = max(3, int(DROP_WINDOW_SEC * fps))

    help_start = None
    confirm_start = None
    fall_hold_start = None
    pending = None
    sent_once = False
    show_debug = True

    torso_hist = deque(maxlen=smooth_len)
    centy_hist = deque(maxlen=max(smooth_len, drop_len))
    centx_hist = deque(maxlen=max(smooth_len, drop_len))
    motion_est = MotionEstimator()

    miss_frames = 0
    last_seen = {'t': None, 'cx': None, 'cy': None}

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = flip_frame(frame)
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        motion_frac, motion_mask = motion_est.update(gray)
        sm_motion = np.mean(motion_est.motion_hist) if motion_est.motion_hist else motion_frac

        pose_res  = pose.process(rgb)
        hands_res = hands.process(rgb)
        now = time.time()

        ui_top = f"motion={sm_motion:.4f}"
        help_trigger = False
        fall_trigger = False

        # --- Pose visible ---
        if pose_res.pose_landmarks:
            miss_frames = 0
            lm = pose_res.pose_landmarks.landmark
            Ls, Rs = lm[11], lm[12]; Lw, Rw = lm[15], lm[16]; Lh, Rh = lm[23], lm[24]

            # HELP: both wrists above their shoulders
            hands_up = (Lw.y < Ls.y and Rw.y < Rs.y)
            if hands_up:
                if help_start is None: help_start = now
                if now - help_start >= HANDS_UP_HOLD and not pending:
                    help_trigger = True
            else:
                help_start = None

            # geometry
            mid_sh = ((Ls.x+Rs.x)/2, (Ls.y+Rs.y)/2)
            mid_hip= ((Lh.x+Rh.x)/2, (Lh.y+Rh.y)/2)
            dx, dy = (mid_hip[0]-mid_sh[0]), (mid_hip[1]-mid_sh[1])
            torso_deg = abs(math.degrees(math.atan2(dy, dx)))  # 90 vertical; 0 horizontal
            centroid_x = (mid_sh[0]+mid_hip[0])/2
            centroid_y = (mid_sh[1]+mid_hip[1])/2

            min_x = min(Ls.x, Rs.x, Lh.x, Rh.x); max_x = max(Ls.x, Rs.x, Lh.x, Rh.x)
            min_y = min(Ls.y, Rs.y, Lh.y, Rh.y); max_y = max(Ls.y, Rs.y, Lh.y, Rh.y)
            bw = max_x - min_x; bh = max_y - min_y
            aspect = bw / (bh + 1e-6)

            torso_hist.append(torso_deg)
            centy_hist.append(centroid_y)
            centx_hist.append(centroid_x)

            t_ang = float(np.mean(torso_hist))
            c_y   = float(np.mean(centy_hist))
            c_x   = float(np.mean(centx_hist))

            sudden_drop = False
            if len(centy_hist) >= drop_len:
                past_cy = centy_hist[-drop_len]
                sudden_drop = (centroid_y - past_cy) > 0.05

            horiz = t_ang <= TORSO_HORIZONTAL_DEG_MAX
            wide  = aspect >= WIDE_ASPECT_MIN
            low   = c_y > LOW_BODY_Y_FRAC
            still = sm_motion <= MOTION_STILL_FRAC_MAX
            core_fall = ((horiz or wide) and (low or sudden_drop) and still)

            if core_fall:
                if fall_hold_start is None: fall_hold_start = now
                elif (now - fall_hold_start) >= FALL_HOLD and not pending:
                    fall_trigger = True
            else:
                fall_hold_start = None

            if show_debug:
                cv2.line(frame, (int(mid_sh[0]*w), int(mid_sh[1]*h)),
                               (int(mid_hip[0]*w), int(mid_hip[1]*h)), (255,180,0), 2)
                dbg = f"torso={t_ang:.1f} aspect={aspect:.2f} cx={c_x:.2f} cy={c_y:.2f} drop={sudden_drop} still={still}"
                cv2.putText(frame, dbg, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)

            last_seen.update({'t': now, 'cx': float(centroid_x), 'cy': float(centroid_y)})

        # --- Pose missing → bottom-exit fall ---
        else:
            miss_frames += 1
            if EXITFALL_ENABLED and last_seen['t'] and miss_frames >= MISS_FRAMES_FOR_EXIT:
                cx, cy = last_seen['cx'], last_seen['cy']
                if len(centy_hist) >= 2:
                    past_cy = centy_hist[-min(len(centy_hist), int(DROP_WINDOW_SEC*FPS_ESTIMATE))]
                    drop_amt = cy - past_cy
                else:
                    drop_amt = 0.0
                central     = EXIT_LAST_X_CENTER_MIN <= cx <= EXIT_LAST_X_CENTER_MAX
                near_bottom = cy >= EXIT_LAST_Y_MIN
                big_down    = drop_amt >= EXIT_DROP_MIN
                scene_ok    = np.mean(motion_est.motion_hist) <= 0.04
                if central and near_bottom and big_down and scene_ok and not pending:
                    fall_trigger = True
                if show_debug:
                    dbg2 = f"[MISSING] miss={miss_frames} cx={cx:.2f} cy={cy:.2f} drop={drop_amt:.3f}"
                    cv2.putText(frame, dbg2, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2)

        # --- Pending & confirm via open palm ---
        if not pending and (help_trigger or fall_trigger):
            pending = {'type': 'HELP' if help_trigger else 'FALL', 'start': time.time()}
            confirm_start = None
            sent_once = False

        if pending:
            remaining = max(0.0, PENDING_TIMEOUT_SECONDS - (time.time() - pending['start']))
            draw_banner(frame, f"PENDING {pending['type']}: show OPEN PALM to CONFIRM",
                        f"Time left: {remaining:.1f}s")
            confirmed_now = False
            if hands_res and hands_res.multi_hand_landmarks:
                any_confirm = any(open_palm(hh) for hh in hands_res.multi_hand_landmarks)
                if any_confirm:
                    if confirm_start is None: confirm_start = time.time()
                    held = time.time() - confirm_start
                    cv2.putText(frame, f"Confirm hold: {held:.1f}s", (12, 54),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,240,255), 2)
                    if held >= CONFIRM_GESTURE_HOLD:
                        confirmed_now = True
                else:
                    confirm_start = None
            if confirmed_now and not sent_once:
                kind = "HELP gesture" if pending['type']=="HELP" else "Fall"
                send_email(kind, frame)   # ✅ send email + attach current snapshot
                sent_once = True
                pending = None

            if remaining <= 0 and pending:
                draw_banner(frame, "No confirm — cancelled.", "Returning to monitoring…", color=(0,130,0))
                pending = None
        else:
            draw_banner(frame, "Monitoring…", ui_top, color=(40,40,40))

        # ---- Add watermark before showing ----
        frame = add_watermark(frame, logo, alpha=0.80, scale=0.22, pos="bottom-right")

        # (Optional) motion mask thumbnail
        thumb = cv2.resize(motion_mask, (160, 90))
        frame[10:10+90, w-10-160:w-10] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(frame, (w-10-160,10), (w-10, 10+90), (80,80,80), 1)
        cv2.putText(frame, "motion", (w-10-156, 10+85), cv2.FONT_HERSHEY_PLAIN, 1.0, (160,160,160), 1)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord('d'): show_debug = not show_debug
        if k == ord('m'): globals()['FLIP_HORIZONTAL'] = not FLIP_HORIZONTAL

        cv2.imshow("CeCureCam (q=quit, d=debug, m=mirror)", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

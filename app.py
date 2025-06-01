from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import uuid
import threading
import time
from datetime import datetime
import json
import tempfile
import shutil
from werkzeug.utils import secure_filename
import logging
import base64

# ML/AI imports
try:
    from sklearn.feature_extraction import image
    from sklearn.cluster import KMeans
    import tensorflow as tf
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML libraries not available. Install with: pip install scikit-learn tensorflow")

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
TEMP_FOLDER = 'temp'
PREVIEW_FOLDER = 'previews'

# Create necessary directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER, PREVIEW_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables for tracking progress
processing_tasks = {}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedVideoStitcher:
    def __init__(self):
        self.stitcher = cv2.Stitcher.create()
        self.sift = cv2.SIFT_create() if hasattr(cv2, 'SIFT_create') else None
        self.orb = cv2.ORB_create()

    def normalize_frame_dimensions(self, frames, method='horizontal'):
        """Normalize frame dimensions to prevent line artifacts"""
        if not frames:
            return []

        if method == 'horizontal':
            target_height = min(frame.shape[0] for frame in frames)
            normalized_frames = []
            for frame in frames:
                if frame.shape[0] != target_height:
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    target_width = int(target_height * aspect_ratio)
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                normalized_frames.append(frame)

        elif method == 'vertical':
            target_width = min(frame.shape[1] for frame in frames)
            normalized_frames = []
            for frame in frames:
                if frame.shape[1] != target_width:
                    aspect_ratio = frame.shape[0] / frame.shape[1]
                    target_height = int(target_width * aspect_ratio)
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                normalized_frames.append(frame)

        elif method == 'grid':
            target_height = min(frame.shape[0] for frame in frames)
            target_width = min(frame.shape[1] for frame in frames)
            normalized_frames = []
            for frame in frames:
                if frame.shape[:2] != (target_height, target_width):
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                normalized_frames.append(frame)
        else:
            normalized_frames = frames

        return normalized_frames

    def add_seamless_blending(self, frames, method='horizontal'):
        """Add seamless blending to reduce visible seams"""
        if len(frames) < 2:
            return frames

        blended_frames = []
        if method == 'horizontal':
            for i, frame in enumerate(frames):
                if i == 0:
                    h, w = frame.shape[:2]
                    mask = np.ones((h, w), dtype=np.float32)
                    fade_width = min(20, w // 10)
                    for x in range(w - fade_width, w):
                        alpha = (w - x) / fade_width
                        mask[:, x] = alpha
                    for c in range(3):
                        frame[:, :, c] = frame[:, :, c] * mask
                elif i == len(frames) - 1:
                    h, w = frame.shape[:2]
                    mask = np.ones((h, w), dtype=np.float32)
                    fade_width = min(20, w // 10)
                    for x in range(fade_width):
                        alpha = x / fade_width
                        mask[:, x] = alpha
                    for c in range(3):
                        frame[:, :, c] = frame[:, :, c] * mask
                else:
                    h, w = frame.shape[:2]
                    mask = np.ones((h, w), dtype=np.float32)
                    fade_width = min(20, w // 10)
                    for x in range(fade_width):
                        alpha = x / fade_width
                        mask[:, x] = alpha
                    for x in range(w - fade_width, w):
                        alpha = (w - x) / fade_width
                        mask[:, x] = alpha
                    for c in range(3):
                        frame[:, :, c] = frame[:, :, c] * mask
                blended_frames.append(frame.astype(np.uint8))
        elif method == 'vertical':
            for i, frame in enumerate(frames):
                if i == 0:
                    h, w = frame.shape[:2]
                    mask = np.ones((h, w), dtype=np.float32)
                    fade_height = min(20, h // 10)
                    for y in range(h - fade_height, h):
                        alpha = (h - y) / fade_height
                        mask[y, :] = alpha
                    for c in range(3):
                        frame[:, :, c] = frame[:, :, c] * mask
                elif i == len(frames) - 1:
                    h, w = frame.shape[:2]
                    mask = np.ones((h, w), dtype=np.float32)
                    fade_height = min(20, h // 10)
                    for y in range(fade_height):
                        alpha = y / fade_height
                        mask[y, :] = alpha
                    for c in range(3):
                        frame[:, :, c] = frame[:, :, c] * mask
                else:
                    h, w = frame.shape[:2]
                    mask = np.ones((h, w), dtype=np.float32)
                    fade_height = min(20, h // 10)
                    for y in range(fade_height):
                        alpha = y / fade_height
                        mask[y, :] = alpha
                    for y in range(h - fade_height, h):
                        alpha = (h - y) / fade_height
                        mask[y, :] = alpha
                    for c in range(3):
                        frame[:, :, c] = frame[:, :, c] * mask
                blended_frames.append(frame.astype(np.uint8))
        else:
            blended_frames = frames
        return blended_frames

    def extract_features(self, frame):
        """Extract features from frame using SIFT or ORB"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.sift is not None:
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        else:
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match features between two frames"""
        if desc1 is None or desc2 is None:
            return []
        try:
            if self.sift is not None:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(desc1, desc2, k=2)
            else:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            return good_matches
        except:
            return []

    def intelligent_stitch(self, frames):
        """AI-powered intelligent stitching with overlap detection"""
        try:
            if len(frames) < 2:
                return frames[0] if frames else None
            frames = self.normalize_frame_dimensions(frames, 'horizontal')
            status, stitched = self.stitcher.stitch(frames)
            if status == cv2.Stitcher_OK:
                return stitched
            else:
                logger.warning(f"OpenCV stitcher failed with status: {status}")
                return np.hstack(frames)  # Simplified fallback
        except Exception as e:
            logger.error(f"Intelligent stitching failed: {e}")
            return np.hstack(frames)

    def enhance_frame_quality(self, frame):
        """Enhanced frame quality improvement"""
        if not ML_AVAILABLE:
            return self.basic_enhance_frame(frame)
        try:
            enhanced = frame.copy()
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            kernel = np.array([[-1, -1, -1, -1, -1],
                               [-1, 2, 2, 2, -1],
                               [-1, 2, 8, 2, -1],
                               [-1, 2, 2, 2, -1],
                               [-1, -1, -1, -1, -1]]) / 8.0
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            a = cv2.addWeighted(a, 1.1, np.zeros_like(a), 0, 0)
            b = cv2.addWeighted(b, 1.1, np.zeros_like(b), 0, 0)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            gamma = 1.2
            lookupTable = np.empty((1, 256), np.uint8)
            for i in range(256):
                lookupTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            enhanced = cv2.LUT(enhanced, lookupTable)
            return enhanced
        except Exception as e:
            logger.error(f"Advanced frame enhancement failed: {e}")
            return self.basic_enhance_frame(frame)

    def basic_enhance_frame(self, frame):
        """Basic frame enhancement without ML dependencies"""
        try:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(frame, -1, kernel)
            enhanced = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=10)
            return enhanced
        except:
            return frame

    def stitch_horizontal(self, frames):
        """Enhanced horizontal stitching with proper alignment"""
        if not frames:
            return None
        frames = self.normalize_frame_dimensions(frames, 'horizontal')
        frames = self.add_seamless_blending(frames, 'horizontal')
        return np.hstack(frames)

    def stitch_vertical(self, frames):
        """Enhanced vertical stitching with proper alignment"""
        if not frames:
            return None
        frames = self.normalize_frame_dimensions(frames, 'vertical')
        frames = self.add_seamless_blending(frames, 'vertical')
        return np.vstack(frames)

    def stitch_grid(self, frames):
        """Enhanced grid-based stitching"""
        num_frames = len(frames)
        if num_frames < 2:
            return frames[0] if frames else None
        cols = int(np.ceil(np.sqrt(num_frames)))
        rows = int(np.ceil(num_frames / cols))
        avg_height = int(np.mean([frame.shape[0] for frame in frames]))
        avg_width = int(np.mean([frame.shape[1] for frame in frames]))
        target_height = max(200, avg_height // 2)
        target_width = max(300, avg_width // 2)
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized)
        padding = 2
        grid_rows = []
        for row in range(rows):
            row_frames = []
            for col in range(cols):
                idx = row * cols + col
                if idx < len(resized_frames):
                    frame = resized_frames[idx]
                    frame = cv2.copyMakeBorder(frame, padding, padding, padding, padding,
                                               cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    row_frames.append(frame)
                else:
                    black_frame = np.zeros((target_height + 2 * padding,
                                            target_width + 2 * padding, 3), dtype=np.uint8)
                    row_frames.append(black_frame)
            if row_frames:
                grid_rows.append(np.hstack(row_frames))
        return np.vstack(grid_rows) if grid_rows else None

def create_preview_frames(video_files, task_id, method='horizontal'):
    """Create preview frames for demo"""
    try:
        logger.info(f"Starting preview generation for task {task_id} with {len(video_files)} videos")
        stitcher = EnhancedVideoStitcher()
        caps = []

        # Open video captures
        for video_file in video_files:
            logger.info(f"Processing video: {video_file}")
            cap = cv2.VideoCapture(video_file)
            if cap.isOpened():
                logger.info(f"Successfully opened video: {video_file}, frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
                caps.append(cap)
            else:
                logger.error(f"Failed to open video: {video_file}")
                return None

        if len(caps) < 2:
            logger.error("Less than 2 valid videos provided")
            return None

        preview_frames = []

        # Extract 1 frame from 50% point
        for cap in caps:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                logger.error(f"Invalid frame count for video")
                return None
            time_points = [0.5]

            cap_frames = []
            for time_point in time_points:
                frame_number = int(frame_count * time_point)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if ret:
                    # Only resize if frame is large
                    height, width = frame.shape[:2]
                    if width > 640 or height > 360:
                        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                    cap_frames.append(frame)
                else:
                    logger.error(f"Failed to read frame {frame_number} from video")
                    continue

            if not cap_frames:
                logger.error("No valid frames extracted from video")
                return None
            preview_frames.append(cap_frames)

        # Close captures
        for cap in caps:
            cap.release()

        # Create stitched preview frames
        stitched_previews = []
        for i in range(len(time_points)):
            frames_at_time = [frames[i] for frames in preview_frames if i < len(frames)]
            logger.info(f"Stitching preview frame {i+1}/{len(time_points)}")
            if method == 'horizontal':
                stitched = stitcher.stitch_horizontal(frames_at_time)
            elif method == 'vertical':
                stitched = stitcher.stitch_vertical(frames_at_time)
            elif method == 'grid':
                stitched = stitcher.stitch_grid(frames_at_time)
            elif method == 'ai_panorama':
                stitched = stitcher.intelligent_stitch(frames_at_time)
            else:
                stitched = stitcher.stitch_horizontal(frames_at_time)

            if stitched is not None:
                # Resize stitched frame if too large
                height, width = stitched.shape[:2]
                if width > 1280 or height > 720:
                    stitched = cv2.resize(stitched, (1280, 720), interpolation=cv2.INTER_AREA)
                stitched_previews.append(stitched)
            else:
                logger.error(f"Failed to stitch preview frame {i+1}")
                return None

        # Save preview frames as images
        preview_paths = []
        for i, frame in enumerate(stitched_previews):
            preview_filename = f"preview_{task_id}_{i}.jpg"
            preview_path = os.path.join(PREVIEW_FOLDER, preview_filename)
            logger.info(f"Saving preview: {preview_path}")
            cv2.imwrite(preview_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            preview_paths.append(f'/preview/{preview_filename}')

        logger.info(f"Preview generation completed for task {task_id}")
        import gc
        gc.collect()
        return preview_paths

    except Exception as e:
        logger.error(f"Preview creation failed: {e}")
        return None

def get_video_quality_params(quality):
    """Get video encoding parameters based on quality setting"""
    quality_params = {
        '720p': {'width': 1280, 'height': 720, 'bitrate': '2500k'},
        '1080p': {'width': 1920, 'height': 1080, 'bitrate': '5000k'},
        '1440p': {'width': 2560, 'height': 1440, 'bitrate': '8000k'},
        '2160p': {'width': 3840, 'height': 2160, 'bitrate': '15000k'},
        'original': None
    }
    return quality_params.get(quality, quality_params['1080p'])

def process_videos_background(task_id, video_files, options):
    """Enhanced background video processing"""
    try:
        processing_tasks[task_id]['status'] = 'Initializing enhanced stitcher...'
        processing_tasks[task_id]['progress'] = 0
        stitcher = EnhancedVideoStitcher()
        caps = []
        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_file}")
            caps.append(cap)

        processing_tasks[task_id]['status'] = 'Analyzing video properties...'
        processing_tasks[task_id]['progress'] = 10

        fps = int(caps[0].get(cv2.CAP_PROP_FPS))
        frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
        max_frames = 1800  # Limit to 1 minute at 30fps
        total_frames = min(min(frame_counts), max_frames)

        first_frames = []
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if ret:
                first_frames.append(frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        normalized_first_frames = stitcher.normalize_frame_dimensions(first_frames, options['method'])

        if options['method'] == 'horizontal':
            output_height = normalized_first_frames[0].shape[0]
            output_width = sum(frame.shape[1] for frame in normalized_first_frames)
        elif options['method'] == 'vertical':
            output_width = normalized_first_frames[0].shape[1]
            output_height = sum(frame.shape[0] for frame in normalized_first_frames)
        elif options['method'] == 'grid':
            cols = int(np.ceil(np.sqrt(len(video_files))))
            rows = int(np.ceil(len(video_files) / cols))
            single_height, single_width = normalized_first_frames[0].shape[:2]
            output_width = single_width * cols + 4 * (cols - 1)
            output_height = single_height * rows + 4 * (rows - 1)
        else:
            test_stitch = stitcher.intelligent_stitch(normalized_first_frames)
            if test_stitch is not None:
                output_height, output_width = test_stitch.shape[:2]
            else:
                output_height = normalized_first_frames[0].shape[0]
                output_width = sum(frame.shape[1] for frame in normalized_first_frames)

        quality_params = get_video_quality_params(options['quality'])
        if quality_params and options['quality'] != 'original':
            scale_factor = min(quality_params['width'] / output_width,
                               quality_params['height'] / output_height)
            if scale_factor < 1:
                output_width = int(output_width * scale_factor)
                output_height = int(output_height * scale_factor)

        output_filename = f"stitched_{task_id}.{options['format']}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        fourcc_map = {
            'mp4': cv2.VideoWriter_fourcc(*'mp4v'),
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            'mov': cv2.VideoWriter_fourcc(*'mp4v'),
            'mkv': cv2.VideoWriter_fourcc(*'XVID')
        }
        fourcc = fourcc_map.get(options['format'], cv2.VideoWriter_fourcc(*'mp4v'))
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

        processing_tasks[task_id]['status'] = 'Processing frames with enhanced stitching...'

        processed_frames = 0
        batch_size = 100
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            for frame_idx in range(batch_start, batch_end):
                frames = []
                all_frames_read = True
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        all_frames_read = False
                        break
                    if options.get('ai_enhancement', False):
                        frame = stitcher.enhance_frame_quality(frame)
                    frames.append(frame)
                if not all_frames_read or len(frames) != len(video_files):
                    break

                if options['method'] == 'horizontal':
                    stitched_frame = stitcher.stitch_horizontal(frames)
                elif options['method'] == 'vertical':
                    stitched_frame = stitcher.stitch_vertical(frames)
                elif options['method'] == 'grid':
                    stitched_frame = stitcher.stitch_grid(frames)
                elif options['method'] == 'ai_panorama':
                    stitched_frame = stitcher.intelligent_stitch(frames)
                else:
                    stitched_frame = stitcher.stitch_horizontal(frames)

                if stitched_frame is not None:
                    if stitched_frame.shape[:2] != (output_height, output_width):
                        stitched_frame = cv2.resize(stitched_frame, (output_width, output_height),
                                                   interpolation=cv2.INTER_LINEAR)
                    out.write(stitched_frame)
                    processed_frames += 1

                progress = int((frame_idx / total_frames) * 85) + 10
                processing_tasks[task_id]['progress'] = progress
                processing_tasks[task_id]['status'] = f'Processing frame {frame_idx + 1}/{total_frames}'
            import gc
            gc.collect()

        for cap in caps:
            cap.release()
        out.release()

        if processed_frames == 0:
            raise Exception("No frames were processed successfully")

        if options.get('include_audio', True):
            processing_tasks[task_id]['status'] = 'Processing audio...'
            processing_tasks[task_id]['progress'] = 95
            try:
                import subprocess
                audio_output = output_path.replace(f'.{options["format"]}', f'_with_audio.{options["format"]}')
                cmd = [
                    'ffmpeg', '-y', '-i', output_path, '-i', video_files[0],
                    '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                    '-shortest', audio_output
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    os.replace(audio_output, output_path)
                else:
                    logger.warning(f"Audio processing failed: {result.stderr}")
            except FileNotFoundError:
                logger.error("FFmpeg not found. Audio processing skipped.")
            except subprocess.TimeoutExpired:
                logger.error("Audio processing timed out.")
            except Exception as e:
                logger.warning(f"Audio processing failed: {e}")

        processing_tasks[task_id]['status'] = 'Complete!'
        processing_tasks[task_id]['progress'] = 100
        processing_tasks[task_id]['success'] = True
        processing_tasks[task_id]['output_url'] = f'/download/{output_filename}'
        processing_tasks[task_id]['processed_frames'] = processed_frames

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        processing_tasks[task_id]['status'] = f'Error: {str(e)}'
        processing_tasks[task_id]['progress'] = 0
        processing_tasks[task_id]['success'] = False
        processing_tasks[task_id]['error'] = str(e)

@app.route('/')
def index():
    """Serve the main web application"""
    try:
        with open('index.html', 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
        <head><title>AI Video Stitcher Pro</title></head>
        <body>
            <h1>🎬 AI Video Stitcher Pro</h1>
            <p>Upload your HTML interface file as 'index.html' or use the API endpoints directly.</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li>POST /process - Start video processing</li>
                <li>POST /start_processing/<task_id> - Start actual processing</li>
                <li>GET /progress/<task_id> - Check progress</li>
                <li>GET /download/<filename> - Download result</li>
            </ul>
        </body>
        </html>
        """

@app.route('/process', methods=['POST'])
def process_videos():
    """Start video processing with preview generation"""
    try:
        task_id = str(uuid.uuid4())
        video_files = []
        for key in request.files:
            if key.startswith('video_'):
                file = request.files[key]
                if file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(UPLOAD_FOLDER, f"{task_id}_{filename}")
                    file.save(filepath)
                    cap = cv2.VideoCapture(filepath)
                    if not cap.isOpened():
                        logger.error(f"Invalid video file: {filename}")
                        return jsonify({'success': False, 'error': f'Invalid video file: {filename}'})
                    cap.release()
                    video_files.append(filepath)

        if len(video_files) < 2:
            return jsonify({'success': False, 'error': 'At least 2 videos required'})

        options = {
            'method': request.form.get('method', 'horizontal'),
            'quality': request.form.get('quality', '1080p'),
            'format': request.form.get('format', 'mp4'),
            'include_audio': request.form.get('include_audio') == 'true',
            'ai_enhancement': request.form.get('ai_enhancement') == 'true',
            'auto_sync': request.form.get('auto_sync') == 'true'
        }

        processing_tasks[task_id] = {
            'status': 'Starting preview generation...',
            'progress': 0,
            'success': False,
            'created_at': datetime.now(),
            'video_files': video_files,
            'options': options
        }

        thread = threading.Thread(
            target=generate_preview_background,
            args=(task_id, video_files, options)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Preview generation started'
        })

    except Exception as e:
        logger.error(f"Process endpoint error: {e}")
        return jsonify({'success': False, 'error': str(e)})

def generate_preview_background(task_id, video_files, options):
    """Generate preview frames in background with timeout"""
    def run_with_timeout():
        try:
            preview_paths = create_preview_frames(video_files, task_id, options['method'])
            processing_tasks[task_id]['preview_paths'] = preview_paths or []
            processing_tasks[task_id]['status'] = 'Preview ready. Ready to process.'
            processing_tasks[task_id]['progress'] = 5
        except Exception as e:
            logger.error(f"Preview generation failed for task {task_id}: {e}")
            processing_tasks[task_id]['status'] = f'Error: {str(e)}'
            processing_tasks[task_id]['success'] = False
            processing_tasks[task_id]['error'] = str(e)

    thread = threading.Thread(target=run_with_timeout)
    thread.daemon = True
    thread.start()
    thread.join(timeout=30)
    if thread.is_alive():
        logger.error(f"Preview generation timed out for task {task_id}")
        processing_tasks[task_id]['status'] = 'Error: Preview generation timed out'
        processing_tasks[task_id]['success'] = False
        processing_tasks[task_id]['error'] = 'Preview generation timed out'

@app.route('/start_processing/<task_id>', methods=['POST'])
def start_processing(task_id):
    """Start the actual video processing after preview approval"""
    try:
        if task_id not in processing_tasks:
            return jsonify({'success': False, 'error': 'Task not found'}), 404
        task = processing_tasks[task_id]
        video_files = task.get('video_files', [])
        options = task.get('options', {})
        if not video_files:
            return jsonify({'success': False, 'error': 'No video files found'}), 400
        thread = threading.Thread(
            target=process_videos_background,
            args=(task_id, video_files, options)
        )
        thread.daemon = True
        thread.start()
        return jsonify({'success': True, 'message': 'Processing started'})
    except Exception as e:
        logger.error(f"Start processing error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """Get processing progress"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
    task = processing_tasks[task_id]
    return jsonify({
        'progress': task.get('progress', 0),
        'status': task.get('status', 'Unknown'),
        'success': task.get('success', False),
        'error': task.get('error'),
        'output_url': task.get('output_url'),
        'preview_urls': task.get('preview_paths', []),
        'processed_frames': task.get('processed_frames', 0)
    })

@app.route('/preview/<filename>')
def get_preview(filename):
    """Serve preview images"""
    try:
        filepath = os.path.join(PREVIEW_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Preview not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed video"""
    try:
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup')
def cleanup_old_files():
    """Clean up old files (call periodically)"""
    try:
        current_time = datetime.now()
        cleaned_count = 0
        for task_id in list(processing_tasks.keys()):
            task = processing_tasks[task_id]
            if (current_time - task['created_at']).total_seconds() > 86400:
                del processing_tasks[task_id]
                cleaned_count += 1
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, PREVIEW_FOLDER]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    try:
                        if os.path.getmtime(filepath) < (current_time.timestamp() - 86400):
                            os.remove(filepath)
                            cleaned_count += 1
                    except OSError:
                        continue
        return jsonify({'success': True, 'cleaned_files': cleaned_count})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ml_available': ML_AVAILABLE,
        'active_tasks': len(processing_tasks),
        'features': {
            'enhanced_stitching': True,
            'seamless_blending': True,
            'ai_enhancement': ML_AVAILABLE,
            'preview_generation': True,
            'multiple_formats': True,
            'audio_processing': True
        }
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB total.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🎬 Enhanced AI Video Stitcher Pro Server Starting...")
    print("📋 New Features:")
    print("   ✅ Enhanced stitching with seamless blending")
    print("   ✅ Proper frame dimension normalization")
    print("   ✅ Preview generation before processing")
    print("   ✅ Multiple stitching methods with line artifact reduction")
    print("   ✅ Advanced AI enhancement" if ML_AVAILABLE else "   ⚠️  ML features disabled")
    print("   ✅ High-quality interpolation")
    print("   ✅ Intelligent grid layout")
    print("   ✅ Real-time progress tracking")
    print("   ✅ Audio processing with FFmpeg")
    print("\n🌐 Access the web app at: http://localhost:5000")
    print("📡 API Endpoints:")
    print("   POST /process - Upload videos and generate preview")
    print("   POST /start_processing/<task_id> - Start actual processing")
    print("   GET /progress/<task_id> - Check processing status")
    print("   GET /preview/<filename> - View preview images")
    print("   GET /download/<filename> - Download final video")
    print("   GET /health - Check system status")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
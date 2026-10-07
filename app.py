"""
High-Performance Web Application Server for Biometric Siamese Fingerprint Recognition
"""

import os
import sys
import io
import json
import base64
import random
import time
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import numpy as np
import cv2
from PIL import Image
import keras
from keras import layers, models

from model import build_siamese_model, load_trained_model, label_to_description, label_to_key, FINGER_NAMES, GENDER_NAMES, HAND_NAMES
from data_generator import load_dataset


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def numpy_to_base64_png(img_array):
    """
    Convert 2D/3D numpy array [0..255] uint8 to base64 PNG data URL.
    """
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)
    img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def base64_to_numpy(base64_str, target_shape=(90, 90)):
    """
    Convert base64 image data URL to normalized numpy float32 array (1, 90, 90, 1).
    """
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    pil_img = Image.open(io.BytesIO(img_data)).convert('L')
    pil_img = pil_img.resize(target_shape, Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return arr.reshape((1, target_shape[0], target_shape[1], 1))


class FingerprintWebServer(BaseHTTPRequestHandler):
    dataset = None
    model = None
    feature_extractor = None
    diff_classifier = None
    x_real_norm = None
    x_real_features = None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Static assets
        if path == '/' or path == '/index.html':
            self.serve_file('web/index.html', 'text/html')
        elif path == '/style.css':
            self.serve_file('web/style.css', 'text/css')
        elif path == '/app.js':
            self.serve_file('web/app.js', 'application/javascript')
        
        # REST APIs
        elif path == '/api/stats':
            self.handle_stats()
        elif path == '/api/random_query':
            self.handle_random_query(query)
        elif path == '/api/get_genuine':
            self.handle_get_genuine(query)
        elif path == '/api/get_imposter':
            self.handle_get_imposter(query)
        elif path == '/api/real_fingerprints':
            self.handle_real_fingerprints(query)
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode('utf-8')) if body else {}
        except Exception:
            data = {}

        if path == '/api/verify':
            self.handle_verify(data)
        elif path == '/api/identify':
            self.handle_identify(data)
        elif path == '/api/augment_test':
            self.handle_augment_test(data)
        else:
            self.send_error(404, "API Not Found")

    def serve_file(self, filepath, content_type):
        if not os.path.exists(filepath):
            self.send_error(404, "File Not Found")
            return
        with open(filepath, 'rb') as f:
            content = f.read()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, data, status_code=200):
        content = json.dumps(data).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def handle_stats(self):
        stats = {
            'status': 'active',
            'num_real': len(self.dataset['x_real']),
            'num_easy': len(self.dataset['x_easy']),
            'num_medium': len(self.dataset['x_medium']),
            'num_hard': len(self.dataset['x_hard']),
            'model_loaded': (self.model is not None)
        }
        self.send_json(stats)

    def handle_random_query(self, query):
        difficulty = query.get('difficulty', ['easy'])[0]
        if difficulty == 'medium':
            arr_x = self.dataset['x_medium']
            arr_y = self.dataset['y_medium']
        elif difficulty == 'hard':
            arr_x = self.dataset['x_hard']
            arr_y = self.dataset['y_hard']
        else:
            arr_x = self.dataset['x_easy']
            arr_y = self.dataset['y_easy']

        rand_idx = random.randint(0, len(arr_x) - 1)
        img = arr_x[rand_idx]
        lbl = arr_y[rand_idx]
        key = label_to_key(lbl)

        meta = {
            'subject_id': int(lbl[0]),
            'gender': GENDER_NAMES.get(int(lbl[1]), 'Male'),
            'hand': HAND_NAMES.get(int(lbl[2]), 'Left'),
            'finger_name': FINGER_NAMES.get(int(lbl[3]), 'Index'),
            'key': key,
            'difficulty': difficulty.capitalize()
        }

        self.send_json({
            'status': 'ok',
            'image': numpy_to_base64_png(img),
            'meta': meta
        })

    def handle_get_genuine(self, query):
        key = query.get('key', [''])[0]
        label_dict = self.dataset['label_real_dict']
        if key in label_dict:
            idx = label_dict[key]
            img = self.dataset['x_real'][idx]
            lbl = self.dataset['y_real'][idx]
            meta = {
                'subject_id': int(lbl[0]),
                'gender': GENDER_NAMES.get(int(lbl[1]), 'Male'),
                'hand': HAND_NAMES.get(int(lbl[2]), 'Left'),
                'finger_name': FINGER_NAMES.get(int(lbl[3]), 'Index'),
                'key': key
            }
            self.send_json({
                'status': 'ok',
                'image': numpy_to_base64_png(img),
                'meta': meta
            })
        else:
            self.send_json({'status': 'error', 'message': 'Key not found'}, 404)

    def handle_get_imposter(self, query):
        exclude_key = query.get('exclude', [''])[0]
        label_dict = self.dataset['label_real_dict']
        keys = list(label_dict.keys())
        while True:
            k = random.choice(keys)
            if k != exclude_key:
                break
        idx = label_dict[k]
        img = self.dataset['x_real'][idx]
        lbl = self.dataset['y_real'][idx]
        meta = {
            'subject_id': int(lbl[0]),
            'gender': GENDER_NAMES.get(int(lbl[1]), 'Male'),
            'hand': HAND_NAMES.get(int(lbl[2]), 'Left'),
            'finger_name': FINGER_NAMES.get(int(lbl[3]), 'Index'),
            'key': k
        }
        self.send_json({
            'status': 'ok',
            'image': numpy_to_base64_png(img),
            'meta': meta
        })

    def handle_real_fingerprints(self, query):
        page = int(query.get('page', [1])[0])
        limit = int(query.get('limit', [24])[0])
        sub_filter = query.get('subject', [''])[0]
        gender_filter = query.get('gender', [''])[0]
        hand_filter = query.get('hand', [''])[0]
        finger_filter = query.get('finger', [''])[0]

        y_real = self.dataset['y_real']
        x_real = self.dataset['x_real']

        # Apply filtering
        filtered_indices = []
        for i, lbl in enumerate(y_real):
            if sub_filter and str(int(lbl[0])) != str(sub_filter).strip():
                continue
            if gender_filter and str(int(lbl[1])) != str(gender_filter).strip():
                continue
            if hand_filter and str(int(lbl[2])) != str(hand_filter).strip():
                continue
            if finger_filter and str(int(lbl[3])) != str(finger_filter).strip():
                continue
            filtered_indices.append(i)

        total_items = len(filtered_indices)
        total_pages = max(1, (total_items + limit - 1) // limit)
        page = min(max(1, page), total_pages)

        start = (page - 1) * limit
        end = start + limit
        page_indices = filtered_indices[start:end]

        items = []
        for idx in page_indices:
            lbl = y_real[idx]
            img = x_real[idx]
            key = label_to_key(lbl)
            items.append({
                'key': key,
                'image': numpy_to_base64_png(img),
                'meta': {
                    'subject_id': int(lbl[0]),
                    'gender': GENDER_NAMES.get(int(lbl[1]), 'Male'),
                    'hand': HAND_NAMES.get(int(lbl[2]), 'Left'),
                    'finger_name': FINGER_NAMES.get(int(lbl[3]), 'Index')
                }
            })

        self.send_json({
            'page': page,
            'total_pages': total_pages,
            'total_items': total_items,
            'items': items
        })

    def handle_verify(self, data):
        q_b64 = data.get('query_image')
        t_b64 = data.get('target_image')
        if not q_b64 or not t_b64:
            self.send_json({'error': 'Missing query_image or target_image'}, 400)
            return

        x1 = base64_to_numpy(q_b64)
        x2 = base64_to_numpy(t_b64)

        pred = float(self.model.predict([x1, x2], verbose=0)[0][0])
        is_match = bool(pred >= 0.5)

        self.send_json({
            'confidence': pred,
            'is_match': is_match,
            'threshold': 0.5
        })

    def handle_identify(self, data):
        q_b64 = data.get('query_image')
        q_key = data.get('query_key', '')
        top_k = int(data.get('top_k', 6))

        if not q_b64:
            self.send_json({'error': 'Missing query_image'}, 400)
            return

        t0 = time.time()
        x1 = base64_to_numpy(q_b64)

        # Fast identification using pre-extracted gallery feature embeddings
        if self.feature_extractor is not None and self.diff_classifier is not None and self.x_real_features is not None:
            # 1. Extract query feature (single forward pass: ~2ms)
            q_feat = self.feature_extractor.predict(x1, verbose=0)
            # 2. Subtract query feature from all gallery features in numpy
            diff_feats = q_feat - self.x_real_features
            # 3. Predict similarity on diff features
            preds = self.diff_classifier.predict(diff_feats, batch_size=512, verbose=0).flatten()
        else:
            # Fallback
            x1_tiled = np.repeat(x1, len(self.dataset['x_real']), axis=0)
            preds = self.model.predict([x1_tiled, self.x_real_norm], batch_size=256, verbose=0).flatten()

        top_indices = np.argsort(preds)[::-1][:top_k]
        latency_ms = int((time.time() - t0) * 1000)

        candidates = []
        for idx in top_indices:
            lbl = self.dataset['y_real'][idx]
            img = self.dataset['x_real'][idx]
            cand_key = label_to_key(lbl)
            candidates.append({
                'key': cand_key,
                'confidence': float(preds[idx]),
                'is_true_match': (cand_key == q_key),
                'image': numpy_to_base64_png(img),
                'meta': {
                    'subject_id': int(lbl[0]),
                    'gender': GENDER_NAMES.get(int(lbl[1]), 'Male'),
                    'hand': HAND_NAMES.get(int(lbl[2]), 'Left'),
                    'finger_name': FINGER_NAMES.get(int(lbl[3]), 'Index')
                }
            })

        self.send_json({
            'latency_ms': latency_ms,
            'candidates': candidates
        })

    def handle_augment_test(self, data):
        q_b64 = data.get('query_image')
        target_key = data.get('target_key', '')
        blur = float(data.get('blur', 0))
        rotate = float(data.get('rotate', 0))
        scale = float(data.get('scale', 1.0))
        shift_x = int(data.get('shift_x', 0))
        shift_y = int(data.get('shift_y', 0))
        noise = float(data.get('noise', 0))

        if not q_b64:
            self.send_json({'error': 'Missing query_image'}, 400)
            return

        # Decode to 90x90 uint8 image
        if ',' in q_b64:
            q_b64 = q_b64.split(',')[1]
        img_data = base64.b64decode(q_b64)
        pil_img = Image.open(io.BytesIO(img_data)).convert('L').resize((90, 90), Image.BILINEAR)
        img_np = np.array(pil_img, dtype=np.uint8)

        # Apply Affine (rotate, scale, shift)
        center = (45, 45)
        M = cv2.getRotationMatrix2D(center, rotate, scale)
        M[0, 2] += shift_x
        M[1, 2] += shift_y
        img_np = cv2.warpAffine(img_np, M, (90, 90), borderValue=255)

        # Apply Gaussian Blur
        if blur > 0.05:
            ksize = int(blur * 4) | 1
            if ksize >= 3:
                img_np = cv2.GaussianBlur(img_np, (ksize, ksize), blur)

        # Apply Salt & Pepper Noise
        if noise > 0.005:
            num_noise = int(noise * 90 * 90)
            for _ in range(num_noise // 2):
                rx, ry = random.randint(0, 89), random.randint(0, 89)
                img_np[ry, rx] = 255
            for _ in range(num_noise // 2):
                rx, ry = random.randint(0, 89), random.randint(0, 89)
                img_np[ry, rx] = 0

        # Run verification against true target
        conf = 0.0
        label_dict = self.dataset['label_real_dict']
        if target_key in label_dict:
            t_idx = label_dict[target_key]
            t_img = self.dataset['x_real'][t_idx]
            x1_in = (img_np.astype(np.float32) / 255.0).reshape((1, 90, 90, 1))
            x2_in = (t_img.astype(np.float32) / 255.0).reshape((1, 90, 90, 1))
            conf = float(self.model.predict([x1_in, x2_in], verbose=0)[0][0])

        self.send_json({
            'distorted_image': numpy_to_base64_png(img_np),
            'confidence': conf,
            'is_match': bool(conf >= 0.5)
        })


def main():
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    print("=" * 60)
    print(" Starting Biometric Fingerprint Recognition Web Server ")
    print("=" * 60)

    # 1. Load dataset
    print("Loading dataset...")
    data = load_dataset('dataset')
    FingerprintWebServer.dataset = data
    FingerprintWebServer.x_real_norm = data['x_real'].astype(np.float32) / 255.0

    # 2. Load model
    model_path = 'result/fingerprint_model.keras'
    if os.path.exists(model_path):
        print(f"Loading trained Siamese CNN model from {model_path}...")
        model = load_trained_model(model_path)
    else:
        print("Trained model checkpoint not found yet. Compiling architecture...")
        model, _ = build_siamese_model()

    FingerprintWebServer.model = model

    # 3. Build fast feature extractor & difference classifier
    try:
        feat_extractor = model.get_layer('shared_feature_extractor')
        diff_in = layers.Input(shape=feat_extractor.output_shape[1:], name='diff_input')
        x = model.get_layer('diff_conv')(diff_in)
        x = model.get_layer('diff_pool')(x)
        x = model.get_layer('flatten')(x)
        x = model.get_layer('dense_1')(x)
        out = model.get_layer('match_probability')(x)
        diff_classifier = models.Model(inputs=diff_in, outputs=out, name='diff_classifier')

        FingerprintWebServer.feature_extractor = feat_extractor
        FingerprintWebServer.diff_classifier = diff_classifier

        print("Pre-computing feature embeddings for 6,000 real fingerprints...")
        t0 = time.time()
        FingerprintWebServer.x_real_features = feat_extractor.predict(FingerprintWebServer.x_real_norm, batch_size=256, verbose=0)
        print(f"Gallery embeddings computed in {time.time() - t0:.2f}s (Shape: {FingerprintWebServer.x_real_features.shape})")
    except Exception as e:
        print(f"Embedding optimization setup fallback: {e}")

    server_address = ('', port)
    httpd = ThreadedHTTPServer(server_address, FingerprintWebServer)
    print(f"\n Server listening at http://localhost:{port} (or http://127.0.0.1:{port})")
    print("=" * 60)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == '__main__':
    main()

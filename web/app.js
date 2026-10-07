/**
 * Biometric Siamese CNN Web Application Frontend Logic
 */

// State
let currentQuery = null;
let currentTarget = null;
let currentAugQuery = null;
let searchProbe = null;
let explorerPage = 1;
let totalExplorerPages = 300;

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initVerificationControls();
  initSearchControls();
  initAugmentControls();
  initExplorerControls();
  initDragAndDrop();

  // Load initial random query and match
  loadRandomQuery('easy');
});

/**
 * Robust image processor: resizes any image (PNG, JPG, BMP, WEBP, etc.) to 90x90 grayscale canvas
 * and returns standard clean base64 data URL.
 */
function processImageFile(file) {
  return new Promise((resolve, reject) => {
    if (!file || !file.type.startsWith('image/')) {
      reject(new Error('Please select a valid image file.'));
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Failed to read image file.'));
    reader.onload = (e) => {
      const img = new Image();
      img.onerror = () => reject(new Error('Invalid or corrupted image format.'));
      img.onload = () => {
        // Create 90x90 canvas
        const canvas = document.createElement('canvas');
        canvas.width = 90;
        canvas.height = 90;
        const ctx = canvas.getContext('2d');

        // Fill black background
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, 90, 90);

        // Aspect fit / center draw
        const scale = Math.min(90 / img.width, 90 / img.height);
        const nw = img.width * scale;
        const nh = img.height * scale;
        const dx = (90 - nw) / 2;
        const dy = (90 - nh) / 2;
        ctx.drawImage(img, dx, dy, nw, nh);

        // Convert to grayscale
        const imgData = ctx.getImageData(0, 0, 90, 90);
        const d = imgData.data;
        for (let i = 0; i < d.length; i += 4) {
          const lum = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
          d[i] = lum;
          d[i + 1] = lum;
          d[i + 2] = lum;
          d[i + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);

        const dataUrl = canvas.toDataURL('image/png');
        resolve(dataUrl);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

// Drag and drop handler
function initDragAndDrop() {
  const queryFrame = document.getElementById('scan-frame-query');
  const targetFrame = document.getElementById('scan-frame-target');
  const searchFrame = document.getElementById('search-probe-frame');

  [queryFrame, targetFrame, searchFrame].forEach(frame => {
    if (!frame) return;
    frame.addEventListener('dragover', (e) => {
      e.preventDefault();
      frame.style.borderColor = '#00f0ff';
      frame.style.boxShadow = '0 0 20px rgba(0, 240, 255, 0.4)';
    });
    frame.addEventListener('dragleave', (e) => {
      e.preventDefault();
      frame.style.borderColor = '';
      frame.style.boxShadow = '';
    });
  });

  if (queryFrame) {
    queryFrame.addEventListener('drop', async (e) => {
      e.preventDefault();
      queryFrame.style.borderColor = '';
      queryFrame.style.boxShadow = '';
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        await handleCustomQueryUpload(e.dataTransfer.files[0]);
      }
    });
  }

  if (targetFrame) {
    targetFrame.addEventListener('drop', async (e) => {
      e.preventDefault();
      targetFrame.style.borderColor = '';
      targetFrame.style.boxShadow = '';
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        await handleCustomTargetUpload(e.dataTransfer.files[0]);
      }
    });
  }

  if (searchFrame) {
    searchFrame.addEventListener('drop', async (e) => {
      e.preventDefault();
      searchFrame.style.borderColor = '';
      searchFrame.style.boxShadow = '';
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        await handleCustomSearchProbeUpload(e.dataTransfer.files[0]);
      }
    });
  }
}

// Tab navigation
function initTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  const panes = document.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      panes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');
      const targetPane = document.getElementById(`pane-${tab.dataset.tab}`);
      if (targetPane) {
        targetPane.classList.add('active');
      }

      if (tab.dataset.tab === 'explorer' && document.getElementById('gallery-grid').children.length === 0) {
        loadExplorerGallery();
      }
      if (tab.dataset.tab === 'augment' && !currentAugQuery) {
        loadAugmentProbe();
      }
    });
  });
}

// 1:1 Verification UI
function initVerificationControls() {
  document.getElementById('btn-rand-query-easy').addEventListener('click', () => loadRandomQuery('easy'));
  document.getElementById('btn-rand-query-med').addEventListener('click', () => loadRandomQuery('medium'));
  document.getElementById('btn-rand-query-hard').addEventListener('click', () => loadRandomQuery('hard'));

  document.getElementById('btn-load-genuine').addEventListener('click', () => loadGenuineTarget());
  document.getElementById('btn-load-imposter').addEventListener('click', () => loadImposterTarget());
  document.getElementById('btn-rand-candidate').addEventListener('click', () => loadRandomCandidate());

  document.getElementById('btn-run-verify').addEventListener('click', () => runVerification());

  // Upload handlers
  const uploadQuery = document.getElementById('input-upload-query');
  if (uploadQuery) {
    uploadQuery.addEventListener('change', async (e) => {
      if (e.target.files && e.target.files[0]) {
        await handleCustomQueryUpload(e.target.files[0]);
      }
    });
  }

  const uploadTarget = document.getElementById('input-upload-target');
  if (uploadTarget) {
    uploadTarget.addEventListener('change', async (e) => {
      if (e.target.files && e.target.files[0]) {
        await handleCustomTargetUpload(e.target.files[0]);
      }
    });
  }
}

async function handleCustomQueryUpload(file) {
  try {
    const base64Data = await processImageFile(file);
    setQueryImage(base64Data, {
      subject_id: 'Custom Upload',
      gender: 'N/A',
      hand: 'User File',
      finger_name: file.name.substring(0, 16),
      difficulty: 'Custom',
      key: 'CUSTOM'
    });
    // If no candidate is loaded yet, load a random candidate from gallery
    if (!currentTarget) {
      loadRandomCandidate();
    }
  } catch (err) {
    alert('Failed to process image: ' + err.message);
  }
}

async function handleCustomTargetUpload(file) {
  try {
    const base64Data = await processImageFile(file);
    setTargetImage(base64Data, {
      subject_id: 'Custom Upload',
      gender: 'N/A',
      hand: 'User File',
      finger_name: file.name.substring(0, 16),
      key: 'CUSTOM_TARGET'
    });
  } catch (err) {
    alert('Failed to process candidate image: ' + err.message);
  }
}

async function loadRandomQuery(difficulty = 'easy') {
  try {
    const res = await fetch(`/api/random_query?difficulty=${difficulty}`);
    const data = await res.json();
    if (data.status === 'ok') {
      setQueryImage(data.image, data.meta);
      loadGenuineTarget();
    }
  } catch (err) {
    console.error('Failed to load random query:', err);
  }
}

function setQueryImage(base64Src, meta) {
  currentQuery = { image: base64Src, meta: meta };
  const img = document.getElementById('query-img');
  img.src = base64Src;
  img.style.display = 'block';
  document.getElementById('query-placeholder').style.display = 'none';

  document.getElementById('query-meta-id').textContent = meta.subject_id !== undefined ? meta.subject_id : '-';
  document.getElementById('query-meta-gender').textContent = meta.gender || '-';
  document.getElementById('query-meta-finger').textContent = `${meta.hand || ''} ${meta.finger_name || ''}`;
  document.getElementById('query-meta-dist').textContent = meta.difficulty || 'Custom Probe';

  document.getElementById('decision-container').style.display = 'none';
}

async function loadGenuineTarget() {
  if (!currentQuery || !currentQuery.meta || !currentQuery.meta.key || currentQuery.meta.key === 'CUSTOM') {
    alert('Custom uploaded probe images do not have a gallery ground truth. You can compare against a "Random Target" or click "Upload Custom Candidate" on the right card to compare two of your own photos.');
    return;
  }
  try {
    const res = await fetch(`/api/get_genuine?key=${currentQuery.meta.key}`);
    const data = await res.json();
    if (data.status === 'ok') {
      setTargetImage(data.image, data.meta);
    }
  } catch (err) {
    console.error('Failed to load genuine target:', err);
  }
}

async function loadImposterTarget() {
  const excludeKey = currentQuery && currentQuery.meta ? currentQuery.meta.key : '';
  try {
    const res = await fetch(`/api/get_imposter?exclude=${excludeKey}`);
    const data = await res.json();
    if (data.status === 'ok') {
      setTargetImage(data.image, data.meta);
    }
  } catch (err) {
    console.error('Failed to load imposter target:', err);
  }
}

async function loadRandomCandidate() {
  try {
    const res = await fetch('/api/get_imposter');
    const data = await res.json();
    if (data.status === 'ok') {
      setTargetImage(data.image, data.meta);
    }
  } catch (err) {
    console.error('Failed to load candidate:', err);
  }
}

function setTargetImage(base64Src, meta) {
  currentTarget = { image: base64Src, meta: meta };
  const img = document.getElementById('target-img');
  img.src = base64Src;
  img.style.display = 'block';
  document.getElementById('target-placeholder').style.display = 'none';

  document.getElementById('target-meta-id').textContent = meta.subject_id !== undefined ? meta.subject_id : '-';
  document.getElementById('target-meta-gender').textContent = meta.gender || '-';
  document.getElementById('target-meta-finger').textContent = `${meta.hand || ''} ${meta.finger_name || ''}`;
  document.getElementById('target-meta-key').textContent = meta.key || '-';

  document.getElementById('decision-container').style.display = 'none';
}

async function runVerification() {
  if (!currentQuery || !currentTarget) {
    alert('Please ensure both Query (Left) and Candidate (Right) fingerprints are loaded.');
    return;
  }

  // Visual scan animation
  const queryFrame = document.getElementById('scan-frame-query');
  const targetFrame = document.getElementById('scan-frame-target');
  queryFrame.classList.add('scanning');
  targetFrame.classList.add('scanning');

  try {
    const res = await fetch('/api/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query_image: currentQuery.image,
        target_image: currentTarget.image
      })
    });
    const result = await res.json();

    setTimeout(() => {
      queryFrame.classList.remove('scanning');
      targetFrame.classList.remove('scanning');
      renderVerificationDecision(result);
    }, 400);

  } catch (err) {
    queryFrame.classList.remove('scanning');
    targetFrame.classList.remove('scanning');
    console.error('Verification failed:', err);
    alert('Verification failed. See console.');
  }
}

function renderVerificationDecision(result) {
  const container = document.getElementById('decision-container');
  const banner = document.getElementById('verdict-banner');
  const icon = document.getElementById('verdict-icon');
  const title = document.getElementById('verdict-title');
  const sub = document.getElementById('verdict-subtitle');
  const confText = document.getElementById('stat-confidence');
  const confBar = document.getElementById('confidence-bar');
  const distText = document.getElementById('stat-distance');

  container.style.display = 'block';

  const score = result.confidence * 100;
  confText.textContent = `${score.toFixed(1)}%`;
  confBar.style.width = `${Math.max(2, score)}%`;
  distText.textContent = (1 - result.confidence).toFixed(4);

  if (result.is_match) {
    banner.className = 'decision-verdict match';
    icon.textContent = '✓';
    title.textContent = 'AUTHENTIC BIOMETRIC MATCH';
    sub.textContent = `Siamese CNN output meets match threshold (${score.toFixed(1)}% >= 50.0%). Same identity confirmed.`;
  } else {
    banner.className = 'decision-verdict mismatch';
    icon.textContent = '✗';
    title.textContent = 'IMPOSTER / MISMATCH REJECTED';
    sub.textContent = `Siamese CNN score (${score.toFixed(1)}% < 50.0%). Fingerprint features do not match candidate identity.`;
  }
}

// 1:N Search UI
function initSearchControls() {
  document.getElementById('btn-search-rand-probe').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/random_query?difficulty=medium');
      const data = await res.json();
      if (data.status === 'ok') {
        searchProbe = data;
        const img = document.getElementById('search-probe-img');
        img.src = data.image;
        img.style.display = 'block';
        document.getElementById('search-probe-placeholder').style.display = 'none';
        document.getElementById('search-target-key').textContent = data.meta.key;
        document.getElementById('search-target-desc').textContent = `${data.meta.gender}, ${data.meta.hand}, ${data.meta.finger_name}`;
      }
    } catch (err) {
      console.error('Failed to pick probe:', err);
    }
  });

  const uploadSearchProbe = document.getElementById('input-upload-search-probe');
  if (uploadSearchProbe) {
    uploadSearchProbe.addEventListener('change', async (e) => {
      if (e.target.files && e.target.files[0]) {
        await handleCustomSearchProbeUpload(e.target.files[0]);
      }
    });
  }

  document.getElementById('btn-run-search').addEventListener('click', async () => {
    if (!searchProbe) {
      alert('Please select or upload a probe fingerprint first.');
      return;
    }
    const status = document.getElementById('search-status');
    const grid = document.getElementById('candidates-grid');
    status.textContent = 'Searching gallery of 6,000 real fingerprints with Siamese CNN...';
    grid.innerHTML = '<div class="text-muted">Computing metric embeddings...</div>';

    try {
      const res = await fetch('/api/identify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query_image: searchProbe.image,
          query_key: searchProbe.meta ? searchProbe.meta.key : '',
          top_k: 6
        })
      });
      const data = await res.json();
      renderSearchResults(data);
    } catch (err) {
      status.textContent = 'Search failed.';
      console.error(err);
    }
  });
}

async function handleCustomSearchProbeUpload(file) {
  try {
    const base64Data = await processImageFile(file);
    searchProbe = {
      image: base64Data,
      meta: {
        key: 'CUSTOM_PROBE',
        gender: 'User Upload',
        hand: 'Custom',
        finger_name: file.name.substring(0, 16)
      }
    };
    const img = document.getElementById('search-probe-img');
    img.src = base64Data;
    img.style.display = 'block';
    document.getElementById('search-probe-placeholder').style.display = 'none';
    document.getElementById('search-target-key').textContent = 'CUSTOM_PROBE';
    document.getElementById('search-target-desc').textContent = file.name;
  } catch (err) {
    alert('Failed to process search probe image: ' + err.message);
  }
}

function renderSearchResults(data) {
  const status = document.getElementById('search-status');
  const grid = document.getElementById('candidates-grid');
  status.textContent = `Search Complete in ${data.latency_ms} ms. Top matching candidates:`;
  grid.innerHTML = '';

  data.candidates.forEach((cand, idx) => {
    const card = document.createElement('div');
    card.className = `candidate-card ${cand.is_true_match ? 'is-match' : ''}`;
    card.innerHTML = `
      <div class="rank-badge">#${idx + 1}</div>
      <img src="${cand.image}" style="width: 100px; height: 100px; object-fit: cover; border-radius: 6px;">
      <div style="font-size: 0.85rem; font-weight: 700; color: ${cand.is_true_match ? '#10b981' : '#fff'};">
        Confidence: ${(cand.confidence * 100).toFixed(1)}%
      </div>
      <div style="font-size: 0.75rem; color: #94a3b8; text-align: center;">
        Key: <strong>${cand.key}</strong><br>
        Subject ${cand.meta.subject_id} (${cand.meta.gender})<br>
        ${cand.meta.hand}, ${cand.meta.finger_name}
      </div>
      ${cand.is_true_match ? '<div style="font-size: 0.72rem; color: #10b981; font-weight: bold;">✓ TRUE IDENTITY</div>' : ''}
    `;
    grid.appendChild(card);
  });
}

// Distortion Simulator
function initAugmentControls() {
  const sliders = ['blur', 'rotate', 'scale', 'shift-x', 'shift-y', 'noise'];
  sliders.forEach(id => {
    const el = document.getElementById(`slider-${id}`);
    const valEl = document.getElementById(`val-${id}`);
    el.addEventListener('input', () => {
      let suffix = '';
      if (id === 'rotate') suffix = '°';
      else if (id === 'scale') suffix = 'x';
      else if (id.includes('shift')) suffix = ' px';
      valEl.textContent = `${el.value}${suffix}`;
    });
  });

  document.getElementById('btn-reset-augment').addEventListener('click', () => {
    document.getElementById('slider-blur').value = 0;
    document.getElementById('slider-rotate').value = 0;
    document.getElementById('slider-scale').value = 1.0;
    document.getElementById('slider-shift-x').value = 0;
    document.getElementById('slider-shift-y').value = 0;
    document.getElementById('slider-noise').value = 0;
    sliders.forEach(id => {
      const el = document.getElementById(`slider-${id}`);
      el.dispatchEvent(new Event('input'));
    });
  });

  document.getElementById('btn-apply-augment').addEventListener('click', () => applyAugmentation());
}

async function loadAugmentProbe() {
  try {
    const res = await fetch('/api/random_query?difficulty=easy');
    const data = await res.json();
    if (data.status === 'ok') {
      currentAugQuery = data;
      document.getElementById('aug-distorted-img').src = data.image;
      const resT = await fetch(`/api/get_genuine?key=${data.meta.key}`);
      const dataT = await resT.json();
      if (dataT.status === 'ok') {
        document.getElementById('aug-ground-truth-img').src = dataT.image;
      }
      applyAugmentation();
    }
  } catch (err) {
    console.error(err);
  }
}

async function applyAugmentation() {
  if (!currentAugQuery) return;

  const params = {
    query_image: currentAugQuery.image,
    target_key: currentAugQuery.meta.key,
    blur: parseFloat(document.getElementById('slider-blur').value),
    rotate: parseFloat(document.getElementById('slider-rotate').value),
    scale: parseFloat(document.getElementById('slider-scale').value),
    shift_x: parseInt(document.getElementById('slider-shift-x').value),
    shift_y: parseInt(document.getElementById('slider-shift-y').value),
    noise: parseFloat(document.getElementById('slider-noise').value)
  };

  try {
    const res = await fetch('/api/augment_test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    const result = await res.json();

    document.getElementById('aug-distorted-img').src = result.distorted_image;
    const score = result.confidence * 100;
    document.getElementById('aug-score-val').textContent = `${score.toFixed(1)}%`;
    document.getElementById('aug-progress-bar').style.width = `${Math.max(2, score)}%`;

    const statusText = document.getElementById('aug-status-text');
    if (result.is_match) {
      statusText.innerHTML = `<span style="color: #10b981; font-weight: bold;">✓ VERIFIED MATCH</span> — Neural network successfully recognized the identity despite synthetic distortion.`;
    } else {
      statusText.innerHTML = `<span style="color: #ef4444; font-weight: bold;">✗ RECOGNITION DEGRADED</span> — Heavy distortion dropped confidence below match threshold.`;
    }
  } catch (err) {
    console.error(err);
  }
}

// Gallery Explorer
function initExplorerControls() {
  document.getElementById('btn-filter-db').addEventListener('click', () => {
    explorerPage = 1;
    loadExplorerGallery();
  });

  document.getElementById('btn-prev-page').addEventListener('click', () => {
    if (explorerPage > 1) {
      explorerPage--;
      loadExplorerGallery();
    }
  });

  document.getElementById('btn-next-page').addEventListener('click', () => {
    if (explorerPage < totalExplorerPages) {
      explorerPage++;
      loadExplorerGallery();
    }
  });
}

async function loadExplorerGallery() {
  const subject = document.getElementById('filter-subject').value;
  const gender = document.getElementById('filter-gender').value;
  const hand = document.getElementById('filter-hand').value;
  const finger = document.getElementById('filter-finger').value;

  const url = `/api/real_fingerprints?page=${explorerPage}&limit=24&subject=${subject}&gender=${gender}&hand=${hand}&finger=${finger}`;
  try {
    const res = await fetch(url);
    const data = await res.json();
    totalExplorerPages = data.total_pages || 1;
    document.getElementById('page-indicator').textContent = `Page ${data.page} of ${totalExplorerPages}`;
    document.getElementById('btn-prev-page').disabled = (data.page <= 1);
    document.getElementById('btn-next-page').disabled = (data.page >= totalExplorerPages);

    const grid = document.getElementById('gallery-grid');
    grid.innerHTML = '';

    data.items.forEach(item => {
      const el = document.createElement('div');
      el.className = 'gallery-item';
      el.innerHTML = `
        <img src="${item.image}" alt="${item.key}">
        <div class="gallery-label">${item.key}</div>
        <div style="font-size: 0.68rem; color: #94a3b8; text-align: center;">${item.meta.hand}, ${item.meta.finger_name}</div>
      `;
      el.addEventListener('click', () => {
        setTargetImage(item.image, item.meta);
        document.getElementById('tab-verify').click();
      });
      grid.appendChild(el);
    });
  } catch (err) {
    console.error('Failed to load gallery:', err);
  }
}

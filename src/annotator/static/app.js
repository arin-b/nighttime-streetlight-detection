const STEPS = [
  {id: 'lamp_boxes', label: 'Mark Lamps', card: 'taskLampBoxes', tool: 'box'},
  {id: 'surfaces', label: 'Mark Other Lights', card: 'taskSurfaces', tool: 'smart'},
  {id: 'public_space', label: 'Mark Road/Footpath', card: 'taskPublicSpace', tool: 'polygon'},
  {id: 'affected', label: 'Mark Lit Area', card: 'taskAffected', tool: 'polygon'},
  {id: 'visibility', label: 'Rate Visibility', card: 'taskVisibility', tool: 'box'},
  {id: 'lux_qa', label: 'Field Lux / Notes', card: 'taskLuxQa', tool: 'point'}
];

const LEGACY_STREETLIGHT_CLASS = 'streetlight';
const LAMP_HEAD_CLASS = 'streetlight_lamp_head';
const POLE_CLASS = 'streetlight_pole';
const POLE_NOT_VISIBLE_FLAG = 'pole_not_visible';

const LABELS = {
  streetlight_lamp_head: 'Lamp head / light source',
  streetlight_pole: 'Pole / visible fixture',
  pole_not_visible: 'Pole not visible',
  building_facade: 'Building wall',
  shopfront: 'Shop light/front',
  window: 'Window light',
  sign_lightbox: 'Bright sign',
  reflective_glass: 'Reflective glass',
  wet_road_reflection: 'Wet road reflection',
  wall_compound_surface: 'Boundary wall',
  vehicle_headlight_region: 'Vehicle headlights',
  unknown_bright_source: 'Other bright thing',
  road: 'Road',
  footpath_sidewalk: 'Footpath/sidewalk',
  crossing: 'Crossing',
  curb: 'Curb',
  median: 'Median',
  verge: 'Road edge/verge',
  vegetation: 'Tree/vegetation',
  vehicle: 'Vehicle',
  building_frontage: 'Building front',
  sign_billboard: 'Sign/board',
  traffic_signal: 'Traffic signal',
  sky: 'Sky',
  wet_reflection_like_road: 'Wet shiny road',
  lit_area: 'Lit area',
  visible: 'Visible',
  partly_visible: 'Partly visible',
  not_visible: 'Not properly visible',
  occluder: 'Blocker/occluder',
  unknown: 'Unknown',
  on: 'On',
  dim: 'Dim',
  off: 'Off',
  flicker: 'Flickering',
  occluded: 'Blocked',
  saturated: 'Too bright / blown out',
  good: 'Good',
  adequate: 'Okay',
  marginal: 'Barely okay',
  poor: 'Poor',
  dark: 'Too dark',
  certain: 'Yes, mostly this lamp',
  mixed: 'Mixed with other lights',
  uncertain: 'Not sure',
  impossible_due_to_confounding: 'Cannot tell',
  P1: 'P1 under lamp',
  P2: 'P2 between lamps',
  P3: 'P3 road point',
  P4: 'P4 footpath point',
  P5: 'P5 darkest patch',
  P6: 'P6 near other light',
  P7: 'P7 opposite side',
  P8: 'P8 shadow/blocked side'
};

const QA_FLAGS = [
  'no_lux_reference',
  'proxy_only',
  'glare',
  'headlight_confounder',
  'shopfront_confounder',
  'wet_reflection',
  'tree_occlusion',
  'exposure_problem',
  POLE_NOT_VISIBLE_FLAG
];

const state = {
  bootstrap: null,
  current: null,
  review: null,
  image: null,
  scale: 1,
  step: 0,
  tool: 'box',
  selectedType: '',
  selectedIndex: -1,
  drawing: null,
  polygonDraft: [],
  previewPoint: null,
  selectedPoint: null,
  proposal: null,
  tutorialMode: false,
  tutorialIndex: 0,
  realIndex: 0,
  dirty: false
};

const $ = (id) => document.getElementById(id);

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

function setMessage(text) {
  $('message').textContent = text || '';
  $('statusStrip').textContent = text || `${STEPS[state.step].label} ready`;
  renderActionControls();
  renderWorkbar();
}

function markDirty() {
  state.dirty = true;
  updateSaveState();
  renderSummary();
}

function updateSaveState(extraText = '') {
  const save = $('saveBtn');
  if (save) {
    save.textContent = state.dirty ? 'Save Changes' : 'Saved';
    save.classList.toggle('primary', state.dirty);
  }
  const target = $('workbarSave');
  if (!target) return;
  if (extraText) {
    target.textContent = extraText;
  } else {
    target.textContent = state.dirty ? 'Unsaved changes' : 'Saved locally';
  }
}

function setHidden(id, hidden) {
  const element = $(id);
  if (element) element.classList.toggle('hidden', hidden);
}

function renderActionControls() {
  const stepId = STEPS[state.step]?.id;
  if ($('boxToolBtn')) $('boxToolBtn').textContent = stepId === 'surfaces' ? 'Other-Light Box' : 'Lamp / Pole Box';
  if ($('finishPolygonBtn')) $('finishPolygonBtn').disabled = state.polygonDraft.length < 3;
  setHidden('proposalActions', !state.proposal);
  setHidden('deleteBoxBtn', !(state.selectedType === 'box' && state.selectedIndex >= 0));
  setHidden('deleteShapeBtn', !(state.selectedType === 'polygon' && state.selectedIndex >= 0) && !state.proposal);
  setHidden('deleteOtherBoxBtn', !(state.selectedType === 'confounder_box' && state.selectedIndex >= 0));
  setHidden('undoPointBtn', !state.polygonDraft.length);
  setHidden('cancelShapeBtn', !state.polygonDraft.length && !state.proposal);
  setHidden('updatePolygonBtn', !['polygon', 'confounder_box'].includes(state.selectedType));
}

function currentActionText() {
  const stepId = STEPS[state.step]?.id;
  if (state.proposal) return 'Review the suggested surface, then keep it or try again';
  if (state.polygonDraft.length) return `Drawing shape: ${state.polygonDraft.length} point${state.polygonDraft.length === 1 ? '' : 's'} placed`;
  if (state.tool === 'box' && stepId === 'surfaces') return 'Drag a box around another light source';
  if (state.tool === 'box') return 'Drag a lamp-head or pole box';
  if (state.tool === 'smart') return 'Drag around a surface for automatic shape finding';
  if (state.tool === 'polygon') return 'Click corners of the visible area, then finish the shape';
  if (state.tool === 'point') return 'Click a measured lux point or select a shape';
  return STEPS[state.step]?.label || 'Ready';
}

function selectedText() {
  if (state.selectedType === 'box' && state.selectedIndex >= 0) {
    const box = state.review?.boxes?.[state.selectedIndex];
    if (box) return `${state.selectedIndex + 1}. ${boxClassLabel(box)} (${box.status || 'candidate'})`;
  }
  if (state.selectedType === 'confounder_box' && state.selectedIndex >= 0) {
    const box = state.review?.confounder_boxes?.[state.selectedIndex];
    if (box) return `${state.selectedIndex + 1}. Other light: ${LABELS[box.surface_type] || box.surface_type || 'untyped'}`;
  }
  if (state.selectedType === 'polygon' && state.selectedIndex >= 0) {
    const polygon = state.review?.polygons?.[state.selectedIndex];
    if (polygon) return `${state.selectedIndex + 1}. Shape: ${LABELS[polygon.surface_type] || polygon.surface_type || 'untyped'}`;
  }
  if (state.selectedType === 'public_region' && state.selectedIndex >= 0) {
    const row = state.review?.measurement?.public_space_regions?.[state.selectedIndex];
    if (row) return `${state.selectedIndex + 1}. Road/footpath: ${LABELS[row.region_type] || row.region_type || 'area'}`;
  }
  if (state.selectedType === 'affected_region' && state.selectedIndex >= 0) {
    const row = state.review?.measurement?.affected_regions?.[state.selectedIndex];
    if (row) return `${state.selectedIndex + 1}. Lit area: ${LABELS[row.region_type] || row.region_type || 'area'}`;
  }
  if (state.selectedType === 'lux_point' && state.selectedIndex >= 0) {
    const point = state.review?.measurement?.lux_points?.[state.selectedIndex];
    if (point) return `${state.selectedIndex + 1}. Lux point: ${LABELS[point.point_type] || point.point_type || 'point'}`;
  }
  if (state.selectedType === 'visibility_label' && state.selectedIndex >= 0) {
    const row = state.review?.measurement?.visibility_labels?.[state.selectedIndex];
    if (row) return `${state.selectedIndex + 1}. Visibility: ${row.track_id || 'untracked'} ${LABELS[row.visibility_class] || row.visibility_class || ''}`;
  }
  if (state.selectedType === 'lamp_status_label' && state.selectedIndex >= 0) {
    const row = state.review?.measurement?.lamp_status?.[state.selectedIndex];
    if (row) return `${state.selectedIndex + 1}. Lamp status: ${row.track_id || 'untracked'} ${LABELS[row.status] || row.status || ''}`;
  }
  if (state.selectedType === 'attribution_label' && state.selectedIndex >= 0) {
    const row = state.review?.measurement?.attribution_labels?.[state.selectedIndex];
    if (row) return `${state.selectedIndex + 1}. Attribution: ${row.track_id || 'untracked'} ${LABELS[row.attribution_class] || row.attribution_class || ''}`;
  }
  if (state.selectedPoint) return `Point at ${Math.round(state.selectedPoint.x)}, ${Math.round(state.selectedPoint.y)}`;
  return 'Nothing selected';
}

function renderWorkbar() {
  if ($('workbarAction')) $('workbarAction').textContent = currentActionText();
  if ($('workbarSelection')) $('workbarSelection').textContent = selectedText();
  if ($('workbarSave') && !$('workbarSave').textContent) updateSaveState();
}

function populateSelect(id, values) {
  const select = $(id);
  select.innerHTML = '';
  for (const value of values) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = LABELS[value] || value;
    select.appendChild(option);
  }
}

async function init() {
  state.bootstrap = await fetchJson('/api/bootstrap');
  updateWorkspaceStatus();
  populateSelect('lampBoxClass', state.bootstrap.lamp_box_classes || [LAMP_HEAD_CLASS, POLE_CLASS]);
  populateSelect('lampDrawClass', state.bootstrap.lamp_box_classes || [LAMP_HEAD_CLASS, POLE_CLASS]);
  populateSelect('surfaceType', state.bootstrap.surface_types);
  populateSelect('lampStatus', state.bootstrap.lamp_status_classes);
  populateSelect('visibilityClass', state.bootstrap.visibility_classes);
  populateSelect('attributionClass', state.bootstrap.attribution_classes);
  populateSelect('luxType', state.bootstrap.lux_point_types);
  populateSelect('regionType', state.bootstrap.public_space_types);
  renderQaChips();
  renderStepper();
  setStep(0);
  if (state.bootstrap.tutorial?.examples?.length && !state.bootstrap.bundle_state?.tutorial_completed) {
    await loadTutorial(0);
  } else {
    await loadItemByIndex(0);
  }
}

function updateWorkspaceStatus() {
  if (!state.bootstrap) return;
  $('workspaceStatus').textContent = `${state.bootstrap.reviewed}/${state.bootstrap.total} reviewed`;
}

async function refreshWorkspaceStatus() {
  try {
    const bootstrap = await fetchJson('/api/bootstrap');
    state.bootstrap.reviewed = bootstrap.reviewed;
    state.bootstrap.total = bootstrap.total;
    updateWorkspaceStatus();
  } catch (_error) {
    // Keep annotation flow moving even if the count refresh fails.
  }
}

function renderStepper() {
  const wrap = $('stepper');
  wrap.innerHTML = '';
  STEPS.forEach((step, index) => {
    const button = document.createElement('button');
    button.className = `step-button ${index === state.step ? 'active' : ''}`;
    button.innerHTML = `<span class="step-number">${index + 1}</span><span>${step.label}</span>`;
    button.onclick = () => setStep(index);
    wrap.appendChild(button);
  });
}

function renderQaChips() {
  const wrap = $('qaChips');
  wrap.innerHTML = '';
  for (const flag of QA_FLAGS) {
    const button = document.createElement('button');
    button.className = 'chip';
    button.textContent = LABELS[flag] || flag.replaceAll('_', ' ');
    button.onclick = () => {
      $('qaFlag').value = flag;
      addQa();
    };
    wrap.appendChild(button);
  }
}

function setStep(index) {
  state.step = Math.max(0, Math.min(index, STEPS.length - 1));
  for (const step of STEPS) $(step.card).classList.add('hidden');
  $(STEPS[state.step].card).classList.remove('hidden');
  renderStepper();
  setTool(STEPS[state.step].tool);
  renderActionControls();
  setMessage('');
}

async function loadItemByIndex(index) {
  state.tutorialMode = false;
  state.realIndex = index;
  await loadPayload(await fetchJson(`/api/item?index=${index}`));
}

async function loadItemByKey(key) {
  state.tutorialMode = false;
  await loadPayload(await fetchJson(`/api/item?key=${encodeURIComponent(key)}`));
}

async function loadTutorial(index) {
  state.tutorialMode = true;
  state.tutorialIndex = index;
  await loadPayload(await fetchJson(`/api/tutorial/item?index=${index}`));
}

async function loadPayload(payload) {
  state.current = payload;
  state.review = payload.review;
  state.review.boxes = state.review.boxes || [];
  for (const box of state.review.boxes) {
    box.class_name = normalizeBoxClass(box.class_name);
    box.parent_pole_box_id = box.parent_pole_box_id || '';
  }
  state.review.confounder_boxes = state.review.confounder_boxes || [];
  state.review.polygons = state.review.polygons || [];
  ensureMeasurement();
  state.selectedIndex = -1;
  state.selectedType = '';
  state.polygonDraft = [];
  state.previewPoint = null;
  state.selectedPoint = null;
  state.proposal = null;
  state.dirty = false;
  updateSaveState('No edits on this frame');
  renderTutorialPanel(payload);
  $('poleNotVisible').checked = hasQaFlag(POLE_NOT_VISIBLE_FLAG);
  $('itemTitle').textContent = payload.item.key;
  $('itemSubtitle').textContent = `${payload.index + 1} / ${payload.total} | ${payload.item.clip_id || 'clip'} | frame ${payload.item.frame_id || ''}`;
  $('prevBtn').disabled = !payload.prev_key;
  $('nextBtn').disabled = !payload.next_key;
  $('prevBtn').onclick = () => {
    if (!payload.prev_key) return;
    if (state.tutorialMode) loadTutorial(Number(payload.prev_key));
    else loadItemByKey(payload.prev_key);
  };
  $('nextBtn').onclick = () => {
    if (!payload.next_key) return;
    if (state.tutorialMode) loadTutorial(Number(payload.next_key));
    else loadItemByKey(payload.next_key);
  };
  await loadImage(payload.item.key);
  renderLists();
  renderSummary();
  renderWorkbar();
}

function renderTutorialPanel(payload) {
  const panel = $('tutorialPanel');
  if (!panel) return;
  if (!payload.tutorial) {
    panel.classList.add('hidden');
    return;
  }
  panel.classList.remove('hidden');
  $('tutorialTitle').textContent = payload.tutorial.title || 'Tutorial';
  $('tutorialLesson').textContent = payload.tutorial.lesson || 'Follow the highlighted correct labels, then start real work.';
  $('tutorialPrevBtn').disabled = !payload.prev_key;
  $('tutorialNextBtn').disabled = !payload.next_key;
  $('tutorialPrevBtn').onclick = () => payload.prev_key && loadTutorial(Number(payload.prev_key));
  $('tutorialNextBtn').onclick = () => payload.next_key && loadTutorial(Number(payload.next_key));
  $('tutorialReplayBtn').onclick = () => {
    state.review = clonePayload(payload.gold_review || payload.review);
    renderLists();
    renderSummary();
    draw();
    setMessage('Correct labels replayed for this tutorial example.');
  };
  $('tutorialSkipBtn').onclick = async () => {
    try {
      await fetchJson('/api/tutorial/complete', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({completed: true})});
    } catch (_error) {
      // Continue to real work even if the completion marker could not be saved.
    }
    state.tutorialMode = false;
    panel.classList.add('hidden');
    await loadItemByIndex(state.realIndex || 0);
  };
}

function clonePayload(payload) {
  return JSON.parse(JSON.stringify(payload));
}

function loadImage(key) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      state.image = img;
      resizeCanvas();
      draw();
      resolve();
    };
    img.onerror = reject;
    img.src = `/image?key=${encodeURIComponent(key)}&t=${Date.now()}`;
  });
}

function resizeCanvas() {
  const canvas = $('canvas');
  const wrap = document.querySelector('.canvas-wrap');
  const maxWidth = Math.max(220, wrap.clientWidth - 24);
  const maxHeight = Math.max(220, wrap.clientHeight - 24);
  state.scale = Math.min(1, maxWidth / state.image.naturalWidth, maxHeight / state.image.naturalHeight);
  canvas.width = Math.round(state.image.naturalWidth * state.scale);
  canvas.height = Math.round(state.image.naturalHeight * state.scale);
}

function imageCoords(event) {
  const rect = $('canvas').getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) / state.scale,
    y: (event.clientY - rect.top) / state.scale
  };
}

function draw() {
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.image) return;
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  drawPolygons(ctx);
  drawBoxes(ctx);
  drawLuxPoints(ctx);
  if (state.proposal) drawProposal(ctx, state.proposal.points, state.proposal.bbox_xyxy);
  if (state.polygonDraft.length) {
    drawPolyline(ctx, state.polygonDraft, '#ffd166', false);
    if (state.previewPoint) drawPolyline(ctx, [state.polygonDraft[state.polygonDraft.length - 1], [state.previewPoint.x, state.previewPoint.y]], '#ffd166', false);
    drawStartPoint(ctx);
  }
  if (state.drawing && (state.drawing.kind === 'box' || state.drawing.kind === 'smart')) {
    drawBox(ctx, state.drawing.box, '#ffd166', state.drawing.kind === 'smart' ? 'smart' : 'new');
  }
}

function drawBoxes(ctx) {
  for (const [index, box] of (state.review.boxes || []).entries()) {
    const isSelected = state.selectedType === 'box' && state.selectedIndex === index;
    const color = isSelected ? '#ffd166' : boxColor(box);
    drawBox(ctx, box.bbox_xyxy, color, `${index + 1}:${boxShortLabel(box)}`);
  }
  for (const [index, box] of (state.review.confounder_boxes || []).entries()) {
    const color = state.selectedType === 'confounder_box' && state.selectedIndex === index ? '#ffd166' : '#f28482';
    const label = LABELS[box.surface_type] || box.surface_type || 'other';
    drawBox(ctx, box.bbox_xyxy, color, `O${index + 1}:${label}`);
  }
}

function drawBox(ctx, rawBox, color, label) {
  const [x1, y1, x2, y2] = rawBox.map(v => v * state.scale);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.fillStyle = color;
  ctx.font = '12px Segoe UI';
  ctx.fillText(label, x1 + 4, Math.max(14, y1 - 4));
}

function drawPolygons(ctx) {
  for (const [index, polygon] of (state.review.polygons || []).entries()) {
    const color = state.selectedType === 'polygon' && state.selectedIndex === index ? '#ffd166' : '#80ed99';
    drawPolyline(ctx, polygon.points || [], color, true, `${index + 1}:${polygon.surface_type || 'polygon'}`);
  }
  for (const row of state.review.measurement?.public_space_regions || []) {
    const index = (state.review.measurement?.public_space_regions || []).indexOf(row);
    const color = state.selectedType === 'public_region' && state.selectedIndex === index ? '#ffd166' : '#9cc8ff';
    drawPolyline(ctx, row.points || [], color, true, row.region_type || 'public');
  }
  for (const row of state.review.measurement?.affected_regions || []) {
    const index = (state.review.measurement?.affected_regions || []).indexOf(row);
    const color = state.selectedType === 'affected_region' && state.selectedIndex === index ? '#ffd166' : '#f28482';
    drawPolyline(ctx, row.points || [], color, true, row.region_type || 'affected');
  }
}

function drawProposal(ctx, points, bbox) {
  ctx.save();
  ctx.globalAlpha = 0.35;
  drawFilledPolygon(ctx, points, '#ffd166');
  ctx.globalAlpha = 1;
  drawPolyline(ctx, points, '#ffd166', true, 'proposal');
  if (bbox) drawBox(ctx, bbox, '#ffd166', 'prompt');
  ctx.restore();
}

function drawFilledPolygon(ctx, points, color) {
  if (!points || points.length < 3) return;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(points[0][0] * state.scale, points[0][1] * state.scale);
  for (const point of points.slice(1)) ctx.lineTo(point[0] * state.scale, point[1] * state.scale);
  ctx.closePath();
  ctx.fill();
}

function drawPolyline(ctx, points, color, closed, label) {
  if (!points || !points.length) return;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(points[0][0] * state.scale, points[0][1] * state.scale);
  for (const point of points.slice(1)) ctx.lineTo(point[0] * state.scale, point[1] * state.scale);
  if (closed && points.length > 2) ctx.closePath();
  ctx.stroke();
  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point[0] * state.scale, point[1] * state.scale, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  if (label) ctx.fillText(label, points[0][0] * state.scale + 4, points[0][1] * state.scale - 4);
}

function drawLuxPoints(ctx) {
  const points = state.review.measurement?.lux_points || [];
  for (const [index, point] of points.entries()) {
    ctx.fillStyle = state.selectedType === 'lux_point' && state.selectedIndex === index ? '#ffd166' : '#ff4d8d';
    ctx.beginPath();
    ctx.arc(point.x * state.scale, point.y * state.scale, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(`${index + 1}:${point.point_type}`, point.x * state.scale + 6, point.y * state.scale - 5);
  }
  if (state.selectedPoint) {
    ctx.fillStyle = '#ffd166';
    ctx.beginPath();
    ctx.arc(state.selectedPoint.x * state.scale, state.selectedPoint.y * state.scale, 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawStartPoint(ctx) {
  if (!state.polygonDraft.length) return;
  const first = state.polygonDraft[0];
  ctx.save();
  ctx.strokeStyle = '#ffd166';
  ctx.fillStyle = 'rgba(255, 209, 102, 0.25)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(first[0] * state.scale, first[1] * state.scale, 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function renderLists() {
  renderBoxList();
  renderParentPoleOptions();
  renderConfounderBoxList();
  renderPolygonList();
  renderMeasurementRegionLists();
  renderLuxPointList();
  renderVisibilityLabelList();
  renderTrackSelectors();
  renderItemList();
}

function renderBoxList() {
  const wrap = $('boxList');
  wrap.innerHTML = '';
  const boxes = state.review.boxes || [];
  if (!boxes.length) {
    wrap.appendChild(emptyRow('No lamp or pole boxes yet'));
    return;
  }
  for (const [index, box] of boxes.entries()) {
    const row = document.createElement('div');
    row.className = `row ${state.selectedType === 'box' && state.selectedIndex === index ? 'active' : ''}`;
    row.innerHTML = `<span>${index + 1}. ${boxClassLabel(box)} - ${box.status || 'candidate'} ${box.track_id || ''}</span><button class="row-delete" title="Delete lamp box">Delete</button>`;
    row.onclick = () => selectBox(index);
    row.querySelector('.row-delete').onclick = (event) => {
      event.stopPropagation();
      deleteBox(index);
    };
    wrap.appendChild(row);
  }
}

function renderParentPoleOptions() {
  const select = $('parentPoleBoxId');
  if (!select) return;
  const current = state.selectedType === 'box' && state.selectedIndex >= 0 ? (state.review.boxes[state.selectedIndex]?.parent_pole_box_id || '') : '';
  select.innerHTML = '';
  const blank = document.createElement('option');
  blank.value = '';
  blank.textContent = 'No linked pole';
  select.appendChild(blank);
  for (const [index, box] of (state.review.boxes || []).entries()) {
    if (normalizeBoxClass(box.class_name) !== POLE_CLASS) continue;
    const option = document.createElement('option');
    option.value = box.box_id || `box_${index + 1}`;
    option.textContent = `${index + 1}. ${box.box_id || 'pole'} ${box.status || ''}`;
    select.appendChild(option);
  }
  select.value = current;
}

function renderConfounderBoxList() {
  const wrap = $('confounderBoxList');
  if (!wrap) return;
  wrap.innerHTML = '';
  const boxes = state.review.confounder_boxes || [];
  if (!boxes.length) {
    wrap.appendChild(emptyRow('No other-light boxes yet'));
    return;
  }
  for (const [index, box] of boxes.entries()) {
    const row = document.createElement('div');
    row.className = `row ${state.selectedType === 'confounder_box' && state.selectedIndex === index ? 'active' : ''}`;
    const label = LABELS[box.surface_type] || box.surface_type || 'other light';
    row.innerHTML = `<span>${index + 1}. ${label}</span><button class="row-delete" title="Delete other-light box">Delete</button>`;
    row.onclick = () => selectConfounderBox(index);
    row.querySelector('.row-delete').onclick = (event) => {
      event.stopPropagation();
      deleteConfounderBox(index);
    };
    wrap.appendChild(row);
  }
}

function renderPolygonList() {
  const wrap = $('polygonList');
  wrap.innerHTML = '';
  const polygons = state.review.polygons || [];
  if (!polygons.length) {
    wrap.appendChild(emptyRow('No surface shapes yet'));
    return;
  }
  for (const [index, polygon] of polygons.entries()) {
    const row = document.createElement('div');
    row.className = `row ${state.selectedType === 'polygon' && state.selectedIndex === index ? 'active' : ''}`;
    const label = LABELS[polygon.surface_type] || polygon.surface_type || 'shape';
    row.innerHTML = `<span>${index + 1}. ${label}</span><button class="row-delete" title="Delete shape">Delete</button>`;
    row.onclick = () => selectPolygon(index);
    row.querySelector('.row-delete').onclick = (event) => {
      event.stopPropagation();
      deletePolygon(index);
    };
    wrap.appendChild(row);
  }
}

function renderMeasurementRegionLists() {
  renderRegionList('publicRegionList', 'public');
  renderRegionList('affectedRegionList', 'affected');
}

function renderRegionList(id, kind) {
  const wrap = $(id);
  if (!wrap) return;
  const measurement = ensureMeasurement();
  const rows = kind === 'public' ? measurement.public_space_regions : measurement.affected_regions;
  wrap.innerHTML = '';
  if (!rows.length) {
    wrap.appendChild(emptyRow(kind === 'public' ? 'No saved road/footpath areas yet' : 'No saved lit areas yet'));
    return;
  }
  for (const [index, row] of rows.entries()) {
    const element = document.createElement('div');
    const selected = state.selectedType === kindRegionType(kind) && state.selectedIndex === index;
    element.className = `row ${selected ? 'active' : ''}`;
    const baseLabel = kind === 'affected' ? 'Lit area' : (LABELS[row.region_type] || row.region_type || 'public area');
    const visibilityText = kind === 'affected' && row.visibility_quality ? ` (${LABELS[row.visibility_quality] || row.visibility_quality})` : '';
    const typeLabel = `${baseLabel}${visibilityText}`;
    const trackText = row.track_id ? ` - ${row.track_id}` : '';
    element.innerHTML = `<span>${index + 1}. ${typeLabel}${trackText}</span><button class="row-delete" title="Delete saved region">Delete</button>`;
    element.onclick = () => selectRegion(kind, index);
    element.querySelector('.row-delete').onclick = (event) => {
      event.stopPropagation();
      deleteRegion(kind, index);
    };
    wrap.appendChild(element);
  }
}

function renderLuxPointList() {
  const wrap = $('luxPointList');
  if (!wrap) return;
  const rows = ensureMeasurement().lux_points || [];
  wrap.innerHTML = '';
  if (!rows.length) {
    wrap.appendChild(emptyRow('No saved lux points yet'));
    return;
  }
  for (const [index, point] of rows.entries()) {
    const element = document.createElement('div');
    element.className = `row ${state.selectedType === 'lux_point' && state.selectedIndex === index ? 'active' : ''}`;
    const label = LABELS[point.point_type] || point.point_type || 'point';
    const value = point.lux_value ? ` - ${point.lux_value} lux` : '';
    element.innerHTML = `<span>${index + 1}. ${label}${value}</span><button class="row-delete" title="Delete lux point">Delete</button>`;
    element.onclick = () => selectLuxPoint(index);
    element.querySelector('.row-delete').onclick = (event) => {
      event.stopPropagation();
      deleteLuxPoint(index);
    };
    wrap.appendChild(element);
  }
}

function renderVisibilityLabelList() {
  const wrap = $('visibilityLabelList');
  if (!wrap) return;
  const measurement = ensureMeasurement();
  const rows = [
    ...(measurement.lamp_status || []).map((row, index) => ({...row, _kind: 'lamp_status', _index: index})),
    ...(measurement.visibility_labels || []).map((row, index) => ({...row, _kind: 'visibility', _index: index})),
    ...(measurement.attribution_labels || []).map((row, index) => ({...row, _kind: 'attribution', _index: index})),
  ];
  wrap.innerHTML = '';
  if (!rows.length) {
    wrap.appendChild(emptyRow('No per-lamp ratings yet'));
    return;
  }
  for (const row of rows) {
    const element = document.createElement('div');
    const selected = state.selectedType === `${row._kind}_label` && state.selectedIndex === row._index;
    element.className = `row ${selected ? 'active' : ''}`;
    const value = row.status || row.visibility_class || row.attribution_class || 'rating';
    element.innerHTML = `<span>${row.track_id || 'untracked'} - ${LABELS[value] || value}</span><button class="row-delete" title="Delete rating">Delete</button>`;
    element.onclick = () => selectVisibilityRow(row._kind, row._index);
    element.querySelector('.row-delete').onclick = (event) => {
      event.stopPropagation();
      deleteVisibilityRow(row._kind, row._index);
    };
    wrap.appendChild(element);
  }
}

function trackOptions() {
  const rows = [{value: '', label: 'No track selected'}];
  for (const [index, box] of (state.review.boxes || []).entries()) {
    if (normalizeBoxClass(box.class_name) !== LAMP_HEAD_CLASS) continue;
    const value = box.track_id || box.box_id || `box_${index + 1}`;
    rows.push({value, label: `${index + 1}. Lamp head - ${value}`});
  }
  return rows;
}

function renderTrackSelectors() {
  for (const id of ['affectedTrackSelect', 'visibilityTrackSelect']) {
    const select = $(id);
    if (!select) continue;
    const current = select.value || $('measurementTrackId')?.value || '';
    select.innerHTML = '';
    for (const row of trackOptions()) {
      const option = document.createElement('option');
      option.value = row.value;
      option.textContent = row.label;
      select.appendChild(option);
    }
    if ([...select.options].some(option => option.value === current)) select.value = current;
  }
}

function kindRegionType(kind) {
  return kind === 'public' ? 'public_region' : 'affected_region';
}

function emptyRow(text) {
  const row = document.createElement('div');
  row.className = 'row empty-row';
  row.textContent = text;
  return row;
}

function renderItemList() {
  const wrap = $('itemList');
  wrap.innerHTML = '';
  const currentIndex = state.current?.index || 0;
  for (let delta = -4; delta <= 4; delta++) {
    const index = currentIndex + delta;
    if (index < 0 || index >= state.current.total) continue;
    const row = document.createElement('div');
    row.className = `row ${index === currentIndex ? 'active' : ''}`;
    row.textContent = `Item ${index + 1}`;
    row.onclick = () => loadItemByIndex(index);
    wrap.appendChild(row);
  }
}

function selectBox(index) {
  state.selectedType = 'box';
  state.selectedIndex = index;
  const box = state.review.boxes[index];
  box.class_name = normalizeBoxClass(box.class_name);
  $('lampBoxClass').value = box.class_name;
  $('boxStatus').value = box.status || 'candidate';
  $('boxTrackId').value = box.track_id || '';
  $('boxNotes').value = box.notes || '';
  renderParentPoleOptions();
  $('parentPoleBoxId').value = box.parent_pole_box_id || '';
  $('parentPoleBoxId').disabled = box.class_name !== LAMP_HEAD_CLASS;
  $('measurementTrackId').value = box.track_id || $('measurementTrackId').value;
  if (box.track_id || box.box_id) setTargetTrack(box.track_id || box.box_id);
  renderLists();
  draw();
  renderActionControls();
  renderWorkbar();
}

function selectConfounderBox(index) {
  state.selectedType = 'confounder_box';
  state.selectedIndex = index;
  const box = state.review.confounder_boxes[index];
  $('surfaceType').value = box.surface_type || state.bootstrap.surface_types[0];
  $('polyBright').checked = !!box.is_bright_source;
  $('polyReflective').checked = !!box.is_reflective;
  $('polyPublic').checked = !!box.is_public_space;
  $('polyConfounds').checked = box.can_confound_streetlight !== false;
  $('polyOverlaps').checked = !!box.overlaps_affected_region;
  $('polyAugAllowed').checked = !!box.augmentation_allowed;
  $('polyMargin').value = box.mask_exclusion_margin_px || 12;
  renderLists();
  draw();
  renderActionControls();
  renderWorkbar();
}

function selectPolygon(index) {
  state.selectedType = 'polygon';
  state.selectedIndex = index;
  const polygon = state.review.polygons[index];
  $('surfaceType').value = polygon.surface_type || state.bootstrap.surface_types[0];
  $('polyBright').checked = !!polygon.is_bright_source;
  $('polyReflective').checked = !!polygon.is_reflective;
  $('polyPublic').checked = !!polygon.is_public_space;
  $('polyConfounds').checked = polygon.can_confound_streetlight !== false;
  $('polyOverlaps').checked = !!polygon.overlaps_affected_region;
  $('polyAugAllowed').checked = !!polygon.augmentation_allowed;
  $('polyMargin').value = polygon.mask_exclusion_margin_px || 12;
  renderLists();
  draw();
  renderActionControls();
  renderWorkbar();
}

function updateSelectedBox() {
  if (state.selectedType !== 'box' || state.selectedIndex < 0) return;
  const box = state.review.boxes[state.selectedIndex];
  box.class_name = normalizeBoxClass($('lampBoxClass').value);
  box.status = $('boxStatus').value;
  box.track_id = $('boxTrackId').value.trim();
  box.parent_pole_box_id = box.class_name === LAMP_HEAD_CLASS ? $('parentPoleBoxId').value : '';
  if (box.class_name === LAMP_HEAD_CLASS && !box.parent_pole_box_id) box.parent_pole_box_id = nearestPoleId(box);
  $('parentPoleBoxId').disabled = box.class_name !== LAMP_HEAD_CLASS;
  box.notes = $('boxNotes').value.trim();
  markDirty();
  renderLists();
  draw();
}

function updateSelectedPolygon() {
  if (state.selectedIndex < 0) return;
  if (state.selectedType === 'polygon') {
    Object.assign(state.review.polygons[state.selectedIndex], polygonMetadata());
  } else if (state.selectedType === 'confounder_box') {
    Object.assign(state.review.confounder_boxes[state.selectedIndex], polygonMetadata(), {surface_type: $('surfaceType').value});
  } else {
    return;
  }
  markDirty();
  renderLists();
  draw();
  renderActionControls();
  renderWorkbar();
}

function polygonMetadata() {
  return {
    surface_type: $('surfaceType').value,
    is_bright_source: $('polyBright').checked,
    is_reflective: $('polyReflective').checked,
    is_public_space: $('polyPublic').checked,
    can_confound_streetlight: $('polyConfounds').checked,
    overlaps_affected_region: $('polyOverlaps').checked,
    augmentation_allowed: $('polyAugAllowed').checked,
    mask_exclusion_margin_px: Number($('polyMargin').value || 12)
  };
}

function ensureMeasurement() {
  if (!state.review.measurement) {
    state.review.measurement = {lamp_status: [], public_space_regions: [], affected_regions: [], visibility_labels: [], attribution_labels: [], lux_points: [], qa_flags: []};
  }
  return state.review.measurement;
}

function addLampStatus() {
  ensureMeasurement().lamp_status.push({track_id: targetTrackId(), status: $('lampStatus').value});
  markDirty();
  renderLists();
  setMessage('Lamp status saved for the selected lamp/track.');
}

function addVisibility() {
  const measurement = ensureMeasurement();
  measurement.visibility_labels.push({track_id: targetTrackId(), visibility_class: $('visibilityClass').value});
  state.selectedType = 'visibility_label';
  state.selectedIndex = measurement.visibility_labels.length - 1;
  markDirty();
  renderLists();
  renderWorkbar();
  setMessage('Visibility saved for the selected lamp/track.');
}

function addAttribution() {
  ensureMeasurement().attribution_labels.push({track_id: targetTrackId(), attribution_class: $('attributionClass').value, evidence: $('attributionEvidence').value.trim()});
  markDirty();
  renderLists();
  setMessage('Attribution saved for the selected lamp/track.');
}

function targetTrackId() {
  return ($('visibilityTrackSelect')?.value || $('affectedTrackSelect')?.value || $('measurementTrackId')?.value || '').trim();
}

function setTargetTrack(value) {
  if ($('measurementTrackId')) $('measurementTrackId').value = value || '';
  if ($('affectedTrackSelect') && [...$('affectedTrackSelect').options].some(option => option.value === value)) $('affectedTrackSelect').value = value;
  if ($('visibilityTrackSelect') && [...$('visibilityTrackSelect').options].some(option => option.value === value)) $('visibilityTrackSelect').value = value;
}

function selectVisibilityRow(kind, index) {
  const measurement = ensureMeasurement();
  const rows = kind === 'lamp_status' ? measurement.lamp_status : kind === 'visibility' ? measurement.visibility_labels : measurement.attribution_labels;
  if (!rows || index < 0 || index >= rows.length) return;
  state.selectedType = `${kind}_label`;
  state.selectedIndex = index;
  setTargetTrack(rows[index].track_id || '');
  renderLists();
  renderWorkbar();
}

function deleteVisibilityRow(kind, index) {
  const measurement = ensureMeasurement();
  const rows = kind === 'lamp_status' ? measurement.lamp_status : kind === 'visibility' ? measurement.visibility_labels : measurement.attribution_labels;
  if (!rows || index < 0 || index >= rows.length) return;
  rows.splice(index, 1);
  state.selectedType = '';
  state.selectedIndex = -1;
  markDirty();
  renderLists();
  renderWorkbar();
  setMessage('Per-lamp rating deleted. Press Save to keep the change.');
}

function deleteAllVisibilityLabels() {
  const measurement = ensureMeasurement();
  const total = (measurement.lamp_status?.length || 0) + (measurement.visibility_labels?.length || 0) + (measurement.attribution_labels?.length || 0);
  if (!total) {
    setMessage('No visibility ratings to delete.');
    return;
  }
  measurement.lamp_status = [];
  measurement.visibility_labels = [];
  measurement.attribution_labels = [];
  state.selectedType = '';
  state.selectedIndex = -1;
  markDirty();
  renderLists();
  renderWorkbar();
  setMessage('All per-lamp visibility ratings deleted. Press Save to keep the change.');
}

function selectedPolygonPoints() {
  if (state.selectedType !== 'polygon' || state.selectedIndex < 0) throw new Error('Select a polygon first.');
  return state.review.polygons[state.selectedIndex].points || [];
}

function addRegion(kind) {
  try {
    const measurement = ensureMeasurement();
    const row = kind === 'public'
      ? {track_id: targetTrackId(), region_type: $('regionType').value, points: selectedPolygonPoints()}
      : {track_id: targetTrackId(), region_type: 'lit_area', visibility_quality: $('affectedVisibility')?.value || 'visible', points: selectedPolygonPoints()};
    const rows = kind === 'public' ? measurement.public_space_regions : measurement.affected_regions;
    rows.push(row);
    markDirty();
    selectRegion(kind, rows.length - 1);
    setMessage(kind === 'public' ? 'Road/footpath area saved. It can be deleted from the saved-area list.' : 'Lit area saved. It can be deleted from the saved-area list.');
  } catch (error) {
    setMessage(error.message);
  }
}

function markAffectedNotVisible() {
  const measurement = ensureMeasurement();
  measurement.affected_regions.push({
    track_id: targetTrackId(),
    region_type: 'lit_area',
    visibility_quality: 'not_visible',
    points: [],
    notes: 'Lit area not properly visible in this frame.'
  });
  markDirty();
  selectRegion('affected', measurement.affected_regions.length - 1);
  setMessage('Lit area marked as not properly visible for the selected lamp/track.');
}

function selectRegion(kind, index) {
  const measurement = ensureMeasurement();
  const rows = kind === 'public' ? measurement.public_space_regions : measurement.affected_regions;
  if (index < 0 || index >= rows.length) return;
  state.selectedType = kindRegionType(kind);
  state.selectedIndex = index;
  const row = rows[index];
  if (row.track_id) $('measurementTrackId').value = row.track_id;
  if (row.track_id) setTargetTrack(row.track_id);
  if (kind === 'public' && row.region_type && $('regionType')) $('regionType').value = row.region_type;
  if (kind === 'affected' && row.visibility_quality && $('affectedVisibility')) $('affectedVisibility').value = row.visibility_quality;
  renderLists();
  draw();
  renderWorkbar();
}

function deleteRegion(kind, index) {
  try {
    const measurement = ensureMeasurement();
    const rows = kind === 'public' ? measurement.public_space_regions : measurement.affected_regions;
    if (index < 0 || index >= rows.length) return;
    rows.splice(index, 1);
    state.selectedType = '';
    state.selectedIndex = -1;
    markDirty();
    renderLists();
    draw();
    renderWorkbar();
    if (kind === 'affected' && $('affectedVisibility')) $('affectedVisibility').value = 'visible';
    setMessage(kind === 'public' ? 'Road/footpath area deleted. Press Save to keep the change.' : 'Lit area deleted. Press Save to keep the change.');
  } catch (error) {
    setMessage(error.message);
  }
}

function deleteAllRegions(kind) {
  const measurement = ensureMeasurement();
  const rows = kind === 'public' ? measurement.public_space_regions : measurement.affected_regions;
  if (!rows.length) {
    setMessage(kind === 'public' ? 'No road/footpath areas to delete.' : 'No lit areas to delete.');
    return;
  }
  rows.splice(0, rows.length);
  state.selectedType = '';
  state.selectedIndex = -1;
  if (kind === 'affected' && $('affectedVisibility')) $('affectedVisibility').value = 'visible';
  markDirty();
  renderLists();
  draw();
  renderWorkbar();
  setMessage(kind === 'public' ? 'All road/footpath areas deleted. Press Save to keep the change.' : 'All lit areas deleted. Press Save to keep the change.');
}

function addLux() {
  if (!state.selectedPoint) {
    setMessage('Use the Point tool and click the image first.');
    return;
  }
  const measurement = ensureMeasurement();
  measurement.lux_points.push({track_id: targetTrackId(), point_type: $('luxType').value, lux_value: $('luxValue').value, x: state.selectedPoint.x, y: state.selectedPoint.y});
  state.selectedPoint = null;
  state.selectedType = 'lux_point';
  state.selectedIndex = measurement.lux_points.length - 1;
  markDirty();
  renderLists();
  draw();
  renderWorkbar();
  setMessage('Lux point saved. It can be deleted from the saved lux point list.');
}

function selectLuxPoint(index) {
  const points = ensureMeasurement().lux_points || [];
  if (index < 0 || index >= points.length) return;
  state.selectedType = 'lux_point';
  state.selectedIndex = index;
  draw();
  renderLists();
  renderWorkbar();
}

function deleteLuxPoint(index) {
  const points = ensureMeasurement().lux_points || [];
  if (index < 0 || index >= points.length) return;
  points.splice(index, 1);
  state.selectedType = '';
  state.selectedIndex = -1;
  markDirty();
  renderLists();
  draw();
  renderWorkbar();
  setMessage('Lux point deleted. Press Save to keep the change.');
}

function deleteAllLuxPoints() {
  const points = ensureMeasurement().lux_points || [];
  if (!points.length) {
    setMessage('No lux points to delete.');
    return;
  }
  points.splice(0, points.length);
  state.selectedType = '';
  state.selectedIndex = -1;
  markDirty();
  renderLists();
  draw();
  renderWorkbar();
  setMessage('All lux points deleted. Press Save to keep the change.');
}

function addQa() {
  const flag = $('qaFlag').value.trim();
  if (!flag) return;
  ensureMeasurement().qa_flags.push({track_id: targetTrackId(), flag});
  $('qaFlag').value = '';
  markDirty();
}

function setQaFlag(flag, enabled) {
  const measurement = ensureMeasurement();
  measurement.qa_flags = measurement.qa_flags || [];
  const existing = measurement.qa_flags.findIndex(row => row.flag === flag && !row.track_id);
  if (enabled && existing < 0) {
    measurement.qa_flags.push({track_id: '', flag});
    markDirty();
  }
  if (!enabled && existing >= 0) {
    measurement.qa_flags.splice(existing, 1);
    markDirty();
  }
  renderSummary();
}

function hasQaFlag(flag) {
  return !!state.review?.measurement?.qa_flags?.some(row => row.flag === flag);
}

function validateAcceptableItem() {
  const boxes = state.review?.boxes || [];
  const accepted = boxes.filter(box => ['accepted', 'fixed'].includes(String(box.status || '')));
  const lampHeads = accepted.filter(box => normalizeBoxClass(box.class_name) === LAMP_HEAD_CLASS);
  const poles = accepted.filter(box => normalizeBoxClass(box.class_name) === POLE_CLASS);
  if (lampHeads.length && !poles.length && !hasQaFlag(POLE_NOT_VISIBLE_FLAG)) {
    setStep(0);
    setMessage('Add a pole/fixture box, or tick "Pole/fixture is not visible enough" before accepting.');
    return false;
  }
  return true;
}

function renderSummary() {
  const boxes = state.review?.boxes || [];
  $('reviewSummary').textContent = JSON.stringify({
    status: state.review?.review_status || 'unreviewed',
    dirty: state.dirty,
    lamp_heads: boxes.filter(box => normalizeBoxClass(box.class_name) === LAMP_HEAD_CLASS).length,
    poles: boxes.filter(box => normalizeBoxClass(box.class_name) === POLE_CLASS).length,
    other_light_boxes: state.review?.confounder_boxes?.length || 0,
    surfaces: state.review?.polygons?.length || 0,
    lamp_status: state.review?.measurement?.lamp_status?.length || 0,
    public_regions: state.review?.measurement?.public_space_regions?.length || 0,
    affected_regions: state.review?.measurement?.affected_regions?.length || 0,
    visibility: state.review?.measurement?.visibility_labels?.length || 0,
    attribution: state.review?.measurement?.attribution_labels?.length || 0,
    lux_points: state.review?.measurement?.lux_points?.length || 0,
    qa_flags: state.review?.measurement?.qa_flags?.length || 0
  }, null, 2);
}

async function saveReview() {
  if (state.tutorialMode) {
    setMessage('Tutorial examples are read-only. Use Start Real Work when ready.');
    return false;
  }
  try {
    await fetchJson('/api/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({key: state.current.item.key, review: state.review})
    });
    state.dirty = false;
    updateSaveState('Saved locally; export queued');
    setMessage('Saved locally. Export is running in the background.');
    renderSummary();
    refreshWorkspaceStatus();
    pollExportStatus();
    return true;
  } catch (error) {
    setMessage(error.message);
    return false;
  }
}

function markStatus(value) {
  if (value === 'accepted' && !validateAcceptableItem()) return;
  state.review.review_status = value;
  if (value === 'needs_review' || value === 'unusable_frame') {
    ensureMeasurement().qa_flags.push({track_id: targetTrackId(), flag: value});
  }
  markDirty();
}

async function acceptSaveNext() {
  if (!validateAcceptableItem()) return;
  state.review.review_status = 'accepted';
  markDirty();
  const saved = await saveReview();
  if (!saved) return;
  if (state.current?.next_key) {
    await loadItemByKey(state.current.next_key);
  } else {
    setMessage('Accepted and saved. This was the last item.');
  }
}

async function pollExportStatus(attempt = 0) {
  try {
    const status = await fetchJson('/api/export-status');
    if (status.status === 'ok') {
      updateSaveState('Export complete');
      return;
    }
    if (status.status === 'error') {
      updateSaveState(`Export failed: ${status.error || 'unknown error'}`);
      return;
    }
    updateSaveState(`Export ${status.status || 'running'}`);
    if (attempt < 8) setTimeout(() => pollExportStatus(attempt + 1), 700);
  } catch (_error) {
    if (attempt < 3) setTimeout(() => pollExportStatus(attempt + 1), 900);
  }
}

function setTool(tool) {
  state.tool = tool;
  for (const id of ['boxToolBtn', 'smartToolBtn', 'polygonToolBtn', 'pointToolBtn']) $(id).classList.remove('active');
  if ($('otherBoxToolBtn')) $('otherBoxToolBtn').classList.remove('active');
  if ($('otherSmartToolBtn')) $('otherSmartToolBtn').classList.remove('active');
  if (tool === 'box') $('boxToolBtn').classList.add('active');
  if (tool === 'smart') $('smartToolBtn').classList.add('active');
  if (tool === 'polygon') $('polygonToolBtn').classList.add('active');
  if (tool === 'point') $('pointToolBtn').classList.add('active');
  if (tool === 'box' && $('otherBoxToolBtn')) $('otherBoxToolBtn').classList.add('active');
  if (tool === 'smart' && $('otherSmartToolBtn')) $('otherSmartToolBtn').classList.add('active');
  renderActionControls();
  renderWorkbar();
}

function finishPolygon() {
  if (state.polygonDraft.length < 3) {
    setMessage('Add at least 3 corners before finishing the shape.');
    return;
  }
  if (!canCloseDraft()) {
    setMessage('That final line would cross the shape. Move the last point first.');
    return;
  }
  addPolygon([...state.polygonDraft], 'manual');
  state.polygonDraft = [];
  state.previewPoint = null;
  renderActionControls();
  renderWorkbar();
}

function addPolygon(points, source) {
  state.review.polygons = state.review.polygons || [];
  const polygon = {
    polygon_id: `poly_${String(state.review.polygons.length + 1).padStart(3, '0')}`,
    points,
    source,
    ...polygonMetadata()
  };
  state.review.polygons.push(polygon);
  markDirty();
  selectPolygon(state.review.polygons.length - 1);
  setMessage('Shape added. Press Save or continue marking this frame.');
}

function deleteSelected() {
  if (state.proposal) {
    state.proposal = null;
    setMessage('Suggested shape removed.');
  } else if (state.selectedType === 'box' && state.selectedIndex >= 0) {
    deleteBox(state.selectedIndex);
    return;
  } else if (state.selectedType === 'confounder_box' && state.selectedIndex >= 0) {
    deleteConfounderBox(state.selectedIndex);
    return;
  } else if (state.selectedType === 'polygon' && state.selectedIndex >= 0) {
    deletePolygon(state.selectedIndex);
    return;
  } else if (state.selectedType === 'public_region' && state.selectedIndex >= 0) {
    deleteRegion('public', state.selectedIndex);
    return;
  } else if (state.selectedType === 'affected_region' && state.selectedIndex >= 0) {
    deleteRegion('affected', state.selectedIndex);
    return;
  } else if (state.selectedType === 'lux_point' && state.selectedIndex >= 0) {
    deleteLuxPoint(state.selectedIndex);
    return;
  } else if (state.selectedType.endsWith('_label') && state.selectedIndex >= 0) {
    deleteVisibilityRow(state.selectedType.replace('_label', ''), state.selectedIndex);
    return;
  }
  state.selectedType = '';
  state.selectedIndex = -1;
  if (!state.proposal) setMessage('Select a box or shape first.');
  renderLists();
  draw();
  renderActionControls();
  renderWorkbar();
}

function deleteBox(index) {
  if (!state.review.boxes || index < 0 || index >= state.review.boxes.length) return;
  state.review.boxes.splice(index, 1);
  state.selectedType = '';
  state.selectedIndex = -1;
  markDirty();
  renderLists();
  draw();
  setMessage('Lamp box deleted. Press Save to keep the change.');
  renderActionControls();
  renderWorkbar();
}

function deleteAllBoxes() {
  state.review.boxes = state.review.boxes || [];
  if (!state.review.boxes.length) {
    setMessage('No lamp/pole boxes to delete.');
    return;
  }
  state.review.boxes.splice(0, state.review.boxes.length);
  state.selectedType = '';
  state.selectedIndex = -1;
  markDirty();
  renderLists();
  draw();
  setMessage('All lamp/pole boxes deleted. Press Save to keep the change.');
  renderActionControls();
  renderWorkbar();
}

function deleteConfounderBox(index) {
  if (!state.review.confounder_boxes || index < 0 || index >= state.review.confounder_boxes.length) return;
  state.review.confounder_boxes.splice(index, 1);
  state.selectedType = '';
  state.selectedIndex = -1;
  markDirty();
  renderLists();
  draw();
  setMessage('Other-light box deleted. Press Save to keep the change.');
  renderActionControls();
  renderWorkbar();
}

function deleteAllConfounderBoxes() {
  state.review.confounder_boxes = state.review.confounder_boxes || [];
  if (!state.review.confounder_boxes.length) {
    setMessage('No other-light boxes to delete.');
    return;
  }
  state.review.confounder_boxes.splice(0, state.review.confounder_boxes.length);
  state.selectedType = '';
  state.selectedIndex = -1;
  markDirty();
  renderLists();
  draw();
  setMessage('All other-light boxes deleted. Press Save to keep the change.');
  renderActionControls();
  renderWorkbar();
}

function deletePolygon(index) {
  if (!state.review.polygons || index < 0 || index >= state.review.polygons.length) return;
  state.review.polygons.splice(index, 1);
  state.selectedType = '';
  state.selectedIndex = -1;
  markDirty();
  renderLists();
  draw();
  setMessage('Shape deleted. Press Save to keep the change.');
  renderActionControls();
  renderWorkbar();
}

function deleteAllPolygons() {
  state.review.polygons = state.review.polygons || [];
  if (!state.review.polygons.length) {
    setMessage('No surface shapes to delete.');
    return;
  }
  state.review.polygons.splice(0, state.review.polygons.length);
  state.selectedType = '';
  state.selectedIndex = -1;
  markDirty();
  renderLists();
  draw();
  setMessage('All surface shapes deleted. Press Save to keep the change.');
  renderActionControls();
  renderWorkbar();
}

function undoLastShape() {
  if (state.review.polygons?.length) {
    deletePolygon(state.review.polygons.length - 1);
  } else {
    setMessage('No shape to undo.');
  }
}

function undoPoint() {
  if (!state.polygonDraft.length) {
    setMessage('No drawing point to undo.');
    return;
  }
  state.polygonDraft.pop();
  state.previewPoint = null;
  draw();
  setMessage('Last point removed.');
}

function cancelShape() {
  state.polygonDraft = [];
  state.previewPoint = null;
  state.proposal = null;
  draw();
  setMessage('Drawing cancelled.');
  renderActionControls();
  renderWorkbar();
}

async function requestAutoPolygon(box) {
  setMessage('Finding the shape...');
  try {
    const payload = await fetchJson('/api/auto-polygon', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        item_key: state.current.item.key,
        bbox_xyxy: box,
        margin_px: Number($('polyMargin').value || 12)
      })
    });
    state.proposal = payload;
    const warningText = payload.warnings?.length ? ` ${payload.warnings.join(' ')}` : '';
    setMessage(`Shape found. Keep it or try again.${warningText}`);
    draw();
    renderActionControls();
  } catch (error) {
    setMessage(error.message);
  }
}

function acceptProposal() {
  if (!state.proposal || !state.proposal.points?.length) {
    setMessage('No proposal to accept.');
    return;
  }
  addPolygon(state.proposal.points, `auto:${state.proposal.engine}`);
  state.proposal = null;
  setMessage('Shape kept.');
  renderActionControls();
}

function discardProposal() {
  state.proposal = null;
  draw();
  setMessage('Try again: drag a new box around the surface.');
  renderActionControls();
}

function canvasMouseDown(event) {
  const point = imageCoords(event);
  if (state.tool === 'box' || state.tool === 'smart') {
    state.drawing = {kind: state.tool === 'smart' ? 'smart' : 'box', start: point, box: [point.x, point.y, point.x, point.y]};
  } else if (state.tool === 'polygon') {
    addDraftPoint(point);
  } else if (state.tool === 'point') {
    if (STEPS[state.step].id === 'lux_qa') {
      state.selectedPoint = point;
      draw();
      renderWorkbar();
      return;
    }
    const selected = polygonIndexAt(point);
    if (selected >= 0) {
      selectPolygon(selected);
    } else {
      state.selectedPoint = point;
      draw();
    }
  }
}

function canvasMouseMove(event) {
  const point = imageCoords(event);
  if (state.tool === 'polygon' && state.polygonDraft.length) {
    state.previewPoint = point;
    draw();
  }
  if (!state.drawing || !['box', 'smart'].includes(state.drawing.kind)) return;
  const start = state.drawing.start;
  state.drawing.box = [Math.min(start.x, point.x), Math.min(start.y, point.y), Math.max(start.x, point.x), Math.max(start.y, point.y)];
  draw();
}

function canvasMouseUp() {
  if (!state.drawing || !['box', 'smart'].includes(state.drawing.kind)) return;
  const box = state.drawing.box;
  const kind = state.drawing.kind;
  const start = state.drawing.start;
  state.drawing = null;
  if (Math.abs(box[2] - box[0]) < 10 / state.scale || Math.abs(box[3] - box[1]) < 10 / state.scale) {
    selectBoxAtPoint(start);
    return;
  }
  if (kind === 'smart') {
    requestAutoPolygon(box);
    return;
  }
  if (STEPS[state.step].id === 'surfaces') {
    addConfounderBox(box);
    return;
  }
  state.review.boxes = state.review.boxes || [];
  const className = normalizeBoxClass($('lampDrawClass')?.value || $('lampBoxClass').value);
  const newBox = {
    box_id: `box_${String(state.review.boxes.length + 1).padStart(3, '0')}`,
    class_name: className,
    bbox_xyxy: box,
    track_id: targetTrackId(),
    parent_pole_box_id: '',
    status: 'fixed',
    source: 'manual',
    notes: ''
  };
  if (className === LAMP_HEAD_CLASS) newBox.parent_pole_box_id = nearestPoleId(newBox);
  state.review.boxes.push(newBox);
  markDirty();
  selectBox(state.review.boxes.length - 1);
  setMessage(className === POLE_CLASS ? 'Pole box added.' : 'Lamp head box added.');
}

function normalizeBoxClass(value) {
  if (!value || value === LEGACY_STREETLIGHT_CLASS) return LAMP_HEAD_CLASS;
  return value;
}

function boxClassLabel(box) {
  return LABELS[normalizeBoxClass(box.class_name)] || normalizeBoxClass(box.class_name);
}

function boxShortLabel(box) {
  return normalizeBoxClass(box.class_name) === POLE_CLASS ? 'pole' : 'head';
}

function boxColor(box) {
  return normalizeBoxClass(box.class_name) === POLE_CLASS ? '#80ed99' : '#4cc9f0';
}

function boxCenter(box) {
  const [x1, y1, x2, y2] = (box.bbox_xyxy || [0, 0, 0, 0]).map(Number);
  return {x: (x1 + x2) / 2, y: (y1 + y2) / 2};
}

function nearestPoleId(lampBox) {
  const poles = (state.review.boxes || [])
    .map((box, index) => ({box, index}))
    .filter(({box}) => normalizeBoxClass(box.class_name) === POLE_CLASS);
  if (!poles.length) return '';
  const lamp = boxCenter(lampBox);
  let best = null;
  for (const candidate of poles) {
    const pole = boxCenter(candidate.box);
    const distance = Math.hypot(lamp.x - pole.x, lamp.y - pole.y);
    if (!best || distance < best.distance) best = {...candidate, distance};
  }
  return best ? best.box.box_id || `box_${best.index + 1}` : '';
}

function selectBoxAtPoint(point) {
  const anyOtherIndex = boxIndexAt(point, state.review.confounder_boxes || []);
  const anyLampIndex = boxIndexAt(point, state.review.boxes || []);
  if (STEPS[state.step].id === 'surfaces') {
    if (anyOtherIndex >= 0) {
      selectConfounderBox(anyOtherIndex);
      setMessage('Other-light box selected. Use Delete Other-Light Box or press Delete.');
      return true;
    }
  }
  if (anyLampIndex >= 0) {
    selectBox(anyLampIndex);
    setMessage('Lamp box selected. Use Delete This Box or press Delete.');
    return true;
  }
  if (anyOtherIndex >= 0) {
    selectConfounderBox(anyOtherIndex);
    setMessage('Other-light box selected. Press Delete to remove it.');
    return true;
  }
  setMessage('Drag to draw a box.');
  return false;
}

function boxIndexAt(point, boxes) {
  const tolerance = Math.max(8 / state.scale, 6);
  for (let index = boxes.length - 1; index >= 0; index--) {
    const box = boxes[index].bbox_xyxy || [];
    if (box.length !== 4) continue;
    const [x1, y1, x2, y2] = box.map(Number);
    if (point.x >= Math.min(x1, x2) - tolerance && point.x <= Math.max(x1, x2) + tolerance && point.y >= Math.min(y1, y2) - tolerance && point.y <= Math.max(y1, y2) + tolerance) {
      return index;
    }
  }
  return -1;
}

function addConfounderBox(box) {
  state.review.confounder_boxes = state.review.confounder_boxes || [];
  const metadata = polygonMetadata();
  state.review.confounder_boxes.push({
    box_id: `other_box_${String(state.review.confounder_boxes.length + 1).padStart(3, '0')}`,
    bbox_xyxy: box,
    source: 'manual_box',
    notes: '',
    ...metadata
  });
  markDirty();
  selectConfounderBox(state.review.confounder_boxes.length - 1);
  setMessage('Other-light box added. Save when ready.');
}

function addDraftPoint(point) {
  const pointArray = [point.x, point.y];
  if (state.polygonDraft.length >= 3 && distance(pointArray, state.polygonDraft[0]) < 12 / state.scale) {
    finishPolygon();
    return;
  }
  if (!canAddPoint(pointArray)) {
    setMessage('That line crosses the shape. Pick another point.');
    return;
  }
  state.polygonDraft.push(pointArray);
  state.previewPoint = null;
  draw();
  renderActionControls();
  renderWorkbar();
}

function canAddPoint(point) {
  const draft = state.polygonDraft;
  if (draft.length < 2) return true;
  const nextSegment = [draft[draft.length - 1], point];
  for (let index = 0; index < draft.length - 2; index++) {
    if (segmentsIntersect(nextSegment[0], nextSegment[1], draft[index], draft[index + 1])) return false;
  }
  return true;
}

function canCloseDraft() {
  const draft = state.polygonDraft;
  if (draft.length < 3) return false;
  const closing = [draft[draft.length - 1], draft[0]];
  for (let index = 1; index < draft.length - 2; index++) {
    if (segmentsIntersect(closing[0], closing[1], draft[index], draft[index + 1])) return false;
  }
  return true;
}

function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function segmentsIntersect(a, b, c, d) {
  if (samePoint(a, c) || samePoint(a, d) || samePoint(b, c) || samePoint(b, d)) return false;
  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);
  if (o1 !== o2 && o3 !== o4) return true;
  return false;
}

function orientation(a, b, c) {
  const value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1]);
  if (Math.abs(value) < 1e-7) return 0;
  return value > 0 ? 1 : 2;
}

function samePoint(a, b) {
  return Math.abs(a[0] - b[0]) < 1e-7 && Math.abs(a[1] - b[1]) < 1e-7;
}

function polygonIndexAt(point) {
  const polygons = state.review.polygons || [];
  for (let index = polygons.length - 1; index >= 0; index--) {
    if (pointInPolygon([point.x, point.y], polygons[index].points || [])) return index;
  }
  return -1;
}

function pointInPolygon(point, polygon) {
  if (!polygon || polygon.length < 3) return false;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    const intersect = ((yi > point[1]) !== (yj > point[1])) &&
      (point[0] < (xj - xi) * (point[1] - yi) / ((yj - yi) || 1e-9) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function bindEvents() {
  $('saveBtn').onclick = saveReview;
  $('acceptedBtn').onclick = acceptSaveNext;
  $('needsReviewBtn').onclick = () => markStatus('needs_review');
  $('unusableBtn').onclick = () => markStatus('unusable_frame');
  $('boxToolBtn').onclick = () => setTool('box');
  $('smartToolBtn').onclick = () => setTool('smart');
  $('polygonToolBtn').onclick = () => setTool('polygon');
  $('pointToolBtn').onclick = () => setTool('point');
  $('finishPolygonBtn').onclick = finishPolygon;
  $('deleteSelectedBtn').onclick = deleteSelected;
  $('deleteBoxBtn').onclick = () => {
    if (state.selectedType === 'box' && state.selectedIndex >= 0) deleteBox(state.selectedIndex);
    else setMessage('Select a lamp box first.');
  };
  $('deleteAllBoxesBtn').onclick = deleteAllBoxes;
  $('deleteShapeBtn').onclick = () => {
    if (state.selectedType === 'polygon' && state.selectedIndex >= 0) deletePolygon(state.selectedIndex);
    else if (state.proposal) deleteSelected();
    else setMessage('Select a surface shape first.');
  };
  $('deleteOtherBoxBtn').onclick = () => {
    if (state.selectedType === 'confounder_box' && state.selectedIndex >= 0) deleteConfounderBox(state.selectedIndex);
    else setMessage('Select an other-light box first.');
  };
  $('deleteAllOtherBoxesBtn').onclick = deleteAllConfounderBoxes;
  $('deleteAllShapesBtn').onclick = deleteAllPolygons;
  $('otherSmartToolBtn').onclick = () => setTool('smart');
  $('otherBoxToolBtn').onclick = () => setTool('box');
  $('undoShapeBtn').onclick = undoLastShape;
  $('undoPointBtn').onclick = undoPoint;
  $('cancelShapeBtn').onclick = cancelShape;
  $('updateBoxBtn').onclick = updateSelectedBox;
  $('lampBoxClass').onchange = () => {
    renderWorkbar();
  };
  $('lampDrawClass').onchange = () => {
    renderWorkbar();
  };
  $('parentPoleBoxId').onchange = updateSelectedBox;
  $('poleNotVisible').onchange = () => setQaFlag(POLE_NOT_VISIBLE_FLAG, $('poleNotVisible').checked);
  $('updatePolygonBtn').onclick = updateSelectedPolygon;
  $('acceptProposalBtn').onclick = acceptProposal;
  $('discardProposalBtn').onclick = discardProposal;
  $('addLampStatusBtn').onclick = addLampStatus;
  $('addVisibilityBtn').onclick = addVisibility;
  $('addAttributionBtn').onclick = addAttribution;
  $('addPublicRegionBtn').onclick = () => addRegion('public');
  $('addAffectedRegionBtn').onclick = () => addRegion('affected');
  $('markAffectedNotVisibleBtn').onclick = markAffectedNotVisible;
  $('affectedTrackSelect').onchange = () => setTargetTrack($('affectedTrackSelect').value);
  $('visibilityTrackSelect').onchange = () => setTargetTrack($('visibilityTrackSelect').value);
  $('deleteAllPublicRegionsBtn').onclick = () => deleteAllRegions('public');
  $('deleteAllAffectedRegionsBtn').onclick = () => deleteAllRegions('affected');
  $('deleteAllVisibilityLabelsBtn').onclick = deleteAllVisibilityLabels;
  $('addLuxBtn').onclick = addLux;
  $('deleteAllLuxPointsBtn').onclick = deleteAllLuxPoints;
  $('addQaBtn').onclick = addQa;
  $('canvas').onmousedown = canvasMouseDown;
  $('canvas').onmousemove = canvasMouseMove;
  $('canvas').onmouseup = canvasMouseUp;
  window.onresize = () => { if (state.image) { resizeCanvas(); draw(); } };
  document.onkeydown = (event) => {
    if (event.target && ['INPUT', 'SELECT'].includes(event.target.tagName)) return;
    if (event.key.toLowerCase() === 's') saveReview();
    if (event.key.toLowerCase() === 'n' && state.current?.next_key) loadItemByKey(state.current.next_key);
    if (event.key.toLowerCase() === 'b') setTool('box');
    if (event.key.toLowerCase() === 'g') setTool('smart');
    if (event.key.toLowerCase() === 'p') setTool('polygon');
    if (event.key === 'Enter') acceptProposal();
    if (event.key === 'Escape') cancelShape();
    if (event.key === 'Delete') {
      event.preventDefault();
      deleteSelected();
    }
    if (event.key === 'Backspace') {
      event.preventDefault();
      if (state.polygonDraft.length) undoPoint();
      else deleteSelected();
    }
  };
}

bindEvents();
init().catch(error => setMessage(error.message));

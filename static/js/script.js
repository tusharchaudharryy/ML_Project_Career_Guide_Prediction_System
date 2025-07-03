document.addEventListener('DOMContentLoaded', () => {
  initSliders();
  initForm();
  initThemeToggle();
});

/**
 * Initialize range sliders with live values
 */
function initSliders() {
  document.querySelectorAll('input[type="range"]').forEach(slider => {
    renderSliderValue(slider);
    slider.addEventListener('input', () => renderSliderValue(slider));
  });
}

/**
 * Render slider value and style badge using CSS variables
 */
function renderSliderValue(slider) {
  const display = document.getElementById(`${slider.id}_value`);
  if (!display) return;
  const val = parseFloat(slider.value);
  display.textContent = slider.value;
  styleBadge(display, val, slider.dataset.trait);
}

/**
 * Style badge based on value ranges using CSS variables
 */
function styleBadge(el, value, trait) {
  // Reset inline styles
  el.style.background = '';
  el.style.color = 'var(--color-text-light)';

  // Determine thresholds from data attributes or defaults
  const thresholds = trait === 'tech'
    ? [2, 4, 6]  // technical: low, medium, high
    : [0.3, 0.6]; // traits: low, medium

  if (trait === 'tech') {
    if (value <= thresholds[0]) el.style.background = 'var(--color-warning)';
    else if (value <= thresholds[1]) el.style.background = 'var(--color-accent)';
    else if (value <= thresholds[2]) el.style.background = 'var(--color-primary)';
    else el.style.background = 'var(--color-success)';
  } else {
    if (value <= thresholds[0]) el.style.background = 'rgba(255,255,255,0.2)';
    else if (value <= thresholds[1]) el.style.background = 'var(--color-accent)';
    else el.style.background = 'var(--color-success)';
  }
}

/**
 * Initialize form submission with spinner and validation
 */
function initForm() {
  const form = document.getElementById('predictionForm');
  if (!form) return;
  form.addEventListener('submit', e => {
    const btn = document.getElementById('predictButton');
    const spinner = document.getElementById('loadingSpinner');
    if (btn && spinner) {
      btn.disabled = true;
      spinner.classList.remove('d-none');
      btn.textContent = 'Predicting...';
      if (!validateForm()) {
        e.preventDefault();
        resetButton(btn, spinner);
      }
    }
  });
}

/**
 * Validate required inputs
 */
function validateForm() {
  let valid = true;
  document.querySelectorAll('input[required]').forEach(input => {
    if (!input.value.trim()) {
      input.classList.add('is-invalid');
      valid = false;
    } else input.classList.remove('is-invalid');
  });
  if (!valid) showAlert('Fill all required fields!', 'error');
  return valid;
}

/**
 * Reset submit button state
 */
function resetButton(btn, spinner) {
  btn.disabled = false;
  btn.textContent = 'Predict My Career';
  spinner.classList.add('d-none');
}

/**
 * Display alert messages dynamically
 */
function showAlert(msg, type='info') {
  const wrapper = document.createElement('div');
  wrapper.className = `alert alert-${type==='error'?'danger':type} show`;
  wrapper.textContent = msg;
  const container = document.querySelector('.container');
  container.insertBefore(wrapper, container.firstChild);
  setTimeout(() => wrapper.remove(), 5000);
}

/**
 * Smooth scroll utility
 */
function smoothScrollTo(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Theme toggle (light/dark) button initializer
 */
function initThemeToggle() {
  const toggle = document.getElementById('themeToggle');
  if (!toggle) return;
  toggle.addEventListener('click', () => {
    document.body.classList.toggle('light-mode');
  });
}

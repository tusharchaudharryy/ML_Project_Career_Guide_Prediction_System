/**
 * Career Prediction System JavaScript
 */

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeSliders();
    initializeForm();
});

/**
 * Initialize all range sliders
 */
function initializeSliders() {
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        // Set initial display value
        updateSliderValue(slider.id, slider.value);
        
        // Add event listener for changes
        slider.addEventListener('input', function() {
            updateSliderValue(this.id, this.value);
        });
    });
}

/**
 * Update slider value display
 * @param {string} skillKey - The skill key identifier
 * @param {string} value - The slider value
 */
function updateSliderValue(skillKey, value) {
    const valueElement = document.getElementById(skillKey + '_value');
    if (valueElement) {
        valueElement.textContent = value;
        
        // Update badge color based on value
        updateBadgeColor(valueElement, value, skillKey);
    }
}

/**
 * Update badge color based on value
 * @param {Element} element - The badge element
 * @param {string} value - The value
 * @param {string} skillKey - The skill key
 */
function updateBadgeColor(element, value, skillKey) {
    // Remove existing color classes
    element.className = 'badge';
    
    // Determine if this is a technical skill (1-7) or personality trait (0-1)
    const isTechnicalSkill = parseFloat(value) > 1;
    
    if (isTechnicalSkill) {
        // Technical skills (1-7 scale)
        const numValue = parseInt(value);
        if (numValue <= 2) {
            element.classList.add('bg-danger');
        } else if (numValue <= 4) {
            element.classList.add('bg-warning');
        } else if (numValue <= 6) {
            element.classList.add('bg-info');
        } else {
            element.classList.add('bg-success');
        }
    } else {
        // Personality traits (0-1 scale)
        const numValue = parseFloat(value);
        if (numValue <= 0.3) {
            element.classList.add('bg-secondary');
        } else if (numValue <= 0.6) {
            element.classList.add('bg-info');
        } else {
            element.classList.add('bg-success');
        }
    }
}

/**
 * Initialize form handling
 */
function initializeForm() {
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
}

/**
 * Handle form submission
 * @param {Event} event - The form submit event
 */
function handleFormSubmit(event) {
    const button = document.getElementById('predictButton');
    const spinner = document.getElementById('loadingSpinner');
    
    if (button && spinner) {
        // Show loading state
        button.disabled = true;
        spinner.classList.remove('d-none');
        button.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Predicting...';
        
        // Validate form before submission
        if (!validateForm()) {
            event.preventDefault();
            resetButton(button);
            return false;
        }
    }
    
    return true;
}

/**
 * Validate form inputs
 * @returns {boolean} - True if form is valid
 */
function validateForm() {
    const requiredFields = document.querySelectorAll('input[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value || field.value.trim() === '') {
            isValid = false;
            field.classList.add('is-invalid');
        } else {
            field.classList.remove('is-invalid');
        }
    });
    
    if (!isValid) {
        showAlert('Please fill in all required fields.', 'error');
    }
    
    return isValid;
}

/**
 * Reset button to original state
 * @param {Element} button - The button element
 */
function resetButton(button) {
    button.disabled = false;
    button.innerHTML = 'Predict My Career';
    
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.classList.add('d-none');
    }
}

/**
 * Show alert message
 * @param {string} message - The message to show
 * @param {string} type - The alert type (success, error, warning, info)
 */
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the container
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
    }
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

/**
 * API call for predictions (for future use)
 * @param {Object} formData - The form data
 * @returns {Promise} - The API response
 */
async function makePredictionAPI(formData) {
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }
        
        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

/**
 * Format form data for API submission
 * @param {FormData} formData - The form data
 * @returns {Object} - Formatted data object
 */
function formatFormDataForAPI(formData) {
    const data = {};
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    return data;
}

/**
 * Smooth scroll to element
 * @param {string} elementId - The element ID to scroll to
 */
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

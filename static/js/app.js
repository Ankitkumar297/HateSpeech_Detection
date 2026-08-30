/**
 * SafeGuard AI - Frontend Application Controller
 * Handles predictions, interactive analytics, admin dashboard, theme toggle, and PDF exports.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize All Subsystems
    initThemeEngine();
    initDetectorStudio();
    initAnalyticsDashboard();
    initAdminPanel();
    initModelInfo();
    initMobileNav();
});

/* ==========================================================================
   1. Theme Engine (Dark / Light Mode)
   ========================================================================== */
function initThemeEngine() {
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const htmlEl = document.documentElement;

    // Load saved theme or default to dark
    const savedTheme = localStorage.getItem('safeguard_theme') || 'dark';
    htmlEl.setAttribute('data-theme', savedTheme);

    themeToggleBtn?.addEventListener('click', () => {
        const currentTheme = htmlEl.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        htmlEl.setAttribute('data-theme', newTheme);
        localStorage.setItem('safeguard_theme', newTheme);
        showToast(`Switched to ${newTheme === 'dark' ? 'Dark' : 'Light'} Mode`, 'info');

        // Refresh Chart.js colors on theme change
        if (window.classDistChart && window.trendChart) {
            updateChartTheme(newTheme);
        }
    });
}

/* ==========================================================================
   2. Live Detector Studio
   ========================================================================== */
let latestPredictionData = null;

function initDetectorStudio() {
    const textarea = document.getElementById('userInputText');
    const charCount = document.getElementById('charCount');
    const wordCount = document.getElementById('wordCount');
    const clearBtn = document.getElementById('clearTextBtn');
    const detectBtn = document.getElementById('detectBtn');
    const errorAlert = document.getElementById('detectorErrorAlert');
    const errorMsg = document.getElementById('detectorErrorMessage');
    const sampleChips = document.querySelectorAll('.sample-chip');
    const nlpToggle = document.getElementById('nlpToggle');
    const nlpBox = document.querySelector('.nlp-breakdown-box');
    const copyResultBtn = document.getElementById('copyResultBtn');
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');

    // Live Word & Char Counter
    const updateCounters = () => {
        const val = textarea.value;
        charCount.textContent = val.length;
        const words = val.trim() ? val.trim().split(/\s+/).length : 0;
        wordCount.textContent = words;
    };

    textarea.addEventListener('input', updateCounters);

    // Keyboard shortcut (Ctrl + Enter)
    textarea.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            detectBtn.click();
        }
    });

    // Clear Text Button
    clearBtn?.addEventListener('click', () => {
        textarea.value = '';
        updateCounters();
        textarea.focus();
    });

    // Quick Sample Chips
    sampleChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const sampleText = chip.getAttribute('data-sample');
            if (sampleText) {
                textarea.value = sampleText;
                updateCounters();
                hideError();
                // Trigger detection smoothly
                triggerDetection(sampleText);
            }
        });
    });

    // Toggle NLP Token Details Accordion
    nlpToggle?.addEventListener('click', () => {
        nlpBox?.classList.toggle('open');
    });

    // Copy Verdict Button
    copyResultBtn?.addEventListener('click', () => {
        if (!latestPredictionData) return;
        const copyText = `Prediction: ${latestPredictionData.prediction} (${latestPredictionData.confidence}% confidence)\nInput: "${latestPredictionData.user_text || textarea.value}"`;
        navigator.clipboard.writeText(copyText).then(() => {
            showToast('Result copied to clipboard', 'success');
        });
    });

    // Download PDF Certificate
    downloadPdfBtn?.addEventListener('click', async () => {
        if (!latestPredictionData) return;
        downloadPdfBtn.disabled = true;
        downloadPdfBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating PDF...';

        try {
            const response = await fetch('/api/export/pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    id: latestPredictionData.id || Math.floor(Math.random() * 9000 + 1000),
                    text: textarea.value,
                    prediction: latestPredictionData.prediction,
                    confidence: latestPredictionData.confidence,
                    probabilities: latestPredictionData.probabilities,
                    timestamp: new Date().toLocaleString()
                })
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `Hate_Speech_Analysis_Report_${latestPredictionData.id || 'export'}.pdf`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
                showToast('PDF report downloaded', 'success');
            } else {
                showToast('Unable to generate PDF report', 'warning');
            }
        } catch (err) {
            console.error('PDF Export Error:', err);
            showToast('Failed to export PDF', 'danger');
        } finally {
            downloadPdfBtn.disabled = false;
            downloadPdfBtn.innerHTML = '<i class="fa-solid fa-file-pdf text-danger"></i> <span>Download PDF Report</span>';
        }
    });

    // Detect Button Click
    detectBtn?.addEventListener('click', () => {
        const text = textarea.value.trim();
        if (!text) {
            showError('Please enter a sentence or choose a sample prompt first.');
            textarea.focus();
            return;
        }
        triggerDetection(text);
    });

    function showError(msg) {
        if (errorAlert && errorMsg) {
            errorMsg.textContent = msg;
            errorAlert.style.display = 'flex';
        }
    }

    function hideError() {
        if (errorAlert) {
            errorAlert.style.display = 'none';
        }
    }

    async function triggerDetection(text) {
        hideError();
        const btnText = detectBtn.querySelector('.btn-text');
        const btnLoader = detectBtn.querySelector('.btn-loader');
        
        detectBtn.disabled = true;
        if (btnText) btnText.style.display = 'none';
        if (btnLoader) btnLoader.style.display = 'inline-flex';

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const data = await res.json();
            if (!res.ok || !data.success) {
                throw new Error(data.error || 'Failed to process text.');
            }

            latestPredictionData = { ...data, user_text: text };
            renderPredictionResult(data);
            showToast(`Result: ${data.prediction} (${data.confidence}%)`, data.severity);

            // Refresh Dashboard & Admin table in background
            refreshDashboardStats();
            refreshAdminTable();
        } catch (err) {
            console.error('Prediction API Error:', err);
            showError(err.message || 'Something went wrong while processing your text.');
        } finally {
            detectBtn.disabled = false;
            if (btnText) btnText.style.display = 'inline-flex';
            if (btnLoader) btnLoader.style.display = 'none';
        }
    }
}

function renderPredictionResult(data) {
    const placeholder = document.getElementById('resultPlaceholder');
    const content = document.getElementById('resultContent');
    const banner = document.getElementById('verdictBanner');
    const verdictClass = document.getElementById('verdictClass');
    const verdictDesc = document.getElementById('verdictDesc');
    const verdictIcon = document.getElementById('verdictIcon');
    const confPercent = document.getElementById('confidencePercent');
    const confBarText = document.getElementById('confidenceBarText');
    const confMeterFill = document.getElementById('confidenceMeterFill');

    const probHateVal = document.getElementById('probHateVal');
    const probHateBar = document.getElementById('probHateBar');
    const probOffensiveVal = document.getElementById('probOffensiveVal');
    const probOffensiveBar = document.getElementById('probOffensiveBar');
    const probNeutralVal = document.getElementById('probNeutralVal');
    const probNeutralBar = document.getElementById('probNeutralBar');

    const stemmedDisplay = document.getElementById('stemmedTextDisplay');
    const tokenBadges = document.getElementById('tokenBadgesContainer');

    if (placeholder) placeholder.style.display = 'none';
    if (content) content.style.display = 'block';

    // Set Verdict Classes & Colors
    banner.className = 'verdict-banner';
    let iconClass = 'fa-circle-check';
    let descText = 'Safe for public consumption and community discourse.';

    if (data.prediction === 'Hate Speech') {
        banner.classList.add('verdict-danger');
        iconClass = 'fa-triangle-exclamation';
        descText = 'Severe violation: Targeted hate speech and toxicity detected.';
    } else if (data.prediction === 'Offensive Language') {
        banner.classList.add('verdict-warning');
        iconClass = 'fa-comment-slash';
        descText = 'Warning: Profanity, vulgarity, or coarse insults detected.';
    } else {
        banner.classList.add('verdict-success');
        iconClass = 'fa-circle-check';
        descText = 'Approved: Content adheres to standard safety guidelines.';
    }

    verdictClass.textContent = data.prediction;
    verdictDesc.textContent = descText;
    if (verdictIcon) verdictIcon.className = `fa-solid ${iconClass}`;
    
    confPercent.textContent = `${data.confidence}%`;
    confBarText.textContent = `${data.confidence}%`;
    confMeterFill.style.width = `${data.confidence}%`;
    confMeterFill.style.backgroundColor = data.color || '#3b82f6';

    // Probabilities
    const probs = data.probabilities || {};
    const hateScore = probs['Hate Speech'] ?? 0;
    const offScore = probs['Offensive Language'] ?? 0;
    const neutScore = probs['Neutral Language'] ?? 0;

    probHateVal.textContent = `${hateScore}%`;
    probHateBar.style.width = `${hateScore}%`;

    probOffensiveVal.textContent = `${offScore}%`;
    probOffensiveBar.style.width = `${offScore}%`;

    probNeutralVal.textContent = `${neutScore}%`;
    probNeutralBar.style.width = `${neutScore}%`;

    // NLP Tokens
    if (stemmedDisplay) {
        stemmedDisplay.textContent = data.stemmed_text || '(no stem tokens remaining)';
    }

    if (tokenBadges && data.tokens) {
        tokenBadges.innerHTML = '';
        data.tokens.forEach(t => {
            const badge = document.createElement('span');
            badge.className = `token-badge ${t.is_stopword ? 'stopword' : ''}`;
            badge.textContent = t.is_stopword ? `${t.original} (stopword)` : `${t.original} ➔ ${t.stemmed}`;
            tokenBadges.appendChild(badge);
        });
    }
}

/* ==========================================================================
   3. Analytics & Chart.js Integration
   ========================================================================== */
let classDistChart = null;
let activityTrendChart = null;

async function initAnalyticsDashboard() {
    await refreshDashboardStats();
}

async function refreshDashboardStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        if (!res.ok || !data.success) return;

        // Animate counter values
        animateCounter('statTotalPredictions', data.total_predictions || 0);
        animateCounter('statHateCount', data.hate_speech_count || 0);
        animateCounter('statOffensiveCount', data.offensive_language_count || 0);
        animateCounter('statNeutralCount', data.neutral_count || 0);

        // Render / Update Charts
        renderDistributionChart(data);
        renderTimelineChart(data.trends || []);
    } catch (err) {
        console.error('Stats Refresh Error:', err);
    }
}

function animateCounter(elementId, targetValue) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const startValue = parseInt(el.textContent, 10) || 0;
    const duration = 600;
    const stepTime = 20;
    const steps = duration / stepTime;
    const increment = (targetValue - startValue) / steps;
    let current = startValue;
    let stepCount = 0;

    const timer = setInterval(() => {
        stepCount++;
        current += increment;
        if (stepCount >= steps) {
            el.textContent = targetValue;
            clearInterval(timer);
        } else {
            el.textContent = Math.round(current);
        }
    }, stepTime);
}

function renderDistributionChart(data) {
    const ctx = document.getElementById('classDistributionChart');
    if (!ctx) return;

    const counts = [
        data.hate_speech_count || 0,
        data.offensive_language_count || 0,
        data.neutral_count || 0
    ];

    if (classDistChart) {
        classDistChart.data.datasets[0].data = counts;
        classDistChart.update();
        return;
    }

    classDistChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Hate Speech', 'Offensive Language', 'Neutral Content'],
            datasets: [{
                data: counts,
                backgroundColor: [
                    '#ef4444',
                    '#f97316',
                    '#10b981'
                ],
                borderColor: 'rgba(0, 0, 0, 0.1)',
                borderWidth: 2,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: getChartTextColor(),
                        font: { family: "'Plus Jakarta Sans', sans-serif", size: 12 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` ${ctx.label}: ${ctx.raw} detections`
                    }
                }
            },
            cutout: '68%'
        }
    });
    window.classDistChart = classDistChart;
}

function renderTimelineChart(trends) {
    const ctx = document.getElementById('activityTrendChart');
    if (!ctx) return;

    const labels = trends.length ? trends.map(t => t.day) : ['Today'];
    const hateData = trends.length ? trends.map(t => t.hate_day) : [0];
    const offData = trends.length ? trends.map(t => t.offensive_day) : [0];
    const neutData = trends.length ? trends.map(t => t.neutral_day) : [0];

    if (activityTrendChart) {
        activityTrendChart.data.labels = labels;
        activityTrendChart.data.datasets[0].data = hateData;
        activityTrendChart.data.datasets[1].data = offData;
        activityTrendChart.data.datasets[2].data = neutData;
        activityTrendChart.update();
        return;
    }

    activityTrendChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Hate Speech',
                    data: hateData,
                    backgroundColor: '#ef4444',
                    borderRadius: 4
                },
                {
                    label: 'Offensive Language',
                    data: offData,
                    backgroundColor: '#f97316',
                    borderRadius: 4
                },
                {
                    label: 'Neutral Content',
                    data: neutData,
                    backgroundColor: '#10b981',
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    stacked: true,
                    grid: { display: false },
                    ticks: { color: getChartTextColor() }
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: getChartTextColor(), stepSize: 1 }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: getChartTextColor(),
                        font: { family: "'Plus Jakarta Sans', sans-serif", size: 12 }
                    }
                }
            }
        }
    });
    window.trendChart = activityTrendChart;
}

function getChartTextColor() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return isDark ? '#94a3b8' : '#475569';
}

function updateChartTheme(theme) {
    const textColor = theme === 'dark' ? '#94a3b8' : '#475569';
    if (classDistChart) {
        classDistChart.options.plugins.legend.labels.color = textColor;
        classDistChart.update();
    }
    if (activityTrendChart) {
        activityTrendChart.options.plugins.legend.labels.color = textColor;
        activityTrendChart.options.scales.x.ticks.color = textColor;
        activityTrendChart.options.scales.y.ticks.color = textColor;
        activityTrendChart.update();
    }
}

/* ==========================================================================
   4. Admin Panel & History
   ========================================================================== */
let adminCurrentPage = 1;
const adminPageSize = 10;
let adminSearchQuery = '';
let adminCategoryFilter = 'All';

function initAdminPanel() {
    const searchInput = document.getElementById('adminSearchInput');
    const categorySelect = document.getElementById('adminCategoryFilter');
    const refreshBtn = document.getElementById('refreshAdminBtn');
    const prevPageBtn = document.getElementById('prevPageBtn');
    const nextPageBtn = document.getElementById('nextPageBtn');

    searchInput?.addEventListener('input', debounce((e) => {
        adminSearchQuery = e.target.value.trim();
        adminCurrentPage = 1;
        refreshAdminTable();
    }, 350));

    categorySelect?.addEventListener('change', (e) => {
        adminCategoryFilter = e.target.value;
        adminCurrentPage = 1;
        refreshAdminTable();
    });

    refreshBtn?.addEventListener('click', () => {
        refreshAdminTable();
        showToast('Records refreshed', 'info');
    });

    prevPageBtn?.addEventListener('click', () => {
        if (adminCurrentPage > 1) {
            adminCurrentPage--;
            refreshAdminTable();
        }
    });

    nextPageBtn?.addEventListener('click', () => {
        adminCurrentPage++;
        refreshAdminTable();
    });

    refreshAdminTable();
}

async function refreshAdminTable() {
    const tableBody = document.getElementById('adminTableBody');
    const paginationInfo = document.getElementById('paginationInfo');
    const currentPageBadge = document.getElementById('currentPageBadge');
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');

    if (!tableBody) return;

    const offset = (adminCurrentPage - 1) * adminPageSize;
    const url = new URL('/api/admin/predictions', window.location.origin);
    url.searchParams.set('limit', adminPageSize);
    url.searchParams.set('offset', offset);
    if (adminSearchQuery) url.searchParams.set('search', adminSearchQuery);
    if (adminCategoryFilter && adminCategoryFilter !== 'All') url.searchParams.set('category', adminCategoryFilter);

    try {
        const res = await fetch(url);
        const data = await res.json();
        if (!res.ok || !data.success) return;

        const items = data.items || [];
        const total = data.total || 0;
        const totalPages = Math.ceil(total / adminPageSize) || 1;

        if (items.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center py-4 text-muted">
                        <i class="fa-solid fa-inbox"></i> No prediction records found matching the criteria.
                    </td>
                </tr>
            `;
        } else {
            tableBody.innerHTML = items.map(item => {
                let badgeClass = 'badge-success';
                if (item.prediction === 'Hate Speech') badgeClass = 'badge-danger';
                else if (item.prediction === 'Offensive Language') badgeClass = 'badge-warning';

                const truncatedText = item.user_text.length > 70 
                    ? item.user_text.substring(0, 70) + '...' 
                    : item.user_text;

                return `
                    <tr>
                        <td><strong>#${item.id}</strong></td>
                        <td title="${escapeHtml(item.user_text)}">${escapeHtml(truncatedText)}</td>
                        <td><span class="badge ${badgeClass}">${item.prediction}</span></td>
                        <td><strong>${item.confidence}%</strong></td>
                        <td>${item.timestamp}</td>
                        <td>
                            <button class="action-btn-del" title="Delete record" onclick="window.handleDeleteRecord(${item.id})">
                                <i class="fa-solid fa-trash-can"></i>
                            </button>
                        </td>
                    </tr>
                `;
            }).join('');
        }

        // Update Pagination Controls
        if (paginationInfo) {
            const start = total === 0 ? 0 : offset + 1;
            const end = Math.min(offset + items.length, total);
            paginationInfo.textContent = `Showing ${start}-${end} of ${total} records`;
        }
        if (currentPageBadge) currentPageBadge.textContent = `Page ${adminCurrentPage} of ${totalPages}`;
        if (prevBtn) prevBtn.disabled = adminCurrentPage <= 1;
        if (nextBtn) nextBtn.disabled = adminCurrentPage >= totalPages;

    } catch (err) {
        console.error('Admin Fetch Error:', err);
    }
}

window.handleDeleteRecord = async function(id) {
    if (!confirm(`Are you sure you want to delete prediction record #${id}?`)) return;

    try {
        const res = await fetch(`/api/admin/predictions/${id}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            showToast(`Record #${id} deleted`, 'success');
            refreshAdminTable();
            refreshDashboardStats();
        } else {
            showToast('Failed to delete record', 'danger');
        }
    } catch (err) {
        console.error('Delete Error:', err);
        showToast('Error deleting record', 'danger');
    }
};

/* ==========================================================================
   5. Model Benchmarks & Metrics
   ========================================================================== */
async function initModelInfo() {
    try {
        const res = await fetch('/api/model/info');
        const data = await res.json();
        if (!res.ok || !data.success) return;

        const lrAcc = document.getElementById('lrAccuracyVal');
        const dtAcc = document.getElementById('dtAccuracyVal');

        if (lrAcc && data.logistic_regression?.accuracy) {
            lrAcc.textContent = `${data.logistic_regression.accuracy}%`;
        }
        if (dtAcc && data.decision_tree?.accuracy) {
            dtAcc.textContent = `${data.decision_tree.accuracy}%`;
        }
    } catch (err) {
        console.error('Model Info Fetch Error:', err);
    }
}

/* ==========================================================================
   6. Contact & Mobile Navigation Utilities
   ========================================================================== */
window.handleContactSubmit = function() {
    const name = document.getElementById('contactName')?.value;
    const form = document.getElementById('contactForm');
    showToast(`Thank you, ${name || 'User'}! Your message has been received.`, 'success');
    if (form) form.reset();
};

function initMobileNav() {
    const mobileBtn = document.getElementById('mobileMenuBtn');
    const navLinks = document.getElementById('navLinks');

    mobileBtn?.addEventListener('click', () => {
        navLinks?.classList.toggle('show');
    });

    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            navLinks?.classList.remove('show');
        });
    });
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let icon = 'fa-circle-info text-primary';
    if (type === 'success') icon = 'fa-circle-check text-success';
    else if (type === 'danger' || type === 'error') icon = 'fa-circle-xmark text-danger';
    else if (type === 'warning') icon = 'fa-triangle-exclamation text-warning';

    toast.innerHTML = `
        <i class="fa-solid ${icon}"></i>
        <span>${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

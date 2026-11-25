document.addEventListener('DOMContentLoaded', () => {
  const payloadNode = document.getElementById('rainfall-analytics-data');
  if (!payloadNode) return;

  let series;
  try {
    series = JSON.parse(payloadNode.textContent);
  } catch (_error) {
    return;
  }

  if (!window.Chart) {
    document.querySelectorAll('.dashboard-chart-plot').forEach((plot) => {
      plot.innerHTML = '<p class="chart-library-error">Chart.js could not load. Check your connection and refresh the dashboard.</p>';
    });
    return;
  }

  const axisColor = '#8390a5';
  const gridColor = 'rgba(139, 157, 183, .17)';
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { intersect: false, mode: 'index' },
    plugins: {
      legend: { labels: { color: '#51617b', usePointStyle: true, boxWidth: 8, padding: 16 } },
      tooltip: { backgroundColor: '#14233c', padding: 11, cornerRadius: 9 },
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: axisColor, maxRotation: 0 } },
      y: { beginAtZero: true, grid: { color: gridColor }, ticks: { color: axisColor, precision: 0 } },
    },
  };

  const draw = (canvasId, config) => {
    const canvas = document.getElementById(canvasId);
    if (canvas) new window.Chart(canvas, config);
  };

  draw('rainfall-seven-day-chart', {
    type: 'line',
    data: {
      labels: (series.recent?.labels || []).map((day) => day.slice(5)),
      datasets: [{
        label: 'Rainfall (mm)',
        data: series.recent?.values || [],
        borderColor: '#3387dc',
        backgroundColor: 'rgba(51, 135, 220, .12)',
        fill: true,
        tension: .34,
        borderWidth: 2.5,
        pointRadius: 3,
        pointHoverRadius: 5,
        pointBackgroundColor: '#fff',
        pointBorderColor: '#3387dc',
      }],
    },
    options: {
      ...commonOptions,
      scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, title: { display: true, text: 'mm', color: axisColor } } },
    },
  });

  const monthlyCanvas = document.getElementById('rainfall-monthly-chart');
  if (monthlyCanvas) {
    const context = monthlyCanvas.getContext('2d');
    const gradient = context.createLinearGradient(0, 0, 0, 240);
    gradient.addColorStop(0, '#58a9f5');
    gradient.addColorStop(1, '#286fc2');
    draw('rainfall-monthly-chart', {
      type: 'bar',
      data: {
        labels: series.monthly?.labels || [],
        datasets: [{ label: 'Rainfall (mm)', data: series.monthly?.values || [], backgroundColor: gradient, borderRadius: 7, maxBarThickness: 34 }],
      },
      options: {
        ...commonOptions,
        plugins: { ...commonOptions.plugins, legend: { display: false } },
        scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, title: { display: true, text: 'mm', color: axisColor } } },
      },
    });
  }

  draw('rainfall-comparison-chart', {
    type: 'line',
    data: {
      labels: series.comparison?.labels || [],
      datasets: [
        { label: 'Actual rainy share (%)', data: series.comparison?.actual || [], borderColor: '#3387dc', backgroundColor: '#3387dc', tension: .25, borderWidth: 2.5, pointRadius: 3 },
        { label: 'Model-predicted rainy share (%)', data: series.comparison?.predicted || [], borderColor: '#25a781', backgroundColor: '#25a781', tension: .25, borderWidth: 2.5, borderDash: [6, 4], pointRadius: 3 },
      ],
    },
    options: {
      ...commonOptions,
      scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, max: 100, ticks: { ...commonOptions.scales.y.ticks, callback: (value) => `${value}%` } } },
    },
  });
});

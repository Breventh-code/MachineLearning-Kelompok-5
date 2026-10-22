/* Prediksi dan visualisasi menggunakan data aktual API; tidak ada hasil buatan. */
(() => {
  const form = document.getElementById('featureForm');
  if (!form) return;
  const errorNode = document.getElementById('apiError');
  const submitButton = document.getElementById('submitBtn');
  const resultEmpty = document.getElementById('resultEmpty');
  const resultLoaded = document.getElementById('resultLoaded');
  const predictionLabel = document.getElementById('predictionLabel');
  const predictionCaption = document.getElementById('predictionCaption');
  const predictionNote = document.getElementById('predictionNote');
  const interpretationTitle = document.getElementById('interpretationTitle');
  const interpretationDescription = document.getElementById('interpretationDescription');
  const interpretationDetail = document.getElementById('interpretationDetail');
  const interpretationCaution = document.getElementById('interpretationCaution');
  const chartTitle = document.getElementById('chartTitle');
  const chartFootnote = document.getElementById('chartFootnote');
  const canvas = document.getElementById('resultChart');
  let chartData = null;

  function hideError() {
    errorNode.textContent = '';
    errorNode.classList.add('is-hidden');
    form.querySelectorAll('.field-error').forEach(node => node.textContent = '');
    form.querySelectorAll('input').forEach(node => node.classList.remove('invalid'));
  }

  function showError(message, details) {
    errorNode.textContent = message || 'Terjadi kesalahan. Silakan coba lagi.';
    errorNode.classList.remove('is-hidden');
    if (details) {
      Object.entries(details).forEach(([field, text]) => {
        const label = form.querySelector(`[data-error-for="${field}"]`);
        const input = form.elements.namedItem(field);
        if (label) label.textContent = String(text);
        if (input && input.classList) input.classList.add('invalid');
      });
    }
  }

  function chartBars(data, maxValue, formatValue) {
    if (!data || !data.length) {
      chartData = null;
      canvas.classList.add('is-hidden');
      return;
    }
    chartData = {data, maxValue, formatValue};
    canvas.classList.remove('is-hidden');
    drawChart();
  }

  function drawChart() {
    if (!chartData) return;
    const {data, maxValue, formatValue} = chartData;
    const width = Math.max(260, canvas.parentElement.clientWidth - 2);
    const height = Math.max(180, data.length * 58 + 28);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const left = 8;
    const right = 70;
    const usableWidth = Math.max(60, width - left - right);
    data.forEach((item, i) => {
      const top = i * 58 + 12;
      const safeValue = Number.isFinite(item.value) ? Math.max(0, item.value) : 0;
      const scale = Math.max(0, Math.min(1, maxValue ? safeValue / maxValue : 0));
      ctx.font = '12px system-ui, sans-serif';
      ctx.textBaseline = 'top';
      ctx.textAlign = 'left';
      ctx.fillStyle = '#cdd8e5';
      ctx.fillText(String(item.label).slice(0, 32), left, top);
      ctx.fillStyle = '#25364a';
      ctx.fillRect(left, top + 22, usableWidth, 10);
      ctx.fillStyle = '#49d9b5';
      ctx.fillRect(left, top + 22, Math.max(0, usableWidth * scale), 10);
      ctx.textAlign = 'right';
      ctx.fillStyle = '#e5f1fa';
      ctx.font = 'bold 12px system-ui, sans-serif';
      ctx.fillText(formatValue(safeValue), width - 9, top + 19);
    });
  }

  // Informasi ini menjelaskan hasil model; bukan pengganti hasil prediksi API.
  // Semua nilai probabilitas dan jarak berasal dari API, bukan angka buatan.
  function updateInterpretation(title, description, detail, caution) {
    if (interpretationTitle) interpretationTitle.textContent = title;
    if (interpretationDescription) interpretationDescription.textContent = description;
    if (interpretationDetail) interpretationDetail.textContent = detail;
    if (interpretationCaution) interpretationCaution.textContent = caution;
  }

  function supervisedExplanation(result) {
    const label = String(result.label || result.prediction || '').trim();
    const key = label.toUpperCase().replace(/[\s_-]+/g, '');
    const explanations = {
      NORMAL: {
        title: 'Prediksi: Bearing Normal',
        description: 'Random Forest menilai pola fitur getaran input paling mirip dengan sampel bearing berkondisi Normal yang dipelajari saat training.'
      },
      BALL: {
        title: 'Prediksi: Ball Fault (Elemen Bola)',
        description: 'Random Forest mengklasifikasikan getaran sebagai Ball Fault, yaitu indikasi gangguan pada elemen bola yang bergulir di dalam bearing.'
      },
      IR: {
        title: 'Prediksi: Inner Race Fault (Jalur Dalam)',
        description: 'Random Forest mengklasifikasikan pola getaran sebagai Inner Race Fault (IR), yaitu indikasi gangguan pada jalur cincin bagian dalam bearing.'
      },
      INNERRACE: {
        title: 'Prediksi: Inner Race Fault (Jalur Dalam)',
        description: 'Random Forest mengklasifikasikan pola getaran sebagai Inner Race Fault (IR), yaitu indikasi gangguan pada jalur cincin bagian dalam bearing.'
      },
      OR: {
        title: 'Prediksi: Outer Race Fault (Jalur Luar)',
        description: 'Random Forest mengklasifikasikan pola getaran sebagai Outer Race Fault (OR), yaitu indikasi gangguan pada jalur cincin bagian luar bearing.'
      },
      OUTERRACE: {
        title: 'Prediksi: Outer Race Fault (Jalur Luar)',
        description: 'Random Forest mengklasifikasikan pola getaran sebagai Outer Race Fault (OR), yaitu indikasi gangguan pada jalur cincin bagian luar bearing.'
      }
    };
    const explain = explanations[key] || {
      title: `Prediksi model: ${label || 'Tidak diketahui'}`,
      description: 'Kelas hasil prediksi ini ditentukan oleh Random Forest dari fitur statistik getaran yang diberikan.'
    };
    const candidates = Array.isArray(result.probabilities) ? result.probabilities : [];
    const selected = candidates.find(item => String(item.label).trim().toUpperCase() === label.toUpperCase());
    const probability = selected ? Number(selected.probability) : NaN;
    const detail = Number.isFinite(probability)
      ? `Probabilitas kelas ${label}: ${(probability * 100).toFixed(1)}%. Persentase ini menunjukkan keluaran model pada input saat ini, bukan akurasi keseluruhan model ataupun tingkat keparahan kerusakan.`
      : 'Probabilitas per kelas tidak tersedia pada respons model. Interpretasi didasarkan pada label prediksi.';
    updateInterpretation(
      explain.title,
      explain.description,
      detail,
      'Catatan: Hasil ini merupakan prediksi berbasis dataset CWRU, bukan konfirmasi kondisi fisik mesin. Validasi lapangan memerlukan pemeriksaan lebih lanjut.'
    );
  }

  // Interpretasi khusus untuk model K-Means saat ini berdasarkan tabulasi 460 data testing
  // yang diberikan tim: Cluster 0 (Ball 138, IR 105, Normal 46, OR 46),
  // Cluster 1 (OR 92, IR 33). Jangan perlakukan komposisi ini sebagai probabilitas input.
  function unsupervisedExplanation(result) {
    const cluster = Number(result.cluster);
    const descriptions = {
      0: {
        title: 'Cluster 0 — Kelompok Pola Getaran Campuran',
        description: 'Sampel ini lebih dekat ke centroid Cluster 0. Pada evaluasi 460 data testing, Cluster 0 berisi 138 Ball, 105 Inner Race, 46 Normal, dan 46 Outer Race. Artinya kelompok ini memiliki campuran kondisi, bukan kelompok bearing normal.',
        detail: 'Komposisi dalam Cluster 0: Ball 41,2%, IR 31,3%, Normal 13,7%, dan OR 13,7% (total 335 data). Ini adalah proporsi data evaluasi, bukan peluang kondisi untuk input yang sedang diuji.'
      },
      1: {
        title: 'Cluster 1 — Kelompok yang Didominasi Outer Race',
        description: 'Sampel ini lebih dekat ke centroid Cluster 1. Pada evaluasi 460 data testing, Cluster 1 memuat 92 sampel Outer Race dan 33 Inner Race; tidak ada sampel Ball atau Normal.',
        detail: 'Komposisi dalam Cluster 1: OR 73,6% dan IR 26,4% (total 125 data). Dominasi OR tidak berarti input ini pasti mengalami Outer Race Fault.'
      }
    };
    const interpretation = descriptions[cluster] || {
      title: `Cluster ${Number.isFinite(cluster) ? cluster : 'tidak diketahui'} — Kelompok Getaran`,
      description: 'Model K-Means menempatkan input pada centroid yang paling dekat dalam ruang fitur. Interpretasi komposisi kelas untuk nomor cluster ini belum tersedia.',
      detail: 'Untuk mengetahui karakter cluster baru, bandingkan hasil clustering dengan label data testing dalam tabel silang.'
    };
    const distances = Array.isArray(result.distances) ? result.distances : [];
    const chosen = distances.find(item => Number(item.cluster) === cluster);
    const distance = chosen ? Number(chosen.distance) : NaN;
    const distanceNote = Number.isFinite(distance)
      ? ` Jarak ke centroid terpilih: ${distance.toFixed(3)} pada ruang fitur terskala. Nilai ini bukan persentase keyakinan ataupun tingkat kerusakan.`
      : ' Jarak centroid tidak tersedia untuk hasil ini.';
    updateInterpretation(
      interpretation.title,
      interpretation.description,
      interpretation.detail + distanceNote,
      'Catatan: Nomor cluster bukan diagnosis. Komposisi di atas mengacu pada 460 data testing dari model K-Means saat ini; jika model di-training ulang atau nomor cluster berubah, interpretasi harus dihitung ulang.'
    );
  }

  function renderSupervised(result) {
    predictionCaption.textContent = 'KLASIFIKASI KONDISI';
    predictionLabel.textContent = result.label || result.prediction;
    predictionNote.textContent = result.note || '';
    supervisedExplanation(result);
    chartTitle.textContent = 'Probabilitas tiap kelas';
    const points = (result.probabilities || []).map(item => ({
      label: item.label, value: Number(item.probability)
    }));
    chartBars(points, 1, value => `${(value * 100).toFixed(1)}%`);
    chartFootnote.textContent = points.length
      ? 'Probabilitas keluaran classifier. Interpretasi tergantung kualitas data dan model.'
      : 'Model tidak menyediakan predict_proba(), sehingga grafik probabilitas tidak tersedia.';
  }

  function renderUnsupervised(result) {
    predictionCaption.textContent = 'CLUSTER TERPILIH';
    predictionLabel.textContent = result.label || `Cluster ${result.cluster}`;
    predictionNote.textContent = result.note || '';
    unsupervisedExplanation(result);
    chartTitle.textContent = 'Jarak ke pusat cluster';
    const points = (result.distances || []).map(item => ({
      label: `Cluster ${item.cluster}`, value: Number(item.distance)
    }));
    const max = Math.max(...points.map(p => p.value), 0.000001);
    chartBars(points, max, value => value.toFixed(2));
    chartFootnote.textContent = points.length
      ? 'Jarak lebih kecil berarti lebih dekat ke pusat cluster dalam ruang fitur model.'
      : 'Model tidak menyediakan transform(), sehingga grafik jarak tidak tersedia.';
  }

  form.addEventListener('submit', async event => {
    event.preventDefault();
    hideError();
    const features = {};
    let isValid = true;
    const inputs = [...form.querySelectorAll('input[name]')];
    for (const input of inputs) {
      const text = input.value.trim();
      const number = Number(text);
      if (!text || !Number.isFinite(number) || !input.checkValidity()) {
        isValid = false;
        input.classList.add('invalid');
        const label = form.querySelector(`[data-error-for="${input.name}"]`);
        if (label) label.textContent = 'Isi dengan angka valid sesuai ketentuan.';
      } else {
        features[input.name] = number;
      }
    }
    if (!isValid) {
      showError('Beberapa fitur kosong atau tidak valid. Periksa form sebelum melanjutkan.');
      return;
    }

    submitButton.disabled = true;
    submitButton.textContent = 'Memproses…';
    try {
      const response = await fetch(form.dataset.endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
        body: JSON.stringify({features}),
      });
      const body = await response.json();
      if (!response.ok || !body.ok) {
        showError(body.error || `Permintaan gagal (HTTP ${response.status}).`, body.details);
        return;
      }
      resultEmpty.classList.add('is-hidden');
      resultLoaded.classList.remove('is-hidden');
      if (form.dataset.mode === 'supervised') renderSupervised(body.result);
      else renderUnsupervised(body.result);
    } catch (_error) {
      showError('Tidak dapat terhubung ke Flask API. Pastikan server masih berjalan.');
    } finally {
      submitButton.disabled = false;
      submitButton.innerHTML = form.dataset.mode === 'supervised'
        ? 'Jalankan klasifikasi <span>→</span>' : 'Cari cluster <span>→</span>';
    }
  });

  form.addEventListener('reset', () => {
    hideError();
    chartData = null;
    resultLoaded.classList.add('is-hidden');
    resultEmpty.classList.remove('is-hidden');
  });
  window.addEventListener('resize', drawChart);
})();

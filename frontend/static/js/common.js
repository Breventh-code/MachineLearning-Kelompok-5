/* Global model status; frontend tidak membuat prediksi palsu bila file model belum ada. */
(async function updateModelStatus() {
  const nodes = document.querySelectorAll('[data-model-status]');
  if (!nodes.length) return;
  try {
    const response = await fetch('/api/status', {headers: {'Accept': 'application/json'}});
    if (!response.ok) throw new Error('Gagal membaca status server.');
    const data = await response.json();
    for (const node of nodes) {
      const kind = node.dataset.modelStatus;
      const item = data.models && data.models[kind];
      if (!item) continue;
      node.textContent = item.ready ? '● Model siap digunakan' : '● Model belum terpasang';
      node.classList.toggle('is-ready', Boolean(item.ready));
      node.classList.toggle('is-missing', !item.ready);
      node.title = item.message;
    }
  } catch (error) {
    nodes.forEach(node => {
      node.textContent = '● Status server tidak tersedia';
      node.classList.add('is-missing');
    });
  }
})();

/* BIZON promo banner + timed modal */
(function(){
  const SITE_URL = 'https://bizon-tech.com/?utm_source=gpu-benchmark&utm_medium=app&utm_campaign=free-tool';
  const BANNER_KEY = 'bizon_banner_closed_at';
  const MODAL_LAST_KEY = 'bizon_modal_last_shown_at';
  const MODAL_INTERVAL_MS = 60 * 1000; // 60s
  const BANNER_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

  function now(){ return Date.now(); }

  function shouldShowBanner(){
    try{
      const closedAt = parseInt(localStorage.getItem(BANNER_KEY) || '0', 10);
      return !closedAt || (now() - closedAt) > BANNER_TTL_MS;
    }catch{ return true; }
  }

  function createBanner(){
    if (!shouldShowBanner()) return;
    const bar = document.createElement('div');
    bar.id = 'bizon-banner';
    bar.style.cssText = 'position:sticky;top:0;z-index:1031;background:#0d6efd;color:#fff;padding:.35rem .75rem;display:flex;align-items:center;justify-content:center;gap:.5rem;box-shadow:0 1px 4px rgba(0,0,0,.15)';
    const link = document.createElement('a');
    link.href = SITE_URL;
    link.target = '_blank';
    link.rel = 'noopener';
    link.style.cssText = 'color:#fff;text-decoration:underline;font-weight:600';
    link.textContent = 'BIZON Workstations for AI — Learn more';
    const close = document.createElement('button');
    close.type = 'button';
    close.ariaLabel = 'Close';
    close.style.cssText = 'margin-left:8px;background:transparent;border:0;color:#fff;font-size:1rem;line-height:1;';
    close.innerHTML = '&times;';
    close.addEventListener('click', () => {
      try{ localStorage.setItem(BANNER_KEY, String(now())); }catch{}
      bar.remove();
    });
    bar.append('Powered by ', link, close);
    // insert after navbar if present, else at top of body
    const nav = document.querySelector('.app-navbar');
    if (nav && nav.parentNode){ nav.parentNode.insertBefore(bar, nav.nextSibling); }
    else { document.body.prepend(bar); }
  }

  function createModal(){
    if (document.getElementById('bizonPromoModal')) return;
    const wrapper = document.createElement('div');
    wrapper.innerHTML = `
<div class="modal fade" id="bizonPromoModal" tabindex="-1" aria-hidden="true">
  <div class="modal-dialog modal-dialog-centered">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Supercharge your AI workloads</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <p>Get a high-performance BIZON workstation optimized for LLMs, Vision, and Training.</p>
        <a class="btn btn-primary" href="${SITE_URL}" target="_blank" rel="noopener">Explore Workstations</a>
      </div>
    </div>
  </div>
</div>`;
    document.body.appendChild(wrapper);
  }

  function canShowModal(){
    try{
      const last = parseInt(localStorage.getItem(MODAL_LAST_KEY) || '0', 10);
      return !last || (now() - last) >= MODAL_INTERVAL_MS;
    }catch{ return true; }
  }

  function showModalIfNeeded(){
    if (!document.hidden && canShowModal()){
      const el = document.getElementById('bizonPromoModal');
      if (!el) return;
      if (typeof bootstrap === 'undefined' || !bootstrap.Modal) return;
      const modal = bootstrap.Modal.getOrCreateInstance(el);
      modal.show();
      try{ localStorage.setItem(MODAL_LAST_KEY, String(now())); }catch{}
    }
  }

  // Init
  document.addEventListener('DOMContentLoaded', () => {
    try{
      // Beta branch: show only the banner; disable recurring modal popups
      createBanner();
    }catch{}
  });
})();

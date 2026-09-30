/* VBench-2.0 on HunyuanVideo-1.5 with the SAGE sampler and the Qwen3.5-9B
   reward. Each spoke carries its own range -- the lowest and highest value any
   row reaches on that dimension -- so dimensions on different scales stay
   readable. Colour does one job: TVRL against the field. */
(function () {
  "use strict";

  var AXES = [
    {k: 'Overall',      lo: 54.09, hi: 57.69, d: 2},
    {k: 'Creativity',   lo: 41.40, hi: 47.36, d: 2},
    {k: 'Common sense', lo: 62.75, hi: 64.88, d: 2},
    {k: 'Control',      lo: 30.26, hi: 31.64, d: 2},
    {k: 'Human action', lo: 88.85, hi: 90.76, d: 2},
    {k: 'Physics',      lo: 45.74, hi: 54.37, d: 2}
  ];
  /*                       Overall Creat. Comm.  Ctrl.  Human  Phys. */
  var SERIES = [
    {name: 'TVRL',      note: 'ours, Qwen3.5-9B critic', v0: '--s1', w: 2.6, dash: '', dots: true,
     v: [57.69, 47.36, 64.31, 31.64, 90.76, 54.37]},
    {name: 'SAGE-GRPO', note: 'same reward, uniform credit', v0: '--s2', w: 1.6, dash: '7 4',
     v: [54.54, 41.68, 64.88, 31.57, 88.85, 45.74]},
    {name: 'Base model', note: 'HunyuanVideo-1.5', v0: '--s3', w: 1.6, dash: '3 4',
     v: [54.09, 41.40, 62.75, 30.26, 88.94, 47.11]}
  ];

  var CX = 280, CY = 232, R = 158, INNER = 0.14;
  var NS = 'http://www.w3.org/2000/svg';
  function el(n, a) {
    var e = document.createElementNS(NS, n);
    for (var k in a) e.setAttribute(k, a[k]);
    return e;
  }
  function pt(i, t) {
    var ang = -Math.PI / 2 + (i / AXES.length) * Math.PI * 2;
    var rr = R * (INNER + (1 - INNER) * t);
    return [CX + Math.cos(ang) * rr, CY + Math.sin(ang) * rr];
  }
  function norm(i, v) {
    var a = AXES[i];
    return Math.max(0, Math.min(1, (v - a.lo) / (a.hi - a.lo)));
  }

  window.TVRLRadar = function (svg, legendHost) {
    if (!svg) return;
    var cs = getComputedStyle(svg.parentElement || svg);
    SERIES.forEach(function (s) {
      s.color = (cs.getPropertyValue(s.v0) || '#FF6A2C').trim();
    });
    var ring = (cs.getPropertyValue('--ringc') || 'rgba(255,255,255,.12)').trim();
    [0.25, 0.5, 0.75, 1].forEach(function (t) {
      svg.appendChild(el('polygon', {
        points: AXES.map(function (_, i) { return pt(i, t).join(','); }).join(' '),
        fill: 'none', stroke: ring}));
    });
    AXES.forEach(function (a, i) {
      var p0 = pt(i, 0), p1 = pt(i, 1);
      svg.appendChild(el('line', {x1: p0[0], y1: p0[1], x2: p1[0], y2: p1[1], stroke: ring}));
      var lp = pt(i, 1.17);
      var anchor = Math.abs(lp[0] - CX) < 6 ? 'middle' : (lp[0] > CX ? 'start' : 'end');
      var t1 = el('text', {x: lp[0], y: lp[1], class: 'axis', 'text-anchor': anchor});
      t1.textContent = a.k;
      svg.appendChild(t1);
      var t2 = el('text', {x: lp[0], y: lp[1] + 14, class: 'rng', 'text-anchor': anchor});
      t2.textContent = a.lo.toFixed(a.d) + ' \u2013 ' + a.hi.toFixed(a.d);
      svg.appendChild(t2);
    });
    SERIES.slice().reverse().forEach(function (s) {
      var pts = s.v.map(function (v, i) { return pt(i, norm(i, v)); });
      svg.appendChild(el('polygon', {
        points: pts.map(function (p) { return p.join(','); }).join(' '),
        fill: s.color, 'fill-opacity': s.dots ? 0.17 : 0.045,
        stroke: s.color, 'stroke-width': s.w, 'stroke-dasharray': s.dash,
        'stroke-linejoin': 'round'
      }));
      if (s.dots) {
        pts.forEach(function (p) {
          svg.appendChild(el('circle', {cx: p[0], cy: p[1], r: 3.6, fill: s.color,
            stroke: (cs.getPropertyValue('--dotring') || '#050505').trim(), 'stroke-width': 1.5}));
        });
      }
    });
    if (legendHost) {
      SERIES.forEach(function (s) {
        var d = document.createElement('div');
        var i = document.createElement('i');
        i.style.background = s.dash
          ? 'repeating-linear-gradient(90deg,' + s.color + ' 0 4px,transparent 4px 8px)'
          : s.color;
        var t = document.createElement('span');
        t.innerHTML = '<b></b><span class="note"></span>';
        t.querySelector('b').innerHTML = (window.TVRLUI && window.TVRLUI.dagHTML)
          ? window.TVRLUI.dagHTML(s.name) : s.name;
        t.querySelector('.note').textContent = s.note;
        d.appendChild(i); d.appendChild(t);
        legendHost.appendChild(d);
      });
    }
  };
})();

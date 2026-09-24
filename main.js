/* Q Project — theme, hero constellation, copy buttons, docs scrollspy */
(function () {
  "use strict";

  var root = document.documentElement;

  /* ---------- Theme toggle ---------- */
  var toggle = document.querySelector(".theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("q-theme", next); } catch (e) {}
      if (typeof window.__qReadColors === "function") window.__qReadColors();
    });
  }

  /* ---------- Language toggle (RU/EN/ZH), shared across every page ---------- */
  var langGroup = document.getElementById("langGroup");
  if (langGroup) {
    var KNOWN_LANGS = ["ru", "en", "zh"];
    var setLang = function (v) {
      document.querySelectorAll("[data-lang]").forEach(function (el) {
        el.classList.toggle("is-active", el.getAttribute("data-lang") === v);
      });
      document.querySelectorAll("#langGroup button").forEach(function (b) {
        b.classList.toggle("is-on", b.getAttribute("data-v") === v);
      });
      document.documentElement.setAttribute("lang", v === "zh" ? "zh-CN" : v);
      try { localStorage.setItem("q-lang", v); } catch (e) {}
    };
    langGroup.addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-v]");
      if (btn) setLang(btn.getAttribute("data-v"));
    });
    var savedLang = null;
    try { savedLang = localStorage.getItem("q-lang"); } catch (e) {}
    var navLang = (navigator.language || "en").slice(0, 2);
    setLang(savedLang && KNOWN_LANGS.indexOf(savedLang) >= 0 ? savedLang : (KNOWN_LANGS.indexOf(navLang) >= 0 ? navLang : "en"));
  }

  /* ---------- Footer year ---------- */
  var yearNow = new Date().getFullYear();
  document.querySelectorAll(".year, #year").forEach(function (el) { el.textContent = yearNow; });

  /* ---------- Waitlist (front-end stub) ---------- */
  var form = document.getElementById("waitlist");
  if (form) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var input = form.querySelector("#email");
      var note = form.querySelector(".waitlist__note");
      var value = (input.value || "").trim();
      var ok = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
      if (!ok) {
        note.style.color = "#e05a5a";
        note.textContent = "Enter a valid email so we can reach you.";
        input.focus();
        return;
      }
      note.style.color = "";
      note.textContent = "You're on the list — we'll email " + value + " when access opens.";
      form.reset();
    });
  }

  /* ---------- Copy buttons on code blocks ---------- */
  document.querySelectorAll(".code").forEach(function (block) {
    var btn = block.querySelector(".code__copy");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var text = (block.getAttribute("data-copy") || "").replace(/&quot;/g, '"');
      var done = function () {
        var old = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = old; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, done);
      } else {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(ta); done();
      }
    });
  });

  /* ---------- Docs scrollspy ---------- */
  var docNav = document.querySelector(".docs__nav");
  if (docNav && "IntersectionObserver" in window) {
    var links = {};
    docNav.querySelectorAll("a").forEach(function (a) {
      var id = a.getAttribute("href").replace("#", "");
      links[id] = a;
    });
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          Object.keys(links).forEach(function (k) { links[k].classList.remove("is-active"); });
          if (links[en.target.id]) links[en.target.id].classList.add("is-active");
        }
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    document.querySelectorAll(".doc-block[id]").forEach(function (b) { spy.observe(b); });
  }

  /* ---------- Hero constellation ---------- */
  var canvas = document.querySelector(".hero__field");
  if (!canvas) return;

  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  var ctx = canvas.getContext("2d");
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var w = 0, h = 0, nodes = [], raf = null, mouse = { x: -999, y: -999 };
  var col = { dot: "", glow: "", line: "", dotA: 0.55, lineA: 0.3 };

  function readColors() {
    var cs = getComputedStyle(root);
    col.dot = cs.getPropertyValue("--dot-rgb").trim();
    col.glow = cs.getPropertyValue("--dot-glow-rgb").trim();
    col.line = cs.getPropertyValue("--line-rgb").trim();
    col.dotA = parseFloat(cs.getPropertyValue("--dot-alpha")) || 0.55;
    col.lineA = parseFloat(cs.getPropertyValue("--line-alpha")) || 0.3;
    if (reduce.matches) draw();
  }
  window.__qReadColors = readColors;

  function size() {
    var rect = canvas.getBoundingClientRect();
    w = rect.width; h = rect.height;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var count = Math.max(28, Math.min(90, Math.round((w * h) / 16000)));
    nodes = [];
    for (var i = 0; i < count; i++) {
      nodes.push({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.28, vy: (Math.random() - 0.5) * 0.28,
        r: Math.random() * 1.6 + 0.6
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    var link = 128;
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > w) n.vx *= -1;
      if (n.y < 0 || n.y > h) n.vy *= -1;

      for (var j = i + 1; j < nodes.length; j++) {
        var m = nodes[j];
        var dx = n.x - m.x, dy = n.y - m.y;
        var d = Math.sqrt(dx * dx + dy * dy);
        if (d < link) {
          var a = (1 - d / link) * col.lineA;
          ctx.strokeStyle = "rgba(" + col.line + "," + a + ")";
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(n.x, n.y); ctx.lineTo(m.x, m.y); ctx.stroke();
        }
      }

      var mdx = n.x - mouse.x, mdy = n.y - mouse.y;
      var glow = Math.sqrt(mdx * mdx + mdy * mdy) < 150;
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = glow ? "rgba(" + col.glow + ", 0.95)" : "rgba(" + col.dot + "," + col.dotA + ")";
      ctx.fill();
    }
  }

  function frame() { draw(); raf = requestAnimationFrame(frame); }

  function start() {
    size();
    if (reduce.matches) { draw(); return; }
    if (!raf) raf = requestAnimationFrame(frame);
  }

  window.addEventListener("resize", function () {
    if (raf) { cancelAnimationFrame(raf); raf = null; }
    start();
  });

  var hero = document.querySelector(".hero");
  hero.addEventListener("pointermove", function (e) {
    var rect = canvas.getBoundingClientRect();
    mouse.x = e.clientX - rect.left; mouse.y = e.clientY - rect.top;
  });
  hero.addEventListener("pointerleave", function () { mouse.x = -999; mouse.y = -999; });

  if ("IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { if (!raf && !reduce.matches) raf = requestAnimationFrame(frame); }
        else if (raf) { cancelAnimationFrame(raf); raf = null; }
      });
    }, { threshold: 0 }).observe(hero);
  }

  readColors();
  start();
})();

# Q Project — website

Static marketing site for [Q Project](https://huggingface.co/q-project), published at **https://q-project-lm.github.io/**.

GitHub Pages publishes `main` from the repository root and runs Jekyll to expand the shared header in `_includes/header.html`. The remaining site is plain HTML/CSS/JS.

## Files
- `index.html` — landing page
- `docs.html` — documentation (quickstart, model reference, fine-tuning, roadmap, API)
- `styles.css` — all styling; theme tokens are at the top (`:root` = light, `[data-theme="dark"]` = dark)
- `main.js` — theme toggle, hero constellation canvas, code copy buttons, docs scrollspy, waitlist
- `assets/q-logo.svg` — logo mark (favicon + in-page)
- `assets/og-image.png` — social share image

## Theming
Light is the default. The toggle in the nav flips light/dark and stores the choice in
`localStorage` under `q-theme`; an inline script in each page's `<head>` applies it before
paint to avoid a flash. To change the default, edit that `|| "light"` fallback in the head
script of each page. Canvas particle colors are read from CSS variables,
so they follow the theme automatically.

## Run locally
With Jekyll installed, run `jekyll serve` and open `http://localhost:4000`. A plain static server does not expand the shared header include.

## Deploy
GitHub Pages deploys `main` from `/` at `https://q-project-lm.github.io/`. No custom domain is configured.

## Known limitation
- **Waitlist** (`main.js`) is a front-end stub — it validates the email and shows a confirmation but does not store anything. Wire the form `submit` to a real endpoint (Formspree, Buttondown, ConvertKit, or your own API) so emails are captured.
- Update `og:url`/`og:image` if the domain changes.

# Q Project — website

Static marketing site for [Q Project](https://huggingface.co/q-project), served at **q.lakomoor.com**.

No build step. Plain HTML/CSS/JS.

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
script of `index.html` and `docs.html`. Canvas particle colors are read from CSS variables,
so they follow the theme automatically.

## Run locally
```bash
python3 -m http.server 8791
# open http://localhost:8791
```

## Deploy
Upload the folder to any static host:
- **Cloudflare Pages / Netlify / Vercel:** drag-and-drop the folder, or connect a repo. Build command: none. Output dir: `/`.
- **GitHub Pages / nginx:** serve the folder root.

Then point `q.lakomoor.com` (CNAME) at the host.

## To finish before launch
- **Waitlist** (`main.js`) is a front-end stub — it validates the email and shows a confirmation but does not store anything. Wire the form `submit` to a real endpoint (Formspree, Buttondown, ConvertKit, or your own API) so emails are captured.
- Update `og:url`/`og:image` if the domain changes.
- The "Q · Next" card intentionally hides details of the model in training.

// One record per visitor per day. That is the whole design.
//
// WHY NOT A COUNTER. KV has no atomic increment: two visitors landing at the
// same moment both read the old number and both write the same new one, and one
// of them is lost. A counter would drift downward forever and there would be no
// way to tell by how much. Counting distinct KEYS instead is exact, because
// writing the same key twice is idempotent.
//
// WHY ONE PER DAY AND NOT ONE PER VIEW. KV's free tier allows a limited number
// of writes a day. If this page ever reaches a front page, one write per view
// would exhaust that in minutes and the tracking would silently stop --
// precisely when the numbers matter most. One write per visitor per day holds
// under a traffic spike.
//
// NO IP IS EVER STORED. The key is a SHA-256 of the address, the user agent and
// the date, truncated. Same visitor on the same day collapses to the same key;
// tomorrow they are a different key and nothing links the two. So this can
// answer "how many people came today" and cannot answer "was this the same
// person as yesterday", which is the right trade for a page like this.

const TTL = 60 * 60 * 24 * 180;   // keep half a year of history

async function visitorKey(request, day) {
  const ip = request.headers.get("cf-connecting-ip") || "";
  const ua = request.headers.get("user-agent") || "";
  const data = new TextEncoder().encode(`${ip}|${ua}|${day}|neurogecko`);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(digest)].slice(0, 10)
    .map((b) => b.toString(16).padStart(2, "0")).join("");
}

function refHost(referrer) {
  if (!referrer) return "direct";
  try {
    const h = new URL(referrer).hostname.replace(/^www\./, "");
    return h.endsWith("neurogecko.pages.dev") ? "direct" : h;
  } catch { return "direct"; }
}

export async function onRequestPost({ request, env }) {
  const res = new Response(JSON.stringify({ ok: true }), {
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
  if (!env.NG) return res;

  let body = {};
  try { body = await request.json(); } catch { /* a beacon may send nothing */ }

  const day = new Date().toISOString().slice(0, 10);
  const key = `v:${day}:${await visitorKey(request, day)}`;

  // Already seen today? Then this is a repeat view and costs no write.
  if (await env.NG.get(key)) return res;

  await env.NG.put(key, JSON.stringify({
    c: (request.cf && request.cf.country) || "??",
    r: refHost(body.ref || request.headers.get("referer")),
    t: new Date().toISOString(),
  }), { expirationTtl: TTL });

  return res;
}

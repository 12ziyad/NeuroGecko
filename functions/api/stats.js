// The numbers behind the admin page. Nothing here answers without the key.
//
// The key lives as a Pages secret (ADMIN_KEY), not in this file and not in the
// repository, because the repository is public. It is compared in constant time
// -- a plain `===` on a secret leaks its length and a little of its content to
// anyone patient enough to time the responses, and that is a free thing to
// avoid.

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}

function sameSecret(a, b) {
  if (typeof a !== "string" || typeof b !== "string") return false;
  const A = new TextEncoder().encode(a), B = new TextEncoder().encode(b);
  let diff = A.length ^ B.length;
  for (let i = 0; i < Math.max(A.length, B.length); i++) {
    diff |= (A[i] || 0) ^ (B[i] || 0);
  }
  return diff === 0;
}

async function listAll(env, prefix, cap = 20000) {
  const out = [];
  let cursor;
  do {
    const page = await env.NG.list({ prefix, cursor, limit: 1000 });
    out.push(...page.keys);
    cursor = page.list_complete ? null : page.cursor;
  } while (cursor && out.length < cap);
  return out;
}

export async function onRequestGet({ request, env }) {
  const key = new URL(request.url).searchParams.get("k") || "";
  if (!env.ADMIN_KEY) return json({ ok: false, error: "no ADMIN_KEY configured" }, 503);
  if (!sameSecret(key, env.ADMIN_KEY)) return json({ ok: false, error: "nope" }, 401);
  if (!env.NG) return json({ ok: false, error: "no KV bound" }, 503);

  // ---- visitors, one key per person per day ----------------------------
  const visitKeys = await listAll(env, "v:");
  const byDay = {};
  for (const k of visitKeys) {
    const day = k.name.slice(2, 12);
    byDay[day] = (byDay[day] || 0) + 1;
  }

  // Countries and referrers need the VALUES, so only read the recent ones --
  // a full read of every visitor since launch would get slower every week for
  // an answer nobody looks at.
  const recent = visitKeys
    .filter((k) => k.name.slice(2, 12) >= isoDaysAgo(30))
    .slice(-1200);
  const country = {}, ref = {};
  await Promise.all(recent.map(async (k) => {
    try {
      const v = JSON.parse(await env.NG.get(k.name) || "{}");
      if (v.c) country[v.c] = (country[v.c] || 0) + 1;
      if (v.r) ref[v.r] = (ref[v.r] || 0) + 1;
    } catch { /* a malformed record is not worth failing the page over */ }
  }));

  // ---- people who asked for one, and what they said --------------------
  const petKeys = await listAll(env, "pet:");
  const TEST = /@(example\.(com|net|org)|test|invalid|localhost)$/i;
  const wantOne = petKeys.filter((k) => !TEST.test(k.name.slice(4))).length;

  const msgKeys = (await listAll(env, "msg:")).slice(-60).reverse();
  const messages = [];
  for (const k of msgKeys) {
    try { messages.push(JSON.parse(await env.NG.get(k.name) || "{}")); }
    catch { /* skip */ }
  }

  const days = Object.keys(byDay).sort();
  const today = isoDaysAgo(0);
  const last7 = days.filter((d) => d >= isoDaysAgo(6))
                    .reduce((s, d) => s + byDay[d], 0);

  return json({
    ok: true,
    totalVisitors: visitKeys.length,
    today: byDay[today] || 0,
    last7,
    daily: days.map((d) => ({ day: d, n: byDay[d] })),
    countries: top(country, 12),
    referrers: top(ref, 12),
    wantOne,
    messages,
    note: "A visitor is one person on one day. Repeat views the same day are "
        + "not counted again, and no IP address is stored anywhere.",
  });
}

function isoDaysAgo(n) {
  const d = new Date(Date.now() - n * 86400000);
  return d.toISOString().slice(0, 10);
}

function top(obj, n) {
  return Object.entries(obj).sort((a, b) => b[1] - a[1]).slice(0, n)
    .map(([name, count]) => ({ name, count }));
}

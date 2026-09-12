// Contact and "want one", as a Cloudflare Pages Function.
//
// WHAT IT DOES. POST stores the message in KV and bumps a counter; GET returns
// the counter so the page can show how many people have asked. If a mail
// provider key is configured it also sends the message on. If one is NOT
// configured it says so in the response rather than returning success -- the
// page then offers the visitor their own mail client, which is the honest
// failure. Nothing here ever reports "sent" for something it did not send.
//
// TO TURN ON EMAIL: add one secret in the Cloudflare dashboard for this Pages
// project, Settings -> Environment variables:
//
//     RESEND_API_KEY = re_xxxxxxxxxxxx      (from resend.com, free tier)
//
// and, if the sending domain is not verified, leave MAIL_FROM unset so it uses
// Resend's own onboarding sender. Nothing else needs changing; the messages are
// already being stored in KV either way and can be read back at any time.

const TO = "ejziyad@gmail.com";
const MAX = 4000;

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

export async function onRequestGet({ env }) {
  if (!env.NG) return json({ ok: true, count: null });
  const count = parseInt((await env.NG.get("pet_count")) || "0", 10) || 0;
  return json({ ok: true, count });
}

export async function onRequestPost({ request, env }) {
  let body;
  try {
    body = await request.json();
  } catch {
    return json({ ok: false, error: "expected JSON" }, 400);
  }

  const kind = body.kind === "adopt" ? "adopt" : "contact";
  const email = String(body.email || "").trim().slice(0, 200);
  const message = String(body.message || "").trim().slice(0, MAX);
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
    return json({ ok: false, error: "that email does not look right" }, 400);
  }

  const at = new Date().toISOString();
  const ray = request.headers.get("cf-ray") || Math.random().toString(36).slice(2);
  const country = request.cf && request.cf.country;

  // 1. Keep it. This happens whether or not mail is configured, so a message is
  //    never lost just because a key is missing.
  let count = null;
  if (env.NG) {
    await env.NG.put(`msg:${at}:${ray}`, JSON.stringify(
      { kind, email, message, at, country, page: String(body.page || "").slice(0, 300) }));
    if (kind === "adopt") {
      // One per email, so the number means people rather than button presses.
      const seen = await env.NG.get(`pet:${email.toLowerCase()}`);
      count = parseInt((await env.NG.get("pet_count")) || "0", 10) || 0;
      if (!seen) {
        count += 1;
        await env.NG.put(`pet:${email.toLowerCase()}`, at);
        await env.NG.put("pet_count", String(count));
      }
    }
  }

  // 2. Send it on, if there is anything to send it with.
  const key = env.RESEND_API_KEY;
  if (!key) {
    return json({
      ok: false,
      stored: !!env.NG,
      count,
      error: "mail is not configured on this deployment",
    }, 503);
  }

  const subject = kind === "adopt"
    ? `NeuroGecko: someone wants one${count ? ` (#${count})` : ""}`
    : "NeuroGecko: a message";
  const text = [
    kind === "adopt" ? "Someone asked for a gecko of their own." : "A message from the site.",
    "",
    `from:    ${email}`,
    country ? `country: ${country}` : null,
    `when:    ${at}`,
    `page:    ${String(body.page || "")}`,
    count ? `count:   ${count} people have now asked` : null,
    "",
    message || "(no message)",
  ].filter(Boolean).join("\n");

  try {
    const r = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        authorization: `Bearer ${key}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        from: env.MAIL_FROM || "NeuroGecko <onboarding@resend.dev>",
        to: [TO],
        reply_to: email,
        subject,
        text,
      }),
    });
    if (!r.ok) {
      const detail = (await r.text()).slice(0, 300);
      return json({ ok: false, stored: !!env.NG, count, error: detail }, 502);
    }
  } catch (err) {
    return json({ ok: false, stored: !!env.NG, count, error: String(err) }, 502);
  }

  return json({ ok: true, count });
}

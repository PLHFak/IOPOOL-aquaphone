// Page « Démarrer l'assistant » hébergée sur Vercel (le PC peut être éteint).
// Secrets dans les variables d'environnement Vercel (jamais dans Git) : RUNPOD_API_KEY, AQUAPHONE_MOT_DE_PASSE.
// Le pod se met en veille tout seul (runpod/veille.sh, avec la clé propre au pod) : ici on ne fait que démarrer.
import crypto from "node:crypto";

export const VOLUME = "hm9zp32n6a", CENTRE = "EU-RO-1", NOM = "aquaphone", ANCIEN_POD = "hn5obumrqbyns1";
export const IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04";
// Par ordre de préférence ; la 1re libre est prise (voir runpod/a_la_demande.py).
export const CARTES = [["NVIDIA GeForce RTX 4090", 2], ["NVIDIA GeForce RTX 4090", 1], ["NVIDIA RTX PRO 4500 Blackwell", 1],
  ["NVIDIA GeForce RTX 5090", 1], ["NVIDIA A100 80GB PCIe", 1], ["NVIDIA A100-SXM4-80GB", 1],
  ["NVIDIA RTX PRO 6000 Blackwell Server Edition", 1]];
const COOKIE = "aquaphone_demarrage";

const secret = () => crypto.createHash("sha256").update("aquaphone-demarrage|" + (process.env.AQUAPHONE_MOT_DE_PASSE || "")).digest();
const jeton = () => crypto.createHmac("sha256", secret()).update("connecte").digest("hex");

export function connecte(req) {
  const c = (req.headers.cookie || "").split(/;\s*/).find((x) => x.startsWith(COOKIE + "="));
  const v = c ? c.slice(COOKIE.length + 1) : "";
  const j = jeton();
  return v.length === j.length && crypto.timingSafeEqual(Buffer.from(v), Buffer.from(j));
}

export function motDePasseOk(saisi) {
  const a = Buffer.from(crypto.createHash("sha256").update(String(saisi || "").trim().toLowerCase()).digest());
  const b = Buffer.from(crypto.createHash("sha256").update(String(process.env.AQUAPHONE_MOT_DE_PASSE || "").trim().toLowerCase()).digest());
  return Boolean(process.env.AQUAPHONE_MOT_DE_PASSE) && crypto.timingSafeEqual(a, b);
}

export function poserCookie(res) {
  res.setHeader("Set-Cookie", `${COOKIE}=${jeton()}; Path=/; Max-Age=${30 * 24 * 3600}; HttpOnly; Secure; SameSite=Lax`);
}

export async function graphql(query, variables = {}) {
  const r = await fetch("https://api.runpod.io/graphql", {
    method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.RUNPOD_API_KEY}` },
    body: JSON.stringify({ query, variables }),
  });
  const d = await r.json();
  if (d.errors) throw new Error(d.errors[0].message || "erreur RunPod");
  return d.data;
}

export async function podActuel() {
  const d = await graphql("query { myself { pods { id name desiredStatus machine { gpuDisplayName } runtime { uptimeInSeconds } } } }");
  return d.myself.pods.find((p) => p.name.startsWith(NOM) && p.desiredStatus === "RUNNING" && p.id !== ANCIEN_POD) || null;
}

export const adresse = (id) => `https://${id}-8765.proxy.runpod.net/`;

export async function serviceRepond(id) {
  try {
    const r = await fetch(adresse(id) + "connexion", { headers: { "User-Agent": "Mozilla/5.0 aquaphone" }, signal: AbortSignal.timeout(8000) });
    return r.status === 200;
  } catch { return false; }
}

import { adresse, connecte, podActuel, serviceRepond } from "./_commun.js";

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  if (!connecte(req)) return res.status(401).json({ phase: "connexion" });
  try {
    const p = await podActuel();
    if (!p) return res.json({ phase: "arrete" });
    const pret = await serviceRepond(p.id);
    res.json({ phase: pret ? "pret" : "demarrage", pod: p.id, adresse: adresse(p.id),
               cartes: p.machine && p.machine.gpuDisplayName, depuis_s: p.runtime ? p.runtime.uptimeInSeconds : 0 });
  } catch (e) {
    res.status(502).json({ phase: "erreur", message: String(e.message || e) });
  }
}

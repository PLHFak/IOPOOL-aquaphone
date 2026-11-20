import { motDePasseOk, poserCookie } from "./_commun.js";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ erreur: "POST" });
  const mdp = (req.body && req.body.mot_de_passe) || "";
  if (!motDePasseOk(mdp)) {
    await new Promise((r) => setTimeout(r, 1500)); // freine les essais au hasard
    return res.status(401).json({ erreur: "Mot de passe incorrect" });
  }
  poserCookie(res);
  res.json({ ok: true });
}

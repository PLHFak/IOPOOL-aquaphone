import { CARTES, CENTRE, IMAGE, NOM, VOLUME, adresse, connecte, graphql, podActuel } from "./_commun.js";

const CREER = `mutation($i: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $i) { id machine { gpuDisplayName } } }`;

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ erreur: "POST" });
  if (!connecte(req)) return res.status(401).json({ phase: "connexion" });
  try {
    const deja = await podActuel();
    if (deja) return res.json({ phase: "demarrage", pod: deja.id, adresse: adresse(deja.id) });
    const erreurs = [];
    for (const [carte, n] of CARTES) {
      const i = {
        name: NOM, cloudType: "SECURE", gpuCount: n, gpuTypeId: carte, dataCenterId: CENTRE, networkVolumeId: VOLUME,
        volumeMountPath: "/workspace", containerDiskInGb: 20, imageName: IMAGE, ports: "22/tcp,8765/http", startSsh: true,
        // le service démarre tout seul et le pod se met en veille tout seul (runpod/au_demarrage.sh, veille.sh)
        dockerArgs: "bash -c 'bash /workspace/iopool-aquaphone/runpod/au_demarrage.sh; exec /start.sh'",
        env: [{ key: "PUBLIC_KEY", value: process.env.CLE_SSH_PUBLIQUE || "" }],
      };
      try {
        const p = (await graphql(CREER, { i })).podFindAndDeployOnDemand;
        return res.json({ phase: "demarrage", pod: p.id, adresse: adresse(p.id), cartes: `${n} × ${carte.replace("NVIDIA ", "")}` });
      } catch (e) { erreurs.push(`${n} × ${carte} : ${e.message}`); }
    }
    res.status(503).json({ phase: "erreur", message: "Aucune carte graphique libre en ce moment, réessayez dans quelques minutes." , detail: erreurs });
  } catch (e) {
    res.status(502).json({ phase: "erreur", message: String(e.message || e) });
  }
}

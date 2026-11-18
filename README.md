# Aquaphone

> **« Aquaphone de Marc »** — l'assistant IA vocal de votre pisciniste.

Le client de piscine ouvre l'application sur son iPhone, voit le visage de son pisciniste (photo animée), lui parle et obtient une réponse vocale personnalisée, basée sur les données de **sa** piscine.

Projet IOPOOL / SuperIopool — THE FAKTORY.

---

## 1. Nom et identité

| Élément | Décision |
|---|---|
| Marque | **Aquaphone** (prononciation « a-kwa-fone », compréhensible dans toutes les langues) |
| Déclinaison | **Aquaphone de [Prénom]** — ex. Aquaphone de Marc, de Sophie, de Julien |
| Présentation à l'écran | « Marc, votre pisciniste, et son assistant » |
| Variante écartée | « Aquoiphone » : même son, mais orthographe non intuitive (recherche App Store / Google) et connotation « à quoi bon » |
| Accroche possible | « Aquaphone de Marc — à quoi bon attendre le lundi ? » |
| À faire | Vérifier disponibilité marque (INPI / EUIPO) et nom de domaine — « aquaphone » désigne aussi un instrument de musique |

---

## 2. Parcours client

1. **Accueil** — « Vous êtes en contact avec l'assistant IA de Marc, votre pisciniste. »
2. **Mise à jour des données** — « J'actualise la connaissance de votre piscine… »
3. **Collecte si pas de sonde iopool** — l'assistant le signale et demande :
   - une photo de la bandelette de test à côté du pot (échelle de couleurs) ;
   - la température de l'eau.
4. **Reformulation** — une phrase courte, seulement si la question est ambiguë.
5. **Réponse** — vocale, avec le visage animé du pisciniste.
6. **Escalade** — si la réponse ne satisfait pas : message envoyé au pisciniste (nommé), qui recontactera le client. Le client peut continuer la conversation en attendant.

---

## 3. Architecture (hypothèse labo)

```
iPhone (web app / PWA)
   │  WebRTC (voix + vidéo, ~1–2 Mbit/s par session)
   ▼
PC labo — RTX 4090 24 Go (fibre)
   ├─ Reconnaissance vocale FR (type Whisper)
   ├─ Synthèse vocale FR — voix clonée du pisciniste
   └─ Animation de la photo (type LivePortrait / MuseTalk)
   │
   ▼
Cerveau : Claude (API)
   ├─ Base de connaissances (RAG) : modes d'emploi, FAQ, traitements, historique des échanges
   ├─ Données récentes de la piscine du client (sonde iopool)
   └─ Analyse photo bandelette (vision)
```

**Pourquoi le cerveau dans le cloud :** un modèle local tenant dans 24 Go à côté de la voix et de l'avatar est nettement moins fiable — compréhension, lecture de bandelette, et surtout risque d'erreur sur un dosage chimique.

### Limites connues du PC labo
- 1 conversation à la fois (2–3 max) — l'animation est la brique la plus gourmande.
- Indisponible quand le PC sert à autre chose (conférences, jeux de caméra) ou redémarre.
- Production à l'échelle → GPU en ligne.

### Objectif de performance
Délai entre la fin de la phrase du client et le début de la réponse : **< 1,5 s** (acceptable jusqu'à 2 s). Nécessite un traitement en flux (réponse générée et prononcée au fil de l'eau).

---

## 4. Labo — premier test

| Élément | Choix |
|---|---|
| Visage | Pierre, dans le rôle du pisciniste (selfie de face, bouche fermée, 1536×2048) |
| Voix | Vidéo de Pierre (~25 s, AAC 44,1 kHz) — à compléter idéalement jusqu'à 1–2 min |
| Cas client | « Mon eau est verte depuis deux jours, j'ai mis du chlore choc et ça ne change rien. » |
| Mesures | Délai à chaque étape : reconnaissance vocale → Claude → synthèse → animation |
| Critères | Réalisme (« est-ce bien Pierre qui parle ? ») et temps de réponse |

Règle : pas d'utilisation du visage ou de la voix d'un tiers sans son accord écrit.

---

## 5. Conformité

- **Transparence IA** (règlement européen sur l'IA) : le client doit savoir qu'il parle à une IA, a fortiori avec un visage animé réaliste.
- **Droit à l'image et à la voix** : accord écrit du pisciniste avant toute utilisation de sa photo et de sa voix.

---

## 6. Prochaines étapes

- [ ] Kit de test sur le PC RTX 4090 (briques à installer + script de bout en bout + mesure des délais)
- [ ] Test labo avec le visage et la voix de Pierre
- [ ] Vérification marque et domaine « Aquaphone »
- [ ] Constitution de la base de connaissances (modes d'emploi, FAQ, traitements)
- [ ] Application iPhone (web app) par-dessus le pipeline validé

---

*Dernière mise à jour : 18 novembre 2026*

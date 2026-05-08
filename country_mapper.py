# country_mapper.py

BENIN_NEWSPAPERS = [
    "24 Heures au Bénin",
    "Agence Bénin Presse",
    "ASSALAM",
    "Banouto",
    "Bénin Intelligent",
    "Benin Web TV",
    "Boulevard des Infos",
    "Daabaaru",
    "Daho-Express",
    "Ehuzu",
    "Fraternité",
    "Islam Hebdo",
    "L'Evénement Précis",
    "L'investigateur",
    "La Nation",
    "La Nouvelle Tribune",
    "La Perche du Nord",
    "Le Béninois Libéré",
    "Le Leader Info Bénin",
    "Le Matinal",
    "Le Parakois",
    "Le Potentiel",
    "Les 4 Vérités",
    "Les Pharaons",
    "Matin Libre",
    "OLOFOFO",
    "Visages du Bénin",
]

BURKINA_FASO_NEWSPAPERS = [
    "Agence d'Information du Burkina",
    "Al Mawadda",
    "An-Nasr Trimestriel",
    "An-Nasr Vendredi",
    "Burkina 24",
    "Carrefour africain",
    "Faso Actu",
    "FasoZine",
    "Infowakat",
    "L'Appel",
    "L'Autre Regard",
    "L'Evénement",
    "L'Observateur",
    "L'Observateur Paalga",
    "La Preuve",
    "Le CERFIste",
    "Le Pays",
    "Le vrai visage de l'islam",
    "LeFaso.net",
    "Mutations",
    "San Finna",
    "Sidwaya",
    "Wakat Séra",
]

COTE_DIVOIRE_NEWSPAPERS = [
    "AJMCI Infos",
    "Al Minbar",
    "Al Muwassat Info",
    "Alif",
    "Agence Ivoirienne de Presse",
    "APA News",
    "Fraternité Hebdo",
    "Fraternité Matin",
    "Ivoire Dimanche",
    "L'Alternative",
    "L'Expression",
    "L'Intelligent d'Abidjan",
    "L'Inter",
    "La Voie",
    "Le Débat Ivoirien",
    "Le Jour",
    "Le Jour Plus",
    "Le Nouveau Réveil",
    "Le Nouvel Horizon",
    "Le Patriote",
    "Le Temps",
    "Nord-Sud",
    "Notre Temps",
    "Notre Voie",
    "Plume Libre",
    "Al-Azan",
    "Allahou Akbar",
    "Bulletin d'information du CNI",
    "Islam Info",
    "Les Échos de l'AEEMCI",
]

NIGER_NEWSPAPERS = [
    "Al Maoulid Info",
    "Al Maoulid Magazine",
    "Al Maoulid Magazine (arabe)",
    "Le Sahel",
]

TOGO_NEWSPAPERS = [
    "AfreePress",
    "Agence Togolaise de Presse",
    "Atopani Express",
    "Courrier du Golfe",
    "Forum Hebdo",
    "L'éveil du Peuple",
    "La Dépêche",
    "La Lettre de Tchaoudjo",
    "La Nouvelle Marche",
    "La Voix de la Nation",
    "Le Démocrate",
    "Le Pacific",
    "Le Rendez-Vous",
    "Togo-Presse",
]

# Reverse-index built once at import: name → country. O(1) lookup vs the
# previous O(n) sequential ``elif`` chain.
_NEWSPAPER_TO_COUNTRY: dict[str, str] = {
    **{name: "Benin" for name in BENIN_NEWSPAPERS},
    **{name: "Burkina Faso" for name in BURKINA_FASO_NEWSPAPERS},
    **{name: "Côte d'Ivoire" for name in COTE_DIVOIRE_NEWSPAPERS},
    **{name: "Niger" for name in NIGER_NEWSPAPERS},
    **{name: "Togo" for name in TOGO_NEWSPAPERS},
}


def get_country_from_newspaper(newspaper_name: str) -> str:
    """Return the country for a known newspaper, or ``""`` if unknown."""
    return _NEWSPAPER_TO_COUNTRY.get(newspaper_name, "")

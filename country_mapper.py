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

NEWSPAPER_TO_COUNTRY = {
    # Example:
    # "Le Monde": "France",
    # "The Guardian": "United Kingdom",
    # "New York Times": "United States",
    # Add mappings for other countries here
}

def get_country_from_newspaper(newspaper_name: str) -> str:
    """
    Retrieves the country for a given newspaper name.
    Checks against lists for specific countries first, then a general dictionary.
    Returns the country name or an empty string if not found.
    """
    if newspaper_name in BENIN_NEWSPAPERS:
        return "Benin"
    elif newspaper_name in BURKINA_FASO_NEWSPAPERS:
        return "Burkina Faso"
    elif newspaper_name in COTE_DIVOIRE_NEWSPAPERS:
        return "Côte d'Ivoire"
    elif newspaper_name in NIGER_NEWSPAPERS:
        return "Niger"
    elif newspaper_name in TOGO_NEWSPAPERS:
        return "Togo"
    return NEWSPAPER_TO_COUNTRY.get(newspaper_name, "")

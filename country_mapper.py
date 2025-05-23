# country_mapper.py

BENIN_NEWSPAPERS = [
    "24h au Bénin",
    "Agence Bénin Presse",
    "Banouto",
    "Bénin Intelligent",
    "Boulevard des Infos",
    "Daho-Express",
    "Ehuzu",
    "Fraternité",
    "L'Evénement Précis",
    "La Nation",
    "La Nouvelle Tribune",
    "Le Matinal",
    "Les Pharaons",
    "Matin Libre",
]

BURKINA_FASO_NEWSPAPERS = [
    "Burkina 24",
    "Carrefour africain",
    "FasoZine",
    "L'Evénement",
    "L'Observateur",
    "L'Observateur Paalga",
    "Le Pays",
    "LeFaso.net",
    "Mutations",
    "San Finna",
    "Sidwaya",
]

COTE_DIVOIRE_NEWSPAPERS = [
    "Agence Ivoirienne de Presse",
    "Fraternité Hebdo",
    "Fraternité Matin",
    "Ivoire Dimanche",
    "L'Alternative",
    "L'Intelligent d'Abidjan",
    "La Voie",
    "Le Jour",
    "Le Jour Plus",
    "Le Nouvel Horizon",
    "Le Patriote",
    "Notre Temps",
    "Notre Voie",
]

NIGER_NEWSPAPERS = [
    "Le Sahel",
]

TOGO_NEWSPAPERS = [
    "Agence Togolaise de Presse",
    "Atopani Express",
    "Courrier du Golfe",
    "Forum Hebdo",
    "L'éveil du Peuple",
    "La Lettre de Tchaoudjo",
    "La Nouvelle Marche",
    "Le Démocrate",
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
